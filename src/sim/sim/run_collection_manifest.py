#!/usr/bin/env python3
"""Run and update rollout rows from a collection manifest."""

from __future__ import annotations

import argparse
import copy
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from sim.collect_task_spec_rollouts import (
    Rollout,
    default_bridge_prepare_command,
    load_yaml as load_task_spec_yaml,
    run_rollout,
)
from sim.validate_collected_rollout import validate_rollout_dir


DELETE_FIELD = object()


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def write_yaml_atomic(path: Path, data: dict[str, Any], worker_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{worker_id}" if worker_id else f".{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_path = path.with_suffix(path.suffix + suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)
    tmp_path.replace(path)


def now_string() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@contextmanager
def manifest_lock(manifest_path: Path, worker_id: str | None, timeout_s: float, poll_s: float = 0.1):
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    owner = f"{worker_id or 'worker'} pid={os.getpid()} time={now_string()}\n"
    deadline = time.monotonic() + timeout_s
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, owner.encode("utf-8"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for manifest lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def find_task(task_spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in task_spec.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    available = [task.get("task_id", "<missing>") for task in task_spec.get("tasks", [])]
    raise ValueError(f"task_id {task_id!r} not found. Available: {available}")


def find_variant(task: dict[str, Any], variant_id: str) -> dict[str, Any]:
    for variant in task.get("trajectory_variants") or []:
        if variant.get("variant_id") == variant_id:
            return variant
    if variant_id == "nominal":
        return {"variant_id": "nominal", "variant_type": "nominal"}
    available = [variant.get("variant_id", "<missing>") for variant in task.get("trajectory_variants") or []]
    raise ValueError(f"variant_id {variant_id!r} not found. Available: {available}")


def row_ready(row: dict[str, Any], include_pose_not_ready: bool) -> bool:
    if include_pose_not_ready:
        return True
    return bool(row.get("pose_variant_ready", True))


def row_status_selected(row: dict[str, Any], retry_failed: bool) -> bool:
    status = str(row.get("status", "pending"))
    return status == "pending" or (retry_failed and status == "failed")


def row_matches_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    assigned_worker = row.get("assigned_worker")
    if args.worker_id and assigned_worker and assigned_worker != args.worker_id:
        return False
    if args.rollout_id and row.get("rollout_id") not in set(args.rollout_id):
        return False
    if args.task_id and row.get("task_id") not in set(args.task_id):
        return False
    if args.variant_id and row.get("variant_id") not in set(args.variant_id):
        return False
    if args.scene_id and row.get("scene_id") not in set(args.scene_id):
        return False
    if args.only_visual_id and row.get("visual_id") not in set(args.only_visual_id):
        return False
    return True


def selected_row_indices(manifest: dict[str, Any], args: argparse.Namespace) -> list[int]:
    selected = []
    for index, row in enumerate(manifest.get("rollouts", [])):
        if not isinstance(row, dict):
            continue
        if not row_status_selected(row, args.retry_failed):
            continue
        if not row_ready(row, args.include_pose_not_ready):
            continue
        if not row_matches_filters(row, args):
            continue
        selected.append(index)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def first_selected_row_index(manifest: dict[str, Any], args: argparse.Namespace) -> int | None:
    indices = selected_row_indices(manifest, argparse.Namespace(**{**vars(args), "limit": 1}))
    return indices[0] if indices else None


def claim_next_row(manifest_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], int] | None:
    with manifest_lock(manifest_path, args.worker_id, args.manifest_lock_timeout_s):
        manifest = load_yaml(manifest_path)
        row_index = first_selected_row_index(manifest, args)
        if row_index is None:
            return None
        row = manifest["rollouts"][row_index]
        row["status"] = "running"
        row["worker_id"] = args.worker_id
        row["claimed_time"] = now_string()
        row["started_time"] = row["claimed_time"]
        row.pop("error", None)
        row.pop("validation", None)
        update_counts(manifest)
        write_yaml_atomic(manifest_path, manifest, args.worker_id)
        return copy.deepcopy(row), row_index


def update_manifest_row_by_id(
    manifest_path: Path,
    rollout_id: str,
    updates: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    with manifest_lock(manifest_path, args.worker_id, args.manifest_lock_timeout_s):
        manifest = load_yaml(manifest_path)
        for row in manifest.get("rollouts", []):
            if isinstance(row, dict) and row.get("rollout_id") == rollout_id:
                for key, value in updates.items():
                    if value is DELETE_FIELD:
                        row.pop(key, None)
                    else:
                        row[key] = value
                update_counts(manifest)
                write_yaml_atomic(manifest_path, manifest, args.worker_id)
                return
        raise ValueError(f"rollout_id {rollout_id!r} not found while updating manifest")


def runtime_task_spec_for_row(
    row: dict[str, Any],
    runtime_dir: Path,
    worker_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_spec_path = Path(str(row["task_spec"])).expanduser().resolve()
    source_spec = load_task_spec_yaml(source_spec_path)
    task = copy.deepcopy(find_task(source_spec, str(row["task_id"])))
    variant = copy.deepcopy(find_variant(task, str(row.get("variant_id", "nominal"))))
    task["trajectory_variants"] = [variant]

    runtime_spec = copy.deepcopy(source_spec)
    runtime_spec["tasks"] = [task]
    runtime_spec["suite_id"] = str(row.get("scene_id") or source_spec.get("suite_id") or source_spec_path.stem)
    runtime_spec["manifest_row"] = {
        "rollout_id": row.get("rollout_id"),
        "trajectory_name": row.get("trajectory_name"),
        "visual_id": row.get("visual_id"),
        "worker_id": worker_id,
        "source_task_spec": source_spec_path.as_posix(),
    }

    scene = dict(runtime_spec.get("scene", {}))
    layout_yaml = row.get("layout_yaml")
    usd_path = row.get("visual_usd") or row.get("generated_usd")
    if layout_yaml:
        scene["source_yaml"] = str(layout_yaml)
        scene["generated_layout_yaml"] = str(layout_yaml)
    if usd_path:
        scene["generated_usd"] = str(usd_path)
    runtime_spec["scene"] = scene

    if isinstance(row.get("collection"), dict):
        collection = dict(runtime_spec.get("collection", {}))
        for key, value in row["collection"].items():
            if value is not None:
                collection[key] = value
        runtime_spec["collection"] = collection

    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{row['rollout_id']}_task_spec.yaml"
    write_yaml_atomic(runtime_path, runtime_spec, worker_id)
    return runtime_path, runtime_spec, task, variant


def bridge_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        isaac_root=args.isaac_root,
        bridge_command_file=args.bridge_command_file,
        bridge_status_file=args.bridge_status_file,
        prepare_timeout_s=args.prepare_timeout_s,
        prepare_topic_timeout_s=args.prepare_topic_timeout_s,
        skip_prepare_topic_wait=args.skip_prepare_topic_wait,
    )


def update_counts(manifest: dict[str, Any]) -> None:
    rollouts = [row for row in manifest.get("rollouts", []) if isinstance(row, dict)]
    counts = dict(manifest.get("counts", {}))
    counts.update(
        {
            "rollouts": len(rollouts),
            "pending": sum(1 for row in rollouts if row.get("status", "pending") == "pending"),
            "running": sum(1 for row in rollouts if row.get("status") == "running"),
            "complete": sum(1 for row in rollouts if row.get("status") == "complete"),
            "failed": sum(1 for row in rollouts if row.get("status") == "failed"),
            "skipped": sum(1 for row in rollouts if row.get("status") == "skipped"),
            "claimed_workers": sorted(
                set(str(row.get("worker_id")) for row in rollouts if row.get("worker_id"))
            ),
        }
    )
    manifest["counts"] = counts
    manifest["updated_time"] = now_string()


def run_manifest_row(
    manifest_path: Path,
    manifest: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
) -> bool:
    row = manifest["rollouts"][row_index]
    return run_claimed_row(manifest_path, copy.deepcopy(row), args, already_marked_running=False)


def effective_base_dir(args: argparse.Namespace, collection: dict[str, Any]) -> Path:
    base_dir = args.base_dir or Path(str(collection.get("base_dir", "sim_datasets/generated")))
    if args.worker_id and args.base_dir is None:
        base_dir = base_dir / args.worker_id
    return base_dir


def run_claimed_row(
    manifest_path: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    already_marked_running: bool,
) -> bool:
    runtime_path, runtime_spec, task, variant = runtime_task_spec_for_row(
        row,
        args.runtime_spec_dir,
        args.worker_id,
    )
    rollout = Rollout(
        task=task,
        variant=variant,
        trajectory_name=str(row.get("trajectory_name") or row["rollout_id"]),
    )
    collection = runtime_spec.get("collection", {})
    base_dir = effective_base_dir(args, collection)
    duration_s = float(args.duration_s if args.duration_s is not None else collection.get("duration_s", 20.0))
    rollout_dir = base_dir / rollout.trajectory_name
    bridge_prepare = None
    if args.use_isaac_bridge:
        bridge_prepare = default_bridge_prepare_command(
            bridge_args(args),
            runtime_path,
            runtime_spec,
            rollout,
        )

    if not already_marked_running:
        update_manifest_row_by_id(
            manifest_path,
            str(row["rollout_id"]),
            {
                "status": "running",
                "worker_id": args.worker_id,
                "started_time": now_string(),
                "runtime_task_spec": runtime_path.as_posix(),
                "error": DELETE_FIELD,
                "validation": DELETE_FIELD,
            },
            args,
        )
    else:
        update_manifest_row_by_id(
            manifest_path,
            str(row["rollout_id"]),
            {"runtime_task_spec": runtime_path.as_posix()},
            args,
        )

    try:
        ok = run_rollout(
            task_spec_path=runtime_path,
            task_spec=runtime_spec,
            rollout=rollout,
            base_dir=base_dir,
            dataset_name=args.dataset_name,
            duration_s=duration_s,
            startup_wait_s=float(args.startup_wait_s),
            stop_wait_s=float(args.stop_wait_s),
            logs_dir=args.logs_dir,
            use_tracker=not args.no_tracker,
            prepare_scene_command=args.prepare_scene_command,
            dry_run=bool(args.dry_run),
            bridge_prepare_command=bridge_prepare,
            wait_for_task_success=not bool(args.fixed_duration_wait),
        )
    except Exception as exc:
        update_manifest_row_by_id(
            manifest_path,
            str(row["rollout_id"]),
            {
                "status": "failed",
                "finished_time": now_string(),
                "error": str(exc),
            },
            args,
        )
        print(f"FAILED: {row.get('rollout_id')}: {exc}")
        return False

    if not ok:
        update_manifest_row_by_id(
            manifest_path,
            str(row["rollout_id"]),
            {
                "status": "failed",
                "finished_time": now_string(),
                "error": "Rollout runner returned failure.",
            },
            args,
        )
        return False

    if args.dry_run:
        update_manifest_row_by_id(
            manifest_path,
            str(row["rollout_id"]),
            {
                "status": "pending",
                "worker_id": DELETE_FIELD,
                "claimed_time": DELETE_FIELD,
                "started_time": DELETE_FIELD,
                "dry_run_time": now_string(),
                "runtime_task_spec": runtime_path.as_posix(),
                "error": DELETE_FIELD,
                "validation": DELETE_FIELD,
            },
            args,
        )
        return True

    if not args.dry_run and not args.skip_validation:
        validation = validate_rollout_dir(
            rollout_dir=rollout_dir,
            expected_task_id=str(row.get("task_id")),
            expected_variant_id=str(row.get("variant_id")),
            min_samples=args.min_samples,
            min_motion_m=args.min_motion_m,
            min_action_chunk_fraction=float(args.min_action_chunk_fraction),
            min_cmd_vel_fraction=float(args.min_cmd_vel_fraction),
            max_mean_abs_action_first_y_m=float(args.max_mean_abs_action_first_y_m),
            max_abs_action_first_y_m=float(args.max_abs_action_first_y_m),
            max_action_chunk_age_s=float(args.max_action_chunk_age_s),
            max_mean_reference_lateral_error_m=float(args.max_mean_reference_lateral_error_m),
            max_reference_lateral_error_m=float(args.max_reference_lateral_error_m),
            max_final_target_distance_m=float(args.max_final_target_distance_m),
            max_black_image_fraction=float(args.max_black_image_fraction),
            min_target_fence_clearance_m=float(args.min_target_fence_clearance_m),
            allow_stationary=bool(args.allow_stationary),
        )
        row["validation"] = {
            "valid": validation.valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "metrics": validation.metrics,
        }
        if not validation.valid:
            update_manifest_row_by_id(
                manifest_path,
                str(row["rollout_id"]),
                {
                    "status": "failed",
                    "finished_time": now_string(),
                    "error": "Collected rollout failed validation: " + "; ".join(validation.errors),
                    "validation": row["validation"],
                },
                args,
            )
            print(f"FAILED VALIDATION: {row.get('rollout_id')}: {'; '.join(validation.errors)}")
            return False

    complete_updates = {
        "status": "complete",
        "finished_time": now_string(),
        "error": DELETE_FIELD,
    }
    if row.get("validation"):
        complete_updates["validation"] = row["validation"]
    update_manifest_row_by_id(manifest_path, str(row["rollout_id"]), complete_updates, args)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rollout-id", action="append")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--variant-id", action="append")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--only-visual-id", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--include-pose-not-ready", action="store_true")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--claim-next", action="store_true")
    parser.add_argument("--manifest-lock-timeout-s", type=float, default=30.0)
    parser.add_argument("--use-isaac-bridge", action="store_true")
    parser.add_argument("--isaac-root", default=None)
    parser.add_argument("--bridge-command-file", default=None)
    parser.add_argument("--bridge-status-file", default=None)
    parser.add_argument("--prepare-timeout-s", type=float, default=90.0)
    parser.add_argument("--prepare-topic-timeout-s", type=float, default=30.0)
    parser.add_argument("--skip-prepare-topic-wait", action="store_true")
    parser.add_argument("--prepare-scene-command", default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="sim_fenceline")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--fixed-duration-wait", action="store_true")
    parser.add_argument("--startup-wait-s", type=float, default=1.0)
    parser.add_argument("--stop-wait-s", type=float, default=5.0)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--runtime-spec-dir", type=Path, default=None)
    parser.add_argument("--no-tracker", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-motion-m", type=float, default=None)
    parser.add_argument("--min-action-chunk-fraction", type=float, default=0.5)
    parser.add_argument("--min-cmd-vel-fraction", type=float, default=0.5)
    parser.add_argument("--max-mean-abs-action-first-y-m", type=float, default=1.25)
    parser.add_argument("--max-abs-action-first-y-m", type=float, default=3.0)
    parser.add_argument("--max-action-chunk-age-s", type=float, default=1.0)
    parser.add_argument("--max-mean-reference-lateral-error-m", type=float, default=2.0)
    parser.add_argument("--max-reference-lateral-error-m", type=float, default=4.0)
    parser.add_argument("--max-final-target-distance-m", type=float, default=2.0)
    parser.add_argument("--max-black-image-fraction", type=float, default=0.05)
    parser.add_argument("--min-target-fence-clearance-m", type=float, default=0.65)
    parser.add_argument("--allow-stationary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_worker_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.worker_id:
        if args.bridge_command_file is None:
            args.bridge_command_file = f"/tmp/isaac_rollout_{args.worker_id}_request.json"
        if args.bridge_status_file is None:
            args.bridge_status_file = f"/tmp/isaac_rollout_{args.worker_id}_status.json"
        if args.logs_dir is None:
            args.logs_dir = Path("logs/sim_rollouts") / args.worker_id
        if args.runtime_spec_dir is None:
            args.runtime_spec_dir = Path("logs/sim_rollouts") / args.worker_id / "runtime_task_specs"
        args.claim_next = True if args.claim_next or args.worker_id else args.claim_next
    else:
        if args.logs_dir is None:
            args.logs_dir = Path("logs/sim_rollouts")
        if args.runtime_spec_dir is None:
            args.runtime_spec_dir = Path("logs/sim_rollouts/runtime_task_specs")
    return args


def main() -> int:
    args = apply_worker_defaults(parse_args())
    manifest_path = args.manifest.resolve()
    if args.claim_next:
        max_rows = args.limit if args.limit is not None else 1
        failures = 0
        completed = 0
        for _ in range(max_rows):
            claimed = claim_next_row(manifest_path, args)
            if claimed is None:
                break
            row, _ = claimed
            ok = run_claimed_row(manifest_path, row, args, already_marked_running=True)
            completed += 1
            if not ok:
                failures += 1
        if completed == 0:
            print("No manifest rows selected.")
            return 1
        print(f"Finished {completed - failures}/{completed} claimed manifest rollout(s).")
        return 1 if failures else 0

    with manifest_lock(manifest_path, args.worker_id, args.manifest_lock_timeout_s):
        manifest = load_yaml(manifest_path)
        row_indices = selected_row_indices(manifest, args)
        if not row_indices:
            print("No manifest rows selected.")
            return 1

    print(f"Selected {len(row_indices)} manifest row(s).")
    failures = 0
    for row_index in row_indices:
        manifest = load_yaml(manifest_path)
        ok = run_manifest_row(manifest_path, manifest, row_index, args)
        if not ok:
            failures += 1
    print(f"Finished {len(row_indices) - failures}/{len(row_indices)} manifest rollout(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

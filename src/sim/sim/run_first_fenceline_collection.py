#!/usr/bin/env python3
"""Run the first fenceline-only collection pipeline.

This script is intentionally narrower than the full manifest tools:

* fenceline generated layout YAMLs only
* corridor layouts excluded by default
* one follow task per layout
* one nominal variant and one deterministic recovery pose variant per layout
* only visual USDs, not base USDs
* one deterministic visual sample per task/variant
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


FOLLOW_TASK_RANK = {
    "follow_fence_sequence": 0,
    "follow_fence": 1,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)


def run_command(command: list[str], label: str, *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("\n" + "=" * 80, flush=True)
    print(f"Stage: {label}", flush=True)
    print("=" * 80, flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    result = subprocess.run([str(part) for part in command], cwd=cwd, check=check)
    if result.returncode == 0:
        print(f"[OK] {label}", flush=True)
    else:
        print(f"[WARN] {label} exited with code {result.returncode}", flush=True)
    return result


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stable_key(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unnamed"


def select_fenceline_layouts(
    layout_root: Path,
    exclude_regex: str | None,
    limit: int | None,
    layout_order: str = "default",
) -> list[Path]:
    pattern = re.compile(exclude_regex) if exclude_regex else None
    selected: list[Path] = []
    paths = [*layout_root.rglob("*.yaml"), *layout_root.rglob("*.yml")]
    if layout_order == "reverse":
        paths = sorted(paths, reverse=True)
    elif layout_order == "gaps-last":
        paths = sorted(paths, key=lambda path: ("fence_gap" in path.as_posix(), path.as_posix()))
    else:
        paths = sorted(paths)

    for path in paths:
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(layout_root.resolve()).as_posix()
        if pattern and pattern.search(relative):
            continue
        try:
            layout = load_yaml(path)
        except Exception as exc:
            print(f"[SKIP] Could not read {path}: {exc}", flush=True)
            continue
        if layout.get("config_type") != "fenceline":
            continue
        if "rover_pose" not in layout:
            continue
        selected.append(path.resolve())
        if limit is not None and len(selected) >= limit:
            break
    return selected


def copy_layouts(layouts: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for layout in layouts:
        destination = output_dir / layout.name
        shutil.copy2(layout, destination)
        print(f"Selected layout: {layout} -> {destination}", flush=True)


def spec_paths(directory: Path) -> list[Path]:
    return sorted([*directory.rglob("*.yaml"), *directory.rglob("*.yml")])


def variant_is_valid(variant: dict[str, Any]) -> bool:
    validation = variant.get("planner_validation")
    return not isinstance(validation, dict) or bool(validation.get("valid", False))


def has_start_delta(variant: dict[str, Any]) -> bool:
    delta = variant.get("start_pose_delta")
    if not isinstance(delta, dict):
        return False
    return any(abs(float(delta.get(key, 0.0))) > 1e-9 for key in ("x_m", "y_m", "yaw_rad"))


def scene_id_for_spec(spec_path: Path, spec: dict[str, Any]) -> str:
    scene = spec.get("scene") if isinstance(spec.get("scene"), dict) else {}
    return str(scene.get("scene_id") or spec.get("suite_id") or spec_path.stem)


def choose_follow_task(spec: dict[str, Any], *, require_valid_nominal: bool = True) -> dict[str, Any] | None:
    candidates = []
    for task in spec.get("tasks", []):
        if not isinstance(task, dict):
            continue
        task_type = str(task.get("task_type", ""))
        if task_type not in FOLLOW_TASK_RANK:
            continue
        variants = task.get("trajectory_variants") or []
        has_nominal = any(
            str(variant.get("variant_id", "")) == "nominal"
            and ((not require_valid_nominal) or variant_is_valid(variant))
            for variant in variants
        )
        if not has_nominal:
            continue
        candidates.append(task)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda task: (
            FOLLOW_TASK_RANK.get(str(task.get("task_type", "")), 99),
            "perimeter" not in str(task.get("task_id", "")),
            "connected" not in str(task.get("task_id", "")),
            str(task.get("task_id", "")),
        ),
    )[0]


def recovery_kind(variant: dict[str, Any]) -> str | None:
    delta = variant.get("start_pose_delta")
    if not isinstance(delta, dict):
        return None
    dx = abs(float(delta.get("x_m", 0.0)))
    dy = abs(float(delta.get("y_m", 0.0)))
    dyaw = abs(float(delta.get("yaw_rad", 0.0)))
    variant_id = str(variant.get("variant_id", "")).lower()
    recovery_case = str(variant.get("recovery_case", "")).lower()
    text = " ".join([variant_id, recovery_case])
    if dyaw > 1e-9 and ("heading" in text or dyaw >= max(dx, dy)):
        return "heading"
    if dx > 1e-9 or dy > 1e-9:
        return "start"
    if dyaw > 1e-9:
        return "heading"
    return None


def recovery_kind_from_variant_id(variant_id: str) -> str | None:
    text = variant_id.lower()
    if "heading" in text or "yaw" in text:
        return "heading"
    if text.startswith("recovery_"):
        return "start"
    return None


def valid_recovery_variants(task: dict[str, Any], *, require_valid: bool = True) -> list[dict[str, Any]]:
    variants = []
    for variant in task.get("trajectory_variants") or []:
        if not isinstance(variant, dict):
            continue
        if str(variant.get("variant_id", "")) == "nominal":
            continue
        if not has_start_delta(variant):
            continue
        if require_valid and not variant_is_valid(variant):
            continue
        data_category = str(variant.get("data_category", ""))
        variant_type = str(variant.get("variant_type", ""))
        if data_category != "recovery" and variant_type != "recovery":
            continue
        variants.append(variant)
    return variants


def choose_recovery_variant(task: dict[str, Any], scene_id: str, sample_seed: int) -> dict[str, Any] | None:
    variants = valid_recovery_variants(task)
    if not variants:
        return None
    return sorted(
        variants,
        key=lambda variant: stable_key(sample_seed, scene_id, task.get("task_id"), variant.get("variant_id")),
    )[0]


def choose_recovery_variants_by_kind(
    task: dict[str, Any],
    scene_id: str,
    sample_seed: int,
    *,
    require_valid: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    variants = valid_recovery_variants(task, require_valid=require_valid)
    for kind in ("start", "heading"):
        candidates = [variant for variant in variants if recovery_kind(variant) == kind]
        if not candidates:
            continue
        chosen = sorted(
            candidates,
            key=lambda variant: stable_key(sample_seed, scene_id, task.get("task_id"), kind, variant.get("variant_id")),
        )[0]
        variant_id = str(chosen.get("variant_id"))
        selected.append(chosen)
        used_ids.add(variant_id)
    if selected:
        return selected

    variants = valid_recovery_variants(task, require_valid=require_valid)
    fallback = None
    if variants:
        fallback = sorted(
            variants,
            key=lambda variant: stable_key(sample_seed, scene_id, task.get("task_id"), variant.get("variant_id")),
        )[0]
    if fallback is None:
        return []
    return [fallback] if str(fallback.get("variant_id")) not in used_ids else []


def specs_by_scene_id(directory: Path | None) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    if directory is None or not directory.exists():
        return indexed
    for spec_path in spec_paths(directory):
        spec = load_yaml(spec_path)
        indexed[scene_id_for_spec(spec_path, spec)] = (spec_path, spec)
    return indexed


def task_by_id(spec: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in spec.get("tasks") or []:
        if isinstance(task, dict) and str(task.get("task_id")) == str(task_id):
            return task
    return None


def build_selection(
    valid_specs_dir: Path,
    sample_seed: int,
    *,
    raw_specs_dir: Path | None = None,
    include_unvalidated_nominal: bool = False,
) -> dict[str, dict[str, Any]]:
    selection: dict[str, dict[str, Any]] = {}
    raw_specs = specs_by_scene_id(raw_specs_dir)
    for spec_path in spec_paths(valid_specs_dir):
        spec = load_yaml(spec_path)
        scene_id = scene_id_for_spec(spec_path, spec)
        task = choose_follow_task(spec)
        if task is None:
            print(f"[SKIP] No valid fenceline follow task in {spec_path}", flush=True)
            continue
        # Treat valid_task_specs as the source of truth. If selected validation
        # trimmed the task to nominal + one start + one heading recovery, do not
        # pull extra recovery variants back in from raw task_specs here.
        recovery_spec_path = spec_path
        recovery_task = task
        recoveries = choose_recovery_variants_by_kind(
            recovery_task,
            scene_id,
            sample_seed,
            require_valid=True,
        )
        recovery_ids = [str(recovery.get("variant_id")) for recovery in recoveries if recovery.get("variant_id")]
        selection[scene_id] = {
            "spec_path": spec_path.as_posix(),
            "recovery_spec_path": recovery_spec_path.as_posix(),
            "planner_validation_status": "validated",
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "nominal_variant_id": "nominal",
            "recovery_variant_id": recovery_ids[0] if recovery_ids else None,
            "recovery_variant_ids": recovery_ids,
        }
    if include_unvalidated_nominal and raw_specs_dir is not None:
        for spec_path in spec_paths(raw_specs_dir):
            spec = load_yaml(spec_path)
            scene_id = scene_id_for_spec(spec_path, spec)
            if scene_id in selection:
                continue
            task = choose_follow_task(spec, require_valid_nominal=False)
            if task is None:
                print(f"[SKIP] No unvalidated fenceline follow task in {spec_path}", flush=True)
                continue
            selection[scene_id] = {
                "spec_path": spec_path.as_posix(),
                "planner_validation_status": "unvalidated",
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "nominal_variant_id": "nominal",
                "recovery_variant_id": None,
                "recovery_variant_ids": [],
            }
    return selection


def selected_variant_ids(selected: dict[str, Any]) -> set[str]:
    ids = {"nominal"}
    recovery_variant_ids = selected.get("recovery_variant_ids")
    if isinstance(recovery_variant_ids, list):
        ids.update(str(item) for item in recovery_variant_ids)
    elif selected.get("recovery_variant_id"):
        ids.add(str(selected["recovery_variant_id"]))
    return ids


def copy_selected_task_spec(selected: dict[str, Any], output_dir: Path) -> None:
    nominal_spec_path = Path(str(selected["spec_path"]))
    recovery_spec_path = Path(str(selected.get("recovery_spec_path") or selected["spec_path"]))
    nominal_spec = load_yaml(nominal_spec_path)
    recovery_spec = load_yaml(recovery_spec_path)
    task_id = str(selected["task_id"])
    nominal_task = task_by_id(nominal_spec, task_id)
    recovery_task = task_by_id(recovery_spec, task_id)
    if nominal_task is None:
        raise ValueError(f"{nominal_spec_path} has no selected task {task_id}")
    if recovery_task is None:
        recovery_task = nominal_task

    keep_variant_ids = selected_variant_ids(selected)
    merged_task = dict(nominal_task)
    merged_variants: list[dict[str, Any]] = []
    for variant_id in keep_variant_ids:
        source_task = nominal_task if variant_id == "nominal" else recovery_task
        for variant in source_task.get("trajectory_variants") or []:
            if isinstance(variant, dict) and str(variant.get("variant_id", "nominal")) == variant_id:
                merged_variants.append(dict(variant))
                break
    merged_variants = sorted(
        merged_variants,
        key=lambda variant: (str(variant.get("variant_id", "")) != "nominal", str(variant.get("variant_id", ""))),
    )
    merged_task["trajectory_variants"] = merged_variants

    merged_spec = dict(nominal_spec)
    merged_spec["tasks"] = [merged_task]
    destination = output_dir / nominal_spec_path.name
    write_yaml(destination, merged_spec)


def prepare_manifest_input_specs(selection: dict[str, dict[str, Any]], output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for selected in selection.values():
        copy_selected_task_spec(selected, output_dir)
    return output_dir


def generated_layout_for_spec(spec_path: Path) -> Path | None:
    spec = load_yaml(spec_path)
    scene = spec.get("scene") if isinstance(spec.get("scene"), dict) else {}
    value = scene.get("generated_layout_yaml") or scene.get("source_yaml")
    if not value:
        return None
    return Path(str(value)).expanduser().resolve()


def expand_selected_recovery_variants(selection: dict[str, dict[str, Any]], pose_variants_dir: Path) -> None:
    for scene_id, selected in sorted(selection.items()):
        recovery_variant_ids = selected.get("recovery_variant_ids")
        if not isinstance(recovery_variant_ids, list):
            recovery_variant_ids = [selected["recovery_variant_id"]] if selected.get("recovery_variant_id") else []
        if not recovery_variant_ids:
            print(f"[SKIP] {scene_id}: no valid recovery variant selected", flush=True)
            continue
        spec_path = Path(str(selected.get("recovery_spec_path") or selected["spec_path"]))
        layout_path = generated_layout_for_spec(spec_path)
        if layout_path is None or not layout_path.exists():
            print(f"[SKIP] {scene_id}: missing generated layout for pose expansion", flush=True)
            continue
        for recovery_variant_id in recovery_variant_ids:
            run_command(
                [
                    "ros2",
                    "run",
                    "sim",
                    "expand_pose_variants",
                    "--layout-yaml",
                    layout_path.as_posix(),
                    "--task-spec",
                    spec_path.as_posix(),
                    "--output-dir",
                    pose_variants_dir.as_posix(),
                    "--task-id",
                    str(selected["task_id"]),
                    "--variant-id",
                    str(recovery_variant_id),
                ],
                label=f"Expand recovery pose variant for {scene_id} / {recovery_variant_id}",
            )


def generated_usd_from_layout(layout_path: Path, isaac_root: Path) -> Path | None:
    layout = load_yaml(layout_path)
    output_dir = layout.get("output_dir")
    output_name = layout.get("output_name")
    if not output_dir or not output_name:
        return None
    filename = str(output_name)
    if not filename.endswith(".usd"):
        filename = f"{filename}.usd"
    return (isaac_root / str(output_dir) / filename).resolve()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate_pose_variant_usds(
    pose_variants_dir: Path,
    isaac_root: Path,
    isaac_python: str,
    variation_config: str,
    visual_sample_seed: int,
    max_visuals_per_pose: int,
) -> None:
    for layout_path in spec_paths(pose_variants_dir):
        base_usd = generated_usd_from_layout(layout_path, isaac_root)
        run_command(
            [
                isaac_python,
                "scripts/isaac_scene_generator.py",
                layout_path.as_posix(),
            ],
            label=f"Generate pose-variant base USD for {layout_path.name}",
            cwd=isaac_root,
        )
        if base_usd is None or not base_usd.exists():
            raise FileNotFoundError(f"Expected pose-variant base USD was not generated for {layout_path}")
        run_command(
            [
                isaac_python,
                "scripts/variation_generator.py",
                relative_to_root(base_usd, isaac_root),
                variation_config,
                "--max-variations",
                str(max_visuals_per_pose),
                "--sample-seed",
                str(visual_sample_seed),
            ],
            label=f"Generate sampled visual USD for {base_usd.name}",
            cwd=isaac_root,
        )


def filter_manifest(
    manifest_all_path: Path,
    manifest_path: Path,
    selection: dict[str, dict[str, Any]],
    duration_policy: dict[str, float],
    excluded_rollout_ids: set[str] | None = None,
    rollouts_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = load_yaml(manifest_all_path)
    rows = []
    seen: set[tuple[str, str]] = set()
    excluded_rollout_ids = excluded_rollout_ids or set()
    for row in manifest.get("rollouts") or []:
        if not isinstance(row, dict):
            continue
        rollout_id = str(row.get("rollout_id") or "")
        if rollout_id in excluded_rollout_ids:
            continue
        scene_id = str(row.get("scene_id") or "")
        variant_id = str(row.get("variant_id", ""))
        variant_type = str(row.get("variant_type") or "")
        data_category = str(row.get("data_category") or "")
        is_nominal = variant_id == "nominal"
        is_recovery = variant_type == "recovery" or data_category == "recovery" or variant_id.startswith("recovery_")
        if not is_nominal and not is_recovery:
            continue
        if row.get("visual_id") == "base":
            continue
        if row.get("requires_pose_variant") and not row.get("pose_variant_ready"):
            continue
        key = (scene_id, str(row.get("task_id") or ""), variant_id)
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        collection = dict(row.get("collection") or {})
        selected = selection.get(scene_id, {})
        collection["duration_s"] = duration_for_rollout_row(row, selected, duration_policy)
        collection["use_isaac_camera_pose_debug"] = True
        collection.setdefault("isaac_pose_debug_topic", "/isaac/scene_pose_debug")
        collection["planner_validation_status"] = selected.get("planner_validation_status", "validated")
        row["collection"] = collection
        row["planner_validation_status"] = selected.get("planner_validation_status", "validated")
        existing_status = existing_rollout_status(row, rollouts_dir)
        if existing_status:
            row["status"] = existing_status
        rows.append(row)

    filtered = dict(manifest)
    filtered["source_manifest"] = manifest_all_path.as_posix()
    filtered["selection_rule"] = (
        "first fenceline collection: one selected follow task per layout; validated specs keep nominal plus selected "
        "start and heading recovery variants when available; unvalidated fallback specs keep nominal only; one non-base "
        "visual each"
    )
    filtered["rollouts"] = rows
    counts = dict(filtered.get("counts", {}))
    counts.update(
        {
            "rollouts": len(rows),
            "pending": len(rows),
            "running": 0,
            "complete": 0,
            "failed": 0,
            "skipped": 0,
            "nominal_rows": sum(1 for row in rows if row.get("variant_id") == "nominal"),
            "recovery_rows": sum(1 for row in rows if row.get("variant_id") != "nominal"),
            "excluded_rollouts": len(excluded_rollout_ids),
        }
    )
    update_existing_rollout_counts(rows, counts)
    filtered["counts"] = counts
    write_yaml(manifest_path, filtered)
    return filtered


def load_excluded_rollout_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def existing_rollout_status(row: dict[str, Any], rollouts_dir: Path | None) -> str | None:
    if rollouts_dir is None:
        return None
    rollout_id = str(row.get("trajectory_name") or row.get("rollout_id") or "")
    if not rollout_id:
        return None
    rollout_dir = rollouts_dir / rollout_id
    metadata_path = rollout_dir / "metadata.json"
    summary_path = rollout_dir / "task_success_wait_summary.json"
    if not metadata_path.exists() or not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return "failed"
    return "complete" if bool(summary.get("success")) else "failed"


def update_existing_rollout_counts(rows: list[dict[str, Any]], counts: dict[str, Any]) -> None:
    counts.update(
        {
            "pending": sum(1 for row in rows if row.get("status", "pending") == "pending"),
            "running": sum(1 for row in rows if row.get("status") == "running"),
            "complete": sum(1 for row in rows if row.get("status") == "complete"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
            "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        }
    )


def duration_for_rollout_row(
    row: dict[str, Any],
    selected: dict[str, Any],
    duration_policy: dict[str, float],
) -> float:
    scene_id = str(row.get("scene_id") or "").lower()
    rollout_id = str(row.get("rollout_id") or "").lower()
    task_id = str(row.get("task_id") or selected.get("task_id") or "").lower()
    text = " ".join([scene_id, rollout_id, task_id])
    if "perimeter_large" in text:
        return duration_policy["large_perimeter"]
    if "rectangle" in text or "perimeter" in task_id:
        return duration_policy["rectangle"]
    if "corner" in text:
        return duration_policy["corner"]
    if "gap" in text or "connected" in task_id:
        return duration_policy["gap"]
    return duration_policy["straight"]


def metadata_field(metadata: dict[str, Any], *keys: str) -> Any:
    value: Any = metadata
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def plot_rollouts(rollouts_dir: Path, plots_dir: Path) -> int:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for metadata_path in sorted(rollouts_dir.rglob("metadata.json")):
        rollout = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        summary_path = rollout / "task_success_wait_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        task_spec = (
            summary.get("task_spec")
            or metadata.get("task_spec")
            or metadata.get("task_spec_path")
            or metadata.get("source_task_spec")
            or metadata_field(metadata, "manifest_row", "source_task_spec")
        )
        task_id = metadata.get("task_id") or metadata_field(metadata, "structured_task", "task_id")
        variant_id = metadata.get("variant_id") or metadata_field(metadata, "selected_variant", "variant_id") or "nominal"
        if not task_spec or not task_id:
            print(f"[SKIP] Missing task spec or task id for {rollout}", flush=True)
            continue
        rollout_plot = rollout / "plan_vs_actual.png"
        result = run_command(
            [
                "ros2",
                "run",
                "sim",
                "plot_task_plan",
                str(task_spec),
                "--task-id",
                str(task_id),
                "--variant-id",
                str(variant_id),
                "--rollout-dir",
                rollout.as_posix(),
                "--output",
                rollout_plot.as_posix(),
            ],
            label=f"Plot rollout {rollout.name}",
            check=False,
        )
        if result.returncode == 0:
            shutil.copy2(rollout_plot, plots_dir / f"{rollout.name}.png")
            plotted += 1
    return plotted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, default=Path("~/isaac_files/configs/generated_layouts"))
    parser.add_argument("--isaac-root", type=Path, default=Path("~/isaac_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("~/sim_datasets/first_fenceline_collection"))
    parser.add_argument("--exclude-layout-regex", default=r"corridor|jetbot")
    parser.add_argument(
        "--layout-order",
        choices=("default", "reverse", "gaps-last"),
        default="default",
        help="Validation/collection order for selected layouts. Use gaps-last to leave fence_gap layouts until the end.",
    )
    parser.add_argument("--limit-layouts", type=int, default=None)
    parser.add_argument("--variant-sample-seed", type=int, default=11)
    parser.add_argument("--visual-sample-seed", type=int, default=23)
    parser.add_argument("--duration-s", type=float, default=90.0, help="Fallback duration for straight/simple layouts.")
    parser.add_argument("--gap-duration-s", type=float, default=120.0)
    parser.add_argument("--corner-duration-s", type=float, default=120.0)
    parser.add_argument("--rectangle-duration-s", type=float, default=180.0)
    parser.add_argument("--large-perimeter-duration-s", type=float, default=260.0)
    parser.add_argument("--prepare-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--validation-planner-preset",
        choices=("full", "fast"),
        default="fast",
        help="Planner validation preset passed to validate_scene_task_specs.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("selected", "all"),
        default="selected",
        help=(
            "'selected' validates only the one task and deterministic nominal/start/heading variants used by this "
            "collection; 'all' runs the older exhaustive task-spec validator."
        ),
    )
    parser.add_argument("--validation-grid-resolution-m", type=float, default=None)
    parser.add_argument("--validation-yaw-resolution-deg", type=float, default=None)
    parser.add_argument("--validation-max-iterations", type=int, default=None)
    parser.add_argument("--validation-subgoal-spacing-m", type=float, default=None)
    parser.add_argument(
        "--validation-per-spec-timeout-s",
        type=float,
        default=480.0,
        help="For selected validation, mark one spec invalid and continue if it exceeds this timeout. 0 disables.",
    )
    parser.add_argument(
        "--validation-full-planner-fallback",
        action="store_true",
        help="When using selected validation, retry fast-planner failures with full planner settings.",
    )
    parser.add_argument("--variation-config", default="configs/variation_configs/variation.yaml")
    parser.add_argument("--isaac-python", default=None)
    parser.add_argument("--max-visuals-per-pose-variant", type=int, default=1)
    parser.add_argument(
        "--include-unvalidated-nominal",
        action="store_true",
        help="If a layout has no valid spec yet, include its raw task spec nominal rollout and rely on Isaac/postprocess rejection.",
    )
    parser.add_argument("--skip-pose-usd-generation", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout_root = args.layout_root.expanduser().resolve()
    isaac_root = args.isaac_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_layouts_dir = output_dir / "selected_layouts"
    task_specs_dir = output_dir / "task_specs"
    valid_specs_dir = output_dir / "valid_task_specs"
    pose_variants_dir = output_dir / "pose_variants"
    rollouts_dir = output_dir / "rollouts"
    plots_dir = output_dir / "plots"
    manifest_all = output_dir / "manifest_all.yaml"
    manifest = output_dir / "manifest.yaml"
    manifest_input_specs_dir = output_dir / "manifest_input_specs"
    selection_path = output_dir / "selected_tasks_and_variants.yaml"
    excluded_rollout_ids_path = output_dir / "crashing_rollout_ids.txt"
    isaac_python = args.isaac_python or sys.executable
    duration_policy = {
        "straight": float(args.duration_s),
        "gap": float(args.gap_duration_s),
        "corner": float(args.corner_duration_s),
        "rectangle": float(args.rectangle_duration_s),
        "large_perimeter": float(args.large_perimeter_duration_s),
    }

    if not args.no_clean:
        clean_output_dir(output_dir)
    for path in (selected_layouts_dir, task_specs_dir, valid_specs_dir, pose_variants_dir, rollouts_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    layouts = select_fenceline_layouts(layout_root, args.exclude_layout_regex, args.limit_layouts, args.layout_order)
    if not layouts:
        raise SystemExit(f"No fenceline layout YAMLs selected from {layout_root}")
    copy_layouts(layouts, selected_layouts_dir)

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "generate_scene_task_specs",
            selected_layouts_dir.as_posix(),
            "--output-dir",
            task_specs_dir.as_posix(),
            "--task-family",
            "following_only",
            "--summary",
        ],
        label="Generate fenceline task specs",
    )

    validation_command = [
            "ros2",
            "run",
            "sim",
            "validate_selected_scene_task_specs" if args.validation_mode == "selected" else "validate_scene_task_specs",
            task_specs_dir.as_posix(),
            "--check-planner",
            "--planner-speed-preset",
            str(args.validation_planner_preset),
            "--allow-invalid",
            "--verbose",
            "--write-valid-output-dir",
            valid_specs_dir.as_posix(),
    ]
    if args.validation_grid_resolution_m is not None:
        validation_command.extend(["--planner-grid-resolution-m", str(args.validation_grid_resolution_m)])
    if args.validation_yaw_resolution_deg is not None:
        validation_command.extend(["--planner-yaw-resolution-deg", str(args.validation_yaw_resolution_deg)])
    if args.validation_max_iterations is not None:
        validation_command.extend(["--planner-max-iterations", str(args.validation_max_iterations)])
    if args.validation_subgoal_spacing_m is not None:
        validation_command.extend(["--planner-subgoal-spacing-m", str(args.validation_subgoal_spacing_m)])
    if args.validation_mode == "selected":
        validation_command.extend(["--per-spec-timeout-s", str(args.validation_per_spec_timeout_s)])
    if args.validation_mode == "selected" and args.validation_full_planner_fallback:
        validation_command.append("--full-planner-fallback")
    run_command(
        validation_command,
        label="Validate fenceline task specs",
    )

    selection = build_selection(
        valid_specs_dir,
        int(args.variant_sample_seed),
        raw_specs_dir=task_specs_dir,
        include_unvalidated_nominal=bool(args.include_unvalidated_nominal),
    )
    if not selection:
        raise SystemExit("No valid fenceline follow tasks selected.")
    write_yaml(selection_path, {"selection": selection})
    print(f"Wrote selected task/variant policy: {selection_path}", flush=True)
    manifest_input_specs = prepare_manifest_input_specs(selection, manifest_input_specs_dir)

    expand_selected_recovery_variants(selection, pose_variants_dir)

    if not args.skip_pose_usd_generation:
        generate_pose_variant_usds(
            pose_variants_dir=pose_variants_dir,
            isaac_root=isaac_root,
            isaac_python=isaac_python,
            variation_config=args.variation_config,
            visual_sample_seed=int(args.visual_sample_seed),
            max_visuals_per_pose=int(args.max_visuals_per_pose_variant),
        )

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "generate_collection_manifest",
            manifest_input_specs.as_posix(),
            "--output",
            manifest_all.as_posix(),
            "--isaac-root",
            isaac_root.as_posix(),
            "--pose-variant-layout-dir",
            pose_variants_dir.as_posix(),
            "--include-visual-variations",
            "--no-base-usd",
            "--max-visuals-per-task-variant",
            "1",
            "--visual-sample-seed",
            str(args.visual_sample_seed),
            "--task-family",
            "following_only",
            "--summary",
        ],
        label="Generate full first-pass manifest",
    )

    excluded_rollout_ids = load_excluded_rollout_ids(excluded_rollout_ids_path)
    if excluded_rollout_ids:
        print(
            f"Excluding {len(excluded_rollout_ids)} known crashing rollout id(s) from {excluded_rollout_ids_path}",
            flush=True,
        )
    filtered = filter_manifest(
        manifest_all,
        manifest,
        selection,
        duration_policy,
        excluded_rollout_ids=excluded_rollout_ids,
        rollouts_dir=rollouts_dir,
    )
    print(f"Wrote filtered first-pass manifest: {manifest}", flush=True)
    print(f"Selected {len(filtered.get('rollouts') or [])} rollout(s).", flush=True)
    print(yaml.safe_dump(filtered.get("counts", {}), sort_keys=True), flush=True)

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "summarize_collection_manifest",
            manifest.as_posix(),
        ],
        label="Summarize filtered first-pass manifest",
        check=False,
    )

    if args.skip_run:
        return 0

    rollout_result = run_command(
        [
            "ros2",
            "run",
            "sim",
            "run_collection_manifest",
            manifest.as_posix(),
            "--retry-failed",
            "--no-tracker",
            "--use-isaac-bridge",
            "--isaac-root",
            isaac_root.as_posix(),
            "--base-dir",
            rollouts_dir.as_posix(),
            "--prepare-timeout-s",
            str(args.prepare_timeout_s),
            *(["--dry-run"] if args.dry_run else []),
        ],
        label="Run first-pass fenceline rollouts",
        check=False,
    )
    if rollout_result.returncode != 0:
        print("Collection reported one or more failed rollouts; plotting any collected outputs.", flush=True)

    if not args.dry_run:
        plotted = plot_rollouts(rollouts_dir, plots_dir)
        print(f"Plotted {plotted} rollout(s) into {plots_dir}", flush=True)

    print("\nFirst fenceline collection pipeline complete.", flush=True)
    print(f"Output folder: {output_dir}", flush=True)
    print(f"Selection: {selection_path}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Rollouts: {rollouts_dir}", flush=True)
    print(f"Plots: {plots_dir}", flush=True)
    return 1 if rollout_result.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

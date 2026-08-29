#!/usr/bin/env python3
"""Validate only the task/variants intended for compact data collection.

This is intentionally much narrower than validate_scene_task_specs:

* one follow task per scene spec
* nominal plus one deterministic start-pose recovery variant
* nominal plus one deterministic heading recovery variant
* Hybrid A* is run only for those selected variants

Accepted variants keep their cached planned_path_xy so rollouts can follow the
same path that validation approved.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
from pathlib import Path
from typing import Any

import yaml

from sim.run_first_fenceline_collection import choose_follow_task
from sim.validate_scene_task_specs import (
    NoAliasDumper,
    expand_inputs,
    get_start_poses,
    load_yaml,
    planner_accepts_variant,
    planner_overrides_from_args,
    validate_task,
)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)


class PerSpecTimeout(RuntimeError):
    pass


def run_with_timeout(timeout_s: float | None, func, *args, **kwargs):
    if timeout_s is None or timeout_s <= 0.0:
        return func(*args, **kwargs)
    if not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs)

    def handle_timeout(signum, frame):
        raise PerSpecTimeout(f"validation exceeded per-spec timeout of {timeout_s:.1f}s")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def stable_key(seed: int, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()


def variant_id(variant: dict[str, Any]) -> str:
    return str(variant.get("variant_id", "nominal"))


def variant_is_recovery(variant: dict[str, Any]) -> bool:
    return (
        str(variant.get("variant_type", "")) == "recovery"
        or str(variant.get("data_category", "")) == "recovery"
        or variant_id(variant).startswith("recovery_")
    )


def start_pose_delta(variant: dict[str, Any]) -> dict[str, float]:
    raw = variant.get("start_pose_delta")
    if not isinstance(raw, dict):
        return {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    return {
        "x_m": float(raw.get("x_m", 0.0)),
        "y_m": float(raw.get("y_m", 0.0)),
        "yaw_rad": float(raw.get("yaw_rad", 0.0)),
    }


def recovery_kind(variant: dict[str, Any]) -> str | None:
    if not variant_is_recovery(variant):
        return None

    delta = start_pose_delta(variant)
    xy_delta = max(abs(delta["x_m"]), abs(delta["y_m"]))
    yaw_delta = abs(delta["yaw_rad"])
    text = " ".join(
        [
            variant_id(variant).lower(),
            str(variant.get("recovery_case", "")).lower(),
        ]
    )

    if "heading" in text or yaw_delta >= 0.75:
        return "heading"
    if xy_delta > 1e-9:
        return "start"
    if yaw_delta > 1e-9:
        return "heading"
    return None


def choose_variant_by_kind(
    task: dict[str, Any],
    kind: str,
    scene_id: str,
    sample_seed: int,
) -> dict[str, Any] | None:
    candidates = [
        variant
        for variant in task.get("trajectory_variants") or []
        if isinstance(variant, dict) and recovery_kind(variant) == kind
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda variant: stable_key(sample_seed, scene_id, task.get("task_id"), kind, variant_id(variant)),
    )[0]


def selected_variants(task: dict[str, Any], scene_id: str, sample_seed: int) -> list[dict[str, Any]]:
    variants = [variant for variant in task.get("trajectory_variants") or [] if isinstance(variant, dict)]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for variant in variants:
        if variant_id(variant) == "nominal":
            selected.append(variant)
            seen.add("nominal")
            break

    for kind in ("start", "heading"):
        chosen = choose_variant_by_kind(task, kind, scene_id, sample_seed)
        if chosen is not None and variant_id(chosen) not in seen:
            selected.append(chosen)
            seen.add(variant_id(chosen))

    return selected


def scene_id_for_spec(spec_path: Path, spec: dict[str, Any]) -> str:
    scene = spec.get("scene") if isinstance(spec.get("scene"), dict) else {}
    return str(scene.get("scene_id") or spec.get("suite_id") or spec_path.stem)


def validate_selected_spec(
    spec_path: Path,
    *,
    max_start_distance_m: float,
    check_expert_support: bool,
    check_planner: bool,
    flip_isaac_y: bool,
    planner_overrides: dict[str, Any],
    sample_seed: int,
    full_planner_fallback: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = load_yaml(spec_path)
    scene_yaml = Path(spec["scene"]["source_yaml"])
    scene = load_yaml(scene_yaml)
    starts = get_start_poses(scene)
    scene_id = scene_id_for_spec(spec_path, spec)

    task = choose_follow_task(spec, require_valid_nominal=False)
    if task is None:
        return spec, [], [{"task": {}, "errors": ["no supported follow task found"]}]

    task_errors = validate_task(task, scene, starts, max_start_distance_m, check_expert_support)
    if task_errors:
        return spec, [], [{"task": task, "errors": task_errors}]

    valid_variants: list[dict[str, Any]] = []
    invalid_variants: list[str] = []
    for variant in selected_variants(task, scene_id, sample_seed):
        retried_with_full_planner = False
        if check_planner:
            accepted, reason, metrics = planner_accepts_variant(
                scene,
                scene_yaml,
                task,
                variant,
                flip_isaac_y,
                planner_overrides=planner_overrides,
            )
            if not accepted and planner_overrides and full_planner_fallback:
                fast_failure_reason = reason
                full_accepted, full_reason, full_metrics = planner_accepts_variant(
                    scene,
                    scene_yaml,
                    task,
                    variant,
                    flip_isaac_y,
                    planner_overrides=None,
                )
                if full_accepted:
                    accepted = True
                    reason = full_reason
                    metrics = dict(full_metrics)
                    metrics["fast_planner_retry_reason"] = fast_failure_reason
                    metrics["selected_validation_fallback"] = "full_planner_after_fast_failure"
                    retried_with_full_planner = True
        else:
            accepted, reason, metrics = True, None, {}

        if accepted:
            checked = dict(variant)
            checked["planner_validation"] = {"checked": bool(check_planner), "valid": True}
            if metrics:
                checked["planner_validation"]["quality"] = metrics
            if check_planner and planner_overrides and not retried_with_full_planner:
                checked["planner_validation"]["planner_preset"] = "fast"
            elif check_planner and retried_with_full_planner:
                checked["planner_validation"]["planner_preset"] = "full_fallback"
            if reason:
                checked["planner_validation"]["note"] = reason
            valid_variants.append(checked)
        else:
            invalid_variants.append(f"variant {variant_id(variant)}: {reason}")

    if not valid_variants:
        return spec, [], [{"task": task, "errors": ["no selected variants passed validation", *invalid_variants]}]

    filtered_task = dict(task)
    filtered_task["trajectory_variants"] = valid_variants
    filtered_task["selected_variant_validation_summary"] = {
        "checked": True,
        "selection_rule": "nominal plus one deterministic start recovery and one deterministic heading recovery",
        "selected_variants": [variant_id(variant) for variant in valid_variants],
        "invalid_selected_variants": invalid_variants,
    }
    return spec, [filtered_task], ([] if not invalid_variants else [{"task": task, "errors": invalid_variants}])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Task spec YAML file(s) or directories.")
    parser.add_argument("--max-start-distance-m", type=float, default=2.0)
    parser.add_argument("--skip-expert-support-check", action="store_true")
    parser.add_argument("--write-valid-output-dir", type=Path, required=True)
    parser.add_argument("--check-planner", action="store_true", default=True)
    parser.add_argument("--no-check-planner", dest="check_planner", action="store_false")
    parser.add_argument(
        "--planner-speed-preset",
        choices=("full", "fast"),
        default="fast",
        help="Default is fast because this command is for quick selected-variant validation.",
    )
    parser.add_argument("--planner-grid-resolution-m", type=float, default=None)
    parser.add_argument("--planner-yaw-resolution-deg", type=float, default=None)
    parser.add_argument("--planner-max-iterations", type=int, default=None)
    parser.add_argument("--planner-subgoal-spacing-m", type=float, default=None)
    parser.add_argument(
        "--per-spec-timeout-s",
        type=float,
        default=0.0,
        help="Skip a single spec if validation takes longer than this many seconds. 0 disables the timeout.",
    )
    parser.add_argument("--variant-sample-seed", type=int, default=11)
    parser.add_argument("--flip-isaac-y", action="store_true")
    parser.add_argument(
        "--full-planner-fallback",
        action="store_true",
        help="Retry failed fast-planner variants with the full planner before rejecting them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Revalidate specs even when the corresponding output YAML already exists.",
    )
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_paths = expand_inputs(args.inputs)
    if not spec_paths:
        raise SystemExit("No task spec YAML files found.")

    planner_overrides = planner_overrides_from_args(args)
    if planner_overrides and args.check_planner:
        print(f"Planner validation overrides: {planner_overrides}", flush=True)

    total_valid = 0
    total_invalid = 0
    total_skipped = 0
    failed_specs = 0
    for spec_path in spec_paths:
        output_path = args.write_valid_output_dir / spec_path.name
        if output_path.exists() and not args.force:
            total_skipped += 1
            print(f"{spec_path}: skipped; selected-validation output already exists at {output_path}", flush=True)
            continue

        try:
            spec, valid_tasks, invalid_tasks = run_with_timeout(
                float(args.per_spec_timeout_s),
                validate_selected_spec,
                spec_path,
                max_start_distance_m=float(args.max_start_distance_m),
                check_expert_support=not bool(args.skip_expert_support_check),
                check_planner=bool(args.check_planner),
                flip_isaac_y=bool(args.flip_isaac_y),
                planner_overrides=planner_overrides,
                sample_seed=int(args.variant_sample_seed),
                full_planner_fallback=bool(args.full_planner_fallback),
            )
        except PerSpecTimeout as exc:
            spec = load_yaml(spec_path)
            valid_tasks = []
            invalid_tasks = [{"task": {}, "errors": [str(exc)]}]
            print(f"{spec_path}: timed out after {float(args.per_spec_timeout_s):.1f}s", flush=True)
        total_valid += len(valid_tasks)
        total_invalid += len(invalid_tasks)
        if invalid_tasks:
            failed_specs += 1

        selected_count = 0
        if valid_tasks:
            selected_count = len(valid_tasks[0].get("trajectory_variants") or [])
        print(
            f"{spec_path}: {len(valid_tasks)} valid selected task(s), "
            f"{selected_count} valid selected variant(s), {len(invalid_tasks)} issue(s)",
            flush=True,
        )
        if args.verbose:
            for invalid in invalid_tasks:
                task = invalid.get("task") or {}
                print(f"  - {task.get('task_id', '<missing id>')}: {'; '.join(invalid['errors'])}", flush=True)

        filtered = dict(spec)
        filtered["tasks"] = valid_tasks
        filtered["validation_summary"] = {
            "source_spec": spec_path.as_posix(),
            "valid_tasks": len(valid_tasks),
            "invalid_tasks_or_selected_variants": len(invalid_tasks),
            "selection_rule": "one follow task; nominal plus one deterministic start recovery and one deterministic heading recovery",
            "variant_sample_seed": int(args.variant_sample_seed),
            "max_start_distance_m": float(args.max_start_distance_m),
            "expert_support_checked": not bool(args.skip_expert_support_check),
            "planner_checked": bool(args.check_planner),
            "planner_overrides": planner_overrides,
            "flip_isaac_y": bool(args.flip_isaac_y),
        }
        write_yaml(output_path, filtered)

    print(
        f"Selected-validation complete: {len(spec_paths)} specs, "
        f"{total_valid} valid tasks, {total_invalid} specs/variant selections with issues, "
        f"{total_skipped} skipped.",
        flush=True,
    )
    if total_invalid and not args.allow_invalid:
        raise SystemExit(f"{failed_specs} spec file(s) had invalid selected tasks/variants.")


if __name__ == "__main__":
    main()

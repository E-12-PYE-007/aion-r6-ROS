#!/usr/bin/env python3
"""Expand task trajectory variants into perturbed Isaac layout YAMLs."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


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


def normalize_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def find_task(task_spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in task_spec.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    available = [task.get("task_id", "<missing>") for task in task_spec.get("tasks", [])]
    raise ValueError(f"task_id {task_id!r} not found. Available: {available}")


def variants_for_task(task: dict[str, Any], requested_ids: set[str] | None) -> list[dict[str, Any]]:
    variants = task.get("trajectory_variants") or []
    selected = []
    for variant in variants:
        variant_id = str(variant.get("variant_id", ""))
        if requested_ids is not None and variant_id not in requested_ids:
            continue
        if has_nonzero_start_delta(variant):
            selected.append(variant)
    return selected


def has_nonzero_start_delta(variant: dict[str, Any]) -> bool:
    delta = variant.get("start_pose_delta")
    if not isinstance(delta, dict):
        return False
    return any(abs(float(delta.get(key, 0.0))) > 1e-9 for key in ("x_m", "y_m", "yaw_rad"))


def yaw_degrees_to_radians(yaw_deg: float) -> float:
    return math.radians(float(yaw_deg))


def yaw_radians_to_degrees(yaw_rad: float) -> float:
    return math.degrees(float(yaw_rad))


def wrap_degrees(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def apply_start_delta(rover_pose: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    if "position" not in rover_pose:
        raise ValueError("layout rover_pose has no position")
    if "yaw" not in rover_pose:
        raise ValueError("layout rover_pose has no yaw")

    delta = variant.get("start_pose_delta") or {}
    position = list(rover_pose["position"])
    while len(position) < 3:
        position.append(0.0)

    yaw_deg = float(rover_pose.get("yaw", 0.0))
    yaw_rad = yaw_degrees_to_radians(yaw_deg)
    dx = float(delta.get("x_m", 0.0))
    dy = float(delta.get("y_m", 0.0))
    dyaw = float(delta.get("yaw_rad", 0.0))

    world_dx = math.cos(yaw_rad) * dx - math.sin(yaw_rad) * dy
    world_dy = math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy

    perturbed = dict(rover_pose)
    perturbed["position"] = [
        float(position[0]) + world_dx,
        float(position[1]) + world_dy,
        float(position[2]),
    ]
    perturbed["yaw"] = wrap_degrees(yaw_deg + yaw_radians_to_degrees(dyaw))
    return perturbed


def output_name_for(layout: dict[str, Any], task_id: str, variant: dict[str, Any]) -> str:
    base_name = str(layout.get("output_name") or "layout").removesuffix(".usd")
    variant_id = normalize_name(str(variant["variant_id"]))
    task_name = normalize_name(task_id)
    return f"{base_name}__task_{task_name}__variant_{variant_id}"


def expand_layout(
    layout_path: Path,
    task_spec_path: Path,
    output_dir: Path,
    task_ids: list[str],
    variant_ids: set[str] | None,
) -> list[Path]:
    layout = load_yaml(layout_path)
    task_spec = load_yaml(task_spec_path)
    if "rover_pose" not in layout:
        raise ValueError(f"{layout_path} is not a generated Isaac layout YAML with rover_pose.")

    written_paths = []
    for task_id in task_ids:
        task = find_task(task_spec, task_id)
        for variant in variants_for_task(task, variant_ids):
            variant_id = str(variant["variant_id"])
            output_layout = copy.deepcopy(layout)
            new_output_name = output_name_for(layout, task_id, variant)
            output_layout["output_name"] = new_output_name
            output_layout["rover_pose"] = apply_start_delta(layout["rover_pose"], variant)
            output_layout["pose_variant"] = {
                "source_layout_yaml": layout_path.as_posix(),
                "source_task_spec": task_spec_path.as_posix(),
                "task_id": task_id,
                "variant_id": variant_id,
                "variant_type": variant.get("variant_type"),
                "recovery_case": variant.get("recovery_case"),
                "start_pose_delta": variant.get("start_pose_delta"),
                "base_rover_pose": layout["rover_pose"],
                "perturbed_rover_pose": output_layout["rover_pose"],
            }
            output_path = output_dir / f"{new_output_name}.yaml"
            write_yaml(output_path, output_layout)
            written_paths.append(output_path)
            print(f"Wrote {output_path}")
    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-yaml", required=True, type=Path, help="Generated Isaac layout YAML with rover_pose.")
    parser.add_argument("--task-spec", required=True, type=Path, help="Task spec YAML containing trajectory variants.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for perturbed layout YAMLs.")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        required=True,
        help="Task ID to expand. May be passed more than once.",
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        dest="variant_ids",
        default=None,
        help="Variant ID to expand. May be passed more than once. Defaults to every variant with a nonzero start_pose_delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_ids = set(args.variant_ids) if args.variant_ids else None
    written = expand_layout(
        layout_path=args.layout_yaml.resolve(),
        task_spec_path=args.task_spec.resolve(),
        output_dir=args.output_dir,
        task_ids=args.task_ids,
        variant_ids=variant_ids,
    )
    print(f"Generated {len(written)} perturbed layout YAML(s).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate one collected sim rollout folder."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} did not contain a JSON object.")
            records.append(record)
    return records


def pose_xy(record: dict[str, Any]) -> tuple[float, float] | None:
    pose = record.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
        return None
    try:
        return float(pose[1]), float(pose[2])
    except (TypeError, ValueError):
        return None


def path_distance(records: list[dict[str, Any]]) -> float:
    distance = 0.0
    previous = None
    for record in records:
        current = pose_xy(record)
        if current is None:
            continue
        if previous is not None:
            distance += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return distance


def displacement(records: list[dict[str, Any]]) -> float:
    points = [point for record in records if (point := pose_xy(record)) is not None]
    if len(points) < 2:
        return 0.0
    return math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])


def fraction_present(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record.get(key) is not None) / len(records)


def validate_images(rollout_dir: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    image_dir = rollout_dir / "img"
    image_files = list(image_dir.glob("*.jpg")) if image_dir.exists() else []
    missing_refs = 0
    for record in records:
        image_name = record.get("image")
        if isinstance(image_name, str) and not (image_dir / image_name).exists():
            missing_refs += 1
    return len(image_files), missing_refs


def task_validation_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    structured_task = metadata.get("structured_task")
    if isinstance(structured_task, dict) and isinstance(structured_task.get("validation"), dict):
        return structured_task["validation"]
    return {}


def validate_rollout_dir(
    rollout_dir: Path,
    expected_task_id: str | None = None,
    expected_variant_id: str | None = None,
    min_samples: int | None = None,
    min_motion_m: float | None = None,
    min_action_chunk_fraction: float = 0.5,
    min_cmd_vel_fraction: float = 0.5,
    allow_stationary: bool = False,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    metadata_path = rollout_dir / "metadata.json"
    poses_path = rollout_dir / "poses.jsonl"
    if not metadata_path.exists():
        errors.append("metadata.json is missing")
        metadata = {}
    else:
        metadata = load_json(metadata_path)
    if not poses_path.exists():
        errors.append("poses.jsonl is missing")
        records: list[dict[str, Any]] = []
    else:
        records = load_jsonl(poses_path)

    task_validation = task_validation_from_metadata(metadata)
    if min_samples is None:
        min_samples = int(task_validation.get("min_samples", 1))
    if min_motion_m is None:
        min_motion_m = float(task_validation.get("require_motion_m", 0.0))
    if allow_stationary:
        min_motion_m = 0.0

    image_count, missing_image_refs = validate_images(rollout_dir, records)
    action_chunk_fraction = fraction_present(records, "action_chunk")
    cmd_vel_fraction = fraction_present(records, "cmd_vel")
    travelled_m = path_distance(records)
    displacement_m = displacement(records)

    metrics.update(
        {
            "sample_count": len(records),
            "image_count": image_count,
            "missing_image_refs": missing_image_refs,
            "action_chunk_fraction": action_chunk_fraction,
            "cmd_vel_fraction": cmd_vel_fraction,
            "path_distance_m": travelled_m,
            "displacement_m": displacement_m,
            "min_samples": min_samples,
            "min_motion_m": min_motion_m,
        }
    )

    if expected_task_id and metadata.get("task_id") != expected_task_id:
        errors.append(f"metadata task_id {metadata.get('task_id')!r} != expected {expected_task_id!r}")
    if expected_variant_id and metadata.get("variant_id") != expected_variant_id:
        errors.append(f"metadata variant_id {metadata.get('variant_id')!r} != expected {expected_variant_id!r}")
    if len(records) < int(min_samples):
        errors.append(f"sample count {len(records)} < required {min_samples}")
    if image_count <= 0:
        errors.append("no JPEG images were saved")
    if missing_image_refs:
        errors.append(f"{missing_image_refs} poses.jsonl image references are missing files")
    if action_chunk_fraction < min_action_chunk_fraction:
        errors.append(
            f"action_chunk present in {action_chunk_fraction:.2%} of samples, "
            f"required {min_action_chunk_fraction:.2%}"
        )
    if cmd_vel_fraction < min_cmd_vel_fraction:
        errors.append(
            f"cmd_vel present in {cmd_vel_fraction:.2%} of samples, required {min_cmd_vel_fraction:.2%}"
        )
    if travelled_m < float(min_motion_m):
        errors.append(f"path distance {travelled_m:.3f}m < required {float(min_motion_m):.3f}m")
    if len(records) and image_count != len(records):
        warnings.append(f"image count {image_count} differs from sample count {len(records)}")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings, metrics=metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--expected-task-id", default=None)
    parser.add_argument("--expected-variant-id", default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-motion-m", type=float, default=None)
    parser.add_argument("--min-action-chunk-fraction", type=float, default=0.5)
    parser.add_argument("--min-cmd-vel-fraction", type=float, default=0.5)
    parser.add_argument("--allow-stationary", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_rollout_dir(
        rollout_dir=args.rollout_dir,
        expected_task_id=args.expected_task_id,
        expected_variant_id=args.expected_variant_id,
        min_samples=args.min_samples,
        min_motion_m=args.min_motion_m,
        min_action_chunk_fraction=args.min_action_chunk_fraction,
        min_cmd_vel_fraction=args.min_cmd_vel_fraction,
        allow_stationary=bool(args.allow_stationary),
    )
    payload = {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "metrics": result.metrics,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{args.rollout_dir}: {'valid' if result.valid else 'invalid'}")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        for error in result.errors:
            print(f"  error: {error}")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export collected sim rollouts to the AG-VLA edge-adapter training manifest.

Input rollout format, produced by sim_dataset_collector:

    rollout_dir/
      img/
        <timestamp>.jpg
      poses.jsonl
      metadata.json

Output per valid rollout:

    rollout_dir/
      target_waypoints.npy  # [T, 8, 3]
      target_async_actions.npy  # [T, 8, 4]
      timestamps.npy        # [T]

and one JSONL manifest where each line points AG-VLA at the rollout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WAYPOINT_CONVENTIONS = ("x_forward_y_left", "async_camera_like")


@dataclass
class ExportResult:
    rollout_dir: Path
    valid_samples: int
    skipped_samples: int
    reason: str | None = None


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


def action_chunk_waypoints(record: dict[str, Any]) -> np.ndarray | None:
    chunk = record.get("action_chunk")
    if not isinstance(chunk, dict):
        return None
    poses = chunk.get("relative_poses")
    if not isinstance(poses, list) or len(poses) != 8:
        return None
    try:
        arr = np.asarray(poses, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.shape != (8, 3) or not np.isfinite(arr).all():
        return None
    return arr


def waypoints_to_async_actions(
    waypoints: np.ndarray,
    *,
    spacing_m: float,
    convention: str,
) -> np.ndarray:
    """Convert robot-frame [x_m, y_m, yaw_rad] chunks to AsyncVLA [x, y, cos, sin].

    Raw sim labels are robot local frame: x forward in metres, y left in metres,
    yaw in radians. AsyncVLA-style actions use normalized x/y plus heading as
    cos/sin so the angle is continuous across the +/-pi boundary.
    """
    if convention not in WAYPOINT_CONVENTIONS:
        raise ValueError(f"Unknown waypoint convention {convention!r}; expected one of {WAYPOINT_CONVENTIONS}")
    if spacing_m <= 0.0:
        raise ValueError(f"async_action_spacing_m must be positive, got {spacing_m}")

    arr = np.asarray(waypoints, dtype=np.float32)
    if arr.shape[-2:] != (8, 3):
        raise ValueError(f"Expected waypoint chunk shape [..., 8, 3], got {arr.shape}")

    x = arr[..., 0] / float(spacing_m)
    y = arr[..., 1] / float(spacing_m)
    yaw = arr[..., 2]

    if convention == "async_camera_like":
        y = -y
        yaw = -yaw

    actions = np.stack([x, y, np.cos(yaw), np.sin(yaw)], axis=-1)
    return actions.astype(np.float32)


def image_exists(rollout_dir: Path, record: dict[str, Any]) -> bool:
    image = record.get("image")
    return isinstance(image, str) and (rollout_dir / "img" / image).exists()


def record_time(record: dict[str, Any]) -> float | None:
    value = record.get("img_time", record.get("anchor_time"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_rollouts(input_root: Path) -> list[Path]:
    if (input_root / "poses.jsonl").exists():
        return [input_root]
    return sorted(path.parent for path in input_root.rglob("poses.jsonl"))


def export_rollout(
    rollout_dir: Path,
    *,
    min_samples: int,
    overwrite: bool,
    async_action_spacing_m: float,
    waypoint_convention: str,
) -> tuple[ExportResult, dict[str, Any] | None]:
    metadata_path = rollout_dir / "metadata.json"
    poses_path = rollout_dir / "poses.jsonl"
    if not metadata_path.exists():
        return ExportResult(rollout_dir, 0, 0, "missing metadata.json"), None
    if not poses_path.exists():
        return ExportResult(rollout_dir, 0, 0, "missing poses.jsonl"), None

    metadata = load_json(metadata_path)
    records = load_jsonl(poses_path)
    images: list[str] = []
    timestamps: list[float] = []
    waypoints: list[np.ndarray] = []
    skipped = 0

    for record in records:
        chunk = action_chunk_waypoints(record)
        time_s = record_time(record)
        if chunk is None or time_s is None or not image_exists(rollout_dir, record):
            skipped += 1
            continue
        images.append(str(Path("img") / str(record["image"])).replace("\\", "/"))
        timestamps.append(time_s)
        waypoints.append(chunk)

    if len(waypoints) < min_samples:
        return ExportResult(rollout_dir, len(waypoints), skipped, f"valid samples < {min_samples}"), None

    target_path = rollout_dir / "target_waypoints.npy"
    async_actions_path = rollout_dir / "target_async_actions.npy"
    timestamps_path = rollout_dir / "timestamps.npy"
    waypoint_array = np.stack(waypoints).astype(np.float32)
    async_actions = waypoints_to_async_actions(
        waypoint_array,
        spacing_m=async_action_spacing_m,
        convention=waypoint_convention,
    )
    if overwrite or not target_path.exists():
        np.save(target_path, waypoint_array)
    if overwrite or not async_actions_path.exists():
        np.save(async_actions_path, async_actions)
    if overwrite or not timestamps_path.exists():
        np.save(timestamps_path, np.asarray(timestamps, dtype=np.float32))

    instruction = (
        metadata.get("language_instruction")
        or metadata.get("instruction")
        or metadata.get("task_id")
        or "navigate safely"
    )
    manifest_record = {
        "episode_id": str(metadata.get("trajectory_name") or rollout_dir.name),
        "root": rollout_dir.resolve().as_posix(),
        "images": images,
        "timestamps_path": "timestamps.npy",
        "instruction": str(instruction),
        "target_waypoints_path": "target_waypoints.npy",
        "target_async_actions_path": "target_async_actions.npy",
        "metadata": {
            "dataset_name": metadata.get("dataset_name"),
            "task_id": metadata.get("task_id"),
            "variant_id": metadata.get("variant_id"),
            "variant_type": metadata.get("variant_type"),
            "recovery_case": metadata.get("recovery_case"),
            "source_format": metadata.get("format", "stream_jsonl"),
            "raw_waypoint_format": "x_forward_m_y_left_m_yaw_rad",
            "async_action_format": "x_over_spacing_y_over_spacing_cos_yaw_sin_yaw",
            "async_action_spacing_m": float(async_action_spacing_m),
            "waypoint_convention": waypoint_convention,
        },
    }
    return ExportResult(rollout_dir, len(waypoints), skipped), manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="One rollout dir or a root containing many rollout dirs.")
    parser.add_argument("--out-manifest", type=Path, required=True, help="Output JSONL manifest for AG-VLA.")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--async-action-spacing-m",
        type=float,
        default=0.125,
        help="Distance scale used when converting waypoint metres to AsyncVLA normalized x/y actions.",
    )
    parser.add_argument(
        "--waypoint-convention",
        choices=WAYPOINT_CONVENTIONS,
        default="x_forward_y_left",
        help="Convention used for converting raw robot-frame waypoint labels to AsyncVLA actions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rollouts = discover_rollouts(args.input_root.resolve())
    if not rollouts:
        print(f"No rollout folders found under {args.input_root}")
        return 1

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    results: list[ExportResult] = []
    for rollout_dir in rollouts:
        result, record = export_rollout(
            rollout_dir,
            min_samples=int(args.min_samples),
            overwrite=bool(args.overwrite),
            async_action_spacing_m=float(args.async_action_spacing_m),
            waypoint_convention=str(args.waypoint_convention),
        )
        results.append(result)
        if record is not None:
            manifest_records.append(record)

    with args.out_manifest.open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record) + "\n")

    total_samples = sum(result.valid_samples for result in results if result.reason is None)
    print(f"Found {len(rollouts)} rollout(s).")
    print(f"Exported {len(manifest_records)} episode(s), total valid samples={total_samples}.")
    print(f"Wrote {args.out_manifest}")

    failures = [result for result in results if result.reason is not None]
    if failures:
        print(f"Skipped {len(failures)} rollout(s):")
        for result in failures[:20]:
            print(f"  {result.rollout_dir}: {result.reason} ({result.valid_samples} valid, {result.skipped_samples} skipped)")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_root": args.input_root.resolve().as_posix(),
            "out_manifest": args.out_manifest.resolve().as_posix(),
            "num_rollouts": len(rollouts),
            "num_exported": len(manifest_records),
            "total_valid_samples": total_samples,
            "results": [result.__dict__ | {"rollout_dir": result.rollout_dir.as_posix()} for result in results],
        }
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if manifest_records else 1


if __name__ == "__main__":
    raise SystemExit(main())

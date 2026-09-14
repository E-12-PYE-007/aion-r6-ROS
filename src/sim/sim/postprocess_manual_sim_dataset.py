#!/usr/bin/env python3
"""Postprocess manually driven Isaac rollouts into action-chunk training labels."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from sim.export_edge_training_manifest import WAYPOINT_CONVENTIONS, waypoints_to_async_actions


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    return records


def record_pose(record: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    pose = record.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 4:
        return None
    x = finite_float(pose[1])
    y = finite_float(pose[2])
    yaw = finite_float(pose[3])
    if x is None or y is None or yaw is None:
        return None
    return np.asarray([x, y], dtype=np.float64), float(yaw)


def record_time(record: dict[str, Any]) -> float | None:
    return finite_float(record.get("img_time", record.get("anchor_time")))


def discover_rollouts(root: Path) -> list[Path]:
    if (root / "poses.jsonl").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("poses.jsonl"))


def cumulative_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def interp_angle(a: float, b: float, t: float) -> float:
    return wrap_to_pi(a + wrap_to_pi(b - a) * t)


def sample_future(points: np.ndarray, yaws: np.ndarray, distances: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    target = min(max(float(target), float(distances[0])), float(distances[-1]))
    index = int(np.searchsorted(distances, target, side="right") - 1)
    index = min(max(index, 0), len(distances) - 2)
    start_d = float(distances[index])
    end_d = float(distances[index + 1])
    t = 0.0 if end_d <= start_d + 1e-9 else (target - start_d) / (end_d - start_d)
    position = points[index] + (points[index + 1] - points[index]) * t
    yaw = interp_angle(float(yaws[index]), float(yaws[index + 1]), t)
    return position, yaw


def world_to_robot(origin: np.ndarray, yaw: float, target: np.ndarray, target_yaw: float) -> list[float]:
    delta = target - origin
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    x = c * float(delta[0]) - s * float(delta[1])
    y = s * float(delta[0]) + c * float(delta[1])
    return [x, y, wrap_to_pi(target_yaw - yaw)]


def image_exists(rollout_dir: Path, record: dict[str, Any]) -> bool:
    image = record.get("image")
    return isinstance(image, str) and (rollout_dir / "img" / image).exists()


def resize_image(image: Image.Image, size: int, mode: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == "stretch":
        return image.resize((size, size), Image.Resampling.LANCZOS)
    width, height = image.size
    if mode == "center_crop":
        crop = min(width, height)
        left = (width - crop) // 2
        top = (height - crop) // 2
        return image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
    if mode == "letterbox":
        scale = min(size / width, size / height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(resized, ((size - resized.size[0]) // 2, (size - resized.size[1]) // 2))
        return canvas
    raise ValueError(f"Unsupported resize mode {mode!r}")


@dataclass
class ManualResult:
    rollout_dir: Path
    accepted: bool
    sample_count: int = 0
    skipped_samples: int = 0
    reason: str | None = None


def postprocess_rollout(
    rollout_dir: Path,
    output_dir: Path,
    *,
    chunk_size: int,
    first_preview_m: float,
    waypoint_spacing_m: float,
    min_samples: int,
    min_motion_m: float,
    max_abs_y_m: float,
    min_x_m: float,
    async_action_spacing_m: float,
    waypoint_convention: str,
    image_size: int,
    resize_mode: str,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[ManualResult, dict[str, Any] | None]:
    metadata = load_json(rollout_dir / "metadata.json")
    raw_records = load_jsonl(rollout_dir / "poses.jsonl")
    records = [record for record in raw_records if record_pose(record) is not None and record_time(record) is not None and image_exists(rollout_dir, record)]
    if len(records) < min_samples:
        return ManualResult(rollout_dir, False, len(records), len(raw_records) - len(records), "not enough image+pose samples"), None

    poses = [record_pose(record) for record in records]
    points = np.asarray([pose[0] for pose in poses if pose is not None], dtype=np.float64)
    yaws = np.asarray([pose[1] for pose in poses if pose is not None], dtype=np.float64)
    distances = cumulative_distances(points)
    if float(distances[-1]) < min_motion_m:
        return ManualResult(rollout_dir, False, len(records), 0, f"motion {distances[-1]:.2f}m < {min_motion_m:.2f}m"), None

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "img").mkdir(exist_ok=True)

    waypoints: list[np.ndarray] = []
    timestamps: list[float] = []
    kept_records: list[dict[str, Any]] = []
    debug_samples: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(records):
        if index >= len(points):
            skipped += 1
            continue
        current_distance = float(distances[index])
        if current_distance + first_preview_m + (chunk_size - 1) * waypoint_spacing_m > float(distances[-1]):
            skipped += 1
            continue
        current_position = points[index]
        current_yaw = float(yaws[index])
        chunk = []
        for waypoint_index in range(chunk_size):
            target_distance = current_distance + first_preview_m + waypoint_index * waypoint_spacing_m
            target_position, target_yaw = sample_future(points, yaws, distances, target_distance)
            chunk.append(world_to_robot(current_position, current_yaw, target_position, target_yaw))
        arr = np.asarray(chunk, dtype=np.float32)
        if not np.isfinite(arr).all() or (arr[:, 0] < min_x_m).any() or (np.abs(arr[:, 1]) > max_abs_y_m).any():
            skipped += 1
            continue
        image = str(record["image"])
        source_image = rollout_dir / "img" / image
        dest_image = output_dir / "img" / image
        if overwrite or not dest_image.exists():
            resized = resize_image(Image.open(source_image), image_size, resize_mode)
            resized.save(dest_image, quality=jpeg_quality)
        waypoints.append(arr)
        timestamps.append(float(record_time(record)))
        kept_records.append(record)
        debug_samples.append(
            {
                "sample_index": index,
                "image": image,
                "time_s": float(record_time(record)),
                "path_progress_m": current_distance,
                "first_waypoint": arr[0].astype(float).tolist(),
                "last_waypoint": arr[-1].astype(float).tolist(),
            }
        )

    if len(waypoints) < min_samples:
        return ManualResult(rollout_dir, False, len(waypoints), skipped, f"generated label samples < {min_samples}"), None

    waypoint_array = np.stack(waypoints).astype(np.float32)
    async_actions = waypoints_to_async_actions(
        waypoint_array,
        spacing_m=async_action_spacing_m,
        convention=waypoint_convention,
    )
    np.save(output_dir / "target_waypoints.npy", waypoint_array)
    np.save(output_dir / "target_async_actions.npy", async_actions)
    np.save(output_dir / "timestamps.npy", np.asarray(timestamps, dtype=np.float32))
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "manual_postprocess": {
                    "label_source": "future_manual_pose_trajectory",
                    "motion_m": float(distances[-1]),
                    "sample_count": len(waypoints),
                    "skipped_samples": skipped,
                    "chunk_size": chunk_size,
                    "first_preview_m": first_preview_m,
                    "waypoint_spacing_m": waypoint_spacing_m,
                    "waypoint_convention": waypoint_convention,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "poses.jsonl").open("w", encoding="utf-8") as f:
        for record in kept_records:
            cleaned = dict(record)
            cleaned.pop("action_chunk", None)
            f.write(json.dumps(cleaned) + "\n")
    with (output_dir / "postprocessed_samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in debug_samples:
            f.write(json.dumps(sample) + "\n")

    manifest_record = {
        "trajectory_dir": output_dir.as_posix(),
        "image_paths": [str(Path("img") / str(record["image"])).replace("\\", "/") for record in kept_records],
        "target_waypoints_path": "target_waypoints.npy",
        "target_async_actions_path": "target_async_actions.npy",
        "timestamps_path": "timestamps.npy",
        "sample_count": len(waypoints),
        "language_instruction": metadata.get("language_instruction", "Follow the fence."),
        "task_id": metadata.get("task_id", "manual_drive"),
        "variant_id": metadata.get("variant_id", "manual"),
        "label_source": "future_manual_pose_trajectory",
        "pose_source": "isaac_camera_pose_debug",
    }
    return ManualResult(rollout_dir, True, len(waypoints), skipped), manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=("letterbox", "center_crop", "stretch"), default="center_crop")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-motion-m", type=float, default=1.5)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--first-preview-m", type=float, default=0.35)
    parser.add_argument("--waypoint-spacing-m", type=float, default=0.18)
    parser.add_argument("--async-action-spacing-m", type=float, default=0.125)
    parser.add_argument("--min-x-m", type=float, default=-0.05)
    parser.add_argument("--max-abs-y-m", type=float, default=2.0)
    parser.add_argument("--waypoint-convention", choices=WAYPOINT_CONVENTIONS, default="x_forward_y_left")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rollouts = discover_rollouts(args.input_root.resolve())
    args.export_root.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or args.out_manifest.with_suffix(".summary.json")
    results: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for rollout in rollouts:
        output_dir = args.export_root / rollout.name
        result, record = postprocess_rollout(
            rollout,
            output_dir,
            chunk_size=args.chunk_size,
            first_preview_m=args.first_preview_m,
            waypoint_spacing_m=args.waypoint_spacing_m,
            min_samples=args.min_samples,
            min_motion_m=args.min_motion_m,
            max_abs_y_m=args.max_abs_y_m,
            min_x_m=args.min_x_m,
            async_action_spacing_m=args.async_action_spacing_m,
            waypoint_convention=args.waypoint_convention,
            image_size=args.image_size,
            resize_mode=args.resize_mode,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
        results.append(
            {
                "rollout_dir": result.rollout_dir.as_posix(),
                "accepted": result.accepted,
                "sample_count": result.sample_count,
                "skipped_samples": result.skipped_samples,
                "reason": result.reason,
            }
        )
        if record:
            manifest.append(record)
    args.out_manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    summary = {
        "input_root": args.input_root.as_posix(),
        "export_root": args.export_root.as_posix(),
        "accepted_rollouts": len(manifest),
        "rejected_rollouts": len(rollouts) - len(manifest),
        "samples": int(sum(row["sample_count"] for row in manifest)),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Accepted {len(manifest)} rollout(s), rejected {len(rollouts) - len(manifest)}.")
    print(f"Wrote manifest: {args.out_manifest}")
    print(f"Wrote summary: {summary_path}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())

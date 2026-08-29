#!/usr/bin/env python3
"""Validate collected rollouts and generate final training labels.

This is intentionally a post-collection step. Raw rollout folders can contain
live ActionChunk messages that were useful for driving/debugging the robot, but
the final dataset labels are regenerated deterministically from the saved
runtime planned path and the recorded pose at each image timestamp.
"""

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

from sim.expert_trajectory_utils import (
    find_task,
    load_yaml,
    path_length,
    project_progress_near,
    sample_distance_action_target,
    world_to_robot,
)
from sim.export_edge_training_manifest import WAYPOINT_CONVENTIONS, waypoints_to_async_actions
from sim.trajectory_profile import TimedTrajectory
from sim.validate_collected_rollout import ValidationResult, resolve_existing_path, validate_rollout_dir


@dataclass
class PostprocessResult:
    rollout_dir: Path
    accepted: bool
    sample_count: int = 0
    skipped_samples: int = 0
    reason: str | None = None
    warnings: list[str] | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class SegmentSelection:
    records: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
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
            if isinstance(record, dict):
                records.append(record)
    return records


def discover_rollouts(input_root: Path) -> list[Path]:
    if (input_root / "poses.jsonl").exists():
        return [input_root]
    return sorted(path.parent for path in input_root.rglob("poses.jsonl"))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def record_pose(record: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    pose = record.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 4:
        return None
    x = finite_float(pose[1])
    y = finite_float(pose[2])
    yaw = finite_float(pose[3])
    if x is None or y is None or yaw is None:
        return None
    return np.asarray([x, y], dtype=np.float64), yaw


def record_time(record: dict[str, Any]) -> float | None:
    value = record.get("img_time", record.get("anchor_time"))
    return finite_float(value)


def image_exists(rollout_dir: Path, record: dict[str, Any]) -> bool:
    image = record.get("image")
    return isinstance(image, str) and (rollout_dir / "img" / image).exists()


def lanczos_resample() -> int:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return int(resampling.LANCZOS)
    return int(Image.LANCZOS)


def letterbox_image(image: Image.Image, size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot resize image with invalid size {image.size}")
    scale = min(size / width, size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, lanczos_resample())
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def resize_image(image: Image.Image, size: int, mode: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == "letterbox":
        return letterbox_image(image, size)
    if mode == "center_crop":
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        cropped = image.crop((left, top, left + side, top + side))
        return cropped.resize((size, size), lanczos_resample())
    if mode == "stretch":
        return image.resize((size, size), lanczos_resample())
    raise ValueError(f"Unsupported resize mode: {mode}")


def closest_point_and_distance(polyline: list[np.ndarray], point: np.ndarray) -> tuple[np.ndarray, float]:
    if not polyline:
        return point.copy(), 0.0
    if len(polyline) == 1:
        nearest = polyline[0]
        return nearest, float(np.linalg.norm(point - nearest))
    best_point = polyline[0]
    best_distance = math.inf
    for index in range(len(polyline) - 1):
        start = polyline[index]
        end = polyline[index + 1]
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-9:
            candidate = start
        else:
            t = float(np.dot(point - start, segment) / length_sq)
            candidate = start + segment * max(0.0, min(1.0, t))
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
    return best_point, best_distance


def transformed_point(point: list[float] | tuple[float, ...], flip_y: bool) -> np.ndarray:
    x = float(point[0])
    y = float(point[1])
    return np.asarray([x, -y if flip_y else y], dtype=np.float64)


def signed_side(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length <= 1e-9:
        return 0.0
    rel = point - start
    return float((segment[0] * rel[1] - segment[1] * rel[0]) / length)


def distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / length_sq)
    nearest = start + segment * max(0.0, min(1.0, t))
    return float(np.linalg.norm(point - nearest))


def target_fence_segments(rollout_dir: Path, metadata: dict[str, Any]) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    structured_task = metadata.get("structured_task") if isinstance(metadata.get("structured_task"), dict) else {}
    task_type = str(structured_task.get("task_type", ""))
    if task_type not in {"follow_fence", "follow_fence_sequence"}:
        return [], 0.0

    task_spec_path = resolve_existing_path(str(metadata.get("task_spec", "")), rollout_dir)
    if task_spec_path is None:
        return [], 0.0

    try:
        task_spec = load_yaml(task_spec_path)
        task_id = str(metadata.get("task_id") or "")
        task = find_task(task_spec, task_id)
        scene_path = Path(task_spec["scene"]["source_yaml"])
        scene = load_yaml(scene_path)
    except Exception:
        return [], 0.0

    target_names = structured_task.get("target_fences")
    if not isinstance(target_names, list) or not target_names:
        target_fence = structured_task.get("target_fence")
        target_names = [target_fence] if isinstance(target_fence, str) else []
    target_name_set = {str(name) for name in target_names}
    flip_scene_y = bool(metadata.get("flip_scene_y", False))
    expected_path_side = str(structured_task.get("path_side", ""))
    expected_sign = 1.0 if expected_path_side == "left" else -1.0 if expected_path_side == "right" else 0.0
    segments = []
    for fence in scene.get("fences", []):
        if fence.get("name") not in target_name_set:
            continue
        segments.append((transformed_point(fence["start"], flip_scene_y), transformed_point(fence["end"], flip_scene_y)))
    return segments, expected_sign


def image_is_nonblack(rollout_dir: Path, image_name: str, *, min_mean: float, min_max: float) -> bool:
    try:
        with Image.open(rollout_dir / "img" / image_name) as image:
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.uint8)
    except Exception:
        return False
    return float(pixels.mean()) >= min_mean and float(pixels.max()) >= min_max


def choose_longest_valid_segment(
    rollout_dir: Path,
    records: list[dict[str, Any]],
    trajectory: TimedTrajectory,
    metadata: dict[str, Any],
    *,
    enabled: bool,
    max_tracking_error_m: float,
    min_progress_m: float,
    min_samples: int,
    min_target_fence_clearance_m: float | None,
    enforce_fence_side: bool,
    check_black_images: bool,
    black_image_min_mean: float,
    black_image_min_max: float,
) -> SegmentSelection:
    if not enabled:
        return SegmentSelection(
            records=records,
            diagnostics=[],
            metrics={
                "enabled": False,
                "input_samples": len(records),
                "selected_samples": len(records),
                "selected_start_index": 0,
                "selected_end_index": max(len(records) - 1, -1),
            },
        )

    fence_segments, expected_side_sign = target_fence_segments(rollout_dir, metadata)
    candidates: list[dict[str, Any]] = []
    previous_progress: float | None = None
    invalid_reasons: dict[str, int] = {}

    for index, record in enumerate(records):
        reason: str | None = None
        pose = record_pose(record)
        time_s = record_time(record)
        image_name = record.get("image")
        if pose is None:
            reason = "missing_pose"
        elif time_s is None:
            reason = "missing_time"
        elif not isinstance(image_name, str) or not image_exists(rollout_dir, record):
            reason = "missing_image"
        position = np.zeros(2, dtype=np.float64)
        yaw = 0.0
        progress = 0.0
        tracking_error = math.inf
        fence_distance: float | None = None
        side_distance: float | None = None
        if reason is None and pose is not None:
            position, yaw = pose
            progress = project_progress_near(
                trajectory.path,
                position,
                previous_progress,
                max_backward_m=0.5,
                max_forward_m=2.0,
            )
            progress = max(float(previous_progress or 0.0), float(progress))
            previous_progress = progress
            _, tracking_error = closest_point_and_distance(trajectory.path, position)
            if tracking_error > max_tracking_error_m:
                reason = "tracking_error"
            if reason is None and fence_segments:
                nearest_distance = math.inf
                nearest_side = 0.0
                for start, end in fence_segments:
                    distance = distance_to_segment(position, start, end)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_side = signed_side(position, start, end)
                fence_distance = nearest_distance
                side_distance = nearest_side
                if min_target_fence_clearance_m is not None and fence_distance < min_target_fence_clearance_m:
                    reason = "target_fence_clearance"
                elif enforce_fence_side and expected_side_sign and fence_distance < 1.8:
                    if side_distance * expected_side_sign < 0.15:
                        reason = "target_fence_side"
            if reason is None and check_black_images and isinstance(image_name, str):
                if not image_is_nonblack(
                    rollout_dir,
                    image_name,
                    min_mean=black_image_min_mean,
                    min_max=black_image_min_max,
                ):
                    reason = "black_image"
        if reason is not None:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        candidates.append(
            {
                "index": index,
                "valid": reason is None,
                "reason": reason,
                "progress_m": progress,
                "tracking_error_m": tracking_error,
                "fence_distance_m": fence_distance,
                "side_distance_m": side_distance,
                "record": record,
            }
        )

    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["valid"]:
            current.append(candidate)
            continue
        if segment_score(current) > segment_score(best):
            best = current
        current = []
    if segment_score(current) > segment_score(best):
        best = current

    if not best:
        selected_records: list[dict[str, Any]] = []
        start_index = -1
        end_index = -1
        progress_delta = 0.0
    else:
        selected_records = [candidate["record"] for candidate in best]
        start_index = int(best[0]["index"])
        end_index = int(best[-1]["index"])
        progress_delta = float(best[-1]["progress_m"] - best[0]["progress_m"])

    accepted_segment = len(selected_records) >= min_samples and progress_delta >= min_progress_m
    metrics = {
        "enabled": True,
        "accepted_segment": accepted_segment,
        "input_samples": len(records),
        "selected_samples": len(selected_records),
        "selected_start_index": start_index,
        "selected_end_index": end_index,
        "selected_progress_m": progress_delta,
        "min_required_progress_m": min_progress_m,
        "min_required_samples": min_samples,
        "max_tracking_error_m": max_tracking_error_m,
        "min_target_fence_clearance_m": min_target_fence_clearance_m,
        "enforce_fence_side": enforce_fence_side,
        "invalid_reasons": invalid_reasons,
        "selected_max_tracking_error_m": max((float(c["tracking_error_m"]) for c in best), default=0.0),
        "selected_min_target_fence_distance_m": min(
            (float(c["fence_distance_m"]) for c in best if c["fence_distance_m"] is not None),
            default=None,
        ),
    }
    return SegmentSelection(records=selected_records, diagnostics=candidates, metrics=metrics)


def segment_score(segment: list[dict[str, Any]]) -> tuple[float, int]:
    if not segment:
        return (0.0, 0)
    progress = float(segment[-1]["progress_m"] - segment[0]["progress_m"])
    return (progress, len(segment))


def copy_clean_rollout(
    source_rollout_dir: Path,
    export_rollout_dir: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    image_size: int,
    resize_mode: str,
    overwrite: bool,
    jpeg_quality: int,
) -> None:
    if export_rollout_dir.exists() and overwrite:
        shutil.rmtree(export_rollout_dir)
    export_rollout_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = export_rollout_dir / "img"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    metadata_source = source_rollout_dir / "metadata.json"
    if metadata_source.exists():
        metadata_destination = export_rollout_dir / "metadata.json"
        if overwrite or not metadata_destination.exists():
            shutil.copy2(metadata_source, metadata_destination)

    if records is None:
        poses_source = source_rollout_dir / "poses.jsonl"
        if poses_source.exists():
            poses_destination = export_rollout_dir / "poses.jsonl"
            if overwrite or not poses_destination.exists():
                shutil.copy2(poses_source, poses_destination)
    else:
        poses_destination = export_rollout_dir / "poses.jsonl"
        if overwrite or not poses_destination.exists():
            with poses_destination.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

    source_img_dir = source_rollout_dir / "img"
    if not source_img_dir.exists():
        raise FileNotFoundError(f"{source_img_dir} is missing")
    selected_images = None
    if records is not None:
        selected_images = {str(record.get("image")) for record in records if isinstance(record.get("image"), str)}
    for source_image in sorted(source_img_dir.iterdir()):
        if not source_image.is_file() or source_image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if selected_images is not None and source_image.name not in selected_images:
            continue
        destination_image = image_output_dir / source_image.name
        if destination_image.exists() and not overwrite:
            continue
        with Image.open(source_image) as image:
            resized = resize_image(image, image_size, resize_mode)
            save_kwargs: dict[str, Any] = {}
            if destination_image.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs = {"quality": int(jpeg_quality), "optimize": True}
            resized.save(destination_image, **save_kwargs)


def load_runtime_trajectory(rollout_dir: Path) -> TimedTrajectory:
    path = rollout_dir / "runtime_planned_path.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; rerun rollouts with runtime_planned_path_output enabled")
    payload = load_json(path)
    raw_path = payload.get("path")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise ValueError(f"{path} does not contain a usable planned path")
    points = [np.asarray([float(point[0]), float(point[1])], dtype=np.float64) for point in raw_path]
    distances = np.concatenate(([0.0], np.cumsum([float(np.linalg.norm(points[i + 1] - points[i])) for i in range(len(points) - 1)])))
    zeros = np.zeros(len(points), dtype=np.float64)
    return TimedTrajectory(path=points, times=distances.copy(), distances=distances, yaws=zeros, speeds=zeros, yaw_rates=zeros)


def generate_action_chunk_for_pose(
    trajectory: TimedTrajectory,
    position: np.ndarray,
    yaw: float,
    progress_m: float,
    *,
    chunk_size: int,
    first_preview_m: float,
    waypoint_spacing_m: float,
) -> tuple[np.ndarray, float]:
    poses: list[list[float]] = []
    last_target_distance: float | None = None
    for index in range(chunk_size):
        preview_m = first_preview_m + index * waypoint_spacing_m
        min_target_distance = (
            last_target_distance + waypoint_spacing_m
            if last_target_distance is not None
            else None
        )
        target_position, target_yaw, target_distance = sample_distance_action_target(
            trajectory,
            position,
            yaw,
            progress_m,
            preview_m,
            min_forward_x_m=0.08,
            max_lateral_y_m=1.75,
            max_forward_search_m=3.0,
            search_step_m=0.15,
            min_target_distance=min_target_distance,
        )
        x, y, theta = world_to_robot(position, yaw, target_position, target_yaw)
        poses.append([float(x), float(y), float(theta)])
        last_target_distance = float(target_distance)
    return np.asarray(poses, dtype=np.float32), float(last_target_distance or progress_m)


def build_labels_for_rollout(
    rollout_dir: Path,
    *,
    output_rollout_dir: Path | None = None,
    min_samples: int,
    overwrite: bool,
    chunk_size: int,
    first_preview_m: float,
    waypoint_spacing_m: float,
    async_action_spacing_m: float,
    waypoint_convention: str,
    segment_filter: bool,
    min_segment_progress_m: float,
    max_segment_tracking_error_m: float,
    min_segment_target_fence_clearance_m: float | None,
    enforce_segment_fence_side: bool,
    check_segment_black_images: bool,
    black_image_min_mean: float,
    black_image_min_max: float,
    image_size: int,
    resize_mode: str,
    jpeg_quality: int,
) -> tuple[PostprocessResult, dict[str, Any] | None]:
    output_dir = output_rollout_dir or rollout_dir
    metadata = load_json(rollout_dir / "metadata.json")
    records = load_jsonl(rollout_dir / "poses.jsonl")
    trajectory = load_runtime_trajectory(rollout_dir)
    segment = choose_longest_valid_segment(
        rollout_dir,
        records,
        trajectory,
        metadata,
        enabled=segment_filter,
        max_tracking_error_m=max_segment_tracking_error_m,
        min_progress_m=min_segment_progress_m,
        min_samples=min_samples,
        min_target_fence_clearance_m=min_segment_target_fence_clearance_m,
        enforce_fence_side=enforce_segment_fence_side,
        check_black_images=check_segment_black_images,
        black_image_min_mean=black_image_min_mean,
        black_image_min_max=black_image_min_max,
    )
    if segment_filter and not bool(segment.metrics.get("accepted_segment")):
        return (
            PostprocessResult(
                rollout_dir=rollout_dir,
                accepted=False,
                sample_count=int(segment.metrics.get("selected_samples", 0)),
                skipped_samples=len(records) - int(segment.metrics.get("selected_samples", 0)),
                reason=(
                    "no clean segment met postprocess thresholds "
                    f"({segment.metrics.get('selected_samples', 0)} samples, "
                    f"{float(segment.metrics.get('selected_progress_m', 0.0)):.3f}m progress)"
                ),
                metrics={"segment": segment.metrics},
            ),
            None,
        )
    records = segment.records
    if output_rollout_dir is not None:
        copy_clean_rollout(
            rollout_dir,
            output_rollout_dir,
            records=records,
            image_size=image_size,
            resize_mode=resize_mode,
            overwrite=overwrite,
            jpeg_quality=jpeg_quality,
        )

    images: list[str] = []
    timestamps: list[float] = []
    waypoints: list[np.ndarray] = []
    debug_records: list[dict[str, Any]] = []
    skipped = 0
    previous_progress: float | None = None

    for index, record in enumerate(records):
        pose = record_pose(record)
        time_s = record_time(record)
        if pose is None or time_s is None or not image_exists(rollout_dir, record):
            skipped += 1
            continue
        position, yaw = pose
        progress = project_progress_near(
            trajectory.path,
            position,
            previous_progress,
            max_backward_m=0.5,
            max_forward_m=2.0,
        )
        progress = max(float(previous_progress or 0.0), float(progress))
        previous_progress = progress
        chunk, target_progress = generate_action_chunk_for_pose(
            trajectory,
            position,
            yaw,
            progress,
            chunk_size=chunk_size,
            first_preview_m=first_preview_m,
            waypoint_spacing_m=waypoint_spacing_m,
        )
        if chunk.shape != (chunk_size, 3) or not np.isfinite(chunk).all():
            skipped += 1
            continue
        images.append(str(Path("img") / str(record["image"])).replace("\\", "/"))
        timestamps.append(float(time_s))
        waypoints.append(chunk)
        debug_records.append(
            {
                "sample_index": index,
                "image": record.get("image"),
                "time_s": float(time_s),
                "path_progress_m": float(progress),
                "target_progress_m": float(target_progress),
                "first_waypoint": chunk[0].astype(float).tolist(),
                "last_waypoint": chunk[-1].astype(float).tolist(),
            }
        )

    if len(waypoints) < min_samples:
        return (
            PostprocessResult(
                rollout_dir=rollout_dir,
                accepted=False,
                sample_count=len(waypoints),
                skipped_samples=skipped,
                reason=f"generated label samples < {min_samples}",
            ),
            None,
        )

    waypoint_array = np.stack(waypoints).astype(np.float32)
    async_actions = waypoints_to_async_actions(
        waypoint_array,
        spacing_m=async_action_spacing_m,
        convention=waypoint_convention,
    )

    outputs = {
        "target_waypoints.npy": waypoint_array,
        "target_async_actions.npy": async_actions,
        "timestamps.npy": np.asarray(timestamps, dtype=np.float32),
    }
    for filename, array in outputs.items():
        output_path = output_dir / filename
        if overwrite or not output_path.exists():
            np.save(output_path, array)

    sample_jsonl = output_dir / "postprocessed_samples.jsonl"
    if overwrite or not sample_jsonl.exists():
        with sample_jsonl.open("w", encoding="utf-8") as f:
            for record in debug_records:
                f.write(json.dumps(record) + "\n")

    label_summary = {
        "label_source": "runtime_planned_path",
        "runtime_planned_path": "runtime_planned_path.json",
        "sample_count": len(waypoints),
        "skipped_samples": skipped,
        "path_length_m": path_length(trajectory.path),
        "chunk_size": chunk_size,
        "first_preview_m": first_preview_m,
        "waypoint_spacing_m": waypoint_spacing_m,
        "async_action_spacing_m": async_action_spacing_m,
        "waypoint_convention": waypoint_convention,
        "target_waypoints_shape": list(waypoint_array.shape),
        "target_async_actions_shape": list(async_actions.shape),
        "image_export": {
            "resized": output_rollout_dir is not None,
        },
        "segment": segment.metrics,
    }
    (output_dir / "postprocess_label_summary.json").write_text(json.dumps(label_summary, indent=2), encoding="utf-8")
    (output_dir / "postprocess_segment_summary.json").write_text(
        json.dumps(
            {
                "rollout_dir": rollout_dir.as_posix(),
                "segment": segment.metrics,
                "sample_diagnostics": [
                    {
                        key: value
                        for key, value in diagnostic.items()
                        if key != "record"
                    }
                    for diagnostic in segment.diagnostics
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    instruction = (
        metadata.get("language_instruction")
        or metadata.get("instruction")
        or metadata.get("task_id")
        or "navigate safely"
    )
    manifest_record = {
        "episode_id": str(metadata.get("trajectory_name") or rollout_dir.name),
        "root": output_dir.resolve().as_posix(),
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
            "label_source": "runtime_planned_path",
            "raw_waypoint_format": "x_forward_m_y_left_m_yaw_rad",
            "async_action_format": "x_over_spacing_y_over_spacing_cos_yaw_sin_yaw",
            "async_action_spacing_m": float(async_action_spacing_m),
            "waypoint_convention": waypoint_convention,
        },
    }
    return (
        PostprocessResult(
            rollout_dir=rollout_dir,
            accepted=True,
            sample_count=len(waypoints),
            skipped_samples=skipped,
            reason=None,
            metrics=label_summary,
        ),
        manifest_record,
    )


def validate_for_final_dataset(rollout_dir: Path, args: argparse.Namespace) -> ValidationResult:
    return validate_rollout_dir(
        rollout_dir=rollout_dir,
        min_samples=args.min_samples,
        min_motion_m=args.min_motion_m,
        min_action_chunk_fraction=0.0,
        min_cmd_vel_fraction=args.min_cmd_vel_fraction,
        max_mean_abs_action_first_y_m=None,
        max_abs_action_first_y_m=None,
        max_action_chunk_age_s=None,
        max_mean_reference_lateral_error_m=args.max_mean_reference_lateral_error_m,
        max_reference_lateral_error_m=args.max_reference_lateral_error_m,
        max_final_target_distance_m=args.max_final_target_distance_m,
        max_black_image_fraction=args.max_black_image_fraction,
        min_target_fence_clearance_m=args.min_target_fence_clearance_m,
    )


def result_payload(result: PostprocessResult) -> dict[str, Any]:
    return {
        "rollout_dir": result.rollout_dir.as_posix(),
        "accepted": result.accepted,
        "sample_count": result.sample_count,
        "skipped_samples": result.skipped_samples,
        "reason": result.reason,
        "warnings": result.warnings or [],
        "metrics": result.metrics or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="One rollout dir or a root containing many rollout dirs.")
    parser.add_argument("--out-manifest", type=Path, required=True, help="Final accepted-rollout JSONL manifest.")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=None,
        help="Optional clean dataset root. Accepted rollouts are copied here with resized images and no diagnostics.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square output image size for --export-root.")
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "center_crop", "stretch"),
        default="letterbox",
        help="Image resize strategy used for --export-root.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-motion-m", type=float, default=None)
    parser.add_argument("--min-cmd-vel-fraction", type=float, default=0.5)
    parser.add_argument("--max-mean-reference-lateral-error-m", type=float, default=1.0)
    parser.add_argument("--max-reference-lateral-error-m", type=float, default=2.5)
    parser.add_argument("--max-final-target-distance-m", type=float, default=2.0)
    parser.add_argument("--max-black-image-fraction", type=float, default=0.05)
    parser.add_argument("--min-target-fence-clearance-m", type=float, default=0.65)
    parser.add_argument(
        "--no-segment-filter",
        action="store_true",
        help="Disable longest-clean-segment filtering and require the whole rollout to pass.",
    )
    parser.add_argument(
        "--no-salvage-failed-rollouts",
        action="store_true",
        help="Reject rollouts that fail whole-rollout validation instead of trying to keep a clean segment.",
    )
    parser.add_argument(
        "--min-segment-progress-m",
        type=float,
        default=2.0,
        help="Minimum reference-path progress required for a salvaged segment.",
    )
    parser.add_argument(
        "--max-segment-tracking-error-m",
        type=float,
        default=1.25,
        help="Maximum distance from the runtime planned path for samples kept in a segment.",
    )
    parser.add_argument(
        "--min-segment-target-fence-clearance-m",
        type=float,
        default=0.65,
        help="Minimum center distance from target fence for samples kept in a segment.",
    )
    parser.add_argument(
        "--no-enforce-segment-fence-side",
        action="store_true",
        help="Do not reject segment samples that cross to the wrong side of the target fence.",
    )
    parser.add_argument(
        "--no-segment-black-image-check",
        action="store_true",
        help="Do not check each kept segment image for all-black frames.",
    )
    parser.add_argument("--black-image-min-mean", type=float, default=2.0)
    parser.add_argument("--black-image-min-max", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--first-preview-m", type=float, default=0.35)
    parser.add_argument("--waypoint-spacing-m", type=float, default=0.18)
    parser.add_argument("--async-action-spacing-m", type=float, default=0.125)
    parser.add_argument(
        "--waypoint-convention",
        choices=WAYPOINT_CONVENTIONS,
        default="x_forward_y_left",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    rollouts = discover_rollouts(input_root)
    if not rollouts:
        print(f"No rollout folders found under {args.input_root}")
        return 1

    if args.export_root is not None:
        args.export_root.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or args.out_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    results: list[PostprocessResult] = []
    for rollout_dir in rollouts:
        validation = validate_for_final_dataset(rollout_dir, args)
        if not validation.valid and (args.no_segment_filter or args.no_salvage_failed_rollouts):
            results.append(
                PostprocessResult(
                    rollout_dir=rollout_dir,
                    accepted=False,
                    reason="; ".join(validation.errors),
                    warnings=validation.warnings,
                    metrics=validation.metrics,
                )
            )
            continue

        try:
            output_rollout_dir = None
            if args.export_root is not None:
                relative = rollout_dir.relative_to(input_root) if rollout_dir != input_root else Path(rollout_dir.name)
                output_rollout_dir = args.export_root / relative
            result, record = build_labels_for_rollout(
                rollout_dir,
                output_rollout_dir=output_rollout_dir,
                min_samples=args.min_samples,
                overwrite=bool(args.overwrite),
                chunk_size=int(args.chunk_size),
                first_preview_m=float(args.first_preview_m),
                waypoint_spacing_m=float(args.waypoint_spacing_m),
                async_action_spacing_m=float(args.async_action_spacing_m),
                waypoint_convention=str(args.waypoint_convention),
                segment_filter=not bool(args.no_segment_filter),
                min_segment_progress_m=float(args.min_segment_progress_m),
                max_segment_tracking_error_m=float(args.max_segment_tracking_error_m),
                min_segment_target_fence_clearance_m=(
                    None
                    if args.min_segment_target_fence_clearance_m is None
                    else float(args.min_segment_target_fence_clearance_m)
                ),
                enforce_segment_fence_side=not bool(args.no_enforce_segment_fence_side),
                check_segment_black_images=not bool(args.no_segment_black_image_check),
                black_image_min_mean=float(args.black_image_min_mean),
                black_image_min_max=float(args.black_image_min_max),
                image_size=int(args.image_size),
                resize_mode=str(args.resize_mode),
                jpeg_quality=int(args.jpeg_quality),
            )
        except Exception as exc:
            result = PostprocessResult(
                rollout_dir=rollout_dir,
                accepted=False,
                reason=f"label generation failed: {exc}",
                warnings=validation.warnings,
                metrics=validation.metrics,
            )
            record = None

        result.warnings = list(validation.warnings) + list(result.warnings or [])
        if result.accepted and result.metrics is not None:
            result.metrics = {"validation": validation.metrics, "labels": result.metrics}
        elif result.metrics is None:
            result.metrics = validation.metrics
        results.append(result)
        if record is not None:
            manifest_records.append(record)

    with args.out_manifest.open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record) + "\n")

    summary = {
        "input_root": args.input_root.resolve().as_posix(),
        "export_root": args.export_root.resolve().as_posix() if args.export_root is not None else None,
        "out_manifest": args.out_manifest.resolve().as_posix(),
        "rollouts_found": len(rollouts),
        "accepted_rollouts": len(manifest_records),
        "rejected_rollouts": len(rollouts) - len(manifest_records),
        "total_samples": sum(result.sample_count for result in results if result.accepted),
        "results": [result_payload(result) for result in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Found {len(rollouts)} rollout(s).")
    print(f"Accepted {len(manifest_records)} rollout(s), rejected {len(rollouts) - len(manifest_records)}.")
    print(f"Wrote manifest: {args.out_manifest}")
    print(f"Wrote summary:  {summary_path}")
    rejected = [result for result in results if not result.accepted]
    if rejected:
        print("First rejected rollout reasons:")
        for result in rejected[:10]:
            print(f"  {result.rollout_dir.name}: {result.reason}")
    return 0 if manifest_records else 1


if __name__ == "__main__":
    raise SystemExit(main())

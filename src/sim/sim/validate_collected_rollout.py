#!/usr/bin/env python3
"""Validate one collected sim rollout folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.expert_trajectory_utils import find_task, find_variant, get_start_pose, load_yaml, path_length, project_progress_near
from sim.validate_scene_task_specs import reference_path_for_task


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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def pose_point(record: dict[str, Any]) -> np.ndarray | None:
    point = pose_xy(record)
    if point is None:
        return None
    return np.asarray([point[0], point[1]], dtype=np.float64)


def displacement(records: list[dict[str, Any]]) -> float:
    points = [point for record in records if (point := pose_xy(record)) is not None]
    if len(points) < 2:
        return 0.0
    return math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])


def fraction_present(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record.get(key) is not None) / len(records)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def row_float(row: dict[str, str], key: str) -> float | None:
    return finite_float(row.get(key))


def diagnostic_spin_stall_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "spin_stall_samples": 0,
            "spin_stall_fraction": 0.0,
            "max_contiguous_spin_stall_s": 0.0,
        }

    spin_samples = 0
    valid_samples = 0
    max_run_s = 0.0
    run_start: float | None = None
    run_end: float | None = None

    for row in rows:
        sim_t = row_float(row, "sim_elapsed_s")
        cmd_w = row_float(row, "cmd_angular_z")
        odom_w = row_float(row, "odom_yaw_rate")
        odom_v = row_float(row, "odom_linear_x")
        if cmd_w is None or odom_w is None or odom_v is None:
            continue
        valid_samples += 1
        is_spin_stall = abs(cmd_w) > 0.35 and abs(odom_w) > 0.15 and abs(odom_v) < 0.06
        if is_spin_stall:
            spin_samples += 1
            if sim_t is not None:
                if run_start is None:
                    run_start = sim_t
                run_end = sim_t
        else:
            if run_start is not None and run_end is not None:
                max_run_s = max(max_run_s, run_end - run_start)
            run_start = None
            run_end = None
    if run_start is not None and run_end is not None:
        max_run_s = max(max_run_s, run_end - run_start)

    return {
        "samples": valid_samples,
        "spin_stall_samples": spin_samples,
        "spin_stall_fraction": spin_samples / valid_samples if valid_samples else 0.0,
        "max_contiguous_spin_stall_s": max_run_s,
    }


def transformed_xy(point: list[float] | tuple[float, ...], flip_y: bool) -> np.ndarray:
    x = float(point[0])
    y = float(point[1])
    return np.asarray([x, -y if flip_y else y], dtype=np.float64)


def signed_side(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    rel = point - start
    length = float(np.linalg.norm(segment))
    if length <= 1e-9:
        return 0.0
    return float((segment[0] * rel[1] - segment[1] * rel[0]) / length)


def distance_to_segment_points(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start)), 0.0
    t = float(np.dot(point - start, segment) / length_sq)
    t_clamped = min(1.0, max(0.0, t))
    nearest = start + segment * t_clamped
    return float(np.linalg.norm(point - nearest)), t_clamped


def fence_follow_geometry_metrics(
    rollout_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    structured_task = metadata.get("structured_task") if isinstance(metadata.get("structured_task"), dict) else {}
    task_type = str(structured_task.get("task_type", ""))
    if task_type not in {"follow_fence", "follow_fence_sequence"}:
        return {}, None

    task_spec_path = resolve_existing_path(str(metadata.get("task_spec", "")), rollout_dir)
    if task_spec_path is None:
        return {}, "fence geometry could not be checked because task spec is unavailable"

    try:
        task_spec = load_yaml(task_spec_path)
        scene_path = Path(task_spec["scene"]["source_yaml"])
        scene = load_yaml(scene_path)
    except Exception as exc:
        return {}, f"fence geometry could not be checked: {exc}"

    target_names = structured_task.get("target_fences")
    if not isinstance(target_names, list) or not target_names:
        target_fence = structured_task.get("target_fence")
        target_names = [target_fence] if isinstance(target_fence, str) else []
    fences = [fence for fence in scene.get("fences", []) if fence.get("name") in set(target_names)]
    if not fences:
        return {}, None

    flip_scene_y = bool(metadata.get("flip_scene_y", False))
    expected_path_side = str(structured_task.get("path_side", ""))
    expected_sign = 1.0 if expected_path_side == "left" else -1.0 if expected_path_side == "right" else 0.0
    segments = [
        (transformed_xy(fence["start"], flip_scene_y), transformed_xy(fence["end"], flip_scene_y))
        for fence in fences
    ]

    distances: list[float] = []
    signed_distances: list[float] = []
    side_violations = 0
    side_checked = 0
    near_fence_samples = 0
    min_distance = math.inf
    min_signed_distance: float | None = None

    for record in records:
        point = pose_point(record)
        if point is None:
            continue
        nearest_distance = math.inf
        nearest_signed = 0.0
        for start, end in segments:
            distance_m, _ = distance_to_segment_points(point, start, end)
            side_m = signed_side(point, start, end)
            if distance_m < nearest_distance:
                nearest_distance = distance_m
                nearest_signed = side_m
        if not math.isfinite(nearest_distance):
            continue
        distances.append(nearest_distance)
        signed_distances.append(nearest_signed)
        if nearest_distance < min_distance:
            min_distance = nearest_distance
            min_signed_distance = nearest_signed
        if nearest_distance < 1.8:
            near_fence_samples += 1
            if expected_sign:
                side_checked += 1
                if nearest_signed * expected_sign < 0.15:
                    side_violations += 1

    if not distances:
        return {}, None

    return {
        "sample_count": len(distances),
        "near_fence_samples": near_fence_samples,
        "min_center_distance_to_target_fence_m": min_distance,
        "min_signed_side_distance_m": min_signed_distance,
        "mean_center_distance_to_target_fence_m": sum(distances) / len(distances),
        "side_violation_count": side_violations,
        "side_checked_samples": side_checked,
        "side_violation_fraction": side_violations / side_checked if side_checked else 0.0,
        "expected_path_side": expected_path_side,
    }, None


def action_chunk_tracking_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    first_y_values: list[float] = []
    first_theta_values: list[float] = []
    age_values: list[float] = []
    for record in records:
        age_s = finite_float(record.get("action_chunk_age_s"))
        if age_s is not None:
            age_values.append(age_s)

        action_chunk = record.get("action_chunk")
        if not isinstance(action_chunk, dict):
            continue
        poses = action_chunk.get("relative_poses")
        if not isinstance(poses, list) or not poses:
            continue
        first_pose = poses[0]
        if not isinstance(first_pose, (list, tuple)) or len(first_pose) < 3:
            continue
        first_y = finite_float(first_pose[1])
        first_theta = finite_float(first_pose[2])
        if first_y is not None:
            first_y_values.append(first_y)
        if first_theta is not None:
            first_theta_values.append(first_theta)

    abs_first_y_values = [abs(value) for value in first_y_values]
    abs_first_theta_values = [abs(value) for value in first_theta_values]
    return {
        "samples_with_first_waypoint": len(first_y_values),
        "mean_abs_first_y_m": sum(abs_first_y_values) / len(abs_first_y_values) if abs_first_y_values else 0.0,
        "max_abs_first_y_m": max(abs_first_y_values, default=0.0),
        "mean_abs_first_theta_rad": (
            sum(abs_first_theta_values) / len(abs_first_theta_values) if abs_first_theta_values else 0.0
        ),
        "max_abs_first_theta_rad": max(abs_first_theta_values, default=0.0),
        "samples_with_action_chunk_age": len(age_values),
        "mean_action_chunk_age_s": sum(age_values) / len(age_values) if age_values else 0.0,
        "max_action_chunk_age_s": max(age_values, default=0.0),
    }


def validate_images(rollout_dir: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    image_dir = rollout_dir / "img"
    image_files = list(image_dir.glob("*.jpg")) if image_dir.exists() else []
    missing_refs = 0
    for record in records:
        image_name = record.get("image")
        if isinstance(image_name, str) and not (image_dir / image_name).exists():
            missing_refs += 1
    return len(image_files), missing_refs


def image_quality_metrics(rollout_dir: Path, max_checked: int = 30) -> tuple[dict[str, Any], str | None]:
    image_dir = rollout_dir / "img"
    image_files = sorted(image_dir.glob("*.jpg")) if image_dir.exists() else []
    if not image_files:
        return {
            "checked_images": 0,
            "black_images": 0,
            "black_image_fraction": 0.0,
            "mean_pixel_value": None,
        }, None

    try:
        from PIL import Image, ImageStat
    except Exception as exc:
        return {}, f"image pixel quality could not be checked because Pillow is unavailable: {exc}"

    if len(image_files) > max_checked:
        step = max((len(image_files) - 1) / max(max_checked - 1, 1), 1.0)
        indices = sorted({int(round(i * step)) for i in range(max_checked)})
        sampled_files = [image_files[min(index, len(image_files) - 1)] for index in indices]
    else:
        sampled_files = image_files

    black_images = 0
    means: list[float] = []
    extrema_maxes: list[float] = []
    for image_path in sampled_files:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                stat = ImageStat.Stat(image)
                channel_mean = sum(float(value) for value in stat.mean) / len(stat.mean)
                extrema_max = max(float(high) for _, high in image.getextrema())
        except Exception:
            continue
        means.append(channel_mean)
        extrema_maxes.append(extrema_max)
        if channel_mean <= 2.0 and extrema_max <= 8.0:
            black_images += 1

    checked = len(means)
    return {
        "checked_images": checked,
        "black_images": black_images,
        "black_image_fraction": black_images / checked if checked else 0.0,
        "mean_pixel_value": sum(means) / checked if checked else None,
        "min_mean_pixel_value": min(means) if means else None,
        "max_pixel_value": max(extrema_maxes) if extrema_maxes else None,
    }, None


def task_validation_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    structured_task = metadata.get("structured_task")
    if isinstance(structured_task, dict) and isinstance(structured_task.get("validation"), dict):
        return structured_task["validation"]
    return {}


def load_diagnostics_summary(rollout_dir: Path) -> dict[str, Any] | None:
    path = rollout_dir / "diagnostics_summary.json"
    if not path.exists():
        return None
    return load_json(path)


def load_task_success_summary(rollout_dir: Path) -> dict[str, Any] | None:
    path = rollout_dir / "task_success_wait_summary.json"
    if not path.exists():
        return None
    return load_json(path)


def resolve_existing_path(path_text: str, rollout_dir: Path) -> Path | None:
    if not path_text:
        return None
    candidates = [Path(path_text)]
    if not Path(path_text).is_absolute():
        candidates.append(rollout_dir / path_text)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def reference_progress_metrics(
    rollout_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    task_spec_path = resolve_existing_path(str(metadata.get("task_spec", "")), rollout_dir)
    if task_spec_path is None:
        return {}, "task spec is unavailable; reference-path progress could not be checked"

    try:
        task_spec = load_yaml(task_spec_path)
        scene_path = Path(task_spec["scene"]["source_yaml"])
        scene = load_yaml(scene_path)
        task_id = str(metadata.get("task_id") or "")
        variant_id = str(metadata.get("variant_id") or "nominal")
        task = find_task(task_spec, task_id)
        variant = find_variant(task, variant_id)
        flip_scene_y = bool(metadata.get("flip_scene_y", False))
        reference_path = reference_path_for_task(scene, scene_path, task, variant, flip_scene_y)
        world_start_position, _ = get_start_pose(scene, task, flip_scene_y)
    except Exception as exc:
        return {}, f"reference-path progress could not be checked: {exc}"

    progress_values: list[float] = []
    lateral_errors: list[float] = []
    previous_progress = 0.0
    for record in records:
        point = pose_point(record)
        if point is None:
            continue
        progress = project_progress_near(
            reference_path,
            point,
            previous_progress,
            max_backward_m=0.5,
            max_forward_m=2.0,
        )
        previous_progress = max(previous_progress, progress)
        progress_values.append(previous_progress)
        lateral_errors.append(float(np.linalg.norm(point - closest_point_on_path(reference_path, previous_progress))))

    if not progress_values:
        return {}, "no usable poses for reference-path progress check"

    max_progress = max(progress_values)
    final_progress = progress_values[-1]
    return {
        "reference_path_length_m": path_length(reference_path),
        "path_progress_m": max_progress,
        "final_path_progress_m": final_progress,
        "start_reference_distance_m": float(np.linalg.norm(pose_point(records[0]) - world_start_position))
        if pose_point(records[0]) is not None
        else None,
        "mean_reference_lateral_error_m": sum(lateral_errors) / len(lateral_errors),
        "max_reference_lateral_error_m": max(lateral_errors, default=0.0),
    }, None


def closest_point_on_path(polyline: list[np.ndarray], progress: float) -> np.ndarray:
    if not polyline:
        return np.zeros(2, dtype=np.float64)
    if len(polyline) == 1:
        return polyline[0]
    remaining = max(float(progress), 0.0)
    for index in range(len(polyline) - 1):
        start = polyline[index]
        end = polyline[index + 1]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        if remaining <= length:
            return start + segment * (remaining / length)
        remaining -= length
    return polyline[-1]


def nested_float(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def nested_int(data: dict[str, Any], *keys: str, default: int = 0) -> int:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def likely_failure_from_diagnostics(summary: dict[str, Any], travelled_m: float, min_motion_m: float) -> str | None:
    action_rate = nested_float(summary, "rates_hz", "action_chunk")
    expert_rate = nested_float(summary, "rates_hz", "expert_cmd_vel")
    cmd_rate = nested_float(summary, "rates_hz", "cmd_vel")
    odom_rate = nested_float(summary, "rates_hz", "odom")
    camera_rate = nested_float(summary, "rates_hz", "camera")
    mean_expert_vx = nested_float(summary, "commands", "mean_expert_linear_x")
    mean_cmd_vx = nested_float(summary, "commands", "mean_cmd_linear_x")
    negative_action_fraction = nested_float(summary, "action_chunk", "negative_x_fraction")

    if action_rate < 1.0:
        return "expert/action_chunk publication is too slow or stopped"
    if odom_rate < 1.0:
        return "Isaac odom is missing or too slow"
    if camera_rate < 1.0:
        return "Isaac camera is missing or too slow"
    if expert_rate >= 1.0 and cmd_rate < 1.0:
        return "tracker is not publishing /cmd_vel"
    if mean_expert_vx > 0.05 and mean_cmd_vx < 0.03:
        return "tracker output is much smaller than expert forward command"
    if negative_action_fraction > 0.25:
        return "many action-chunk future waypoints have negative x in robot frame"
    if travelled_m < min_motion_m and mean_cmd_vx > 0.05:
        return "cmd_vel is being published but Isaac rover is not moving as expected"
    return None


def validate_rollout_dir(
    rollout_dir: Path,
    expected_task_id: str | None = None,
    expected_variant_id: str | None = None,
    min_samples: int | None = None,
    min_motion_m: float | None = None,
    min_action_chunk_fraction: float = 0.5,
    min_cmd_vel_fraction: float = 0.5,
    max_mean_abs_action_first_y_m: float | None = 1.25,
    max_abs_action_first_y_m: float | None = 3.0,
    max_action_chunk_age_s: float | None = 1.0,
    max_mean_reference_lateral_error_m: float | None = 2.0,
    max_reference_lateral_error_m: float | None = 4.0,
    max_final_target_distance_m: float | None = 2.0,
    max_required_progress_shortfall_m: float = 0.0,
    max_black_image_fraction: float | None = 0.05,
    min_target_fence_clearance_m: float | None = 0.65,
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
    image_quality, image_quality_warning = image_quality_metrics(rollout_dir)
    diagnostics_summary = load_diagnostics_summary(rollout_dir)
    diagnostics_rows = load_csv_rows(rollout_dir / "diagnostics.csv")
    task_success_summary = load_task_success_summary(rollout_dir)
    action_chunk_fraction = fraction_present(records, "action_chunk")
    cmd_vel_fraction = fraction_present(records, "cmd_vel")
    tracking_metrics = action_chunk_tracking_metrics(records)
    travelled_m = path_distance(records)
    displacement_m = displacement(records)
    reference_metrics, reference_warning = reference_progress_metrics(rollout_dir, metadata, records)
    spin_stall_metrics = diagnostic_spin_stall_metrics(diagnostics_rows)
    fence_geometry_metrics, fence_geometry_warning = fence_follow_geometry_metrics(rollout_dir, metadata, records)
    if not reference_metrics and task_success_summary is not None:
        summary_progress = finite_float(task_success_summary.get("path_progress_m"))
        summary_length = finite_float(task_success_summary.get("reference_path_length_m"))
        if summary_progress is not None:
            reference_metrics["path_progress_m"] = summary_progress
            reference_metrics["final_path_progress_m"] = summary_progress
        if summary_length is not None:
            reference_metrics["reference_path_length_m"] = summary_length

    structured_task = metadata.get("structured_task") if isinstance(metadata.get("structured_task"), dict) else {}
    success_condition = structured_task.get("success_condition") if isinstance(structured_task, dict) else {}
    success_type = str(success_condition.get("type", "")) if isinstance(success_condition, dict) else ""
    distance_success_types = {"reach_path_end", "pass_point", "pass_point_and_continue"}
    use_reference_progress = (
        success_type in distance_success_types
        and not allow_stationary
        and finite_float(reference_metrics.get("path_progress_m")) is not None
    )
    required_progress_m = float(min_motion_m)
    reference_path_length_m = finite_float(reference_metrics.get("reference_path_length_m"))
    if use_reference_progress and reference_path_length_m is not None:
        required_progress_m = min(required_progress_m, max(reference_path_length_m - 0.25, 0.0))
    progress_shortfall_tolerance_m = max(0.0, float(max_required_progress_shortfall_m))
    accepted_required_progress_m = max(0.0, required_progress_m - progress_shortfall_tolerance_m)

    metrics.update(
        {
            "sample_count": len(records),
            "image_count": image_count,
            "missing_image_refs": missing_image_refs,
            "image_quality": image_quality,
            "action_chunk_fraction": action_chunk_fraction,
            "cmd_vel_fraction": cmd_vel_fraction,
            "tracking": tracking_metrics,
            "path_distance_m": travelled_m,
            "displacement_m": displacement_m,
            "min_samples": min_samples,
            "min_motion_m": min_motion_m,
            "success_type": success_type,
            "reference_progress": reference_metrics,
            "required_progress_m": required_progress_m,
            "accepted_required_progress_m": accepted_required_progress_m,
            "max_required_progress_shortfall_m": progress_shortfall_tolerance_m,
            "spin_stall": spin_stall_metrics,
            "fence_geometry": fence_geometry_metrics,
        }
    )
    if diagnostics_summary is not None:
        metrics["diagnostics"] = diagnostics_summary
    else:
        warnings.append("diagnostics_summary.json is missing")
    if task_success_summary is not None:
        metrics["task_success_wait_summary"] = task_success_summary
    if reference_warning is not None:
        warnings.append(reference_warning)
    if fence_geometry_warning is not None:
        warnings.append(fence_geometry_warning)
    if image_quality_warning is not None:
        warnings.append(image_quality_warning)

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
    black_fraction = finite_float(image_quality.get("black_image_fraction")) if image_quality else None
    checked_images = int(image_quality.get("checked_images", 0)) if image_quality else 0
    if (
        max_black_image_fraction is not None
        and checked_images > 0
        and black_fraction is not None
        and black_fraction > float(max_black_image_fraction)
    ):
        errors.append(
            f"black image fraction {black_fraction:.2%} > allowed {float(max_black_image_fraction):.2%} "
            f"({image_quality.get('black_images', 0)}/{checked_images} checked images)"
        )
    if action_chunk_fraction < min_action_chunk_fraction:
        errors.append(
            f"action_chunk present in {action_chunk_fraction:.2%} of samples, "
            f"required {min_action_chunk_fraction:.2%}"
        )
    if cmd_vel_fraction < min_cmd_vel_fraction:
        errors.append(
            f"cmd_vel present in {cmd_vel_fraction:.2%} of samples, required {min_cmd_vel_fraction:.2%}"
        )
    if tracking_metrics["samples_with_first_waypoint"] == 0 and action_chunk_fraction >= min_action_chunk_fraction:
        errors.append("action_chunk messages are present but contain no usable first waypoint tracking data")
    if (
        max_mean_abs_action_first_y_m is not None
        and tracking_metrics["mean_abs_first_y_m"] > float(max_mean_abs_action_first_y_m)
    ):
        errors.append(
            f"mean first-waypoint lateral tracking error "
            f"{tracking_metrics['mean_abs_first_y_m']:.3f}m > allowed "
            f"{float(max_mean_abs_action_first_y_m):.3f}m"
        )
    if (
        max_abs_action_first_y_m is not None
        and tracking_metrics["max_abs_first_y_m"] > float(max_abs_action_first_y_m)
    ):
        errors.append(
            f"max first-waypoint lateral tracking error "
            f"{tracking_metrics['max_abs_first_y_m']:.3f}m > allowed "
            f"{float(max_abs_action_first_y_m):.3f}m"
        )
    if (
        max_action_chunk_age_s is not None
        and tracking_metrics["samples_with_action_chunk_age"] > 0
        and tracking_metrics["max_action_chunk_age_s"] > float(max_action_chunk_age_s)
    ):
        errors.append(
            f"max action_chunk age {tracking_metrics['max_action_chunk_age_s']:.3f}s > allowed "
            f"{float(max_action_chunk_age_s):.3f}s"
        )
    if tracking_metrics["samples_with_action_chunk_age"] == 0 and action_chunk_fraction >= min_action_chunk_fraction:
        warnings.append("action_chunk_age_s is missing; freshness could not be checked for this rollout")
    if use_reference_progress:
        progress_m = float(reference_metrics["path_progress_m"])
        if progress_m < accepted_required_progress_m:
            errors.append(
                f"path progress {progress_m:.3f}m < required {required_progress_m:.3f}m "
                f"(accepted shortfall {progress_shortfall_tolerance_m:.3f}m)"
            )
    elif travelled_m < float(min_motion_m):
        errors.append(f"path distance {travelled_m:.3f}m < required {float(min_motion_m):.3f}m")
    mean_reference_error = finite_float(reference_metrics.get("mean_reference_lateral_error_m"))
    max_reference_error = finite_float(reference_metrics.get("max_reference_lateral_error_m"))
    if (
        max_mean_reference_lateral_error_m is not None
        and mean_reference_error is not None
        and mean_reference_error > float(max_mean_reference_lateral_error_m)
    ):
        errors.append(
            f"mean reference tracking error {mean_reference_error:.3f}m > allowed "
            f"{float(max_mean_reference_lateral_error_m):.3f}m"
        )
    if (
        max_reference_lateral_error_m is not None
        and max_reference_error is not None
        and max_reference_error > float(max_reference_lateral_error_m)
    ):
        errors.append(
            f"max reference tracking error {max_reference_error:.3f}m > allowed "
            f"{float(max_reference_lateral_error_m):.3f}m"
        )
    final_target_distance = (
        finite_float(task_success_summary.get("target_distance_m"))
        if isinstance(task_success_summary, dict)
        else None
    )
    progress_m = finite_float(reference_metrics.get("path_progress_m"))
    if (
        max_final_target_distance_m is not None
        and final_target_distance is not None
        and progress_m is not None
        and progress_m < accepted_required_progress_m
        and final_target_distance > float(max_final_target_distance_m)
    ):
        errors.append(
            f"final target distance {final_target_distance:.3f}m > allowed "
            f"{float(max_final_target_distance_m):.3f}m before required progress was met"
        )
    if (
        spin_stall_metrics["samples"] > 0
        and (
            spin_stall_metrics["spin_stall_fraction"] > 0.12
            or spin_stall_metrics["max_contiguous_spin_stall_s"] > 4.0
        )
    ):
        errors.append(
            "spin/stall detected: "
            f"{spin_stall_metrics['spin_stall_fraction']:.2%} of diagnostics samples, "
            f"max continuous {spin_stall_metrics['max_contiguous_spin_stall_s']:.2f}s"
        )
    if fence_geometry_metrics:
        min_fence_distance = finite_float(fence_geometry_metrics.get("min_center_distance_to_target_fence_m"))
        side_fraction = finite_float(fence_geometry_metrics.get("side_violation_fraction"))
        if (
            min_target_fence_clearance_m is not None
            and min_fence_distance is not None
            and min_fence_distance < float(min_target_fence_clearance_m)
        ):
            errors.append(
                f"target-fence clearance too small: center distance {min_fence_distance:.3f}m < "
                f"required {float(min_target_fence_clearance_m):.3f}m"
            )
        if side_fraction is not None and side_fraction > 0.02:
            errors.append(
                f"target-fence side violation: {side_fraction:.2%} of near-fence samples crossed/entered wrong side"
            )
    if len(records) and image_count != len(records):
        warnings.append(f"image count {image_count} differs from sample count {len(records)}")
    if diagnostics_summary is not None:
        if nested_int(diagnostics_summary, "messages", "action_chunk") == 0:
            warnings.append("diagnostics: no /vla/action_chunk messages observed")
        if nested_int(diagnostics_summary, "messages", "cmd_vel") == 0:
            warnings.append("diagnostics: no /cmd_vel messages observed")
        if nested_int(diagnostics_summary, "messages", "frame_debug") == 0:
            warnings.append("diagnostics: no /expert/frame_debug messages observed")
        isaac_pose_messages = nested_int(diagnostics_summary, "messages", "isaac_pose_debug")
        if isaac_pose_messages == 0:
            warnings.append("diagnostics: no /isaac/scene_pose_debug messages observed")
        if nested_float(diagnostics_summary, "rates_hz", "action_chunk") < 2.0:
            warnings.append(
                f"diagnostics: action_chunk rate {nested_float(diagnostics_summary, 'rates_hz', 'action_chunk'):.2f} Hz"
            )
        if nested_float(diagnostics_summary, "rates_hz", "cmd_vel") < 2.0:
            warnings.append(f"diagnostics: cmd_vel rate {nested_float(diagnostics_summary, 'rates_hz', 'cmd_vel'):.2f} Hz")
        failure_distance = (
            float(reference_metrics["path_progress_m"])
            if use_reference_progress and finite_float(reference_metrics.get("path_progress_m")) is not None
            else travelled_m
        )
        likely_failure = likely_failure_from_diagnostics(diagnostics_summary, failure_distance, required_progress_m)
        if likely_failure is not None:
            warnings.append(f"likely failure stage: {likely_failure}")

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
    parser.add_argument("--max-mean-abs-action-first-y-m", type=float, default=1.25)
    parser.add_argument("--max-abs-action-first-y-m", type=float, default=3.0)
    parser.add_argument("--max-action-chunk-age-s", type=float, default=1.0)
    parser.add_argument("--max-mean-reference-lateral-error-m", type=float, default=2.0)
    parser.add_argument("--max-reference-lateral-error-m", type=float, default=4.0)
    parser.add_argument("--max-final-target-distance-m", type=float, default=2.0)
    parser.add_argument("--max-required-progress-shortfall-m", type=float, default=0.0)
    parser.add_argument("--max-black-image-fraction", type=float, default=0.05)
    parser.add_argument("--min-target-fence-clearance-m", type=float, default=0.65)
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
        max_mean_abs_action_first_y_m=args.max_mean_abs_action_first_y_m,
        max_abs_action_first_y_m=args.max_abs_action_first_y_m,
        max_action_chunk_age_s=args.max_action_chunk_age_s,
        max_mean_reference_lateral_error_m=args.max_mean_reference_lateral_error_m,
        max_reference_lateral_error_m=args.max_reference_lateral_error_m,
        max_final_target_distance_m=args.max_final_target_distance_m,
        max_required_progress_shortfall_m=args.max_required_progress_shortfall_m,
        max_black_image_fraction=args.max_black_image_fraction,
        min_target_fence_clearance_m=args.min_target_fence_clearance_m,
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

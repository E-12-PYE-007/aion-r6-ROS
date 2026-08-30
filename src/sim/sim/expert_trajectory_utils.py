#!/usr/bin/env python3
"""Shared geometry and ROS helpers for expert trajectory publishers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def find_task(task_spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in task_spec.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    available = [task.get("task_id", "<missing>") for task in task_spec.get("tasks", [])]
    raise ValueError(f"task_id {task_id!r} not found. Available: {available}")


def find_variant(task: dict[str, Any], variant_id: str) -> dict[str, Any]:
    variants = task.get("trajectory_variants") or []
    for variant in variants:
        if variant.get("variant_id") == variant_id:
            return variant
    if variant_id == "nominal":
        return {"variant_id": "nominal", "variant_type": "nominal"}
    available = [variant.get("variant_id", "<missing>") for variant in variants]
    raise ValueError(f"variant_id {variant_id!r} not found. Available: {available}")


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_pose(msg, flip_isaac_y: bool, flip_yaw: bool = False) -> tuple[np.ndarray, float]:
    x = float(msg.pose.pose.position.x)
    y = float(msg.pose.pose.position.y)
    yaw = yaw_from_quaternion(msg.pose.pose.orientation)
    if flip_isaac_y:
        y = -y
        yaw = -yaw
    elif flip_yaw:
        yaw = -yaw
    return np.asarray([x, y], dtype=np.float64), yaw


def yaw_value(raw_yaw: Any, flip_isaac_y: bool) -> float:
    yaw = float(raw_yaw or 0.0)
    if abs(yaw) > math.tau:
        yaw = math.radians(yaw)
    if flip_isaac_y:
        yaw = -yaw
    return wrap_to_pi(yaw)


def get_start_pose(scene: dict[str, Any], task: dict[str, Any], flip_isaac_y: bool) -> tuple[np.ndarray, float]:
    starts = scene.get("rover_poses")
    start_name = task.get("start_pose")
    if isinstance(starts, dict) and isinstance(start_name, str) and start_name in starts:
        start = starts[start_name]
    elif isinstance(scene.get("rover_pose"), dict):
        start = scene["rover_pose"]
    else:
        raise ValueError("Scene does not define a usable rover start pose.")
    return point2(start["position"], flip_isaac_y), yaw_value(start.get("yaw", 0.0), flip_isaac_y)


def local_odom_to_world(
    local_position: np.ndarray,
    local_yaw: float,
    world_start_position: np.ndarray,
    world_start_yaw: float,
) -> tuple[np.ndarray, float]:
    world_position = world_start_position + rotate(local_position, world_start_yaw)
    world_yaw = wrap_to_pi(world_start_yaw + local_yaw)
    return world_position, world_yaw


def point2(point: list[float] | tuple[float, ...], flip_isaac_y: bool = False) -> np.ndarray:
    x = float(point[0])
    y = float(point[1])
    if flip_isaac_y:
        y = -y
    return np.asarray([x, y], dtype=np.float64)


def rotate(point: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray([c * point[0] - s * point[1], s * point[0] + c * point[1]], dtype=np.float64)


def segment_lengths(polyline: list[np.ndarray]) -> np.ndarray:
    if len(polyline) < 2:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        [np.linalg.norm(polyline[i + 1] - polyline[i]) for i in range(len(polyline) - 1)],
        dtype=np.float64,
    )


def path_length(polyline: list[np.ndarray]) -> float:
    return float(np.sum(segment_lengths(polyline)))


def project_progress(polyline: list[np.ndarray], point: np.ndarray) -> float:
    lengths = segment_lengths(polyline)
    best_progress = 0.0
    best_distance = float("inf")
    cumulative = 0.0
    for i, length in enumerate(lengths):
        if length <= 1e-9:
            continue
        start = polyline[i]
        end = polyline[i + 1]
        direction = (end - start) / length
        t = float(np.dot(point - start, direction))
        t = min(max(t, 0.0), float(length))
        closest = start + t * direction
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative + t
        cumulative += float(length)
    return best_progress


def project_progress_near(
    polyline: list[np.ndarray],
    point: np.ndarray,
    previous_progress_m: float | None = None,
    *,
    max_backward_m: float = 0.5,
    max_forward_m: float = 2.0,
) -> float:
    """Project onto a path near previous progress to avoid jumps on loops/detours."""
    if previous_progress_m is None:
        return project_progress(polyline, point)

    lengths = segment_lengths(polyline)
    if len(lengths) == 0:
        return 0.0

    total = path_length(polyline)
    window_start = max(0.0, float(previous_progress_m) - max_backward_m)
    window_end = min(total, float(previous_progress_m) + max_forward_m)
    best_progress = window_start
    best_distance = float("inf")
    cumulative = 0.0

    for i, length_value in enumerate(lengths):
        length = float(length_value)
        if length <= 1e-9:
            continue
        segment_start_progress = cumulative
        segment_end_progress = cumulative + length
        cumulative = segment_end_progress
        if segment_end_progress < window_start or segment_start_progress > window_end:
            continue

        start = polyline[i]
        end = polyline[i + 1]
        direction = (end - start) / length
        t = float(np.dot(point - start, direction))
        unclamped_progress = segment_start_progress + t
        progress = min(max(unclamped_progress, window_start), window_end)
        local_t = min(max(progress - segment_start_progress, 0.0), length)
        closest = start + local_t * direction
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_progress = progress

    return best_progress


def crop_path_at_progress(polyline: list[np.ndarray], progress: float) -> list[np.ndarray]:
    """Return the remaining portion of a polyline starting at a path-progress value."""
    if len(polyline) < 2:
        return list(polyline)
    total = path_length(polyline)
    progress = min(max(float(progress), 0.0), total)
    start_position, _ = sample_path_pose(polyline, progress)

    cropped = [start_position]
    lengths = segment_lengths(polyline)
    cumulative = 0.0
    for index, length in enumerate(lengths):
        next_cumulative = cumulative + float(length)
        if next_cumulative > progress + 1e-9:
            for point in polyline[index + 1:]:
                if float(np.linalg.norm(point - cropped[-1])) > 1e-6:
                    cropped.append(point)
            break
        cumulative = next_cumulative

    if len(cropped) == 1 and float(np.linalg.norm(polyline[-1] - cropped[-1])) > 1e-6:
        cropped.append(polyline[-1])
    return cropped


def crop_loop_path_at_progress(polyline: list[np.ndarray], progress: float) -> list[np.ndarray]:
    """Return a full loop path starting near progress and wrapping back to start."""
    if len(polyline) < 2:
        return list(polyline)
    total = path_length(polyline)
    progress = min(max(float(progress), 0.0), total)
    start_position, _ = sample_path_pose(polyline, progress)

    lengths = segment_lengths(polyline)
    segment_index = 0
    cumulative = 0.0
    for index, length in enumerate(lengths):
        next_cumulative = cumulative + float(length)
        if progress <= next_cumulative + 1e-9:
            segment_index = index
            break
        cumulative = next_cumulative

    loop = [start_position]
    ordered_points = list(polyline[segment_index + 1 :]) + list(polyline[: segment_index + 1])
    for point in ordered_points:
        if float(np.linalg.norm(point - loop[-1])) > 1e-6:
            loop.append(point)
    if float(np.linalg.norm(start_position - loop[-1])) > 1e-6:
        loop.append(start_position)
    return loop


def orient_and_crop_path_from_start(
    polyline: list[np.ndarray],
    start_position: np.ndarray,
    start_yaw: float,
    *,
    allow_reverse: bool = True,
) -> list[np.ndarray]:
    """Choose the path direction that agrees with the start heading and crop passed path."""
    if len(polyline) < 2:
        return list(polyline)

    def score(candidate: list[np.ndarray]) -> tuple[float, float, float]:
        progress = project_progress(candidate, start_position)
        cropped = crop_path_at_progress(candidate, progress)
        if len(cropped) < 2:
            return (math.inf, math.inf, math.inf)
        preview_progress = min(0.35, path_length(cropped))
        _, path_yaw = sample_path_pose(cropped, preview_progress)
        heading_error = abs(wrap_to_pi(path_yaw - start_yaw))
        start_distance = float(np.linalg.norm(cropped[0] - start_position))
        remaining = path_length(cropped)
        return (heading_error, start_distance, -remaining)

    candidates = [list(polyline)]
    if allow_reverse:
        candidates.append(list(reversed(polyline)))
    best = min(candidates, key=score)
    return crop_path_at_progress(best, project_progress(best, start_position))


def orient_loop_path_from_start(
    polyline: list[np.ndarray],
    start_position: np.ndarray,
    start_yaw: float,
    *,
    allow_reverse: bool = True,
) -> list[np.ndarray]:
    """Choose loop direction from start and preserve the whole loop."""
    if len(polyline) < 2:
        return list(polyline)

    def score(candidate: list[np.ndarray]) -> tuple[float, float, float]:
        progress = project_progress(candidate, start_position)
        loop = crop_loop_path_at_progress(candidate, progress)
        if len(loop) < 2:
            return (math.inf, math.inf, math.inf)
        preview_progress = min(0.35, path_length(loop))
        _, path_yaw = sample_path_pose(loop, preview_progress)
        heading_error = abs(wrap_to_pi(path_yaw - start_yaw))
        start_distance = float(np.linalg.norm(loop[0] - start_position))
        return (heading_error, start_distance, -path_length(loop))

    candidates = [list(polyline)]
    if allow_reverse:
        candidates.append(list(reversed(polyline)))
    best = min(candidates, key=score)
    return crop_loop_path_at_progress(best, project_progress(best, start_position))


def sample_path_pose(polyline: list[np.ndarray], progress: float) -> tuple[np.ndarray, float]:
    lengths = segment_lengths(polyline)
    total = float(np.sum(lengths))
    progress = min(max(progress, 0.0), total)
    cumulative = 0.0
    for i, length in enumerate(lengths):
        if i == len(lengths) - 1 or progress <= cumulative + float(length):
            local = progress - cumulative
            start = polyline[i]
            end = polyline[i + 1]
            direction = (end - start) / max(float(length), 1e-9)
            position = start + local * direction
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            return position, yaw
        cumulative += float(length)
    direction = polyline[-1] - polyline[-2]
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    return polyline[-1], yaw


def reference_subgoals_for_path(reference_path: list[np.ndarray], settings: dict[str, Any]) -> list[tuple[np.ndarray, float]]:
    """Sample planner subgoals on drivable straights, not hard corner vertices.

    Sparse subgoals placed directly on sharp offset-polyline corners can force
    Hybrid A* to invent small hooks. For sharp vertices, skip corner-adjacent
    targets entirely and let the planner connect between straight-section goals.
    """
    total_length = path_length(reference_path)
    spacing = float(settings.get("planner_subgoal_spacing_m", 3.0))
    endpoint_margin = min(float(settings.get("planner_subgoal_endpoint_margin_m", 0.5)), max(total_length * 0.25, 0.0))
    vertex_margin = float(settings.get("planner_subgoal_vertex_margin_m", 0.5))
    corner_angle_deg = float(settings.get("planner_subgoal_corner_angle_deg", 45.0))
    corner_exclusion_m = float(settings.get("planner_subgoal_corner_exclusion_m", max(vertex_margin, 1.0)))
    max_progress = max(total_length - endpoint_margin, 0.0)
    if spacing <= 0.0 or max_progress <= spacing:
        return [sample_path_pose(reference_path, max_progress)]

    vertex_progress_values: list[float] = []
    sharp_vertex_progress_values: list[float] = []
    cumulative = 0.0
    for index in range(len(reference_path) - 1):
        segment_length = float(np.linalg.norm(reference_path[index + 1] - reference_path[index]))
        cumulative += segment_length
        if index >= len(reference_path) - 2:
            continue
        vertex_progress_values.append(cumulative)
        prev_vec = reference_path[index + 1] - reference_path[index]
        next_vec = reference_path[index + 2] - reference_path[index + 1]
        prev_len = float(np.linalg.norm(prev_vec))
        next_len = float(np.linalg.norm(next_vec))
        if prev_len <= 1e-9 or next_len <= 1e-9:
            continue
        prev_dir = prev_vec / prev_len
        next_dir = next_vec / next_len
        dot = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
        turn_angle_deg = math.degrees(math.acos(dot))
        if turn_angle_deg >= corner_angle_deg:
            sharp_vertex_progress_values.append(cumulative)

    progress_values = list(np.arange(spacing, total_length, spacing))
    progress_values.append(max_progress)

    adjusted_progress_values: list[float] = []
    for progress in sorted(progress_values):
        adjusted = min(float(progress), max_progress)
        if any(abs(adjusted - vertex_progress) < corner_exclusion_m for vertex_progress in sharp_vertex_progress_values):
            continue
        if any(abs(adjusted - vertex_progress) < vertex_margin for vertex_progress in vertex_progress_values):
            continue
        if adjusted <= 1e-6:
            continue
        if not adjusted_progress_values or adjusted > adjusted_progress_values[-1] + 1e-6:
            adjusted_progress_values.append(adjusted)
    return [sample_path_pose(reference_path, float(progress)) for progress in adjusted_progress_values]


def offset_polyline(polyline: list[np.ndarray], offset_m: float, side: str) -> list[np.ndarray]:
    if len(polyline) < 2:
        return polyline
    offset_points = []
    for i, point in enumerate(polyline):
        if i == 0:
            direction = polyline[1] - polyline[0]
        elif i == len(polyline) - 1:
            direction = polyline[-1] - polyline[-2]
        else:
            prev_dir = polyline[i] - polyline[i - 1]
            next_dir = polyline[i + 1] - polyline[i]
            direction = prev_dir / max(np.linalg.norm(prev_dir), 1e-9) + next_dir / max(np.linalg.norm(next_dir), 1e-9)
            if np.linalg.norm(direction) < 1e-9:
                direction = next_dir
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        left_normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        normal = left_normal if side == "left" else -left_normal
        offset_points.append(point + offset_m * normal)
    return offset_points


def world_to_robot(anchor_position: np.ndarray, anchor_yaw: float, point: np.ndarray, yaw: float) -> tuple[float, float, float]:
    delta = point - anchor_position
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    x_robot = cos_yaw * float(delta[0]) + sin_yaw * float(delta[1])
    y_robot = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
    return x_robot, y_robot, wrap_to_pi(yaw - anchor_yaw)


def build_action_chunk(
    node,
    current_position: np.ndarray,
    current_yaw: float,
    path: list[np.ndarray],
    seq_num: int,
    waypoint_spacing_m: float,
    frame_id: str = "base_link",
) -> object:
    from aion_msgs.msg import ActionChunk
    from geometry_msgs.msg import Pose2D

    progress = project_progress(path, current_position)
    total = path_length(path)
    msg = ActionChunk()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.seq_num = seq_num

    for index in range(1, len(msg.relative_poses) + 1):
        target_progress = min(progress + index * waypoint_spacing_m, total)
        target_position, target_yaw = sample_path_pose(path, target_progress)
        x, y, theta = world_to_robot(current_position, current_yaw, target_position, target_yaw)
        pose = Pose2D()
        pose.x = float(x)
        pose.y = float(y)
        pose.theta = float(theta)
        msg.relative_poses[index - 1] = pose
    return msg


def build_timed_action_chunk(
    node,
    trajectory,
    current_position: np.ndarray,
    current_yaw: float,
    current_progress_time_s: float,
    future_time_offsets_s: list[float],
    seq_num: int,
    frame_id: str = "base_link",
) -> object:
    from aion_msgs.msg import ActionChunk
    from geometry_msgs.msg import Pose2D

    msg = ActionChunk()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.seq_num = seq_num
    offsets = list(future_time_offsets_s)
    while len(offsets) < len(msg.relative_poses):
        offsets.append(offsets[-1] if offsets else 0.3)

    total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
    min_forward_x_m = 0.03
    min_preview_distance_m = 0.25
    min_preview_spacing_m = 0.12
    search_step_m = 0.15
    max_forward_search_m = 2.0
    current_distance = float(np.interp(current_progress_time_s, trajectory.times, trajectory.distances))

    for index in range(len(msg.relative_poses)):
        target_position, target_yaw, _ = sample_timed_action_target(
            trajectory,
            current_position,
            current_yaw,
            current_progress_time_s,
            float(offsets[index]),
            min_forward_x_m=min_forward_x_m,
            min_preview_distance_m=min_preview_distance_m + index * min_preview_spacing_m,
            max_forward_search_m=max_forward_search_m,
            search_step_m=search_step_m,
            current_distance=current_distance,
        )
        x, y, theta = world_to_robot(current_position, current_yaw, target_position, target_yaw)
        pose = Pose2D()
        pose.x = float(x)
        pose.y = float(y)
        pose.theta = float(theta)
        msg.relative_poses[index] = pose
    return msg


def build_distance_action_chunk(
    node,
    trajectory,
    current_position: np.ndarray,
    current_yaw: float,
    current_progress_m: float,
    seq_num: int,
    frame_id: str = "base_link",
    first_preview_m: float = 0.35,
    waypoint_spacing_m: float = 0.18,
) -> object:
    from aion_msgs.msg import ActionChunk
    from geometry_msgs.msg import Pose2D

    msg = ActionChunk()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.seq_num = seq_num

    total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
    min_forward_x_m = 0.08
    max_lateral_y_m = 1.75
    search_step_m = 0.15
    max_forward_search_m = 3.0

    last_target_distance: float | None = None
    last_safe_pose: tuple[float, float, float] | None = None
    for index in range(len(msg.relative_poses)):
        preview_m = first_preview_m + index * waypoint_spacing_m
        target_position, target_yaw, target_distance = sample_distance_action_target(
            trajectory,
            current_position,
            current_yaw,
            current_progress_m,
            preview_m,
            min_forward_x_m=min_forward_x_m,
            max_lateral_y_m=max_lateral_y_m,
            max_forward_search_m=max_forward_search_m,
            search_step_m=search_step_m,
            min_target_distance=(
                last_target_distance + waypoint_spacing_m
                if last_target_distance is not None
                else None
            ),
        )
        x, y, theta = world_to_robot(current_position, current_yaw, target_position, target_yaw)
        if (
            not action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m)
            and last_safe_pose is not None
        ):
            x, y, theta = last_safe_pose
        elif not action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m):
            x, y, theta = local_reacquisition_pose(x, y, min_forward_x_m, max_lateral_y_m)
            last_safe_pose = (float(x), float(y), float(theta))
        elif action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m):
            last_target_distance = target_distance
            last_safe_pose = (float(x), float(y), float(theta))
        pose = Pose2D()
        pose.x = float(x)
        pose.y = float(y)
        pose.theta = float(theta)
        msg.relative_poses[index] = pose
    return msg


def sample_distance_action_target(
    trajectory,
    current_position: np.ndarray,
    current_yaw: float,
    current_progress_m: float,
    preview_m: float,
    *,
    min_forward_x_m: float = 0.08,
    max_lateral_y_m: float = 1.75,
    max_forward_search_m: float = 3.0,
    max_backward_reacquire_m: float = 2.5,
    search_step_m: float = 0.15,
    min_target_distance: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Sample by path distance ahead of current progress, forcing a usable forward target."""
    total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
    current_progress_m = min(max(float(current_progress_m), 0.0), total_distance)
    target_distance = min(current_progress_m + max(float(preview_m), 0.0), total_distance)
    if min_target_distance is not None:
        target_distance = max(target_distance, min(max(float(min_target_distance), 0.0), total_distance))
    target_position, target_yaw = sample_path_pose(trajectory.path, target_distance)
    x, y, _ = world_to_robot(current_position, current_yaw, target_position, target_yaw)

    searched = 0.0
    while (
        not action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m)
        and searched < max_forward_search_m
        and target_distance < total_distance
    ):
        searched += search_step_m
        target_distance = min(target_distance + search_step_m, total_distance)
        target_position, target_yaw = sample_path_pose(trajectory.path, target_distance)
        x, y, _ = world_to_robot(current_position, current_yaw, target_position, target_yaw)

    if not action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m):
        reacquired = nearest_forward_path_target(
            trajectory,
            current_position,
            current_yaw,
            current_progress_m,
            min_forward_x_m=min_forward_x_m,
            max_lateral_y_m=max_lateral_y_m,
            max_backward_m=max_backward_reacquire_m,
            max_forward_m=max_forward_search_m,
            search_step_m=search_step_m,
            min_target_distance=min_target_distance,
        )
        if reacquired is not None:
            target_position, target_yaw, target_distance = reacquired

    return target_position, target_yaw, target_distance


def action_target_is_trackable(
    x: float,
    y: float,
    min_forward_x_m: float,
    max_lateral_y_m: float,
) -> bool:
    return (
        math.isfinite(float(x))
        and math.isfinite(float(y))
        and float(x) > float(min_forward_x_m)
        and abs(float(y)) <= float(max_lateral_y_m)
    )


def local_reacquisition_pose(
    x: float,
    y: float,
    min_forward_x_m: float,
    max_lateral_y_m: float,
) -> tuple[float, float, float]:
    forward_x = max(float(min_forward_x_m) + 0.12, min(max(float(x), 0.2), 0.45))
    lateral_y = min(max(float(y), -float(max_lateral_y_m)), float(max_lateral_y_m))
    theta = math.atan2(lateral_y, forward_x)
    return forward_x, lateral_y, theta


def nearest_forward_path_target(
    trajectory,
    current_position: np.ndarray,
    current_yaw: float,
    current_progress_m: float,
    *,
    min_forward_x_m: float,
    max_lateral_y_m: float,
    max_backward_m: float,
    max_forward_m: float,
    search_step_m: float,
    min_target_distance: float | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Find a nearby path point that is genuinely in front of the rover frame.

    Progress projection can legitimately be ahead of the robot when the rover is
    off-path or sideways around an obstacle. In that case, continuing to sample
    farther along the path can leave every ActionChunk point behind the rover,
    which makes the tracker rotate in place. This local scan gives the controller
    a forward reacquisition target without changing the monotonic progress state.
    """
    total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
    if total_distance <= 0.0:
        return None

    step = max(float(search_step_m), 0.02)
    lower = max(0.0, float(current_progress_m) - max(float(max_backward_m), 0.0))
    upper = min(total_distance, float(current_progress_m) + max(float(max_forward_m), 0.0))
    if min_target_distance is not None:
        lower = max(lower, min(max(float(min_target_distance), 0.0), total_distance))
    if lower > upper:
        return None

    best: tuple[float, np.ndarray, float, float] | None = None
    count = max(1, int(math.ceil((upper - lower) / step)))
    for i in range(count + 1):
        distance = min(upper, lower + i * step)
        position, yaw = sample_path_pose(trajectory.path, distance)
        x, y, theta = world_to_robot(current_position, current_yaw, position, yaw)
        if not action_target_is_trackable(x, y, min_forward_x_m, max_lateral_y_m):
            continue
        euclidean = float(np.linalg.norm(position - current_position))
        # Prefer nearby, forward, low-lateral-error targets. The tiny path term
        # keeps later chunk entries ordered when several candidates are similar.
        score = euclidean + 0.35 * abs(float(y)) + 0.03 * abs(distance - current_progress_m)
        if best is None or score < best[0]:
            best = (score, position, yaw, distance)

    if best is None:
        return None
    _, position, yaw, distance = best
    return position, yaw, distance


def sample_timed_action_target(
    trajectory,
    current_position: np.ndarray,
    current_yaw: float,
    current_progress_time_s: float,
    future_offset_s: float,
    *,
    min_forward_x_m: float = 0.03,
    min_preview_distance_m: float = 0.25,
    max_forward_search_m: float = 2.0,
    search_step_m: float = 0.15,
    current_distance: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Sample a future target while avoiding tiny or behind-robot previews."""
    total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
    target_time = current_progress_time_s + float(future_offset_s)
    if current_distance is None:
        current_distance = float(np.interp(current_progress_time_s, trajectory.times, trajectory.distances))
    timed_distance = float(np.interp(target_time, trajectory.times, trajectory.distances))
    target_distance = max(timed_distance, float(current_distance) + min_preview_distance_m)
    target_distance = min(target_distance, total_distance)
    target_position, target_yaw = sample_path_pose(trajectory.path, target_distance)
    x, _, _ = world_to_robot(current_position, current_yaw, target_position, target_yaw)

    searched = 0.0
    while x <= min_forward_x_m and searched < max_forward_search_m and target_distance < total_distance:
        searched += search_step_m
        target_distance = min(target_distance + search_step_m, total_distance)
        target_position, target_yaw = sample_path_pose(trajectory.path, target_distance)
        x, _, _ = world_to_robot(current_position, current_yaw, target_position, target_yaw)

    return target_position, target_yaw, target_distance


def fence_by_name(scene: dict[str, Any], name: str) -> dict[str, Any]:
    for fence in scene.get("fences", []):
        if fence.get("name") == name:
            return fence
    raise ValueError(f"Fence {name!r} not found.")


def road_by_name(scene: dict[str, Any], name: str) -> dict[str, Any]:
    for road in scene.get("roads", []):
        if road.get("name") == name:
            return road
    raise ValueError(f"Road {name!r} not found.")


def segment_polyline(segment: dict[str, Any], flip_isaac_y: bool) -> list[np.ndarray]:
    return [point2(segment["start"], flip_isaac_y), point2(segment["end"], flip_isaac_y)]


def concat_segments(segments: list[dict[str, Any]], flip_isaac_y: bool) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for segment in segments:
        start, end = segment_polyline(segment, flip_isaac_y)
        if not points:
            points.append(start)
        elif np.linalg.norm(points[-1] - start) > 1e-6:
            points.append(start)
        points.append(end)
    return points


def offset_segment_sequence(
    segments: list[dict[str, Any]],
    flip_isaac_y: bool,
    offset_m: float,
    side: str,
) -> list[np.ndarray]:
    """Offset each segment and round connected joins on the driving line."""
    def append_point(path: list[np.ndarray], point: np.ndarray) -> None:
        if not path or float(np.linalg.norm(path[-1] - point)) > 1e-6:
            path.append(point)

    def same_point(a: np.ndarray, b: np.ndarray) -> bool:
        return float(np.linalg.norm(a - b)) <= 1e-6

    def append_corner_arc(path: list[np.ndarray], center: np.ndarray, end_point: np.ndarray) -> None:
        if not path:
            append_point(path, end_point)
            return
        start_point = path[-1]
        radius_start = float(np.linalg.norm(start_point - center))
        radius_end = float(np.linalg.norm(end_point - center))
        if radius_start <= 1e-6 or radius_end <= 1e-6:
            append_point(path, end_point)
            return
        radius = 0.5 * (radius_start + radius_end)
        start_angle = math.atan2(float(start_point[1] - center[1]), float(start_point[0] - center[0]))
        end_angle = math.atan2(float(end_point[1] - center[1]), float(end_point[0] - center[0]))
        delta = wrap_to_pi(end_angle - start_angle)
        if abs(delta) < math.radians(5.0) or abs(delta) > math.radians(170.0):
            append_point(path, end_point)
            return
        steps = max(2, int(math.ceil(abs(delta) / math.radians(12.0))))
        for step in range(1, steps + 1):
            angle = start_angle + delta * (float(step) / float(steps))
            append_point(path, center + radius * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64))

    points: list[np.ndarray] = []
    segment_paths: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
    for segment in segments:
        original_start, original_end = segment_polyline(segment, flip_isaac_y)
        segment_path = offset_polyline(segment_polyline(segment, flip_isaac_y), offset_m, side)
        if not segment_path:
            continue
        segment_paths.append((original_start, original_end, segment_path))

    for index, (_, original_end, segment_path) in enumerate(segment_paths):
        if not points:
            for point in segment_path:
                append_point(points, point)
            continue
        previous_original_end = segment_paths[index - 1][1]
        original_start = segment_paths[index][0]
        if same_point(previous_original_end, original_start):
            append_corner_arc(points, original_start, segment_path[0])
        else:
            append_point(points, segment_path[0])
        append_point(points, segment_path[-1])

    if len(segment_paths) > 2 and same_point(segment_paths[-1][1], segment_paths[0][0]) and points:
        append_corner_arc(points, segment_paths[0][0], points[0])
    return points


def densify_polyline(polyline: list[np.ndarray], max_segment_length_m: float) -> list[np.ndarray]:
    if len(polyline) < 2 or max_segment_length_m <= 0.0:
        return list(polyline)
    dense = [polyline[0]]
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        steps = max(1, int(math.ceil(length / max_segment_length_m)))
        for step in range(1, steps + 1):
            point = start + segment * (float(step) / float(steps))
            if float(np.linalg.norm(point - dense[-1])) > 1e-6:
                dense.append(point)
    return dense


def line_path_from_points(points: list[list[float]], flip_isaac_y: bool) -> list[np.ndarray]:
    return [point2(point, flip_isaac_y) for point in points]


def get_asset_bbox(scene: dict[str, Any], asset_group: str, asset_name: str, scene_path: Path) -> list[float] | None:
    assets = scene.get("assets", {})
    direct = assets.get(asset_group)
    if isinstance(direct, dict) and "bbox_size" in direct:
        return direct["bbox_size"]
    if isinstance(direct, dict) and asset_name in direct and "bbox_size" in direct[asset_name]:
        return direct[asset_name]["bbox_size"]

    asset_ref = assets.get(asset_group)
    if isinstance(asset_ref, str):
        asset_name = asset_ref

    library_path = scene.get("asset_library")
    if not library_path:
        return None
    library = Path(library_path)
    if not library.is_absolute():
        library = (scene_path.parent / library).resolve()
        if not library.exists():
            library = (Path("C:/Users/miahv/Documents/Capstone_Project/isaac") / library_path).resolve()
    if not library.exists():
        return None
    library_data = load_yaml(library)
    entry = library_data.get("assets", {}).get(asset_group, {}).get(asset_name)
    if isinstance(entry, dict):
        return entry.get("bbox_size")
    return None

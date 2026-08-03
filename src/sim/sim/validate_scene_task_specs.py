#!/usr/bin/env python3
"""Validate generated rollout task specs against Isaac scene geometry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.collision_map import CollisionMap
from sim.expert_trajectory_utils import (
    concat_segments,
    fence_by_name,
    get_asset_bbox,
    offset_polyline,
    path_length,
    point2,
    road_by_name,
    rotate,
    sample_path_pose,
)
from sim.hybrid_astar import HybridAStarPlanner, Pose


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


FENCELINE_EXPERT_TASKS = {
    "follow_fence",
    "follow_and_turn",
    "follow_fence_sequence",
    "follow_corridor",
    "pass_through_gap",
    "stop_at_gap",
    "stop_at_landmark",
    "switch_sides",
    "hold_position",
}

ROAD_EXPERT_TASKS = {
    "follow_road",
    "follow_and_turn",
    "approach_target",
    "stop_at_landmark",
    "hold_position",
}

SHEDLINE_EXPERT_TASKS = {
    "follow_shed_side",
    "approach_target",
    "stop_at_landmark",
    "hold_position",
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


def xy(point: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def heading_of(segment: dict[str, Any]) -> float:
    start = xy(segment["start"])
    end = xy(segment["end"])
    return math.atan2(end[1] - start[1], end[0] - start[0])


def turn_direction(first: dict[str, Any], second: dict[str, Any]) -> str:
    delta = wrap_to_pi(heading_of(second) - heading_of(first))
    if delta > 0.15:
        return "left"
    if delta < -0.15:
        return "right"
    return "straight"


def opposite_side(side: str) -> str:
    return "right" if side == "left" else "left"


def nearest_point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    sx, sy = start
    ex, ey = end
    px, py = point
    vx = ex - sx
    vy = ey - sy
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return start, 0.0
    t = ((px - sx) * vx + (py - sy) * vy) / length_sq
    t = min(1.0, max(0.0, t))
    return (sx + t * vx, sy + t * vy), t


def distance_to_segment(point: tuple[float, float], segment: dict[str, Any]) -> float:
    nearest, _ = nearest_point_on_segment(point, xy(segment["start"]), xy(segment["end"]))
    return distance(point, nearest)


def side_of_segment(point: tuple[float, float], segment: dict[str, Any]) -> str:
    sx, sy = xy(segment["start"])
    ex, ey = xy(segment["end"])
    px, py = point
    cross = (ex - sx) * (py - sy) - (ey - sy) * (px - sx)
    if cross >= 0.0:
        return "left"
    return "right"


def yaw_of_pose(pose: dict[str, Any]) -> float:
    return float(pose.get("yaw", 0.0))


def robot_relative_side_to_segment(start_pose: dict[str, Any], segment: dict[str, Any]) -> str:
    start_point = xy(start_pose["position"])
    nearest, _ = nearest_point_on_segment(start_point, xy(segment["start"]), xy(segment["end"]))
    dx = nearest[0] - start_point[0]
    dy = nearest[1] - start_point[1]
    yaw = yaw_of_pose(start_pose)
    y_robot = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if y_robot >= 0.0:
        return "left"
    return "right"


def path_side_for_task(
    task: dict[str, Any],
    flip_isaac_y: bool,
    *,
    path_key: str = "path_side",
    follow_key: str = "follow_side",
) -> str:
    side = task.get(path_key)
    if side not in {"left", "right"}:
        side = opposite_side(task.get(follow_key, "left"))
    if flip_isaac_y:
        side = opposite_side(side)
    return side


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> list[float]:
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0]


def get_start_poses(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(scene.get("rover_poses"), dict):
        return scene["rover_poses"]
    if isinstance(scene.get("rover_pose"), dict):
        return {"scene_rover_pose": scene["rover_pose"]}
    return {}


def infer_config_type(scene: dict[str, Any]) -> str:
    config_type = scene.get("config_type")
    if isinstance(config_type, str) and config_type:
        return config_type
    if scene.get("fences"):
        return "fenceline"
    if scene.get("roads"):
        return "road"
    if scene.get("shed"):
        return "shedline"
    return "unknown"


def get_by_name(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("name") == name:
            return item
    return None


def connected_segment_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    max_endpoint_gap_m: float,
) -> bool:
    return distance(xy(first["end"]), xy(second["start"])) <= max_endpoint_gap_m


def detected_gap(
    before: dict[str, Any],
    after: dict[str, Any],
    min_gap_m: float = 0.4,
    max_heading_delta_rad: float = 0.2,
) -> tuple[bool, float, list[float]]:
    before_end = xy(before["end"])
    after_start = xy(after["start"])
    gap_distance = distance(before_end, after_start)
    if gap_distance < min_gap_m:
        return False, gap_distance, midpoint(before_end, after_start)

    heading_delta = abs(wrap_to_pi(heading_of(after) - heading_of(before)))
    connection_heading = math.atan2(after_start[1] - before_end[1], after_start[0] - before_end[0])
    connection_delta = abs(wrap_to_pi(connection_heading - heading_of(before)))
    is_gap = heading_delta <= max_heading_delta_rad and connection_delta <= max_heading_delta_rad
    return is_gap, gap_distance, midpoint(before_end, after_start)


def validate_start_pose(task: dict[str, Any], starts: dict[str, dict[str, Any]]) -> list[str]:
    start_pose = task.get("start_pose")
    if not isinstance(start_pose, str):
        return ["missing start_pose"]
    if start_pose not in starts:
        return [f"start_pose {start_pose!r} not found in scene"]
    if "position" not in starts[start_pose]:
        return [f"start_pose {start_pose!r} has no position"]
    return []


def validate_expert_support(task: dict[str, Any], scene: dict[str, Any]) -> list[str]:
    task_type = task.get("task_type")
    config_type = infer_config_type(scene)
    supported_by_scene = {
        "fenceline": FENCELINE_EXPERT_TASKS,
        "road": ROAD_EXPERT_TASKS,
        "shedline": SHEDLINE_EXPERT_TASKS,
    }
    supported = supported_by_scene.get(config_type, set())
    if task_type not in supported:
        return [f"task_type {task_type!r} is scene-plausible but no current {config_type} expert supports it yet"]
    return []


def validate_follow_fence(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
) -> list[str]:
    errors = []
    fence = get_by_name(scene.get("fences") or [], str(task.get("target_fence")))
    if fence is None:
        return [f"target_fence {task.get('target_fence')!r} not found"]

    start_pose = starts[task["start_pose"]]
    start = xy(start_pose["position"])
    start_distance = distance_to_segment(start, fence)
    if start_distance > max_start_distance_m:
        errors.append(
            f"start_pose is {start_distance:.2f}m from target_fence {fence['name']}, "
            f"max allowed is {max_start_distance_m:.2f}m"
        )

    follow_side = task.get("follow_side")
    actual_follow_side = robot_relative_side_to_segment(start_pose, fence)
    if follow_side in {"left", "right"} and follow_side != actual_follow_side:
        errors.append(
            f"follow_side is {follow_side!r}, but fence is on the robot's {actual_follow_side} side"
        )

    path_side = task.get("path_side")
    actual_path_side = side_of_segment(start, fence)
    if path_side in {"left", "right"} and path_side != actual_path_side:
        errors.append(f"path_side is {path_side!r}, but start pose is on the segment's {actual_path_side} side")
    return errors


def validate_stop_at_fence_end(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
) -> list[str]:
    if "target_fence" not in task:
        return []
    return validate_follow_fence(task, scene, starts, max_start_distance_m)


def validate_follow_and_turn(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
) -> list[str]:
    target_segments = task.get("target_fences") or task.get("target_roads")
    if not isinstance(target_segments, list) or len(target_segments) != 2:
        return ["follow_and_turn requires exactly two target_fences or target_roads"]
    first = get_by_name(scene.get("fences") or scene.get("roads") or [], target_segments[0])
    second = get_by_name(scene.get("fences") or scene.get("roads") or [], target_segments[1])
    if first is None or second is None:
        return [f"target segments {target_segments!r} not found"]

    endpoint_gap_limit = 0.05 if infer_config_type(scene) == "fenceline" else 5.0
    errors = []
    if not connected_segment_pair(first, second, endpoint_gap_limit):
        errors.append(f"target segments are not connected within {endpoint_gap_limit:.2f}m")

    expected_turn = turn_direction(first, second)
    if expected_turn == "straight":
        errors.append("target segments do not form a turn")
    elif task.get("turn_direction") != expected_turn:
        errors.append(f"turn_direction is {task.get('turn_direction')!r}, expected {expected_turn!r}")

    start = xy(starts[task["start_pose"]]["position"])
    start_distance = distance_to_segment(start, first)
    if start_distance > max_start_distance_m:
        errors.append(f"start_pose is {start_distance:.2f}m from first segment, max allowed is {max_start_distance_m:.2f}m")
    return errors


def validate_follow_fence_sequence(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
) -> list[str]:
    target_fences = task.get("target_fences")
    if not isinstance(target_fences, list) or len(target_fences) < 3:
        return ["follow_fence_sequence requires at least three target_fences"]
    fences = scene.get("fences") or []
    resolved = [get_by_name(fences, name) for name in target_fences]
    if any(fence is None for fence in resolved):
        return [f"target_fences {target_fences!r} not found"]
    errors = []
    for first, second in zip(resolved, resolved[1:]):
        if not connected_segment_pair(first, second, 0.05):
            errors.append(f"target_fences {first['name']} and {second['name']} are not connected")
    start = xy(starts[task["start_pose"]]["position"])
    start_distance = distance_to_segment(start, resolved[0])
    if start_distance > max_start_distance_m:
        errors.append(f"start_pose is {start_distance:.2f}m from first fence, max allowed is {max_start_distance_m:.2f}m")
    return errors


def validate_follow_corridor(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
) -> list[str]:
    corridor_fences = task.get("corridor_fences")
    if not isinstance(corridor_fences, list) or len(corridor_fences) != 2:
        return ["follow_corridor requires exactly two corridor_fences"]
    fences = scene.get("fences") or []
    first = get_by_name(fences, corridor_fences[0])
    second = get_by_name(fences, corridor_fences[1])
    if first is None or second is None:
        return [f"corridor_fences {corridor_fences!r} not found"]
    heading_delta = abs(wrap_to_pi(heading_of(first) - heading_of(second)))
    if min(heading_delta, abs(math.pi - heading_delta)) > 0.12:
        return ["corridor_fences are not parallel"]
    first_heading = heading_of(first)
    direction = (math.cos(first_heading), math.sin(first_heading))
    normal = (-direction[1], direction[0])
    first_start = xy(first["start"])
    first_length = distance(first_start, xy(first["end"]))
    second_projections = [
        (point[0] - first_start[0]) * direction[0] + (point[1] - first_start[1]) * direction[1]
        for point in (xy(second["start"]), xy(second["end"]))
    ]
    overlap = min(first_length, max(second_projections)) - max(0.0, min(second_projections))
    if overlap < 1.0:
        return ["corridor_fences do not overlap along their length"]
    lateral_offsets = [
        (point[0] - first_start[0]) * normal[0] + (point[1] - first_start[1]) * normal[1]
        for point in (xy(second["start"]), xy(second["end"]))
    ]
    lateral_separation = abs(sum(lateral_offsets) * 0.5)
    if lateral_separation < 1.2 or lateral_separation > 4.0:
        return [f"corridor_fences lateral separation {lateral_separation:.2f}m is outside supported range"]
    start = xy(starts[task["start_pose"]]["position"])
    first_distance = distance_to_segment(start, first)
    second_distance = distance_to_segment(start, second)
    if max(first_distance, second_distance) > 4.0:
        return ["start_pose is too far from the corridor fences"]
    return []


def validate_gap_task(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
) -> list[str]:
    target_gap = task.get("target_gap")
    if not isinstance(target_gap, dict):
        return ["gap task requires target_gap"]
    before = get_by_name(scene.get("fences") or [], str(target_gap.get("before_fence")))
    after = get_by_name(scene.get("fences") or [], str(target_gap.get("after_fence")))
    if before is None or after is None:
        return [f"gap fences {target_gap!r} not found"]

    errors = []
    is_gap, gap_width, center = detected_gap(before, after)
    if not is_gap:
        errors.append(f"before/after fences do not form a collinear gap, gap_width={gap_width:.2f}m")
    if task.get("task_type") == "switch_sides" and gap_width < 1.0:
        errors.append(f"switch_sides needs at least 1.00m gap width, found {gap_width:.2f}m")

    requested_center = target_gap.get("approximate_center")
    if isinstance(requested_center, list) and distance(xy(requested_center), xy(center)) > 0.25:
        errors.append(f"target_gap approximate_center does not match detected center {center}")

    start_pose = starts[task["start_pose"]]
    start = xy(start_pose["position"])
    start_distance = distance_to_segment(start, before)
    if start_distance > max_start_distance_m:
        errors.append(f"start_pose is {start_distance:.2f}m from gap approach fence, max allowed is {max_start_distance_m:.2f}m")
    follow_side = task.get("follow_side") or task.get("entry_side")
    actual_follow_side = robot_relative_side_to_segment(start_pose, before)
    if follow_side in {"left", "right"} and follow_side != actual_follow_side:
        errors.append(
            f"follow/entry side is {follow_side!r}, but gap approach fence is on the robot's {actual_follow_side} side"
        )
    path_side = task.get("path_side") or task.get("entry_path_side")
    actual_path_side = side_of_segment(start, before)
    if path_side in {"left", "right"} and path_side != actual_path_side:
        errors.append(f"path side is {path_side!r}, but start pose is on the segment's {actual_path_side} side")
    return errors


def validate_hold_position(task: dict[str, Any]) -> list[str]:
    success_condition = task.get("success_condition")
    if not isinstance(success_condition, dict):
        return ["hold_position requires success_condition"]
    if success_condition.get("type") != "remain_near_start":
        return ["hold_position success_condition.type should be remain_near_start"]
    return []


def validate_task(
    task: dict[str, Any],
    scene: dict[str, Any],
    starts: dict[str, dict[str, Any]],
    max_start_distance_m: float,
    check_expert_support: bool,
) -> list[str]:
    errors = []
    for key in ["task_id", "task_type", "instruction"]:
        if key not in task:
            errors.append(f"missing {key}")
    if errors:
        return errors

    if check_expert_support:
        errors.extend(validate_expert_support(task, scene))

    task_type = task["task_type"]
    if task_type != "hold_position":
        errors.extend(validate_start_pose(task, starts))
        if errors:
            return errors
    else:
        errors.extend(validate_start_pose(task, starts))
        errors.extend(validate_hold_position(task))
        return errors

    if task_type == "follow_fence":
        errors.extend(validate_follow_fence(task, scene, starts, max_start_distance_m))
    elif task_type == "stop_at_landmark":
        errors.extend(validate_stop_at_fence_end(task, scene, starts, max_start_distance_m))
    elif task_type == "follow_and_turn":
        errors.extend(validate_follow_and_turn(task, scene, starts, max_start_distance_m))
    elif task_type == "follow_fence_sequence":
        errors.extend(validate_follow_fence_sequence(task, scene, starts, max_start_distance_m))
    elif task_type == "follow_corridor":
        errors.extend(validate_follow_corridor(task, scene, starts))
    elif task_type in {"pass_through_gap", "stop_at_gap", "switch_sides"}:
        errors.extend(validate_gap_task(task, scene, starts, max_start_distance_m))
    return errors


def yaw_value(raw_yaw: Any, flip_isaac_y: bool) -> float:
    yaw = float(raw_yaw or 0.0)
    if abs(yaw) > math.tau:
        yaw = math.radians(yaw)
    if flip_isaac_y:
        yaw = -yaw
    return yaw


def scene_start_pose(scene: dict[str, Any], task: dict[str, Any], flip_isaac_y: bool) -> tuple[np.ndarray, float]:
    starts = get_start_poses(scene)
    start_name = task.get("start_pose")
    if isinstance(start_name, str) and start_name in starts:
        start = starts[start_name]
    elif isinstance(scene.get("rover_pose"), dict):
        start = scene["rover_pose"]
    else:
        raise ValueError("Scene does not define a usable rover start pose.")
    return point2(start["position"], flip_isaac_y), yaw_value(start.get("yaw", 0.0), flip_isaac_y)


def shifted_start_pose(
    scene: dict[str, Any],
    task: dict[str, Any],
    variant: dict[str, Any],
    flip_isaac_y: bool,
) -> tuple[np.ndarray, float]:
    start, yaw = scene_start_pose(scene, task, flip_isaac_y)
    delta = variant.get("start_pose_delta") or {}
    local_delta = np.asarray([float(delta.get("x_m", 0.0)), float(delta.get("y_m", 0.0))], dtype=np.float64)
    return start + rotate(local_delta, yaw), wrap_to_pi(yaw + float(delta.get("yaw_rad", 0.0)))


def shed_center(scene: dict[str, Any], flip_isaac_y: bool) -> np.ndarray:
    shed = scene.get("shed", {})
    return point2(shed.get("position", [0.0, 0.0, 0.0]), flip_isaac_y)


def shed_yaw(scene: dict[str, Any], flip_isaac_y: bool) -> float:
    return yaw_value(scene.get("shed", {}).get("yaw", 0.0), flip_isaac_y)


def shed_side_path(
    scene: dict[str, Any],
    scene_yaml: Path,
    task: dict[str, Any],
    offset_m: float,
    flip_isaac_y: bool,
) -> list[np.ndarray]:
    bbox = get_asset_bbox(scene, "shed", "shed", scene_yaml)
    if bbox is None:
        raise ValueError("Could not resolve shed bbox_size from scene assets or asset library.")
    half_x = float(bbox[0]) * 0.5
    half_y = float(bbox[1]) * 0.5
    center = shed_center(scene, flip_isaac_y)
    yaw = shed_yaw(scene, flip_isaac_y)
    side = task.get("shed_side", "north")
    if side == "nearest":
        side = "north"
    local_segments = {
        "north": [np.asarray([-half_x, half_y + offset_m]), np.asarray([half_x, half_y + offset_m])],
        "south": [np.asarray([half_x, -half_y - offset_m]), np.asarray([-half_x, -half_y - offset_m])],
        "east": [np.asarray([half_x + offset_m, half_y]), np.asarray([half_x + offset_m, -half_y])],
        "west": [np.asarray([-half_x - offset_m, -half_y]), np.asarray([-half_x - offset_m, half_y])],
    }
    if side not in local_segments:
        raise ValueError(f"Unsupported shed side {side!r}")
    return [center + rotate(point, yaw) for point in local_segments[side]]


def reference_path_for_task(
    scene: dict[str, Any],
    scene_yaml: Path,
    task: dict[str, Any],
    variant: dict[str, Any],
    flip_isaac_y: bool,
) -> list[np.ndarray]:
    task_type = task["task_type"]
    offset_m = float(variant.get("preferred_offset_m", 0.8))
    if task_type == "follow_fence":
        fence = fence_by_name(scene, task["target_fence"])
        return offset_polyline(
            concat_segments([fence], flip_isaac_y),
            offset_m,
            path_side_for_task(task, flip_isaac_y),
        )
    if task_type == "follow_and_turn" and "target_fences" in task:
        fences = [fence_by_name(scene, name) for name in task["target_fences"]]
        return offset_polyline(
            concat_segments(fences, flip_isaac_y),
            offset_m,
            path_side_for_task(task, flip_isaac_y),
        )
    if task_type == "follow_fence_sequence":
        fences = [fence_by_name(scene, name) for name in task["target_fences"]]
        return offset_polyline(
            concat_segments(fences, flip_isaac_y),
            offset_m,
            path_side_for_task(task, flip_isaac_y),
        )
    if task_type == "follow_corridor":
        left, right = [fence_by_name(scene, name) for name in task["corridor_fences"]]
        left_path = concat_segments([left], flip_isaac_y)
        right_path = concat_segments([right], flip_isaac_y)
        return [(left_path[0] + right_path[0]) * 0.5, (left_path[-1] + right_path[-1]) * 0.5]
    if task_type in {"pass_through_gap", "stop_at_gap", "switch_sides"}:
        gap = task["target_gap"]
        before = fence_by_name(scene, gap["before_fence"])
        after = fence_by_name(scene, gap["after_fence"])
        side = path_side_for_task(
            task,
            flip_isaac_y,
            path_key="entry_path_side" if "entry_path_side" in task else "path_side",
            follow_key="entry_side" if "entry_side" in task else "follow_side",
        )
        before_path = offset_polyline(concat_segments([before], flip_isaac_y), offset_m, side)
        if task_type == "stop_at_gap":
            return [before_path[0], before_path[-1]]
        after_side = path_side_for_task(
            task,
            flip_isaac_y,
            path_key="exit_path_side",
            follow_key="exit_side",
        )
        after_path = offset_polyline(concat_segments([after], flip_isaac_y), offset_m, after_side)
        return [before_path[0], before_path[-1], point2(gap["approximate_center"], flip_isaac_y), after_path[0], after_path[-1]]
    if task_type == "stop_at_landmark" and "target_fence" in task:
        fence = fence_by_name(scene, task["target_fence"])
        return offset_polyline(
            concat_segments([fence], flip_isaac_y),
            offset_m,
            path_side_for_task(task, flip_isaac_y),
        )
    if task_type == "follow_road":
        return concat_segments([road_by_name(scene, task["target_road"])], flip_isaac_y)
    if task_type == "follow_and_turn" and "target_roads" in task:
        return concat_segments([road_by_name(scene, name) for name in task["target_roads"]], flip_isaac_y)
    if task_type == "approach_target" and "target_point" in task:
        start, _ = scene_start_pose(scene, task, flip_isaac_y)
        return [start, point2(task["target_point"], flip_isaac_y)]
    if task_type == "stop_at_landmark" and "target_road" in task:
        return concat_segments([road_by_name(scene, task["target_road"])], flip_isaac_y)
    if task_type == "follow_shed_side":
        return shed_side_path(scene, scene_yaml, task, offset_m, flip_isaac_y)
    if task_type == "stop_at_landmark" and scene.get("shed"):
        path = shed_side_path(scene, scene_yaml, task, offset_m, flip_isaac_y)
        return [path[0], path[len(path) // 2]]
    if task_type == "hold_position":
        start, _ = scene_start_pose(scene, task, flip_isaac_y)
        return [start, start + np.asarray([0.001, 0.0], dtype=np.float64)]
    raise ValueError(f"Cannot resolve reference path for task_type {task_type!r}")


def reference_subgoals(reference_path: list[np.ndarray], settings: dict[str, Any]) -> list[tuple[np.ndarray, float]]:
    total_length = path_length(reference_path)
    spacing = float(settings.get("planner_subgoal_spacing_m", 2.0))
    if spacing <= 0.0 or total_length <= spacing:
        return [sample_path_pose(reference_path, total_length)]
    progress_values = list(np.arange(spacing, total_length, spacing))
    progress_values.append(total_length)
    return [sample_path_pose(reference_path, float(progress)) for progress in progress_values]


def build_planner(collision_map: CollisionMap, settings: dict[str, Any]) -> HybridAStarPlanner:
    return HybridAStarPlanner(
        collision_map,
        grid_resolution_m=float(settings.get("grid_resolution_m", 0.25)),
        yaw_resolution_rad=math.radians(float(settings.get("yaw_resolution_deg", 15.0))),
        step_size_m=float(settings.get("step_size_m", 0.35)),
        min_turn_radius_m=float(settings.get("min_turn_radius_m", 0.75)),
        goal_tolerance_m=float(settings.get("goal_tolerance_m", 0.35)),
        max_iterations=int(settings.get("hybrid_astar_max_iterations", 20000)),
        allow_reverse=bool(settings.get("allow_reverse", False)),
    )


def plan_through_subgoals(
    planner: HybridAStarPlanner,
    start_position: np.ndarray,
    start_yaw: float,
    subgoals: list[tuple[np.ndarray, float]],
) -> bool:
    start_pose = Pose(float(start_position[0]), float(start_position[1]), float(start_yaw))
    for goal_position, goal_yaw in subgoals:
        segment = planner.plan(
            start_pose,
            Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw)),
        )
        if segment is None:
            return False
        start_pose = Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw))
    return True


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / length_sq)
    t = min(1.0, max(0.0, t))
    nearest = start + t * segment
    return float(np.linalg.norm(point - nearest))


def fence_clearance_error(
    scene: dict[str, Any],
    point: np.ndarray,
    required_clearance_m: float,
    flip_isaac_y: bool,
) -> str | None:
    for fence in scene.get("fences") or []:
        start = point2(fence["start"], flip_isaac_y)
        end = point2(fence["end"], flip_isaac_y)
        clearance = point_to_segment_distance(point, start, end)
        if clearance < required_clearance_m:
            return (
                f"start pose is {clearance:.2f}m from fence {fence.get('name', '<unnamed>')}, "
                f"needs at least {required_clearance_m:.2f}m inflated clearance"
            )
    return None


def planner_accepts_variant(
    scene: dict[str, Any],
    scene_yaml: Path,
    task: dict[str, Any],
    variant: dict[str, Any],
    flip_isaac_y: bool,
) -> tuple[bool, str | None]:
    if task.get("task_type") == "hold_position":
        return True, None
    try:
        reference_path = reference_path_for_task(scene, scene_yaml, task, variant, flip_isaac_y)
        start_position, start_yaw = shifted_start_pose(scene, task, variant, flip_isaac_y)
        settings = variant.get("planner_settings") or {}
        required_clearance = float(settings.get("robot_radius_m", 0.35)) + float(settings.get("obstacle_padding_m", 0.25))
        clearance_error = fence_clearance_error(scene, start_position, required_clearance, flip_isaac_y)
        if clearance_error is not None:
            return False, clearance_error
        subgoals = reference_subgoals(reference_path, settings)
        collision_map = CollisionMap.from_scene(
            scene,
            scene_yaml,
            [start_position] + [position for position, _ in subgoals] + reference_path,
            flip_isaac_y,
            robot_radius_m=float(settings.get("robot_radius_m", 0.35)),
            obstacle_padding_m=float(settings.get("obstacle_padding_m", 0.25)),
        )
        planner = build_planner(collision_map, settings)
        planned = plan_through_subgoals(planner, start_position, start_yaw, subgoals)
    except Exception as exc:
        return False, str(exc)
    if not planned:
        return False, "Hybrid A* failed to plan through reference subgoals"
    return True, None


def filter_planner_valid_variants(
    scene: dict[str, Any],
    scene_yaml: Path,
    task: dict[str, Any],
    flip_isaac_y: bool,
) -> tuple[dict[str, Any], list[str]]:
    variants = task.get("trajectory_variants") or [{"variant_id": "nominal", "variant_type": "nominal"}]
    valid_variants = []
    errors = []
    for variant in variants:
        accepted, reason = planner_accepts_variant(scene, scene_yaml, task, variant, flip_isaac_y)
        if accepted:
            checked_variant = dict(variant)
            checked_variant["planner_validation"] = {"checked": True, "valid": True}
            valid_variants.append(checked_variant)
        else:
            errors.append(f"variant {variant.get('variant_id', '<missing>')}: {reason}")
    filtered = dict(task)
    filtered["trajectory_variants"] = valid_variants
    if not valid_variants:
        return filtered, ["no trajectory_variants passed planner validation"] + errors
    if errors:
        filtered["planner_validation_summary"] = {
            "checked": True,
            "valid_variants": len(valid_variants),
            "invalid_variants": len(errors),
        }
    return filtered, []


def validate_spec(
    spec_path: Path,
    max_start_distance_m: float,
    check_expert_support: bool,
    check_planner: bool,
    flip_isaac_y: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = load_yaml(spec_path)
    scene_yaml = Path(spec["scene"]["source_yaml"])
    scene = load_yaml(scene_yaml)
    starts = get_start_poses(scene)

    valid_tasks = []
    invalid_tasks = []
    seen_ids = set()
    for task in spec.get("tasks", []):
        errors = []
        task_id = task.get("task_id")
        if task_id in seen_ids:
            errors.append(f"duplicate task_id {task_id!r}")
        seen_ids.add(task_id)
        errors.extend(validate_task(task, scene, starts, max_start_distance_m, check_expert_support))
        if errors:
            invalid_tasks.append({"task": task, "errors": errors})
        else:
            valid_task = task
            if check_planner:
                valid_task, planner_errors = filter_planner_valid_variants(scene, scene_yaml, task, flip_isaac_y)
                errors.extend(planner_errors)
            if errors:
                invalid_tasks.append({"task": task, "errors": errors})
            else:
                valid_tasks.append(valid_task)
    return spec, valid_tasks, invalid_tasks


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.yaml")))
            paths.extend(sorted(input_path.rglob("*.yml")))
        else:
            paths.append(input_path)
    return sorted(set(path.resolve() for path in paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Task spec YAML file(s) or directories.")
    parser.add_argument("--max-start-distance-m", type=float, default=2.0)
    parser.add_argument(
        "--skip-expert-support-check",
        action="store_true",
        help="Only check scene geometry, even for task types without implemented expert publishers.",
    )
    parser.add_argument(
        "--write-valid-output-dir",
        type=Path,
        default=None,
        help="If set, write filtered specs containing only valid tasks into this directory.",
    )
    parser.add_argument(
        "--check-planner",
        action="store_true",
        help="Run Hybrid A* for each trajectory variant and filter variants that cannot be planned.",
    )
    parser.add_argument(
        "--no-flip-isaac-y",
        action="store_true",
        help="Validate planner geometry without flipping Isaac's Y axis.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Return success even when invalid tasks are found.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every invalid task and reason.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_paths = expand_inputs(args.inputs)
    if not spec_paths:
        raise SystemExit("No task spec YAML files found.")

    total_valid = 0
    total_invalid = 0
    failed_specs = 0
    for spec_path in spec_paths:
        spec, valid_tasks, invalid_tasks = validate_spec(
            spec_path,
            max_start_distance_m=args.max_start_distance_m,
            check_expert_support=not args.skip_expert_support_check,
            check_planner=args.check_planner,
            flip_isaac_y=not args.no_flip_isaac_y,
        )
        total_valid += len(valid_tasks)
        total_invalid += len(invalid_tasks)
        if invalid_tasks:
            failed_specs += 1
        print(f"{spec_path}: {len(valid_tasks)} valid, {len(invalid_tasks)} invalid")
        if args.verbose:
            for invalid in invalid_tasks:
                task = invalid["task"]
                print(f"  - {task.get('task_id', '<missing id>')}: {'; '.join(invalid['errors'])}")

        if args.write_valid_output_dir is not None:
            filtered = dict(spec)
            filtered["tasks"] = valid_tasks
            filtered["validation_summary"] = {
                "source_spec": spec_path.as_posix(),
                "valid_tasks": len(valid_tasks),
                "invalid_tasks": len(invalid_tasks),
                "max_start_distance_m": args.max_start_distance_m,
                "expert_support_checked": not args.skip_expert_support_check,
                "planner_checked": args.check_planner,
                "flip_isaac_y": not args.no_flip_isaac_y,
            }
            write_yaml(args.write_valid_output_dir / spec_path.name, filtered)

    print(f"Validated {len(spec_paths)} specs: {total_valid} valid tasks, {total_invalid} invalid tasks.")
    if total_invalid and not args.allow_invalid:
        raise SystemExit(f"{failed_specs} spec file(s) contain invalid tasks.")


if __name__ == "__main__":
    main()

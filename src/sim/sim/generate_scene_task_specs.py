#!/usr/bin/env python3
"""Generate rollout task specs from Isaac scene YAML files."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


DEFAULT_COLLECTION = {
    "base_dir": "C:/Users/miahv/Documents/Capstone_Project/sim_datasets/generated",
    "duration_s": 20,
    "camera_topic": "/vla/cam",
    "odom_topic": "/sim_odom",
    "cmd_vel_topic": "/cmd_vel",
    "action_chunk_topic": "/vla/action_chunk",
    "sample_frequency_hz": 3.0,
}

DEFAULT_EXPERT = {
    "publisher": "fenceline_action_chunk_publisher",
    "controller": "sim_waypoint_tracker",
    "preferred_offset_m": 0.8,
    "waypoint_spacing_m": 0.18,
    "publish_rate_hz": 3.0,
}

RECOVERY_JITTER_RANGES = {
    "recovery_left_offset": {
        "x_m": (0.0, 0.0),
        "y_m": (0.10, 0.25),
        "yaw_rad": (0.20, 0.50),
    },
    "recovery_right_offset": {
        "x_m": (0.0, 0.0),
        "y_m": (-0.25, -0.10),
        "yaw_rad": (-0.50, -0.20),
    },
    "recovery_wrong_heading": {
        "x_m": (0.0, 0.0),
        "y_m": (0.0, 0.0),
        "yaw_rad": (0.60, 1.00),
    },
    "recovery_too_close": {
        "x_m": (0.0, 0.0),
        "y_m": (-0.50, -0.25),
        "yaw_rad": (0.10, 0.35),
    },
    "recovery_too_far": {
        "x_m": (0.0, 0.0),
        "y_m": (0.30, 0.70),
        "yaw_rad": (-0.35, 0.10),
    },
    "recovery_bad_approach_heading": {
        "x_m": (-0.50, -0.15),
        "y_m": (0.0, 0.0),
        "yaw_rad": (0.80, 1.40),
    },
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


def scene_id_from_path(path: Path, scene: dict[str, Any]) -> str:
    output_name = scene.get("output_name")
    if isinstance(output_name, str) and output_name:
        return output_name
    return path.stem


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


def normalize_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def xy(point: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> list[float]:
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0]


def nearest_point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    px, py = point
    sx, sy = start
    ex, ey = end
    vx = ex - sx
    vy = ey - sy
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-9:
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
    return "left" if cross >= 0.0 else "right"


def yaw_of_pose(pose: dict[str, Any]) -> float:
    return float(pose.get("yaw", 0.0))


def robot_relative_side_to_segment(start_pose: dict[str, Any], segment: dict[str, Any]) -> str:
    start_point = xy(start_pose["position"])
    nearest, _ = nearest_point_on_segment(start_point, xy(segment["start"]), xy(segment["end"]))
    dx = nearest[0] - start_point[0]
    dy = nearest[1] - start_point[1]
    yaw = yaw_of_pose(start_pose)
    y_robot = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return "left" if y_robot >= 0.0 else "right"


def heading_of(segment: dict[str, Any]) -> float:
    start = xy(segment["start"])
    end = xy(segment["end"])
    return math.atan2(end[1] - start[1], end[0] - start[0])


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def turn_direction(first: dict[str, Any], second: dict[str, Any]) -> str:
    delta = wrap_to_pi(heading_of(second) - heading_of(first))
    if delta > 0.15:
        return "left"
    if delta < -0.15:
        return "right"
    return "straight"


def infer_side_from_start(start_pose_name: str, default: str = "left") -> str:
    name = start_pose_name.lower()
    if "right" in name:
        return "right"
    if "left" in name:
        return "left"
    return default


def opposite_side(side: str) -> str:
    return "right" if side == "left" else "left"


def get_start_poses(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(scene.get("rover_poses"), dict):
        return scene["rover_poses"]
    if isinstance(scene.get("rover_pose"), dict):
        return {"scene_rover_pose": scene["rover_pose"]}
    return {}


def instruction_pack(primary: str, variants: list[str]) -> dict[str, Any]:
    return {
        "instruction": primary,
        "instruction_variants": variants,
    }


def validation(min_progress_m: float = 2.0, min_samples: int = 30) -> dict[str, Any]:
    return {
        "min_samples": min_samples,
        "require_action_chunk": True,
        "require_motion_m": min_progress_m,
    }


DEFAULT_PLANNER_SETTINGS = {
    "robot_radius_m": 0.32,
    "obstacle_padding_m": 0.08,
    "grid_resolution_m": 0.25,
    "yaw_resolution_deg": 15.0,
    "subgoal_yaw_tolerance_deg": 180.0,
    "step_size_m": 0.35,
    "min_turn_radius_m": 0.75,
    "goal_tolerance_m": 0.35,
    "planner_subgoal_spacing_m": 3.0,
    "planner_subgoal_lateral_search_m": 2.0,
    "planner_subgoal_longitudinal_search_m": 2.0,
    "planner_subgoal_search_step_m": 0.5,
    "planner_subgoal_max_candidates": 48,
    "planner_subgoal_min_clearance_m": 0.15,
    "planner_subgoal_vertex_margin_m": 0.5,
    "planner_subgoal_endpoint_margin_m": 0.5,
    "hybrid_astar_max_iterations": 20000,
    "allow_reverse": False,
    "fence_offset_cost_weight": 0.6,
    "fence_offset_cost_deadband_m": 0.15,
    "fence_offset_cost_max_error_m": 2.0,
    "obstacle_clearance_cost_weight": 0.5,
    "obstacle_clearance_cost_distance_m": 0.4,
    "quality_max_nudged_subgoals": 8,
    "quality_max_nudge_m": 2.0,
    "quality_max_mean_nudge_m": 1.25,
    "quality_max_path_length_ratio": 4.0,
}

DEFAULT_SPEED_PROFILE = {
    "max_speed_mps": 0.3,
    "max_yaw_rate_radps": 0.45,
    "max_accel_mps2": 0.25,
    "max_decel_mps2": 0.35,
    "max_angular_accel_radps2": 0.6,
    "min_profile_speed_mps": 0.03,
    "stop_at_end": True,
}

TERMINAL_TASK_TYPES = {"hold_position", "stop_at_gap", "stop_at_landmark"}
AMBIGUOUS_TASK_TYPES = {"pass_through_gap", "switch_sides"}
EXCLUDED_TASK_TYPES = {"follow_and_turn"}
TURN_TASK_TYPES = {"follow_fence_sequence"}
FOLLOWING_TASK_TYPES = {
    "follow_fence",
    "follow_fence_sequence",
    "follow_road",
    "follow_shed_side",
}


def unique_tags(*tag_groups: list[str] | tuple[str, ...] | None) -> list[str]:
    tags: list[str] = []
    for group in tag_groups:
        if not group:
            continue
        for tag in group:
            normalized = normalize_name(str(tag))
            if normalized and normalized not in tags:
                tags.append(normalized)
    return tags


def default_task_category(task_type: str) -> str:
    if task_type in TERMINAL_TASK_TYPES:
        return "terminal"
    if task_type in AMBIGUOUS_TASK_TYPES:
        return "ambiguous"
    return "normal"


def default_task_tags(task_type: str) -> list[str]:
    tags = [task_type]
    if task_type in TERMINAL_TASK_TYPES:
        tags.append("terminal")
    if task_type in AMBIGUOUS_TASK_TYPES:
        tags.append("ambiguous")
    if task_type in TURN_TASK_TYPES:
        tags.extend(["turning", "curved_reference"])
    if task_type in {"follow_fence", "follow_road", "follow_shed_side"}:
        tags.append("path_following")
    if task_type in {"approach_target", "pass_through_gap", "switch_sides"}:
        tags.append("goal_directed")
    return unique_tags(tags)


def terminal_speed_profile() -> dict[str, Any]:
    return {
        "max_speed_mps": 0.22,
        "max_yaw_rate_radps": 0.35,
        "max_decel_mps2": 0.2,
        "min_profile_speed_mps": 0.02,
        "stop_at_end": True,
    }


def variant_settings(
    variant_id: str,
    variant_type: str,
    preferred_offset_m: float,
    planner_overrides: dict[str, Any] | None = None,
    speed_overrides: dict[str, Any] | None = None,
    start_pose_delta: dict[str, float] | None = None,
    recovery_case: str | None = None,
    data_category: str = "normal",
    scenario_tags: list[str] | None = None,
) -> dict[str, Any]:
    planner_settings = dict(DEFAULT_PLANNER_SETTINGS)
    if planner_overrides:
        planner_settings.update(planner_overrides)
    speed_profile = dict(DEFAULT_SPEED_PROFILE)
    if speed_overrides:
        speed_profile.update(speed_overrides)
    return {
        "variant_id": variant_id,
        "variant_type": variant_type,
        "data_category": data_category,
        "scenario_tags": unique_tags(scenario_tags),
        "recovery_case": recovery_case,
        "preferred_offset_m": preferred_offset_m,
        "start_pose_delta": start_pose_delta or {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        "planner_settings": planner_settings,
        "speed_profile": speed_profile,
    }


def sample_recovery_jitter_delta(variant_id: str, rng: random.Random) -> dict[str, float] | None:
    ranges = RECOVERY_JITTER_RANGES.get(variant_id)
    if ranges is None:
        return None

    delta = {}
    for key in ("x_m", "y_m", "yaw_rad"):
        low, high = ranges[key]
        delta[key] = float(low if abs(high - low) <= 1e-12 else rng.uniform(low, high))
    return delta


def add_recovery_jitter_variants_to_task(
    task: dict[str, Any],
    jitter_count: int,
    jitter_seed: int,
) -> None:
    if jitter_count <= 0:
        return

    variants = task.get("trajectory_variants")
    if not isinstance(variants, list):
        return

    task_id = str(task.get("task_id", "task"))
    jittered_variants = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        variant_id = str(variant.get("variant_id", ""))
        if variant_id not in RECOVERY_JITTER_RANGES:
            continue
        for index in range(1, jitter_count + 1):
            rng = random.Random(f"{jitter_seed}:{task_id}:{variant_id}:{index}")
            delta = sample_recovery_jitter_delta(variant_id, rng)
            if delta is None:
                continue
            jittered = dict(variant)
            jittered["variant_id"] = f"{variant_id}_jitter_{index:02d}"
            jittered["start_pose_delta"] = delta
            jittered["source_variant_id"] = variant_id
            jittered["jitter_index"] = index
            jittered["scenario_tags"] = unique_tags(
                variant.get("scenario_tags", []),
                ["recovery_jitter", "sampled_start_pose"],
            )
            jittered_variants.append(jittered)

    variants.extend(jittered_variants)


def add_recovery_jitter_variants(
    tasks: list[dict[str, Any]],
    jitter_count: int,
    jitter_seed: int,
) -> list[dict[str, Any]]:
    for task in tasks:
        add_recovery_jitter_variants_to_task(task, jitter_count, jitter_seed)
    return tasks


def trajectory_variants_for_task(
    task_type: str,
    task_category: str | None = None,
    task_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    category = task_category or default_task_category(task_type)
    base_tags = unique_tags(task_tags, default_task_tags(task_type))
    variants = [
        variant_settings(
            "nominal",
            "nominal",
            0.8,
            data_category=category,
            scenario_tags=unique_tags(base_tags, ["nominal"]),
        ),
        variant_settings(
            "cautious_wide_clearance",
            "clearance",
            1.0,
            planner_overrides={"obstacle_padding_m": 0.15, "min_turn_radius_m": 0.85},
            speed_overrides={"max_speed_mps": 0.25, "max_yaw_rate_radps": 0.35},
            data_category="ambiguous",
            scenario_tags=unique_tags(base_tags, ["ambiguous", "wide_clearance", "alternate_offset"]),
        ),
        variant_settings(
            "normal_tight_clearance",
            "clearance",
            0.65,
            planner_overrides={"obstacle_padding_m": 0.04},
            data_category="ambiguous",
            scenario_tags=unique_tags(base_tags, ["ambiguous", "tight_clearance", "alternate_offset"]),
        ),
    ]
    if task_type == "hold_position":
        return variants[:1]

    if task_type in TERMINAL_TASK_TYPES:
        variants.extend([
            variant_settings(
                "terminal_slow_approach",
                "terminal",
                0.8,
                speed_overrides=terminal_speed_profile(),
                data_category="terminal",
                scenario_tags=unique_tags(base_tags, ["terminal", "slow_approach", "precise_stop"]),
            ),
            variant_settings(
                "terminal_cautious_offset",
                "terminal",
                1.0,
                planner_overrides={"obstacle_padding_m": 0.12, "min_turn_radius_m": 0.85},
                speed_overrides=terminal_speed_profile(),
                data_category="terminal",
                scenario_tags=unique_tags(base_tags, ["terminal", "cautious_stop", "alternate_offset"]),
            ),
        ])

    variants.extend([
        variant_settings(
            "recovery_left_offset",
            "recovery",
            0.8,
            start_pose_delta={"x_m": 0.0, "y_m": 0.15, "yaw_rad": 0.35},
            recovery_case="lost_target_left",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "lateral_offset", "wrong_heading"]),
        ),
        variant_settings(
            "recovery_right_offset",
            "recovery",
            0.8,
            start_pose_delta={"x_m": 0.0, "y_m": -0.15, "yaw_rad": -0.35},
            recovery_case="lost_target_right",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "lateral_offset", "wrong_heading"]),
        ),
        variant_settings(
            "recovery_wrong_heading",
            "recovery",
            0.8,
            speed_overrides={"max_speed_mps": 0.25, "max_yaw_rate_radps": 0.35},
            start_pose_delta={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.85},
            recovery_case="wrong_heading",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "wrong_heading"]),
        ),
        variant_settings(
            "recovery_too_close",
            "recovery",
            0.8,
            speed_overrides={"max_speed_mps": 0.22, "max_yaw_rate_radps": 0.35},
            start_pose_delta={"x_m": 0.0, "y_m": -0.35, "yaw_rad": 0.2},
            recovery_case="start_too_close",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "start_too_close"]),
        ),
        variant_settings(
            "recovery_too_far",
            "recovery",
            0.8,
            speed_overrides={"max_speed_mps": 0.25, "max_yaw_rate_radps": 0.35},
            start_pose_delta={"x_m": 0.0, "y_m": 0.45, "yaw_rad": -0.2},
            recovery_case="start_too_far",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "start_too_far"]),
        ),
        variant_settings(
            "recovery_bad_approach_heading",
            "recovery",
            0.8,
            speed_overrides={"max_speed_mps": 0.2, "max_yaw_rate_radps": 0.4},
            start_pose_delta={"x_m": -0.25, "y_m": 0.0, "yaw_rad": 1.2},
            recovery_case="bad_approach_heading",
            data_category="recovery",
            scenario_tags=unique_tags(base_tags, ["recovery", "bad_approach", "wrong_heading"]),
        ),
    ])
    if task_type in {"pass_through_gap", "switch_sides"}:
        for variant in variants:
            variant["planner_settings"].update({
                "planner_subgoal_spacing_m": 4.0,
                "planner_subgoal_longitudinal_search_m": 2.5,
                "min_turn_radius_m": min(float(variant["planner_settings"].get("min_turn_radius_m", 0.75)), 0.5),
                "allow_reverse": True,
            })
    return variants


def make_task(task_id: str, task_type: str, **fields: Any) -> dict[str, Any]:
    data_category = fields.pop("data_category", default_task_category(task_type))
    scenario_tags = unique_tags(default_task_tags(task_type), fields.pop("scenario_tags", None))
    task = {
        "task_id": normalize_name(task_id),
        "task_type": task_type,
        "data_category": data_category,
        "scenario_tags": scenario_tags,
    }
    task.update(fields)
    task.setdefault("trajectory_variants", trajectory_variants_for_task(task_type, data_category, scenario_tags))
    return task


def add_domain_tags(tasks: list[dict[str, Any]], *domain_tags: str) -> list[dict[str, Any]]:
    for task in tasks:
        task["scenario_tags"] = unique_tags(task.get("scenario_tags", []), list(domain_tags))
        for variant in task.get("trajectory_variants", []):
            variant["scenario_tags"] = unique_tags(variant.get("scenario_tags", []), list(domain_tags))
    return tasks


def connected_segments(
    segments: list[dict[str, Any]],
    max_endpoint_gap_m: float = 0.05,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for i, first in enumerate(segments):
        first_end = xy(first["end"])
        for j, second in enumerate(segments):
            if i == j:
                continue
            if distance(first_end, xy(second["start"])) <= max_endpoint_gap_m:
                pairs.append((first, second))
    return pairs


def collinear_gaps(segments: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], list[float]]]:
    gaps = []
    for i, first in enumerate(segments):
        first_end = xy(first["end"])
        first_heading = heading_of(first)
        for j, second in enumerate(segments):
            if i == j:
                continue
            second_start = xy(second["start"])
            gap_distance = distance(first_end, second_start)
            if gap_distance < 0.4:
                continue
            heading_delta = abs(wrap_to_pi(heading_of(second) - first_heading))
            connection_heading = math.atan2(second_start[1] - first_end[1], second_start[0] - first_end[0])
            connection_delta = abs(wrap_to_pi(connection_heading - first_heading))
            if heading_delta < 0.2 and connection_delta < 0.2:
                gaps.append((first, second, midpoint(first_end, second_start)))
    return gaps


def endpoint_key(point: list[float] | tuple[float, ...], precision: int = 3) -> tuple[float, float]:
    x, y = xy(point)
    return round(x, precision), round(y, precision)


def ordered_connected_chain(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    remaining = {segment["name"]: segment for segment in segments}
    chain = [segments[0]]
    remaining.pop(segments[0]["name"], None)
    while remaining:
        end_key = endpoint_key(chain[-1]["end"])
        next_name = None
        for name, segment in remaining.items():
            if endpoint_key(segment["start"]) == end_key:
                next_name = name
                break
        if next_name is None:
            break
        chain.append(remaining.pop(next_name))
    return chain


def is_closed_chain(chain: list[dict[str, Any]]) -> bool:
    return len(chain) >= 3 and endpoint_key(chain[0]["start"]) == endpoint_key(chain[-1]["end"])


def rotate_chain(chain: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    if not chain:
        return []
    start_index = int(start_index) % len(chain)
    return chain[start_index:] + chain[:start_index]


def start_region_label(start_name: str) -> str:
    name = start_name.lower()
    if "inside" in name:
        return "inside"
    if "outside" in name:
        return "outside"
    if "between" in name:
        return "between"
    if "gate" in name:
        return "near_gate"
    return "unknown"


def add_sequence_tasks(tasks: list[dict[str, Any]], fences: list[dict[str, Any]], starts: dict[str, dict[str, Any]]) -> None:
    chain = ordered_connected_chain(fences)
    if len(chain) < 3:
        return
    scene_is_closed = is_closed_chain(chain)
    task_kind = "perimeter" if scene_is_closed else "multi_turn"
    for start_name in starts:
        start_pose = starts[start_name]
        start_point = xy(start_pose["position"])
        start_index, start_distance = min(
            enumerate(chain),
            key=lambda item: distance_to_segment(start_point, item[1]),
        )
        if start_distance > 2.0:
            continue
        task_chain = rotate_chain(chain, start_index) if scene_is_closed else chain[start_index:]
        if len(task_chain) < 3:
            continue
        target_names = [segment["name"] for segment in task_chain]
        first_segment = task_chain[0]
        path_side = side_of_segment(start_point, first_segment)
        follow_side = robot_relative_side_to_segment(start_pose, first_segment)
        if scene_is_closed:
            primary = "Follow the fence perimeter around the enclosure."
            variants = [
                "Follow the boundary of the fenced area.",
                "Continue around the fenceline perimeter.",
                "Track the fence line around the enclosure.",
            ]
        else:
            primary = "Follow the fenceline around the boundary."
            variants = [
                "Continue along the connected fence line.",
                "Keep following the fenceline through the bends.",
                "Track the fence around the next sides.",
            ]
        tasks.append(make_task(
            f"follow_{task_kind}_{task_chain[0]['name']}_to_{task_chain[-1]['name']}_from_{start_name}",
            "follow_fence_sequence",
            **instruction_pack(
                primary,
                variants,
            ),
            start_pose=start_name,
            target_fences=target_names,
            follow_side=follow_side,
            path_side=path_side,
            travel_direction="forward",
            sequence_type=task_kind,
            success_condition={
                "type": "reach_path_end",
                "target_point": task_chain[-1]["end"],
                "min_progress_m": max(
                    3.0,
                    sum(distance(xy(fence["start"]), xy(fence["end"])) for fence in task_chain) * 0.5,
                ),
            },
            validation=validation(),
        ))


def generate_fenceline_tasks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    fences = scene.get("fences") or []
    starts = get_start_poses(scene)
    tasks: list[dict[str, Any]] = []
    chain = ordered_connected_chain(fences)
    prefer_sequence_tasks = len(chain) >= 3

    add_sequence_tasks(tasks, fences, starts)

    for start_name in starts:
        start_pose = starts[start_name]
        start_point = xy(start_pose["position"])
        for fence in fences:
            if prefer_sequence_tasks:
                continue
            if distance_to_segment(start_point, fence) > 2.0:
                continue
            path_side = side_of_segment(start_point, fence)
            inferred_side = robot_relative_side_to_segment(start_pose, fence)
            fence_name = fence["name"]
            tasks.append(make_task(
                f"follow_{fence_name}_{inferred_side}_from_{start_name}",
                "follow_fence",
                **instruction_pack(
                    f"Follow the fence on your {inferred_side}.",
                    [
                        f"Drive forward while keeping the fence on your {inferred_side}.",
                        f"Track the fence line ahead on the {inferred_side} side.",
                        f"Continue along the {inferred_side} side of the fence.",
                    ],
                ),
                start_pose=start_name,
                target_fence=fence_name,
                follow_side=inferred_side,
                path_side=path_side,
                travel_direction="forward",
                success_condition={
                    "type": "reach_path_end",
                    "min_progress_m": max(1.0, distance(xy(fence["start"]), xy(fence["end"])) - 1.0),
                },
                validation=validation(),
            ))

            tasks.append(make_task(
                f"stop_at_end_of_{fence_name}_from_{start_name}",
                "stop_at_landmark",
                **instruction_pack(
                    "Stop when you reach the end of the fence.",
                    [
                        "Drive along the fence and stop at its end.",
                        "Follow the fence until it ends, then hold position.",
                        "Stop beside the final fence post.",
                    ],
                ),
                start_pose=start_name,
                target_fence=fence_name,
                landmark={"type": "fence_end", "point": fence["end"]},
                follow_side=inferred_side,
                path_side=path_side,
                travel_direction="forward",
                success_condition={
                    "type": "stop_near_point",
                    "target_point": fence["end"],
                    "tolerance_m": 0.5,
                    "max_speed_mps": 0.05,
                },
                validation=validation(),
            ))

        tasks.append(make_task(
            f"hold_position_from_{start_name}",
            "hold_position",
            **instruction_pack(
                "Hold position.",
                [
                    "Stop here.",
                    "Wait in place.",
                    "Stay where you are.",
                ],
            ),
            start_pose=start_name,
            success_condition={
                "type": "remain_near_start",
                "tolerance_m": 0.25,
                "max_speed_mps": 0.05,
            },
            validation=validation(min_progress_m=0.0, min_samples=5),
        ))

    for first, second in connected_segments(fences):
        direction = turn_direction(first, second)
        if direction == "straight":
            continue
        for start_name in starts:
            start_pose = starts[start_name]
            start_point = xy(start_pose["position"])
            if distance_to_segment(start_point, first) > 2.0:
                continue
            path_side = side_of_segment(start_point, first)
            follow_side = robot_relative_side_to_segment(start_pose, first)
            tasks.append(make_task(
                f"follow_{first['name']}_around_{direction}_turn_to_{second['name']}_from_{start_name}",
                "follow_and_turn",
                **instruction_pack(
                    f"Follow the fence around the {direction} corner.",
                    [
                        f"Turn {direction} with the fence line.",
                        "Keep following the fence as it bends.",
                        f"Track the fence through the {direction} turn.",
                    ],
                ),
                start_pose=start_name,
                target_fences=[first["name"], second["name"]],
                follow_side=follow_side,
                path_side=path_side,
                travel_direction="forward",
                turn_direction=direction,
                success_condition={
                    "type": "reach_path_end",
                    "target_point": second["end"],
                    "min_progress_m": 3.0,
                },
                validation=validation(),
            ))

    for before, after, center in collinear_gaps(fences):
        for start_name in starts:
            start_pose = starts[start_name]
            start_point = xy(start_pose["position"])
            if distance_to_segment(start_point, before) > 2.0:
                continue
            path_side = side_of_segment(start_point, before)
            side = robot_relative_side_to_segment(start_pose, before)
            region = start_region_label(start_name)
            pass_primary = "Drive through the gate." if "gate" in start_name.lower() else "Drive through the opening in the fence."
            pass_variants = [
                "Pass between the two fence sections.",
                "Go through the gap and continue forward.",
                "Navigate through the fence gap ahead.",
            ]
            if region == "outside":
                pass_variants.insert(0, "Enter the fenced area through the gate.")
            elif region == "inside":
                pass_variants.insert(0, "Exit the fenced area through the gate.")
            tasks.append(make_task(
                f"pass_gap_between_{before['name']}_and_{after['name']}_from_{start_name}",
                "pass_through_gap",
                **instruction_pack(
                    pass_primary,
                    pass_variants,
                ),
                start_pose=start_name,
                target_gap={
                    "before_fence": before["name"],
                    "after_fence": after["name"],
                    "approximate_center": center,
                },
                follow_side=side,
                path_side=path_side,
                gate_intent="enter" if region == "outside" else "exit" if region == "inside" else "pass_through",
                travel_direction="forward",
                success_condition={
                    "type": "pass_point",
                    "target_point": center,
                    "min_progress_m": 4.0,
                },
                validation=validation(),
            ))
            tasks.append(make_task(
                f"stop_at_gap_between_{before['name']}_and_{after['name']}_from_{start_name}",
                "stop_at_gap",
                **instruction_pack(
                    "Stop at the opening in the fence.",
                    [
                        "Drive up to the fence gap and stop.",
                        "Approach the opening between the fence sections and stop there.",
                        "Move forward to the gap in the fence, then hold position.",
                    ],
                ),
                start_pose=start_name,
                target_gap={
                    "before_fence": before["name"],
                    "after_fence": after["name"],
                    "approximate_center": center,
                },
                follow_side=side,
                path_side=path_side,
                travel_direction="forward",
                success_condition={
                    "type": "stop_near_point",
                    "target_point": center,
                    "tolerance_m": 0.5,
                    "max_speed_mps": 0.05,
                },
                validation=validation(),
            ))
            tasks.append(make_task(
                f"cross_over_gap_between_{before['name']}_and_{after['name']}_from_{start_name}",
                "switch_sides",
                **instruction_pack(
                    f"Move to the other side of the fence through the gap, then follow it on your {opposite_side(side)}.",
                    [
                        f"Cross through the opening, then keep the fence on your {opposite_side(side)}.",
                        "Use the fence gap to move to the other side.",
                        f"Pass through the gap and continue along the {opposite_side(side)} side of the fence.",
                    ],
                ),
                start_pose=start_name,
                target_gap={
                    "before_fence": before["name"],
                    "after_fence": after["name"],
                    "approximate_center": center,
                },
                entry_side=side,
                exit_side=opposite_side(side),
                entry_path_side=path_side,
                exit_path_side=opposite_side(path_side),
                travel_direction="forward",
                success_condition={
                    "type": "pass_point_and_continue",
                    "target_point": center,
                    "min_progress_m": 5.0,
                },
                validation=validation(),
            ))

    return add_domain_tags(tasks, "fenceline")


def generate_road_tasks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    roads = scene.get("roads") or []
    starts = get_start_poses(scene)
    tasks: list[dict[str, Any]] = []

    for start_name in starts:
        for road in roads:
            road_name = road["name"]
            tasks.append(make_task(
                f"follow_{road_name}_from_{start_name}",
                "follow_road",
                **instruction_pack(
                    "Follow the road.",
                    [
                        "Stay on the road ahead.",
                        "Continue along the path.",
                        "Drive along the track.",
                    ],
                ),
                start_pose=start_name,
                target_road=road_name,
                travel_direction="forward",
                success_condition={
                    "type": "reach_path_end",
                    "target_point": road["end"],
                    "min_progress_m": max(1.0, distance(xy(road["start"]), xy(road["end"])) - 1.0),
                },
                validation=validation(),
            ))
            tasks.append(make_task(
                f"approach_start_of_{road_name}_from_{start_name}",
                "approach_target",
                **instruction_pack(
                    "Drive to the start of the road.",
                    [
                        "Approach the beginning of the track.",
                        "Move toward the road entrance.",
                        "Drive up to where the road starts.",
                    ],
                ),
                start_pose=start_name,
                target_road=road_name,
                target_point=road["start"],
                success_condition={
                    "type": "reach_point",
                    "target_point": road["start"],
                    "tolerance_m": 0.5,
                },
                validation=validation(),
            ))
            tasks.append(make_task(
                f"stop_at_end_of_{road_name}_from_{start_name}",
                "stop_at_landmark",
                **instruction_pack(
                    "Stop at the end of the road.",
                    [
                        "Follow the road and stop at the end.",
                        "Drive along the track, then hold position at the end.",
                        "Stop when the road ends.",
                    ],
                ),
                start_pose=start_name,
                target_road=road_name,
                landmark={"type": "road_end", "point": road["end"]},
                success_condition={
                    "type": "stop_near_point",
                    "target_point": road["end"],
                    "tolerance_m": 0.5,
                    "max_speed_mps": 0.05,
                },
                validation=validation(),
            ))

        tasks.append(make_task(
            f"hold_position_from_{start_name}",
            "hold_position",
            **instruction_pack("Hold position.", ["Stop here.", "Wait in place.", "Stay where you are."]),
            start_pose=start_name,
            success_condition={"type": "remain_near_start", "tolerance_m": 0.25, "max_speed_mps": 0.05},
            validation=validation(min_progress_m=0.0, min_samples=5),
        ))

    for first, second in connected_segments(roads, max_endpoint_gap_m=5.0):
        direction = turn_direction(first, second)
        if direction == "straight":
            continue
        for start_name in starts:
            tasks.append(make_task(
                f"follow_road_around_{direction}_turn_from_{start_name}",
                "follow_and_turn",
                **instruction_pack(
                    f"Turn with the road to the {direction}.",
                    [
                        f"Follow the road as it bends {direction}.",
                        f"Stay on the track through the {direction} turn.",
                        f"Continue along the road around the {direction} curve.",
                    ],
                ),
                start_pose=start_name,
                target_roads=[first["name"], second["name"]],
                turn_direction=direction,
                travel_direction="forward",
                success_condition={"type": "reach_path_end", "target_point": second["end"], "min_progress_m": 3.0},
                validation=validation(),
            ))

    return add_domain_tags(tasks, "road")


def shed_side_from_start(start_pose_name: str) -> str:
    name = start_pose_name.lower()
    for side in ["north", "south", "east", "west"]:
        if side in name:
            return side
    return "nearest"


def generate_shedline_tasks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    starts = get_start_poses(scene)
    shed = scene.get("shed") or {}
    shed_position = shed.get("position", [0.0, 0.0, 0.0])
    tasks: list[dict[str, Any]] = []

    for start_name in starts:
        side = shed_side_from_start(start_name)
        tasks.append(make_task(
            f"follow_shed_{side}_side_from_{start_name}",
            "follow_shed_side",
            **instruction_pack(
                f"Follow the {side} side of the shed." if side != "nearest" else "Follow the side of the shed.",
                [
                    "Drive along the shed wall.",
                    "Keep the shed beside you and continue forward.",
                    "Track the side of the shed.",
                ],
            ),
            start_pose=start_name,
            target_shed="shed",
            shed_side=side,
            travel_direction="forward",
            success_condition={"type": "reach_path_end", "min_progress_m": 2.0},
            validation=validation(),
        ))
        tasks.append(make_task(
            f"approach_shed_from_{start_name}",
            "approach_target",
            **instruction_pack(
                "Drive toward the shed.",
                [
                    "Approach the shed.",
                    "Move closer to the shed.",
                    "Drive up to the shed.",
                ],
            ),
            start_pose=start_name,
            target_shed="shed",
            target_point=shed_position,
            success_condition={"type": "reach_point", "target_point": shed_position, "tolerance_m": 0.75},
            validation=validation(),
        ))
        tasks.append(make_task(
            f"stop_beside_shed_from_{start_name}",
            "stop_at_landmark",
            **instruction_pack(
                "Stop beside the shed.",
                [
                    "Drive along the shed and stop beside it.",
                    "Stop next to the shed wall.",
                    "Hold position beside the shed.",
                ],
            ),
            start_pose=start_name,
            target_shed="shed",
            landmark={"type": "shed_side", "side": side},
            success_condition={"type": "stop_near_shed", "max_speed_mps": 0.05},
            validation=validation(),
        ))
        tasks.append(make_task(
            f"hold_position_from_{start_name}",
            "hold_position",
            **instruction_pack("Hold position.", ["Stop here.", "Wait in place.", "Stay where you are."]),
            start_pose=start_name,
            success_condition={"type": "remain_near_start", "tolerance_m": 0.25, "max_speed_mps": 0.05},
            validation=validation(min_progress_m=0.0, min_samples=5),
        ))

    return add_domain_tags(tasks, "shedline")


def obstacle_tasks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _disabled_obstacle_tasks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Kept for later if we add explicit obstacle-language expert support."""
    obstacles = scene.get("obstacles")
    starts = get_start_poses(scene)
    if not isinstance(obstacles, list):
        return []

    tasks: list[dict[str, Any]] = []
    for start_name in starts:
        for index, obstacle in enumerate(obstacles):
            if not isinstance(obstacle, dict) or "position" not in obstacle:
                continue
            obstacle_type = str(obstacle.get("type", "obstacle"))
            obstacle_name = str(obstacle.get("name", f"{obstacle_type}_{index:02d}"))
            obstacle_id = normalize_name(f"{obstacle_type}_{obstacle_name}_{index:02d}")
            position = obstacle["position"]
            tasks.append(make_task(
                f"stop_before_{obstacle_id}_from_{start_name}",
                "stop_at_landmark",
                **instruction_pack(
                    f"Stop before the {obstacle_type.replace('_', ' ')}.",
                    [
                        f"Drive toward the {obstacle_type.replace('_', ' ')} and stop before it.",
                        f"Approach the {obstacle_type.replace('_', ' ')} without hitting it.",
                        f"Stop near the {obstacle_type.replace('_', ' ')}.",
                    ],
                ),
                start_pose=start_name,
                target_obstacle={"type": obstacle_type, "name": obstacle_name, "position": position},
                success_condition={
                    "type": "stop_near_point",
                    "target_point": position,
                    "tolerance_m": 0.7,
                    "max_speed_mps": 0.05,
                },
                validation=validation(),
            ))
            tasks.append(make_task(
                f"avoid_{obstacle_id}_from_{start_name}",
                "avoid_obstacle_while_following",
                **instruction_pack(
                    f"Go around the {obstacle_type.replace('_', ' ')} and continue forward.",
                    [
                        f"Avoid the {obstacle_type.replace('_', ' ')} while continuing ahead.",
                        f"Pass the {obstacle_type.replace('_', ' ')} safely and keep going.",
                        f"Steer around the {obstacle_type.replace('_', ' ')}.",
                    ],
                ),
                start_pose=start_name,
                target_obstacle={"type": obstacle_type, "name": obstacle_name, "position": position},
                success_condition={"type": "pass_obstacle", "target_point": position, "clearance_m": 0.4},
                validation=validation(),
            ))
    return tasks


def filter_tasks_for_family(tasks: list[dict[str, Any]], task_family: str) -> list[dict[str, Any]]:
    tasks = [task for task in tasks if task.get("task_type") not in EXCLUDED_TASK_TYPES]
    if task_family == "all_supported":
        return tasks
    if task_family == "following_only":
        return [task for task in tasks if task.get("task_type") in FOLLOWING_TASK_TYPES]
    raise ValueError(f"Unsupported task_family: {task_family!r}")


def generate_tasks(scene: dict[str, Any], task_family: str = "following_only") -> list[dict[str, Any]]:
    config_type = infer_config_type(scene)
    if config_type == "fenceline":
        tasks = generate_fenceline_tasks(scene)
    elif config_type == "road":
        tasks = generate_road_tasks(scene)
    elif config_type == "shedline":
        tasks = generate_shedline_tasks(scene)
    else:
        raise ValueError(f"Unsupported config_type: {config_type!r}")
    tasks.extend(obstacle_tasks(scene))
    return filter_tasks_for_family(dedupe_tasks(tasks), task_family)


def dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id in seen:
            continue
        seen.add(task_id)
        unique.append(task)
    return unique


def build_spec(
    scene_path: Path,
    scene: dict[str, Any],
    task_family: str = "following_only",
    recovery_jitter_count: int = 0,
    recovery_jitter_seed: int = 17,
) -> dict[str, Any]:
    scene_id = scene_id_from_path(scene_path, scene)
    config_type = infer_config_type(scene)
    generated_usd = None
    if scene.get("output_dir") and scene.get("output_name"):
        generated_usd = f"{scene['output_dir']}/{scene['output_name']}.usd"
    tasks = generate_tasks(scene, task_family)
    add_recovery_jitter_variants(tasks, recovery_jitter_count, recovery_jitter_seed)

    return {
        "spec_version": 0.1,
        "suite_id": f"{scene_id}_tasks",
        "scene": {
            "scene_id": scene_id,
            "config_type": config_type,
            "source_yaml": scene_path.as_posix(),
            "generated_layout_yaml": scene_path.as_posix() if "rover_pose" in scene else None,
            "generated_usd": generated_usd,
            "notes": "Generated from Isaac scene YAML.",
        },
        "collection": DEFAULT_COLLECTION.copy(),
        "expert": DEFAULT_EXPERT.copy(),
        "tasks": tasks,
    }


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
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Isaac scene YAML file(s) or directories containing YAML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/sim/config/generated_task_specs"),
        help="Directory where generated task spec YAML files will be written.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact task count summary after writing files.",
    )
    parser.add_argument(
        "--task-family",
        choices=("following_only", "all_supported"),
        default="following_only",
        help=(
            "Task set to generate. The default keeps only path-following tasks and "
            "excludes stop, hold, gap-pass, switch-side, and approach-target tasks."
        ),
    )
    parser.add_argument(
        "--recovery-jitter-count",
        type=int,
        default=0,
        help=(
            "Add this many sampled recovery start-pose variants per fixed recovery variant. "
            "Default 0 keeps the deterministic fixed variants only."
        ),
    )
    parser.add_argument(
        "--recovery-jitter-seed",
        type=int,
        default=17,
        help="Deterministic seed used when --recovery-jitter-count is greater than 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No YAML files found.")

    total_tasks = 0
    for scene_path in input_paths:
        scene = load_yaml(scene_path)
        if infer_config_type(scene) == "unknown":
            continue
        spec = build_spec(
            scene_path,
            scene,
            args.task_family,
            recovery_jitter_count=max(0, int(args.recovery_jitter_count)),
            recovery_jitter_seed=int(args.recovery_jitter_seed),
        )
        output_path = args.output_dir / f"{spec['scene']['scene_id']}_task_spec.yaml"
        write_yaml(output_path, spec)
        task_count = len(spec["tasks"])
        total_tasks += task_count
        print(f"Wrote {output_path} ({task_count} tasks)")

    if args.summary:
        print(f"Generated {total_tasks} tasks from {len(input_paths)} YAML files.")


if __name__ == "__main__":
    main()

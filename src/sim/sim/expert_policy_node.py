#!/usr/bin/env python3
"""Base ROS node for task-spec-driven expert trajectory publishers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from sim.expert_trajectory_utils import (
    build_distance_action_chunk,
    build_timed_action_chunk,
    fence_by_name,
    find_variant,
    find_task,
    get_start_pose,
    local_odom_to_world,
    load_yaml,
    odom_to_pose,
    orient_and_crop_path_from_start,
    orient_loop_path_from_start,
    path_length,
    point2,
    project_progress,
    project_progress_near,
    sample_path_pose,
    sample_distance_action_target,
    sample_timed_action_target,
    wrap_to_pi,
    world_to_robot,
    yaw_from_quaternion,
)
from sim.collision_map import CollisionMap
from sim.hybrid_astar import HybridAStarPlanner, Pose
from sim.trajectory_profile import TimedTrajectory, build_timed_trajectory, resample_path, shortcut_smooth


def shifted_subgoal_candidates(
    position: np.ndarray,
    yaw: float,
    max_lateral_m: float,
    max_longitudinal_m: float,
    step_m: float = 0.25,
) -> list[np.ndarray]:
    direction = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    normal = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    lateral_steps = int(np.floor(max_lateral_m / step_m))
    longitudinal_steps = int(np.floor(max_longitudinal_m / step_m))
    local_offsets = [(0.0, 0.0)]
    for lat_index in range(-lateral_steps, lateral_steps + 1):
        for lon_index in range(-longitudinal_steps, longitudinal_steps + 1):
            lateral = lat_index * step_m
            longitudinal = lon_index * step_m
            if abs(lateral) < 1e-9 and abs(longitudinal) < 1e-9:
                continue
            local_offsets.append((lateral, longitudinal))

    local_offsets.sort(key=lambda item: (abs(item[0]) + 1.5 * abs(item[1]), abs(item[0]), abs(item[1])))
    return [position + normal * lateral + direction * longitudinal for lateral, longitudinal in local_offsets]


def clear_reference_subgoals(
    collision_map: CollisionMap,
    subgoals: list[tuple[np.ndarray, float]],
    max_lateral_m: float,
    max_longitudinal_m: float,
) -> tuple[list[tuple[np.ndarray, float]], int]:
    cleared = []
    nudged_count = 0
    for position, yaw in subgoals:
        clear_position = None
        for candidate in shifted_subgoal_candidates(position, yaw, max_lateral_m, max_longitudinal_m):
            if not collision_map.is_collision(candidate):
                clear_position = candidate
                break
        if clear_position is None:
            return [], nudged_count
        if float(np.linalg.norm(clear_position - position)) > 1e-6:
            nudged_count += 1
        cleared.append((clear_position, yaw))
    return cleared, nudged_count


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / length_sq)
    t = min(1.0, max(0.0, t))
    nearest = start + t * segment
    return float(np.linalg.norm(point - nearest))


def signed_side_of_segment(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    direction = segment_end - segment_start
    relative = point - segment_start
    return float(direction[0]) * float(relative[1]) - float(direction[1]) * float(relative[0])


def candidate_preserves_side(
    original: np.ndarray,
    candidate: np.ndarray,
    side_constraint_segments: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    if not side_constraint_segments:
        return True
    segment_start, segment_end = min(
        side_constraint_segments,
        key=lambda segment: point_to_segment_distance(original, segment[0], segment[1]),
    )
    original_side = signed_side_of_segment(original, segment_start, segment_end)
    candidate_side = signed_side_of_segment(candidate, segment_start, segment_end)
    if abs(original_side) < 1e-6:
        return True
    return original_side * candidate_side > 0.0


def distance_to_nearest_segment(point: np.ndarray, segments: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not segments:
        return math.inf
    return min(point_to_segment_distance(point, start, end) for start, end in segments)


def path_min_distance_to_segments(path: list[np.ndarray], segments: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not path or not segments:
        return math.inf
    return min(distance_to_nearest_segment(point, segments) for point in path)


def side_constraint_segments_for_task(
    scene: dict,
    task: dict,
    flip_isaac_y: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    task_type = task.get("task_type")
    fence_names: list[str] = []
    if task_type in {"follow_fence", "stop_at_landmark"} and task.get("target_fence"):
        fence_names.append(str(task["target_fence"]))
    elif task_type in {"follow_and_turn", "follow_fence_sequence"}:
        fence_names.extend(str(name) for name in task.get("target_fences", []))
    elif task_type in {"pass_through_gap", "stop_at_gap", "switch_sides"}:
        gap = task.get("target_gap") or {}
        if gap.get("before_fence"):
            fence_names.append(str(gap["before_fence"]))
        if gap.get("after_fence"):
            fence_names.append(str(gap["after_fence"]))

    segments = []
    for name in fence_names:
        fence = fence_by_name(scene, name)
        segments.append((point2(fence["start"], flip_isaac_y), point2(fence["end"], flip_isaac_y)))
    if task_type == "follow_fence_sequence":
        fences = [fence_by_name(scene, str(name)) for name in task.get("target_fences", [])]
        for before, after in zip(fences[:-1], fences[1:]):
            before_start = point2(before["start"], flip_isaac_y)
            before_end = point2(before["end"], flip_isaac_y)
            after_start = point2(after["start"], flip_isaac_y)
            after_end = point2(after["end"], flip_isaac_y)
            gap_distance = float(np.linalg.norm(after_start - before_end))
            if gap_distance < 0.4:
                continue
            before_heading = math.atan2(float(before_end[1] - before_start[1]), float(before_end[0] - before_start[0]))
            after_heading = math.atan2(float(after_end[1] - after_start[1]), float(after_end[0] - after_start[0]))
            gap_heading = math.atan2(float(after_start[1] - before_end[1]), float(after_start[0] - before_end[0]))
            if (
                abs(math.atan2(math.sin(after_heading - before_heading), math.cos(after_heading - before_heading))) <= 0.2
                and abs(math.atan2(math.sin(gap_heading - before_heading), math.cos(gap_heading - before_heading))) <= 0.2
            ):
                segments.append((before_end, after_start))
    return segments


class ExpertPolicyNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.declare_parameter("task_spec", Parameter.Type.STRING)
        self.declare_parameter("task_id", Parameter.Type.STRING)
        self.declare_parameter("variant_id", "nominal")
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("action_chunk_topic", "/vla/action_chunk")
        self.declare_parameter("expert_cmd_vel_topic", "/expert/cmd_vel")
        self.declare_parameter("frame_debug_topic", "/expert/frame_debug")
        self.declare_parameter("runtime_planned_path_output", "")
        self.declare_parameter("waypoint_spacing_m", 0.18)
        self.declare_parameter("first_preview_m", 0.9)
        self.declare_parameter("expert_path_lookahead_m", 0.5)
        self.declare_parameter("expert_min_tracking_speed_mps", 0.16)
        self.declare_parameter("expert_heading_slowdown_rad", 0.8)
        self.declare_parameter("expert_tracking_max_yaw_rate_radps", 0.3)
        self.declare_parameter("path_progress_motion_slack_m", 0.6)
        self.declare_parameter("max_expert_tracking_error_m", 0.85)
        self.declare_parameter("max_target_lateral_error_m", 1.1)
        self.declare_parameter("max_tracking_target_distance_m", 1.25)
        self.declare_parameter("recovery_lookahead_m", 0.45)
        self.declare_parameter("recovery_speed_mps", 0.10)
        self.declare_parameter("future_time_offsets_s", [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])
        self.declare_parameter("publish_rate_hz", 3.0)
        self.declare_parameter("flip_isaac_y", False)
        self.declare_parameter("flip_scene_y", False)
        self.declare_parameter("flip_runtime_odom_y", False)
        self.declare_parameter("flip_runtime_odom_yaw", True)
        self.declare_parameter("use_hybrid_astar", True)
        self.declare_parameter("robot_radius_m", 0.32)
        self.declare_parameter("obstacle_padding_m", 0.08)
        self.declare_parameter("grid_resolution_m", 0.25)
        self.declare_parameter("yaw_resolution_deg", 15.0)
        self.declare_parameter("subgoal_yaw_tolerance_deg", 180.0)
        self.declare_parameter("step_size_m", 0.35)
        self.declare_parameter("min_turn_radius_m", 0.75)
        self.declare_parameter("goal_tolerance_m", 0.35)
        self.declare_parameter("planner_subgoal_spacing_m", 3.0)
        self.declare_parameter("planner_subgoal_lateral_search_m", 2.0)
        self.declare_parameter("planner_subgoal_longitudinal_search_m", 2.0)
        self.declare_parameter("planner_subgoal_search_step_m", 0.5)
        self.declare_parameter("planner_subgoal_max_candidates", 48)
        self.declare_parameter("planner_subgoal_min_clearance_m", 0.15)
        self.declare_parameter("planner_fence_min_clearance_m", 0.65)
        self.declare_parameter("planner_subgoal_vertex_margin_m", 0.5)
        self.declare_parameter("planner_subgoal_endpoint_margin_m", 0.5)
        self.declare_parameter("hybrid_astar_max_iterations", 20000)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("fence_offset_cost_weight", 0.6)
        self.declare_parameter("fence_offset_cost_deadband_m", 0.15)
        self.declare_parameter("fence_offset_cost_max_error_m", 2.0)
        self.declare_parameter("fence_min_clearance_cost_weight", 35.0)
        self.declare_parameter("obstacle_clearance_cost_weight", 0.5)
        self.declare_parameter("obstacle_clearance_cost_distance_m", 0.4)
        self.declare_parameter("max_speed_mps", 0.3)
        self.declare_parameter("expert_speed_limit_mps", 0.3)
        self.declare_parameter("max_yaw_rate_radps", 0.45)
        self.declare_parameter("max_accel_mps2", 0.25)
        self.declare_parameter("max_decel_mps2", 0.35)
        self.declare_parameter("max_angular_accel_radps2", 0.6)
        self.declare_parameter("min_profile_speed_mps", 0.03)
        self.declare_parameter("stop_at_end", True)

        self.task_spec_path = Path(str(self.get_parameter("task_spec").value))
        if not self.task_spec_path.exists():
            raise RuntimeError(f"task_spec parameter must point to an existing YAML file: {self.task_spec_path}")
        self.task_id = str(self.get_parameter("task_id").value)
        if not self.task_id:
            raise RuntimeError("task_id parameter is required.")

        self.task_spec = load_yaml(self.task_spec_path)
        self.scene_path = Path(self.task_spec["scene"]["source_yaml"])
        self.scene = load_yaml(self.scene_path)
        self.task = find_task(self.task_spec, self.task_id)
        self.variant_id = str(self.get_parameter("variant_id").value)
        self.variant = find_variant(self.task, self.variant_id)

        self.flip_scene_y = bool(self.get_parameter("flip_scene_y").value)
        self.flip_isaac_y = self.flip_scene_y
        self.flip_runtime_odom_y = bool(self.get_parameter("flip_runtime_odom_y").value)
        self.flip_runtime_odom_yaw = bool(self.get_parameter("flip_runtime_odom_yaw").value)
        self.world_start_position, self.world_start_yaw = get_start_pose(self.scene, self.task, self.flip_scene_y)
        self.waypoint_spacing_m = float(self.get_parameter("waypoint_spacing_m").value)
        self.first_preview_m = float(self.get_parameter("first_preview_m").value)
        self.expert_path_lookahead_m = max(0.25, float(self.get_parameter("expert_path_lookahead_m").value))
        self.expert_min_tracking_speed_mps = max(
            0.0,
            float(self.get_parameter("expert_min_tracking_speed_mps").value),
        )
        self.expert_heading_slowdown_rad = max(
            0.05,
            float(self.get_parameter("expert_heading_slowdown_rad").value),
        )
        self.expert_tracking_max_yaw_rate_radps = max(
            0.05,
            float(self.get_parameter("expert_tracking_max_yaw_rate_radps").value),
        )
        self.future_time_offsets_s = [
            float(value)
            for value in self.get_parameter("future_time_offsets_s").value
        ]
        self.path = self.prepare_reference_path(self.resolve_path())
        if len(self.path) < 2 or path_length(self.path) < 1e-6:
            raise RuntimeError(f"Task {self.task_id} resolved to an empty path.")
        self.use_hybrid_astar = bool(self.get_parameter("use_hybrid_astar").value)
        self.planned_path: Optional[list[np.ndarray]] = None
        self.trajectory: Optional[TimedTrajectory] = None
        self.planning_attempted = False

        self.current_position: Optional[np.ndarray] = None
        self.current_yaw: Optional[float] = None
        self.path_progress_m = 0.0
        self.path_progress_motion_slack_m = max(
            0.0,
            float(self.get_parameter("path_progress_motion_slack_m").value),
        )
        self.max_expert_tracking_error_m = max(
            0.0,
            float(self.get_parameter("max_expert_tracking_error_m").value),
        )
        self.max_target_lateral_error_m = max(
            0.25,
            float(self.get_parameter("max_target_lateral_error_m").value),
        )
        self.max_tracking_target_distance_m = max(
            0.5,
            float(self.get_parameter("max_tracking_target_distance_m").value),
        )
        self.recovery_lookahead_m = max(
            0.1,
            float(self.get_parameter("recovery_lookahead_m").value),
        )
        self.recovery_speed_mps = max(
            0.0,
            float(self.get_parameter("recovery_speed_mps").value),
        )
        self.path_progress_anchor_position: Optional[np.ndarray] = None
        self.path_progress_distance_budget_m = 0.0
        self.latest_tracking_error_m: Optional[float] = None
        self.latest_off_path = False
        self.latest_odom_frame_debug: dict[str, object] = {}
        self.seq_num = 1

        self.publisher = self.create_publisher(ActionChunk, str(self.get_parameter("action_chunk_topic").value), 10)
        self.frame_debug_publisher = self.create_publisher(String, str(self.get_parameter("frame_debug_topic").value), 10)
        runtime_planned_path_output = str(self.get_parameter("runtime_planned_path_output").value)
        self.runtime_planned_path_output = (
            Path(runtime_planned_path_output) if runtime_planned_path_output else None
        )
        self.cmd_vel_publisher = None
        expert_cmd_vel_topic = str(self.get_parameter("expert_cmd_vel_topic").value)
        if expert_cmd_vel_topic:
            self.cmd_vel_publisher = self.create_publisher(Twist, expert_cmd_vel_topic, 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_chunk)
        self.get_logger().info(
            f"{node_name} loaded task_id={self.task_id} variant_id={self.variant_id} "
            f"reference_path_length={path_length(self.path):.2f}m"
        )

    def resolve_path(self) -> list[np.ndarray]:
        raise NotImplementedError

    def prepare_reference_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        if (
            self.task.get("task_type") == "follow_fence_sequence"
            and self.task.get("sequence_type") == "perimeter"
        ):
            return orient_loop_path_from_start(
                path,
                self.world_start_position,
                self.world_start_yaw,
                allow_reverse=True,
            )
        if (
            self.task.get("task_type") == "follow_shed_side"
            and self.task.get("shed_side") == "perimeter"
        ):
            return orient_loop_path_from_start(
                path,
                self.world_start_position,
                self.world_start_yaw,
                allow_reverse=False,
            )
        if self.task.get("task_type") in {
            "follow_fence",
            "follow_fence_sequence",
            "follow_road",
            "follow_shed_side",
            "stop_at_landmark",
        }:
            return orient_and_crop_path_from_start(
                path,
                self.world_start_position,
                self.world_start_yaw,
                allow_reverse=True,
            )
        return path

    def odom_callback(self, msg: Odometry) -> None:
        raw_x = float(msg.pose.pose.position.x)
        raw_y = float(msg.pose.pose.position.y)
        raw_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        local_position, local_yaw = odom_to_pose(
            msg,
            self.flip_runtime_odom_y,
            self.flip_runtime_odom_yaw,
        )
        world_position, world_yaw = local_odom_to_world(
            local_position,
            local_yaw,
            self.world_start_position,
            self.world_start_yaw,
        )
        self.current_position = world_position
        self.current_yaw = world_yaw
        self.latest_odom_frame_debug = {
            "raw_odom_x": raw_x,
            "raw_odom_y": raw_y,
            "raw_odom_yaw": raw_yaw,
            "flip_isaac_y": self.flip_isaac_y,
            "flip_scene_y": self.flip_scene_y,
            "flip_runtime_odom_y": self.flip_runtime_odom_y,
            "flip_runtime_odom_yaw": self.flip_runtime_odom_yaw,
            "local_x_after_flip": float(local_position[0]),
            "local_y_after_flip": float(local_position[1]),
            "local_yaw_after_flip": float(local_yaw),
            "world_x": float(world_position[0]),
            "world_y": float(world_position[1]),
            "world_yaw": float(world_yaw),
        }

    def active_path(self) -> list[np.ndarray]:
        if self.use_hybrid_astar and self.planned_path is not None:
            return self.planned_path
        return self.path

    def active_trajectory(self) -> TimedTrajectory:
        if self.trajectory is None:
            self.trajectory = self.profile_path(self.active_path())
        return self.trajectory

    def profile_path(self, path: list[np.ndarray]) -> TimedTrajectory:
        speed_profile = self.variant.get("speed_profile", {})
        requested_max_speed_mps = float(speed_profile.get("max_speed_mps", self.get_parameter("max_speed_mps").value))
        expert_speed_limit_mps = float(self.get_parameter("expert_speed_limit_mps").value)
        max_speed_mps = min(requested_max_speed_mps, expert_speed_limit_mps)
        return build_timed_trajectory(
            path,
            max_speed_mps=max_speed_mps,
            max_yaw_rate_radps=float(speed_profile.get("max_yaw_rate_radps", self.get_parameter("max_yaw_rate_radps").value)),
            max_accel_mps2=float(speed_profile.get("max_accel_mps2", self.get_parameter("max_accel_mps2").value)),
            max_decel_mps2=float(speed_profile.get("max_decel_mps2", self.get_parameter("max_decel_mps2").value)),
            max_angular_accel_radps2=float(
                speed_profile.get("max_angular_accel_radps2", self.get_parameter("max_angular_accel_radps2").value)
            ),
            min_speed_mps=float(speed_profile.get("min_profile_speed_mps", self.get_parameter("min_profile_speed_mps").value)),
            stop_at_end=bool(speed_profile.get("stop_at_end", self.get_parameter("stop_at_end").value)),
        )

    def planner_setting(self, name: str, default_parameter: str):
        planner_settings = self.variant.get("planner_settings", {})
        return planner_settings.get(name, self.get_parameter(default_parameter).value)

    def reference_subgoals(self) -> list[tuple[np.ndarray, float]]:
        total_length = path_length(self.path)
        spacing = float(self.planner_setting("planner_subgoal_spacing_m", "planner_subgoal_spacing_m"))
        endpoint_margin = min(
            float(self.planner_setting("planner_subgoal_endpoint_margin_m", "planner_subgoal_endpoint_margin_m")),
            max(total_length * 0.25, 0.0),
        )
        vertex_margin = float(self.planner_setting("planner_subgoal_vertex_margin_m", "planner_subgoal_vertex_margin_m"))
        max_progress = max(total_length - endpoint_margin, 0.0)
        if spacing <= 0.0 or max_progress <= spacing:
            return [sample_path_pose(self.path, max_progress)]

        vertex_progress_values = []
        cumulative = 0.0
        for index in range(len(self.path) - 1):
            cumulative += float(np.linalg.norm(self.path[index + 1] - self.path[index]))
            if index < len(self.path) - 2:
                vertex_progress_values.append(cumulative)

        progress_values = list(np.arange(spacing, total_length, spacing))
        progress_values.append(max_progress)
        adjusted_progress_values = []
        for progress in progress_values:
            adjusted = min(float(progress), max_progress)
            for vertex_progress in vertex_progress_values:
                if abs(adjusted - vertex_progress) < vertex_margin:
                    adjusted = max(0.0, vertex_progress - vertex_margin)
                    break
            if not adjusted_progress_values or adjusted > adjusted_progress_values[-1] + 1e-6:
                adjusted_progress_values.append(adjusted)
        return [sample_path_pose(self.path, float(progress)) for progress in adjusted_progress_values]

    def build_planner(self, collision_map: CollisionMap) -> HybridAStarPlanner:
        return HybridAStarPlanner(
            collision_map,
            grid_resolution_m=float(self.planner_setting("grid_resolution_m", "grid_resolution_m")),
            yaw_resolution_rad=np.deg2rad(float(self.planner_setting("yaw_resolution_deg", "yaw_resolution_deg"))),
            step_size_m=float(self.planner_setting("step_size_m", "step_size_m")),
            min_turn_radius_m=float(self.planner_setting("min_turn_radius_m", "min_turn_radius_m")),
            goal_tolerance_m=float(self.planner_setting("goal_tolerance_m", "goal_tolerance_m")),
            yaw_tolerance_rad=np.deg2rad(
                float(self.planner_setting("subgoal_yaw_tolerance_deg", "subgoal_yaw_tolerance_deg"))
            ),
            max_iterations=int(self.planner_setting("hybrid_astar_max_iterations", "hybrid_astar_max_iterations")),
            allow_reverse=bool(self.planner_setting("allow_reverse", "allow_reverse")),
            point_cost_fn=self.combined_point_cost_fn(collision_map),
        )

    def combined_point_cost_fn(self, collision_map: CollisionMap) -> Callable[[np.ndarray], float] | None:
        cost_fns = [
            cost_fn
            for cost_fn in [
                self.fence_offset_cost_fn(),
                self.obstacle_clearance_cost_fn(collision_map),
            ]
            if cost_fn is not None
        ]
        if not cost_fns:
            return None

        def point_cost(point: np.ndarray) -> float:
            return sum(float(cost_fn(point)) for cost_fn in cost_fns)

        return point_cost

    def fence_offset_cost_fn(self) -> Callable[[np.ndarray], float] | None:
        weight = float(self.planner_setting("fence_offset_cost_weight", "fence_offset_cost_weight"))
        danger_weight = float(
            self.planner_setting("fence_min_clearance_cost_weight", "fence_min_clearance_cost_weight")
        )
        segments = side_constraint_segments_for_task(self.scene, self.task, self.flip_scene_y)
        if not segments:
            return None

        preferred_offset_m = float(self.variant.get("preferred_offset_m", 0.8))
        min_clearance_m = max(
            0.0,
            float(self.planner_setting("planner_fence_min_clearance_m", "planner_fence_min_clearance_m")),
        )
        deadband_m = max(0.0, float(self.planner_setting("fence_offset_cost_deadband_m", "fence_offset_cost_deadband_m")))
        max_error_m = max(
            deadband_m,
            float(self.planner_setting("fence_offset_cost_max_error_m", "fence_offset_cost_max_error_m")),
        )

        def point_cost(point: np.ndarray) -> float:
            distance_to_fence = distance_to_nearest_segment(point, segments)
            danger_cost = 0.0
            if min_clearance_m > 0.0 and distance_to_fence < min_clearance_m:
                danger_cost = danger_weight * (min_clearance_m - distance_to_fence + 1.0)
            if weight <= 0.0:
                return danger_cost
            offset_error = abs(distance_to_fence - preferred_offset_m)
            if offset_error <= deadband_m:
                return danger_cost
            return danger_cost + weight * min(offset_error - deadband_m, max_error_m)

        return point_cost

    def obstacle_clearance_cost_fn(self, collision_map: CollisionMap) -> Callable[[np.ndarray], float] | None:
        weight = float(self.planner_setting("obstacle_clearance_cost_weight", "obstacle_clearance_cost_weight"))
        desired_clearance_m = float(
            self.planner_setting("obstacle_clearance_cost_distance_m", "obstacle_clearance_cost_distance_m")
        )
        if weight <= 0.0 or desired_clearance_m <= 0.0:
            return None

        def point_cost(point: np.ndarray) -> float:
            clearance_m = collision_map.obstacle_clearance(point, include_fences=False)
            if not math.isfinite(clearance_m) or clearance_m >= desired_clearance_m:
                return 0.0
            return weight * (desired_clearance_m - max(clearance_m, 0.0)) / desired_clearance_m

        return point_cost

    def plan_through_subgoals(
        self,
        planner: HybridAStarPlanner,
        subgoals: list[tuple[np.ndarray, float]],
        collision_map: CollisionMap,
    ) -> list[np.ndarray] | None:
        if self.current_position is None or self.current_yaw is None:
            return None
        if collision_map.is_collision(self.current_position):
            self.get_logger().warn(f"Hybrid A* start pose {self.current_position.tolist()} is in collision")
            return None
        planned_path: list[np.ndarray] = []
        start_pose = Pose(float(self.current_position[0]), float(self.current_position[1]), float(self.current_yaw))
        max_lateral_m = float(self.planner_setting("planner_subgoal_lateral_search_m", "planner_subgoal_lateral_search_m"))
        max_longitudinal_m = float(
            self.planner_setting("planner_subgoal_longitudinal_search_m", "planner_subgoal_longitudinal_search_m")
        )
        search_step_m = float(self.planner_setting("planner_subgoal_search_step_m", "planner_subgoal_search_step_m"))
        max_candidates = int(self.planner_setting("planner_subgoal_max_candidates", "planner_subgoal_max_candidates"))
        min_clearance_m = float(
            self.planner_setting("planner_subgoal_min_clearance_m", "planner_subgoal_min_clearance_m")
        )
        min_fence_clearance_m = float(
            self.planner_setting("planner_fence_min_clearance_m", "planner_fence_min_clearance_m")
        )
        nudged_count = 0
        side_constraint_segments = side_constraint_segments_for_task(self.scene, self.task, self.flip_scene_y)

        for index, (goal_position, goal_yaw) in enumerate(subgoals):
            selected_position = None
            selected_segment = None
            attempted_candidates = 0
            free_candidates = 0
            collision_candidates = 0
            wrong_side_candidates = 0
            low_clearance_candidates = 0
            for candidate in shifted_subgoal_candidates(
                goal_position,
                goal_yaw,
                max_lateral_m,
                max_longitudinal_m,
                step_m=search_step_m,
            ):
                if collision_map.is_collision(candidate):
                    collision_candidates += 1
                    continue
                if collision_map.obstacle_clearance(candidate, include_fences=False) < min_clearance_m:
                    low_clearance_candidates += 1
                    continue
                if (
                    min_fence_clearance_m > 0.0
                    and side_constraint_segments
                    and distance_to_nearest_segment(candidate, side_constraint_segments) < min_fence_clearance_m
                ):
                    low_clearance_candidates += 1
                    continue
                if not candidate_preserves_side(goal_position, candidate, side_constraint_segments):
                    wrong_side_candidates += 1
                    continue
                free_candidates += 1
                if attempted_candidates >= max_candidates:
                    continue
                attempted_candidates += 1
                segment = planner.plan(
                    start_pose,
                    Pose(float(candidate[0]), float(candidate[1]), float(goal_yaw)),
                )
                if segment is not None:
                    selected_position = candidate
                    selected_segment = segment
                    break
            if selected_position is None or selected_segment is None:
                self.get_logger().warn(
                    f"Hybrid A* could not reach subgoal {index} near {goal_position.tolist()} "
                    f"within {max_lateral_m:.2f}m lateral / {max_longitudinal_m:.2f}m longitudinal search "
                    f"({attempted_candidates}/{free_candidates} free candidates attempted, "
                    f"{collision_candidates} colliding candidates, {low_clearance_candidates} low-clearance candidates, "
                    f"{wrong_side_candidates} wrong-side candidates)"
                )
                return None
            if float(np.linalg.norm(selected_position - goal_position)) > 1e-6:
                nudged_count += 1
            if planned_path:
                planned_path.extend(selected_segment[1:])
            else:
                planned_path.extend(selected_segment)
            if np.linalg.norm(planned_path[-1] - selected_position) > 1e-6:
                planned_path.append(selected_position)
            start_pose = Pose(float(selected_position[0]), float(selected_position[1]), float(goal_yaw))
        if nudged_count:
            self.get_logger().info(f"Hybrid A* used {nudged_count} nearby reachable subgoals")
        return planned_path

    def maybe_plan_path(self) -> None:
        if not self.use_hybrid_astar or self.planning_attempted:
            return
        if self.current_position is None or self.current_yaw is None:
            return
        self.planning_attempted = True

        subgoals = self.reference_subgoals()
        reference_points = [self.current_position] + [position for position, _ in subgoals] + self.path
        planner_settings = self.variant.get("planner_settings", {})
        collision_map = CollisionMap.from_scene(
            self.scene,
            self.scene_path,
            reference_points,
            self.flip_scene_y,
            robot_radius_m=float(planner_settings.get("robot_radius_m", self.get_parameter("robot_radius_m").value)),
            obstacle_padding_m=float(
                planner_settings.get("obstacle_padding_m", self.get_parameter("obstacle_padding_m").value)
            ),
        )
        planner = self.build_planner(collision_map)
        planned = self.plan_through_subgoals(planner, subgoals, collision_map)
        if planned is None:
            message = "Hybrid A* subgoal planning failed"
            self.get_logger().error(f"{message}; strict collection mode is stopping the expert policy")
            raise RuntimeError(message)
        if not self.disable_shortcut_smoothing_for_task():
            planned = shortcut_smooth(planned, collision_map.is_collision)
        planned = resample_path(planned, max(self.waypoint_spacing_m * 0.5, 0.1))
        planned_length = path_length(planned)
        reference_length = path_length(self.path)
        side_constraint_segments = side_constraint_segments_for_task(self.scene, self.task, self.flip_scene_y)
        min_fence_clearance_m = float(
            self.planner_setting("planner_fence_min_clearance_m", "planner_fence_min_clearance_m")
        )
        if min_fence_clearance_m > 0.0 and side_constraint_segments:
            planned_min_fence_distance = path_min_distance_to_segments(planned, side_constraint_segments)
            if planned_min_fence_distance < min_fence_clearance_m:
                message = (
                    f"Hybrid A* runtime path is too close to target fence: "
                    f"min_distance={planned_min_fence_distance:.2f}m "
                    f"required={min_fence_clearance_m:.2f}m"
                )
                self.get_logger().error(message)
                raise RuntimeError(message)
        if self.requires_full_reference_progress() and planned_length < 0.75 * reference_length:
            message = (
                f"Hybrid A* runtime path is too short for ordered task: "
                f"planned_length={planned_length:.2f}m reference_length={reference_length:.2f}m"
            )
            self.get_logger().error(message)
            raise RuntimeError(message)
        self.planned_path = planned
        self.trajectory = self.profile_path(planned)
        self.write_runtime_planned_path(planned, subgoals, collision_map)
        self.path_progress_m = 0.0
        self.path_progress_anchor_position = None
        self.path_progress_distance_budget_m = 0.0
        self.get_logger().info(
            f"Hybrid A* planned {len(planned)} points via {len(subgoals)} subgoals "
            f"around {len(collision_map.obstacles)} obstacles, "
            f"path_length={path_length(planned):.2f}m duration={self.trajectory.duration():.2f}s"
        )

    def write_runtime_planned_path(
        self,
        planned: list[np.ndarray],
        subgoals: list[tuple[np.ndarray, float]],
        collision_map: CollisionMap,
    ) -> None:
        if self.runtime_planned_path_output is None:
            return
        try:
            self.runtime_planned_path_output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "task_spec": self.task_spec_path.as_posix(),
                "scene_yaml": self.scene_path.as_posix(),
                "task_id": self.task_id,
                "variant_id": self.variant_id,
                "flip_scene_y": bool(self.flip_scene_y),
                "flip_runtime_odom_y": bool(self.flip_runtime_odom_y),
                "flip_runtime_odom_yaw": bool(self.flip_runtime_odom_yaw),
                "path_length_m": float(path_length(planned)),
                "point_count": len(planned),
                "collision_inflation_m": float(collision_map.inflation_m),
                "path": [[float(point[0]), float(point[1])] for point in planned],
                "subgoals": [
                    {"position": [float(position[0]), float(position[1])], "yaw": float(yaw)}
                    for position, yaw in subgoals
                ],
            }
            self.runtime_planned_path_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.get_logger().info(f"Wrote runtime planned path to {self.runtime_planned_path_output}")
        except Exception as exc:
            self.get_logger().warn(f"Could not write runtime planned path: {exc}")

    def requires_full_reference_progress(self) -> bool:
        task_type = self.task.get("task_type")
        if task_type == "follow_fence_sequence":
            return True
        if task_type == "follow_shed_side" and self.task.get("shed_side") == "perimeter":
            return True
        return False

    def disable_shortcut_smoothing_for_task(self) -> bool:
        task_type = self.task.get("task_type")
        if task_type == "follow_fence_sequence":
            return True
        if task_type == "follow_shed_side" and self.task.get("shed_side") == "perimeter":
            return True
        if task_type in {"pass_through_gap", "switch_sides"}:
            return True
        return False

    def current_profile_time(self, trajectory: TimedTrajectory) -> float:
        progress_m = self.current_path_progress(trajectory.path)
        return float(np.interp(progress_m, trajectory.distances, trajectory.times))

    def current_path_progress(self, path: list[np.ndarray]) -> float:
        if self.current_position is None:
            return 0.0
        projected_progress = project_progress_near(
            path,
            self.current_position,
            self.path_progress_m,
            max_backward_m=0.5,
            max_forward_m=2.0,
        )
        closest_position, _ = sample_path_pose(path, projected_progress)
        tracking_error_m = float(np.linalg.norm(self.current_position - closest_position))
        self.latest_tracking_error_m = tracking_error_m
        self.latest_off_path = (
            self.max_expert_tracking_error_m > 0.0
            and tracking_error_m > self.max_expert_tracking_error_m
        )

        if self.path_progress_anchor_position is None:
            self.path_progress_anchor_position = self.current_position.copy()
        else:
            moved_m = float(np.linalg.norm(self.current_position - self.path_progress_anchor_position))
            if math.isfinite(moved_m):
                self.path_progress_distance_budget_m += moved_m
            self.path_progress_anchor_position = self.current_position.copy()

        max_motion_consistent_progress = (
            self.path_progress_distance_budget_m + self.path_progress_motion_slack_m
        )
        if self.latest_off_path:
            return self.path_progress_m
        progress = min(projected_progress, max_motion_consistent_progress)
        self.path_progress_m = max(self.path_progress_m, progress)
        return self.path_progress_m

    def select_direct_tracking_target(
        self,
        trajectory: TimedTrajectory,
        progress_m: float,
    ) -> tuple[np.ndarray, float, float]:
        if self.current_position is None or self.current_yaw is None:
            return sample_path_pose(trajectory.path, progress_m)

        total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
        start_distance = min(total_distance, float(progress_m) + self.expert_path_lookahead_m)
        upper_distance = min(total_distance, float(progress_m) + self.max_tracking_target_distance_m)
        if start_distance >= total_distance or upper_distance <= start_distance:
            position, yaw = sample_path_pose(trajectory.path, total_distance)
            return position, yaw, total_distance

        best: tuple[float, np.ndarray, float, float] | None = None
        step_m = 0.15
        count = max(1, int(math.ceil((upper_distance - start_distance) / step_m)))
        for index in range(count + 1):
            distance = min(upper_distance, start_distance + index * step_m)
            position, yaw = sample_path_pose(trajectory.path, distance)
            x, y, theta = world_to_robot(self.current_position, self.current_yaw, position, yaw)
            euclidean_distance = float(np.linalg.norm(position - self.current_position))
            if (
                x <= 0.10
                or abs(y) > self.max_target_lateral_error_m
                or euclidean_distance > self.max_tracking_target_distance_m
            ):
                continue
            preferred_x_error = abs(x - self.expert_path_lookahead_m)
            score = (
                0.7 * abs(y)
                + 0.25 * preferred_x_error
                + 0.10 * abs(theta)
                + 0.15 * euclidean_distance
                + 0.02 * (distance - start_distance)
            )
            if best is None or score < best[0]:
                best = (score, position, yaw, distance)

        if best is not None:
            _, position, yaw, distance = best
            return position, yaw, distance

        position, yaw = sample_path_pose(trajectory.path, progress_m)
        return position, yaw, progress_m

    def select_recovery_tracking_target(
        self,
        trajectory: TimedTrajectory,
    ) -> tuple[np.ndarray, float, float]:
        if self.current_position is None:
            return sample_path_pose(trajectory.path, self.path_progress_m)
        total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
        closest_progress = project_progress(trajectory.path, self.current_position)
        target_distance = min(total_distance, closest_progress + self.recovery_lookahead_m)
        position, yaw = sample_path_pose(trajectory.path, target_distance)
        return position, yaw, target_distance

    def publish_expert_cmd_vel(self, trajectory: TimedTrajectory, progress_m: float) -> None:
        if self.cmd_vel_publisher is None or self.current_position is None or self.current_yaw is None:
            return
        total_distance = float(trajectory.distances[-1]) if len(trajectory.distances) else 0.0
        msg = Twist()
        if total_distance <= 1e-6 or total_distance - progress_m <= 0.15:
            self.cmd_vel_publisher.publish(msg)
            return
        recovering = bool(self.latest_off_path)
        if recovering:
            target_position, target_yaw, target_distance = self.select_recovery_tracking_target(trajectory)
            self.get_logger().warn(
                f"Recovering toward planned path: "
                f"tracking_error={self.latest_tracking_error_m:.2f}m "
                f"limit={self.max_expert_tracking_error_m:.2f}m"
            )
        else:
            target_position, target_yaw, target_distance = self.select_direct_tracking_target(trajectory, progress_m)
        relative_x, relative_y, relative_theta = world_to_robot(
            self.current_position,
            self.current_yaw,
            target_position,
            target_yaw,
        )

        profile_time_s = float(np.interp(target_distance, trajectory.distances, trajectory.times))
        _, _, profile_speed, _ = trajectory.sample(profile_time_s)
        max_speed = min(
            float(self.get_parameter("expert_speed_limit_mps").value),
            float(self.get_parameter("max_speed_mps").value),
        )
        speed = min(max(float(profile_speed), self.expert_min_tracking_speed_mps), max_speed)
        if recovering:
            speed = min(speed, self.recovery_speed_mps)

        heading_error = math.atan2(float(relative_y), max(float(relative_x), 0.05))
        distance_sq = max(float(relative_x * relative_x + relative_y * relative_y), 1e-5)
        curvature = 2.0 * float(relative_y) / distance_sq
        max_yaw_rate = min(
            self.expert_tracking_max_yaw_rate_radps,
            float(self.get_parameter("max_yaw_rate_radps").value),
        )

        steering_error = wrap_to_pi(1.05 * heading_error + 0.35 * relative_theta)
        if recovering:
            steering_error = heading_error
        abs_steering_error = abs(steering_error)
        if abs_steering_error > self.expert_heading_slowdown_rad:
            slowdown_span = max(math.pi - self.expert_heading_slowdown_rad, 1e-6)
            slowdown = 1.0 - 0.55 * min(
                1.0,
                (abs_steering_error - self.expert_heading_slowdown_rad) / slowdown_span,
            )
            speed = max(self.expert_min_tracking_speed_mps, speed * slowdown)

        if abs(curvature) > 1e-6:
            speed = min(speed, max(self.expert_min_tracking_speed_mps, max_yaw_rate / abs(curvature)))

        yaw_rate = 0.9 * steering_error
        if abs_steering_error > 1.8:
            speed = min(speed, self.expert_min_tracking_speed_mps)

        msg.linear.x = float(max(0.0, min(speed, max_speed)))
        command_yaw_rate = -yaw_rate if self.flip_runtime_odom_yaw else yaw_rate
        msg.angular.z = float(max(-max_yaw_rate, min(max_yaw_rate, command_yaw_rate)))
        self.cmd_vel_publisher.publish(msg)

    def publish_chunk(self) -> None:
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn("No sim odom received yet; not publishing ActionChunk")
            return
        self.maybe_plan_path()
        trajectory = self.active_trajectory()
        progress_m = self.current_path_progress(trajectory.path)
        self.publish_frame_debug(trajectory, progress_m)
        msg = build_distance_action_chunk(
            self,
            trajectory,
            self.current_position,
            self.current_yaw,
            progress_m,
            self.seq_num,
            first_preview_m=self.first_preview_m,
            waypoint_spacing_m=self.waypoint_spacing_m,
        )
        first_pose = msg.relative_poses[0]
        if first_pose.x < 0.08:
            self.get_logger().warn(
                f"ActionChunk first waypoint is still too close/behind: "
                f"x={first_pose.x:.3f} y={first_pose.y:.3f} theta={first_pose.theta:.3f} "
                f"progress_m={progress_m:.3f}/{path_length(trajectory.path):.3f}"
            )
        self.publisher.publish(msg)
        self.publish_expert_cmd_vel(trajectory, progress_m)
        self.seq_num += 1

    def publish_frame_debug(self, trajectory: TimedTrajectory, progress_m: float) -> None:
        if self.current_position is None or self.current_yaw is None:
            return
        if self.latest_off_path:
            target_position, target_yaw, target_distance = self.select_recovery_tracking_target(trajectory)
            target_source = "recovery"
        else:
            target_position, target_yaw, target_distance = self.select_direct_tracking_target(trajectory, progress_m)
            target_source = "tracking"
        delta_world = target_position - self.current_position
        relative_x, relative_y, relative_theta = world_to_robot(
            self.current_position,
            self.current_yaw,
            target_position,
            target_yaw,
        )
        payload = {
            **self.latest_odom_frame_debug,
            "current_world_x": float(self.current_position[0]),
            "current_world_y": float(self.current_position[1]),
            "current_world_yaw": float(self.current_yaw),
            "target_world_x": float(target_position[0]),
            "target_world_y": float(target_position[1]),
            "target_world_yaw": float(target_yaw),
            "target_path_distance_m": float(target_distance),
            "target_source": target_source,
            "current_path_progress_m": float(progress_m),
            "tracking_error_m": float(self.latest_tracking_error_m or 0.0),
            "off_path": bool(self.latest_off_path),
            "max_expert_tracking_error_m": float(self.max_expert_tracking_error_m),
            "delta_world_x": float(delta_world[0]),
            "delta_world_y": float(delta_world[1]),
            "relative_x": float(relative_x),
            "relative_y": float(relative_y),
            "relative_theta": float(relative_theta),
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.frame_debug_publisher.publish(msg)

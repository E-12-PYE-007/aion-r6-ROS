#!/usr/bin/env python3
"""Base ROS node for task-spec-driven expert trajectory publishers."""

from __future__ import annotations

import json
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
    build_timed_action_chunk,
    fence_by_name,
    find_variant,
    find_task,
    get_start_pose,
    local_odom_to_world,
    load_yaml,
    odom_to_pose,
    path_length,
    point2,
    sample_path_pose,
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
        self.declare_parameter("waypoint_spacing_m", 0.18)
        self.declare_parameter("future_time_offsets_s", [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])
        self.declare_parameter("publish_rate_hz", 3.0)
        self.declare_parameter("flip_isaac_y", True)
        self.declare_parameter("flip_scene_y", True)
        self.declare_parameter("flip_runtime_odom_y", False)
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
        self.declare_parameter("planner_subgoal_vertex_margin_m", 0.5)
        self.declare_parameter("planner_subgoal_endpoint_margin_m", 0.5)
        self.declare_parameter("hybrid_astar_max_iterations", 20000)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("fence_offset_cost_weight", 0.6)
        self.declare_parameter("fence_offset_cost_deadband_m", 0.15)
        self.declare_parameter("fence_offset_cost_max_error_m", 2.0)
        self.declare_parameter("max_speed_mps", 0.35)
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
        self.world_start_position, self.world_start_yaw = get_start_pose(self.scene, self.task, self.flip_scene_y)
        self.waypoint_spacing_m = float(self.get_parameter("waypoint_spacing_m").value)
        self.future_time_offsets_s = [
            float(value)
            for value in self.get_parameter("future_time_offsets_s").value
        ]
        self.path = self.resolve_path()
        if len(self.path) < 2 or path_length(self.path) < 1e-6:
            raise RuntimeError(f"Task {self.task_id} resolved to an empty path.")
        self.use_hybrid_astar = bool(self.get_parameter("use_hybrid_astar").value)
        self.planned_path: Optional[list[np.ndarray]] = None
        self.trajectory: Optional[TimedTrajectory] = None
        self.planning_attempted = False

        self.current_position: Optional[np.ndarray] = None
        self.current_yaw: Optional[float] = None
        self.latest_odom_frame_debug: dict[str, object] = {}
        self.seq_num = 1

        self.publisher = self.create_publisher(ActionChunk, str(self.get_parameter("action_chunk_topic").value), 10)
        self.frame_debug_publisher = self.create_publisher(String, str(self.get_parameter("frame_debug_topic").value), 10)
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

    def odom_callback(self, msg: Odometry) -> None:
        raw_x = float(msg.pose.pose.position.x)
        raw_y = float(msg.pose.pose.position.y)
        raw_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        local_position, local_yaw = odom_to_pose(msg, self.flip_runtime_odom_y)
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
        return build_timed_trajectory(
            path,
            max_speed_mps=float(speed_profile.get("max_speed_mps", self.get_parameter("max_speed_mps").value)),
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
            point_cost_fn=self.fence_offset_cost_fn(),
        )

    def fence_offset_cost_fn(self) -> Callable[[np.ndarray], float] | None:
        weight = float(self.planner_setting("fence_offset_cost_weight", "fence_offset_cost_weight"))
        if weight <= 0.0:
            return None
        segments = side_constraint_segments_for_task(self.scene, self.task, self.flip_scene_y)
        if not segments:
            return None

        preferred_offset_m = float(self.variant.get("preferred_offset_m", 0.8))
        deadband_m = max(0.0, float(self.planner_setting("fence_offset_cost_deadband_m", "fence_offset_cost_deadband_m")))
        max_error_m = max(
            deadband_m,
            float(self.planner_setting("fence_offset_cost_max_error_m", "fence_offset_cost_max_error_m")),
        )

        def point_cost(point: np.ndarray) -> float:
            distance_to_fence = min(point_to_segment_distance(point, start, end) for start, end in segments)
            offset_error = abs(distance_to_fence - preferred_offset_m)
            if offset_error <= deadband_m:
                return 0.0
            return weight * min(offset_error - deadband_m, max_error_m)

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
        nudged_count = 0
        side_constraint_segments = side_constraint_segments_for_task(self.scene, self.task, self.flip_scene_y)

        for index, (goal_position, goal_yaw) in enumerate(subgoals):
            selected_position = None
            selected_segment = None
            attempted_candidates = 0
            free_candidates = 0
            collision_candidates = 0
            wrong_side_candidates = 0
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
                    f"{collision_candidates} colliding candidates, {wrong_side_candidates} wrong-side candidates)"
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
        planned = shortcut_smooth(planned, collision_map.is_collision)
        planned = resample_path(planned, max(self.waypoint_spacing_m * 0.5, 0.1))
        self.planned_path = planned
        self.trajectory = self.profile_path(planned)
        self.get_logger().info(
            f"Hybrid A* planned {len(planned)} points via {len(subgoals)} subgoals "
            f"around {len(collision_map.obstacles)} obstacles, "
            f"path_length={path_length(planned):.2f}m duration={self.trajectory.duration():.2f}s"
        )

    def current_profile_time(self, trajectory: TimedTrajectory) -> float:
        progress_m = self.current_path_progress(trajectory.path)
        return float(np.interp(progress_m, trajectory.distances, trajectory.times))

    def current_path_progress(self, path: list[np.ndarray]) -> float:
        from sim.expert_trajectory_utils import project_progress

        if self.current_position is None:
            return 0.0
        return project_progress(path, self.current_position)

    def publish_expert_cmd_vel(self, trajectory: TimedTrajectory, profile_time_s: float) -> None:
        if self.cmd_vel_publisher is None:
            return
        _, _, speed, yaw_rate = trajectory.sample(profile_time_s)
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(yaw_rate)
        self.cmd_vel_publisher.publish(msg)

    def publish_chunk(self) -> None:
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn("No sim odom received yet; not publishing ActionChunk")
            return
        self.maybe_plan_path()
        trajectory = self.active_trajectory()
        profile_time_s = self.current_profile_time(trajectory)
        self.publish_frame_debug(trajectory, profile_time_s)
        msg = build_timed_action_chunk(
            self,
            trajectory,
            self.current_position,
            self.current_yaw,
            profile_time_s,
            self.future_time_offsets_s,
            self.seq_num,
        )
        self.publisher.publish(msg)
        self.publish_expert_cmd_vel(trajectory, profile_time_s)
        self.seq_num += 1

    def publish_frame_debug(self, trajectory: TimedTrajectory, profile_time_s: float) -> None:
        if self.current_position is None or self.current_yaw is None:
            return
        first_offset_s = float(self.future_time_offsets_s[0]) if self.future_time_offsets_s else 0.3
        target_position, target_yaw, _, _ = trajectory.sample(profile_time_s + first_offset_s)
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
            "delta_world_x": float(delta_world[0]),
            "delta_world_y": float(delta_world[1]),
            "relative_x": float(relative_x),
            "relative_y": float(relative_y),
            "relative_theta": float(relative_theta),
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.frame_debug_publisher.publish(msg)

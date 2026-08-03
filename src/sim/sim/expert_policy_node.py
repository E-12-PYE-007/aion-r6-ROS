#!/usr/bin/env python3
"""Base ROS node for task-spec-driven expert trajectory publishers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from sim.expert_trajectory_utils import (
    build_timed_action_chunk,
    find_variant,
    find_task,
    load_yaml,
    odom_to_pose,
    path_length,
    sample_path_pose,
)
from sim.collision_map import CollisionMap
from sim.hybrid_astar import HybridAStarPlanner, Pose
from sim.trajectory_profile import TimedTrajectory, build_timed_trajectory, resample_path, shortcut_smooth


class ExpertPolicyNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.declare_parameter("task_spec", Parameter.Type.STRING)
        self.declare_parameter("task_id", Parameter.Type.STRING)
        self.declare_parameter("variant_id", "nominal")
        self.declare_parameter("odom_topic", "sim_odom")
        self.declare_parameter("action_chunk_topic", "/vla/action_chunk")
        self.declare_parameter("expert_cmd_vel_topic", "/expert/cmd_vel")
        self.declare_parameter("waypoint_spacing_m", 0.18)
        self.declare_parameter("future_time_offsets_s", [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])
        self.declare_parameter("publish_rate_hz", 3.0)
        self.declare_parameter("flip_isaac_y", True)
        self.declare_parameter("use_hybrid_astar", True)
        self.declare_parameter("robot_radius_m", 0.35)
        self.declare_parameter("obstacle_padding_m", 0.25)
        self.declare_parameter("grid_resolution_m", 0.25)
        self.declare_parameter("yaw_resolution_deg", 15.0)
        self.declare_parameter("step_size_m", 0.35)
        self.declare_parameter("min_turn_radius_m", 0.75)
        self.declare_parameter("goal_tolerance_m", 0.35)
        self.declare_parameter("planner_subgoal_spacing_m", 2.0)
        self.declare_parameter("hybrid_astar_max_iterations", 20000)
        self.declare_parameter("allow_reverse", False)
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

        self.flip_isaac_y = bool(self.get_parameter("flip_isaac_y").value)
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
        self.seq_num = 1

        self.publisher = self.create_publisher(ActionChunk, str(self.get_parameter("action_chunk_topic").value), 10)
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
        self.current_position, self.current_yaw = odom_to_pose(msg, self.flip_isaac_y)

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
        if spacing <= 0.0 or total_length <= spacing:
            return [sample_path_pose(self.path, total_length)]

        progress_values = list(np.arange(spacing, total_length, spacing))
        progress_values.append(total_length)
        return [sample_path_pose(self.path, float(progress)) for progress in progress_values]

    def build_planner(self, collision_map: CollisionMap) -> HybridAStarPlanner:
        return HybridAStarPlanner(
            collision_map,
            grid_resolution_m=float(self.planner_setting("grid_resolution_m", "grid_resolution_m")),
            yaw_resolution_rad=np.deg2rad(float(self.planner_setting("yaw_resolution_deg", "yaw_resolution_deg"))),
            step_size_m=float(self.planner_setting("step_size_m", "step_size_m")),
            min_turn_radius_m=float(self.planner_setting("min_turn_radius_m", "min_turn_radius_m")),
            goal_tolerance_m=float(self.planner_setting("goal_tolerance_m", "goal_tolerance_m")),
            max_iterations=int(self.planner_setting("hybrid_astar_max_iterations", "hybrid_astar_max_iterations")),
            allow_reverse=bool(self.planner_setting("allow_reverse", "allow_reverse")),
        )

    def plan_through_subgoals(
        self,
        planner: HybridAStarPlanner,
        subgoals: list[tuple[np.ndarray, float]],
    ) -> list[np.ndarray] | None:
        if self.current_position is None or self.current_yaw is None:
            return None
        planned_path: list[np.ndarray] = []
        start_pose = Pose(float(self.current_position[0]), float(self.current_position[1]), float(self.current_yaw))

        for goal_position, goal_yaw in subgoals:
            segment = planner.plan(
                start_pose,
                Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw)),
            )
            if segment is None:
                return None
            if planned_path:
                planned_path.extend(segment[1:])
            else:
                planned_path.extend(segment)
            if np.linalg.norm(planned_path[-1] - goal_position) > 1e-6:
                planned_path.append(goal_position)
            start_pose = Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw))
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
            self.flip_isaac_y,
            robot_radius_m=float(planner_settings.get("robot_radius_m", self.get_parameter("robot_radius_m").value)),
            obstacle_padding_m=float(
                planner_settings.get("obstacle_padding_m", self.get_parameter("obstacle_padding_m").value)
            ),
        )
        planner = self.build_planner(collision_map)
        planned = self.plan_through_subgoals(planner, subgoals)
        if planned is None:
            self.get_logger().warn("Hybrid A* subgoal planning failed; falling back to geometric reference path")
            self.trajectory = self.profile_path(self.path)
            return
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

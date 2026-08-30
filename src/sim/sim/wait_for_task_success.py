#!/usr/bin/env python3
"""Wait until a rollout reaches its task success condition or a max sim-time timeout."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sim.expert_trajectory_utils import (
    find_task,
    find_variant,
    get_start_pose,
    load_yaml,
    local_odom_to_world,
    odom_to_pose,
    path_length,
    point2,
    project_progress_near,
    sample_path_pose,
    wrap_to_pi,
)
from sim.validate_scene_task_specs import reference_path_for_task


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


class TaskSuccessWaiter(Node):
    def __init__(self) -> None:
        super().__init__("wait_for_task_success")
        self.declare_parameter("task_spec", "")
        self.declare_parameter("task_id", "")
        self.declare_parameter("variant_id", "nominal")
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("isaac_pose_debug_topic", "/isaac/scene_pose_debug")
        self.declare_parameter("use_isaac_camera_pose_debug", False)
        self.declare_parameter("max_duration_s", 60.0)
        self.declare_parameter("fallback_duration_s", 20.0)
        self.declare_parameter("wall_timeout_s", 900.0)
        self.declare_parameter("min_odom_messages", 2)
        self.declare_parameter("success_margin_m", 0.0)
        self.declare_parameter("perimeter_success_margin_m", 5.0)
        self.declare_parameter("target_tolerance_m", 0.5)
        self.declare_parameter("max_success_tracking_error_m", 1.25)
        self.declare_parameter("max_progress_tracking_error_m", 2.0)
        self.declare_parameter("flip_scene_y", False)
        self.declare_parameter("flip_runtime_odom_y", False)
        self.declare_parameter("flip_runtime_odom_yaw", True)
        self.declare_parameter("summary_path", "")

        self.task_spec_path = Path(str(self.get_parameter("task_spec").value))
        self.task_id = str(self.get_parameter("task_id").value)
        self.variant_id = str(self.get_parameter("variant_id").value)
        self.max_duration_s = float(self.get_parameter("max_duration_s").value)
        self.fallback_duration_s = float(self.get_parameter("fallback_duration_s").value)
        self.wall_timeout_s = float(self.get_parameter("wall_timeout_s").value)
        self.min_odom_messages = int(self.get_parameter("min_odom_messages").value)
        self.success_margin_m = float(self.get_parameter("success_margin_m").value)
        self.perimeter_success_margin_m = max(
            0.0,
            float(self.get_parameter("perimeter_success_margin_m").value),
        )
        self.target_tolerance_m = float(self.get_parameter("target_tolerance_m").value)
        self.max_success_tracking_error_m = max(
            0.0,
            float(self.get_parameter("max_success_tracking_error_m").value),
        )
        self.max_progress_tracking_error_m = max(
            self.max_success_tracking_error_m,
            float(self.get_parameter("max_progress_tracking_error_m").value),
        )
        self.flip_scene_y = bool(self.get_parameter("flip_scene_y").value)
        self.flip_runtime_odom_y = bool(self.get_parameter("flip_runtime_odom_y").value)
        self.flip_runtime_odom_yaw = bool(self.get_parameter("flip_runtime_odom_yaw").value)
        self.use_isaac_camera_pose_debug = bool(self.get_parameter("use_isaac_camera_pose_debug").value)
        summary_value = str(self.get_parameter("summary_path").value)
        self.summary_path = Path(summary_value) if summary_value else None

        (
            self.required_distance_m,
            self.success_type,
            self.target_position,
            self.world_start_position,
            self.world_start_yaw,
            self.reference_path,
        ) = self.load_success_condition()
        self.reference_path_length_m = path_length(self.reference_path) if self.reference_path is not None else None
        self.target_duration_s = self.max_duration_s if self.required_distance_m is not None else self.fallback_duration_s

        self.start_wall_s = time.monotonic()
        self.start_stamp_s: float | None = None
        self.latest_stamp_s: float | None = None
        self.previous_xy: tuple[float, float] | None = None
        self.initial_camera_debug_position: np.ndarray | None = None
        self.initial_camera_debug_yaw: float | None = None
        self.latest_world_position: np.ndarray | None = None
        self.latest_world_yaw: float | None = None
        self.latest_target_distance_m: float | None = None
        self.latest_tracking_error_m: float | None = None
        self.max_tracking_error_m = 0.0
        self.distance_travelled_m = 0.0
        self.path_progress_m = 0.0
        self.odom_messages = 0
        self.pose_messages = 0
        self.done = False
        self.success = False
        self.timed_out = False
        self.stop_reason = "running"

        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            10,
        )
        if self.use_isaac_camera_pose_debug:
            self.create_subscription(
                String,
                str(self.get_parameter("isaac_pose_debug_topic").value),
                self.isaac_pose_debug_callback,
                10,
            )

        if self.required_distance_m is None:
            self.get_logger().warn(
                f"Task {self.task_id!r} does not expose a distance-based success condition; "
                f"falling back to {self.fallback_duration_s:.1f}s sim time."
            )

        if self.use_isaac_camera_pose_debug:
            self.get_logger().info(
                "Task success/progress will use /isaac/scene_pose_debug camera_world_pose; "
                "/sim_odom is used only for sim-time heartbeat."
            )
        else:
            target_text = ""
            if self.target_position is not None:
                target_text = f", target_tolerance_m={self.target_tolerance_m:.2f}"
            self.get_logger().info(
                f"Waiting for task {self.task_id!r}: path_progress_m >= "
                f"{self.required_distance_m:.2f}m{target_text}, max_duration_s={self.max_duration_s:.1f}"
            )

    def load_success_condition(
        self,
    ) -> tuple[float | None, str | None, np.ndarray | None, np.ndarray | None, float | None, list[np.ndarray] | None]:
        if not self.task_spec_path.exists() or not self.task_id:
            return None, None, None, None, None, None
        task_spec = load_yaml(self.task_spec_path)
        scene_path = Path(task_spec["scene"]["source_yaml"])
        scene = load_yaml(scene_path)
        task = find_task(task_spec, self.task_id)
        variant = find_variant(task, self.variant_id)
        world_start_position, world_start_yaw = get_start_pose(scene, task, self.flip_scene_y)
        reference_path = None
        try:
            reference_path = reference_path_for_task(scene, scene_path, task, variant, self.flip_scene_y)
        except Exception as exc:
            self.get_logger().warn(f"Could not resolve reference path for success progress: {exc}")
        success_condition = task.get("success_condition")
        if not isinstance(success_condition, dict):
            return None, None, None, world_start_position, world_start_yaw, reference_path
        success_type = str(success_condition.get("type", ""))
        if success_type in {"reach_path_end", "pass_point", "pass_point_and_continue"}:
            required = finite_float(success_condition.get("min_progress_m"), 0.0) - self.success_margin_m
            if self.is_perimeter_follow_task(task):
                required -= self.perimeter_success_margin_m
            target_raw = success_condition.get("target_point")
            target_position = point2(target_raw, self.flip_scene_y) if isinstance(target_raw, (list, tuple)) else None
            if reference_path is not None:
                required = min(required, max(path_length(reference_path) - 0.25, 0.0))
                if success_type == "reach_path_end":
                    target_position = reference_path[-1]
            if success_type == "reach_path_end" and not self.requires_target_tolerance(task, success_condition):
                target_position = None
            return max(required, 0.0), success_type, target_position, world_start_position, world_start_yaw, reference_path
        return None, success_type, None, world_start_position, world_start_yaw, reference_path

    @staticmethod
    def is_perimeter_follow_task(task: dict[str, Any]) -> bool:
        task_id = str(task.get("task_id", ""))
        scenario_tags = {str(tag) for tag in task.get("scenario_tags", []) if tag is not None}
        return task_id == "follow_fence_perimeter_from_scene_rover_pose" or "perimeter" in scenario_tags

    @staticmethod
    def requires_target_tolerance(task: dict[str, Any], success_condition: dict[str, Any]) -> bool:
        if bool(success_condition.get("require_target_tolerance", False)):
            return True
        task_type = str(task.get("task_type", ""))
        data_category = str(task.get("data_category", ""))
        scenario_tags = {str(tag) for tag in task.get("scenario_tags", []) if tag is not None}
        return (
            task_type.startswith("stop")
            or task_type in {"stop_at_landmark", "stop_at_gap"}
            or data_category == "terminal"
            or "terminal" in scenario_tags
        )

    def odom_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s <= 0.0:
            return

        self.odom_messages += 1
        if self.start_stamp_s is None:
            self.start_stamp_s = stamp_s
        self.latest_stamp_s = stamp_s

        if self.use_isaac_camera_pose_debug:
            self.update_completion()
            return

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self.update_distance((x, y))

        if self.world_start_position is not None and self.world_start_yaw is not None:
            local_position, local_yaw = odom_to_pose(
                msg,
                self.flip_runtime_odom_y,
                self.flip_runtime_odom_yaw,
            )
            world_position, _ = local_odom_to_world(
                local_position,
                local_yaw,
                self.world_start_position,
                self.world_start_yaw,
            )
            self.update_world_progress(world_position, None)

        self.update_completion()

    def isaac_pose_debug_callback(self, msg: String) -> None:
        if not self.use_isaac_camera_pose_debug:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring malformed /isaac/scene_pose_debug JSON")
            return
        camera_pose = data.get("camera_world_pose")
        if not isinstance(camera_pose, dict):
            return
        if self.world_start_position is None or self.world_start_yaw is None:
            return
        try:
            camera_position = np.array(
                [
                    float(camera_pose["x"]),
                    float(camera_pose["y"]),
                ],
                dtype=float,
            )
            camera_yaw = float(camera_pose["yaw"])
        except (KeyError, TypeError, ValueError):
            return

        if self.initial_camera_debug_position is None or self.initial_camera_debug_yaw is None:
            self.initial_camera_debug_position = camera_position
            self.initial_camera_debug_yaw = camera_yaw

        camera_delta = camera_position - self.initial_camera_debug_position
        world_position = self.world_start_position + camera_delta
        world_yaw = wrap_to_pi(
            self.world_start_yaw + wrap_to_pi(camera_yaw - self.initial_camera_debug_yaw)
        )
        self.pose_messages += 1
        self.update_distance((float(world_position[0]), float(world_position[1])))
        self.update_world_progress(world_position, world_yaw)
        self.update_completion()

    def update_distance(self, xy: tuple[float, float]) -> None:
        if self.previous_xy is not None:
            self.distance_travelled_m += math.hypot(xy[0] - self.previous_xy[0], xy[1] - self.previous_xy[1])
        self.previous_xy = xy

    def update_world_progress(self, world_position: np.ndarray, world_yaw: float | None) -> None:
        self.latest_world_position = world_position
        self.latest_world_yaw = world_yaw
        if self.reference_path is not None:
            progress = project_progress_near(
                self.reference_path,
                world_position,
                self.path_progress_m,
                max_backward_m=0.5,
                max_forward_m=2.0,
            )
            closest_position, _ = sample_path_pose(self.reference_path, progress)
            self.latest_tracking_error_m = float(np.linalg.norm(world_position - closest_position))
            self.max_tracking_error_m = max(self.max_tracking_error_m, self.latest_tracking_error_m)
            if self.latest_tracking_error_m <= self.max_progress_tracking_error_m:
                self.path_progress_m = max(self.path_progress_m, progress)
        if self.target_position is not None:
            self.latest_target_distance_m = float(np.linalg.norm(world_position - self.target_position))

    def update_completion(self) -> None:
        sim_elapsed_s = self.sim_elapsed_s()
        reached_required_distance = (
            self.required_distance_m is not None
            and self.path_progress_m >= self.required_distance_m
        )
        tracking_is_close = (
            self.reference_path is None
            or self.latest_tracking_error_m is None
            or self.latest_tracking_error_m <= self.max_success_tracking_error_m
        )
        reached_target = (
            self.target_position is None
            or (
                self.latest_target_distance_m is not None
                and self.latest_target_distance_m <= self.target_tolerance_m
            )
        )
        if reached_required_distance and reached_target and tracking_is_close:
            self.success = True
            self.stop_reason = "success_distance_and_target" if self.target_position is not None else "success_progress"
            self.done = self.has_enough_messages()
        elif sim_elapsed_s >= self.target_duration_s and self.has_enough_messages():
            self.stop_reason = "max_duration_reached"
            self.done = True

    def has_enough_messages(self) -> bool:
        if self.odom_messages < self.min_odom_messages:
            return False
        if self.use_isaac_camera_pose_debug and self.pose_messages < self.min_odom_messages:
            return False
        return True

    def check_wall_timeout(self) -> None:
        if time.monotonic() - self.start_wall_s > self.wall_timeout_s:
            self.timed_out = True
            self.stop_reason = "wall_timeout"
            self.done = True

    def sim_elapsed_s(self) -> float:
        if self.start_stamp_s is None or self.latest_stamp_s is None:
            return 0.0
        return max(0.0, self.latest_stamp_s - self.start_stamp_s)

    def summary(self) -> dict[str, Any]:
        return {
            "task_spec": self.task_spec_path.as_posix(),
            "task_id": self.task_id,
            "success_type": self.success_type,
            "required_distance_m": self.required_distance_m,
            "distance_travelled_m": self.distance_travelled_m,
            "path_progress_m": self.path_progress_m,
            "reference_path_length_m": self.reference_path_length_m,
            "target_tolerance_m": self.target_tolerance_m if self.target_position is not None else None,
            "target_distance_m": self.latest_target_distance_m,
            "latest_tracking_error_m": self.latest_tracking_error_m,
            "max_tracking_error_m": self.max_tracking_error_m,
            "max_success_tracking_error_m": self.max_success_tracking_error_m,
            "max_progress_tracking_error_m": self.max_progress_tracking_error_m,
            "success_margin_m": self.success_margin_m,
            "perimeter_success_margin_m": self.perimeter_success_margin_m,
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "latest_world_position": (
                self.latest_world_position.tolist() if self.latest_world_position is not None else None
            ),
            "latest_world_yaw": self.latest_world_yaw,
            "pose_source": "isaac_camera_pose_debug" if self.use_isaac_camera_pose_debug else "sim_odom",
            "success": self.success,
            "stop_reason": self.stop_reason,
            "sim_elapsed_s": self.sim_elapsed_s(),
            "max_duration_s": self.max_duration_s,
            "fallback_duration_s": self.fallback_duration_s,
            "wall_timeout_s": self.wall_timeout_s,
            "odom_messages": self.odom_messages,
            "pose_messages": self.pose_messages,
        }

    def write_summary(self) -> None:
        if self.summary_path is None:
            return
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")


def main(args=None) -> int:
    rclpy.init(args=args)
    node = TaskSuccessWaiter()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.check_wall_timeout()
        node.write_summary()
        summary = node.summary()
        if node.timed_out:
            node.get_logger().error(
                f"Timed out after {node.wall_timeout_s:.1f}s wall time; "
                f"observed {node.sim_elapsed_s():.2f}s sim time, "
                f"distance={node.distance_travelled_m:.2f}m, "
                f"path_progress={node.path_progress_m:.2f}m."
            )
            return 1
        node.get_logger().info(
            f"Task wait complete: reason={summary['stop_reason']} success={node.success} "
            f"sim_elapsed={node.sim_elapsed_s():.2f}s distance={node.distance_travelled_m:.2f}m "
            f"path_progress={node.path_progress_m:.2f}m."
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

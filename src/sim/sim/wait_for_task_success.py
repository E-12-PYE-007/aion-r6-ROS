#!/usr/bin/env python3
"""Wait until a rollout reaches its task success condition or a max sim-time timeout."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

from sim.expert_trajectory_utils import find_task, load_yaml


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
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("max_duration_s", 60.0)
        self.declare_parameter("fallback_duration_s", 20.0)
        self.declare_parameter("wall_timeout_s", 900.0)
        self.declare_parameter("min_odom_messages", 2)
        self.declare_parameter("success_margin_m", 0.0)
        self.declare_parameter("summary_path", "")

        self.task_spec_path = Path(str(self.get_parameter("task_spec").value))
        self.task_id = str(self.get_parameter("task_id").value)
        self.max_duration_s = float(self.get_parameter("max_duration_s").value)
        self.fallback_duration_s = float(self.get_parameter("fallback_duration_s").value)
        self.wall_timeout_s = float(self.get_parameter("wall_timeout_s").value)
        self.min_odom_messages = int(self.get_parameter("min_odom_messages").value)
        self.success_margin_m = float(self.get_parameter("success_margin_m").value)
        summary_value = str(self.get_parameter("summary_path").value)
        self.summary_path = Path(summary_value) if summary_value else None

        self.required_distance_m, self.success_type = self.load_required_distance()
        self.target_duration_s = self.max_duration_s if self.required_distance_m is not None else self.fallback_duration_s

        self.start_wall_s = time.monotonic()
        self.start_stamp_s: float | None = None
        self.latest_stamp_s: float | None = None
        self.previous_xy: tuple[float, float] | None = None
        self.distance_travelled_m = 0.0
        self.odom_messages = 0
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

        if self.required_distance_m is None:
            self.get_logger().warn(
                f"Task {self.task_id!r} does not expose a distance-based success condition; "
                f"falling back to {self.fallback_duration_s:.1f}s sim time."
            )
        else:
            self.get_logger().info(
                f"Waiting for task {self.task_id!r}: distance_travelled_m >= "
                f"{self.required_distance_m:.2f}m, max_duration_s={self.max_duration_s:.1f}"
            )

    def load_required_distance(self) -> tuple[float | None, str | None]:
        if not self.task_spec_path.exists() or not self.task_id:
            return None, None
        task_spec = load_yaml(self.task_spec_path)
        task = find_task(task_spec, self.task_id)
        success_condition = task.get("success_condition")
        if not isinstance(success_condition, dict):
            return None, None
        success_type = str(success_condition.get("type", ""))
        if success_type in {"reach_path_end", "pass_point", "pass_point_and_continue"}:
            required = finite_float(success_condition.get("min_progress_m"), 0.0) - self.success_margin_m
            return max(required, 0.0), success_type
        return None, success_type

    def odom_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s <= 0.0:
            return
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        self.odom_messages += 1
        if self.start_stamp_s is None:
            self.start_stamp_s = stamp_s
        self.latest_stamp_s = stamp_s

        if self.previous_xy is not None:
            self.distance_travelled_m += math.hypot(x - self.previous_xy[0], y - self.previous_xy[1])
        self.previous_xy = (x, y)

        sim_elapsed_s = self.sim_elapsed_s()
        if self.required_distance_m is not None and self.distance_travelled_m >= self.required_distance_m:
            self.success = True
            self.stop_reason = "success_distance"
            self.done = self.odom_messages >= self.min_odom_messages
        elif sim_elapsed_s >= self.target_duration_s and self.odom_messages >= self.min_odom_messages:
            self.stop_reason = "max_duration_reached"
            self.done = True

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
            "success": self.success,
            "stop_reason": self.stop_reason,
            "sim_elapsed_s": self.sim_elapsed_s(),
            "max_duration_s": self.max_duration_s,
            "fallback_duration_s": self.fallback_duration_s,
            "wall_timeout_s": self.wall_timeout_s,
            "odom_messages": self.odom_messages,
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
                f"distance={node.distance_travelled_m:.2f}m."
            )
            return 1
        node.get_logger().info(
            f"Task wait complete: reason={summary['stop_reason']} success={node.success} "
            f"sim_elapsed={node.sim_elapsed_s():.2f}s distance={node.distance_travelled_m:.2f}m."
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

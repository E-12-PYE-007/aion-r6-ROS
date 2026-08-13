#!/usr/bin/env python3
"""Wait until /sim_odom header timestamps advance by a requested duration."""

from __future__ import annotations

import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SimDurationWaiter(Node):
    def __init__(self) -> None:
        super().__init__("wait_for_sim_duration")
        self.declare_parameter("duration_s", 20.0)
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("wall_timeout_s", 300.0)
        self.declare_parameter("min_odom_messages", 2)

        self.duration_s = float(self.get_parameter("duration_s").value)
        self.wall_timeout_s = float(self.get_parameter("wall_timeout_s").value)
        self.min_odom_messages = int(self.get_parameter("min_odom_messages").value)
        self.start_wall_s = time.monotonic()
        self.start_stamp_s: float | None = None
        self.latest_stamp_s: float | None = None
        self.odom_messages = 0
        self.done = False
        self.timed_out = False

        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            10,
        )

    def odom_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s <= 0.0:
            return
        self.odom_messages += 1
        if self.start_stamp_s is None:
            self.start_stamp_s = stamp_s
        self.latest_stamp_s = stamp_s
        sim_elapsed_s = self.latest_stamp_s - self.start_stamp_s
        if self.odom_messages >= self.min_odom_messages and sim_elapsed_s >= self.duration_s:
            self.done = True

    def check_timeout(self) -> None:
        if time.monotonic() - self.start_wall_s > self.wall_timeout_s:
            self.timed_out = True
            self.done = True

    def sim_elapsed_s(self) -> float:
        if self.start_stamp_s is None or self.latest_stamp_s is None:
            return 0.0
        return max(0.0, self.latest_stamp_s - self.start_stamp_s)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SimDurationWaiter()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.check_timeout()
        if node.timed_out:
            node.get_logger().error(
                f"Timed out after {node.wall_timeout_s:.1f}s wall time waiting for "
                f"{node.duration_s:.1f}s sim time; observed {node.sim_elapsed_s():.2f}s sim time "
                f"from {node.odom_messages} odom messages."
            )
            return 1
        node.get_logger().info(
            f"Observed {node.sim_elapsed_s():.2f}s sim time from {node.odom_messages} odom messages."
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Direct /cmd_vel to /sim_odom actuation smoke test for Isaac rollouts."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def unwrap_delta(angle: float, reference: float) -> float:
    return math.atan2(math.sin(angle - reference), math.cos(angle - reference))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class ActuationSmokeTest(Node):
    def __init__(self) -> None:
        super().__init__("actuation_smoke_test")
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("output", "actuation_smoke_test.json")
        self.declare_parameter("duration_s", 3.0)
        self.declare_parameter("settle_duration_s", 0.5)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("wall_timeout_s", 120.0)
        self.declare_parameter("angular_test_radps", 0.3)
        self.declare_parameter("linear_test_mps", 0.2)
        self.declare_parameter("wheel_debug_topic", "/isaac/wheel_actuation_debug")
        self.declare_parameter("wheel_radius_m", 0.08)
        self.declare_parameter("wheel_distance_m", 0.32634)

        self.output_path = Path(str(self.get_parameter("output").value))
        self.duration_s = float(self.get_parameter("duration_s").value)
        self.settle_duration_s = float(self.get_parameter("settle_duration_s").value)
        self.publish_period_s = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.wall_timeout_s = float(self.get_parameter("wall_timeout_s").value)
        self.angular_test_radps = float(self.get_parameter("angular_test_radps").value)
        self.linear_test_mps = float(self.get_parameter("linear_test_mps").value)
        self.wheel_radius_m = float(self.get_parameter("wheel_radius_m").value)
        self.wheel_distance_m = float(self.get_parameter("wheel_distance_m").value)

        self.cmd_publisher = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("wheel_debug_topic").value),
            self.wheel_debug_callback,
            10,
        )

        self.latest_odom: dict[str, float] | None = None
        self.odom_history: list[dict[str, float]] = []
        self.command_history: list[dict[str, float]] = []
        self.wheel_debug_history: list[dict[str, Any]] = []

    def odom_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s <= 0.0:
            return
        pose = msg.pose.pose
        twist = msg.twist.twist
        record = {
            "stamp_s": stamp_s,
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": yaw_from_quaternion(pose.orientation),
            "linear_x": float(twist.linear.x),
            "linear_y": float(twist.linear.y),
            "yaw_rate": float(twist.angular.z),
        }
        self.latest_odom = record
        self.odom_history.append(record)

    def wheel_debug_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring malformed wheel actuation debug JSON")
            return
        if isinstance(data, dict):
            data.setdefault("receive_stamp_s", self.latest_odom["stamp_s"] if self.latest_odom is not None else None)
            self.wheel_debug_history.append(data)

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_publisher.publish(msg)
        if self.latest_odom is not None:
            self.command_history.append(
                {
                    "stamp_s": self.latest_odom["stamp_s"],
                    "linear_x": float(linear_x),
                    "angular_z": float(angular_z),
                }
            )

    def wait_for_odom(self) -> None:
        start_wall_s = time.monotonic()
        while rclpy.ok() and self.latest_odom is None:
            if time.monotonic() - start_wall_s > self.wall_timeout_s:
                raise TimeoutError("Timed out waiting for odom before actuation smoke test.")
            rclpy.spin_once(self, timeout_sec=0.1)

    def run_for_sim_time(self, linear_x: float, angular_z: float, duration_s: float) -> list[dict[str, float]]:
        self.wait_for_odom()
        start_stamp_s = float(self.latest_odom["stamp_s"])
        segment_odom: list[dict[str, float]] = []
        start_wall_s = time.monotonic()
        next_publish_s = 0.0
        while rclpy.ok():
            if time.monotonic() - start_wall_s > self.wall_timeout_s:
                raise TimeoutError(
                    f"Timed out while waiting for {duration_s:.2f}s sim time; "
                    f"observed {self.latest_odom['stamp_s'] - start_stamp_s:.2f}s."
                )
            now = time.monotonic()
            if now >= next_publish_s:
                self.publish_cmd(linear_x, angular_z)
                next_publish_s = now + self.publish_period_s
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_odom is None:
                continue
            if self.latest_odom["stamp_s"] >= start_stamp_s:
                segment_odom.append(dict(self.latest_odom))
            if self.latest_odom["stamp_s"] - start_stamp_s >= duration_s:
                break
        self.publish_cmd(0.0, 0.0)
        return segment_odom

    def settle(self) -> None:
        if self.settle_duration_s > 0.0:
            self.run_for_sim_time(0.0, 0.0, self.settle_duration_s)

    def summarize_segment(
        self,
        name: str,
        commanded_linear_x: float,
        commanded_angular_z: float,
        odom_records: list[dict[str, float]],
    ) -> dict[str, Any]:
        if len(odom_records) < 2:
            return {
                "name": name,
                "valid": False,
                "reason": "fewer than two odom records",
                "commanded_linear_x": commanded_linear_x,
                "commanded_angular_z": commanded_angular_z,
            }

        first = odom_records[0]
        last = odom_records[-1]
        sim_duration_s = max(0.0, last["stamp_s"] - first["stamp_s"])
        yaw_change = unwrap_delta(last["yaw"], first["yaw"])
        distance_m = 0.0
        for prev, curr in zip(odom_records, odom_records[1:]):
            distance_m += math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])

        yaw_rates = [record["yaw_rate"] for record in odom_records]
        linear_x_values = [record["linear_x"] for record in odom_records]
        wheel_debug = self.summarize_wheel_debug(
            commanded_linear_x,
            commanded_angular_z,
            first["stamp_s"],
            last["stamp_s"],
        )
        mean_cmd_angular = commanded_angular_z
        mean_cmd_linear = commanded_linear_x
        mean_odom_yaw_rate = mean(yaw_rates)
        mean_odom_linear_x = mean(linear_x_values)

        return {
            "name": name,
            "valid": True,
            "commanded_linear_x": commanded_linear_x,
            "commanded_angular_z": commanded_angular_z,
            "sim_duration_s": sim_duration_s,
            "initial_x": first["x"],
            "initial_y": first["y"],
            "initial_yaw": first["yaw"],
            "final_x": last["x"],
            "final_y": last["y"],
            "final_yaw": last["yaw"],
            "yaw_change": yaw_change,
            "distance_m": distance_m,
            "mean_commanded_linear_x": mean_cmd_linear,
            "mean_commanded_angular_z": mean_cmd_angular,
            "mean_odom_linear_x": mean_odom_linear_x,
            "max_odom_linear_x": max(linear_x_values),
            "mean_odom_yaw_rate": mean_odom_yaw_rate,
            "max_abs_odom_yaw_rate": max(abs(value) for value in yaw_rates),
            "angular_response_ratio": (
                mean_odom_yaw_rate / mean_cmd_angular if abs(mean_cmd_angular) > 1e-9 else None
            ),
            "linear_response_ratio": (
                mean_odom_linear_x / mean_cmd_linear if abs(mean_cmd_linear) > 1e-9 else None
            ),
            "wheel_actuation": wheel_debug,
            "odom_messages": len(odom_records),
        }

    def expected_wheel_targets(self, linear_x: float, angular_z: float) -> dict[str, float]:
        left = (linear_x - angular_z * self.wheel_distance_m * 0.5) / self.wheel_radius_m
        right = (linear_x + angular_z * self.wheel_distance_m * 0.5) / self.wheel_radius_m
        return {
            "lf_wheel_joint": left,
            "lb_wheel_joint": left,
            "rf_wheel_joint": right,
            "rb_wheel_joint": right,
        }

    def summarize_wheel_debug(
        self,
        commanded_linear_x: float,
        commanded_angular_z: float,
        start_stamp_s: float,
        end_stamp_s: float,
    ) -> dict[str, Any]:
        samples = [
            sample
            for sample in self.wheel_debug_history
            if sample.get("receive_stamp_s") is not None
            and start_stamp_s <= float(sample["receive_stamp_s"]) <= end_stamp_s
        ]
        expected = self.expected_wheel_targets(commanded_linear_x, commanded_angular_z)
        wheel_names = ["lf_wheel_joint", "lb_wheel_joint", "rf_wheel_joint", "rb_wheel_joint"]
        wheels: dict[str, Any] = {}
        for wheel_name in wheel_names:
            measured_values = []
            drive_targets = []
            commanded_values = []
            for sample in samples:
                wheel = (sample.get("wheels") or {}).get(wheel_name) or {}
                measured = wheel.get("measured_velocity_radps")
                drive_target = wheel.get("drive_target_velocity")
                commanded = wheel.get("commanded_velocity_radps")
                if measured is not None:
                    measured_values.append(float(measured))
                if drive_target is not None:
                    drive_targets.append(float(drive_target))
                if commanded is not None:
                    commanded_values.append(float(commanded))
            expected_target = expected[wheel_name]
            mean_measured = mean(measured_values)
            mean_drive_target = mean(drive_targets)
            mean_commanded = mean(commanded_values) if commanded_values else expected_target
            wheels[wheel_name] = {
                "expected_commanded_velocity_radps": expected_target,
                "mean_reported_commanded_velocity_radps": mean_commanded,
                "mean_measured_velocity_radps": mean_measured,
                "max_abs_measured_velocity_radps": max([abs(value) for value in measured_values], default=0.0),
                "mean_drive_target_velocity": mean_drive_target if drive_targets else None,
                "measured_to_expected_ratio": (
                    mean_measured / expected_target if abs(expected_target) > 1e-9 and measured_values else None
                ),
                "drive_target_to_expected_ratio": (
                    mean_drive_target / expected_target if abs(expected_target) > 1e-9 and drive_targets else None
                ),
                "samples": len(measured_values),
            }
        return {
            "wheel_debug_topic": str(self.get_parameter("wheel_debug_topic").value),
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_distance_m": self.wheel_distance_m,
            "samples": len(samples),
            "wheels": wheels,
        }

    def run_tests(self) -> dict[str, Any]:
        self.wait_for_odom()
        tests = [
            ("turn_left", 0.0, self.angular_test_radps),
            ("turn_right", 0.0, -self.angular_test_radps),
            ("straight", self.linear_test_mps, 0.0),
        ]
        results = []
        for name, linear_x, angular_z in tests:
            self.get_logger().info(
                f"Running {name}: linear.x={linear_x:.3f}, angular.z={angular_z:.3f} "
                f"for {self.duration_s:.2f}s sim time"
            )
            odom_records = self.run_for_sim_time(linear_x, angular_z, self.duration_s)
            results.append(self.summarize_segment(name, linear_x, angular_z, odom_records))
            self.settle()

        self.publish_cmd(0.0, 0.0)
        return {
            "odom_topic": str(self.get_parameter("odom_topic").value),
            "cmd_vel_topic": str(self.get_parameter("cmd_vel_topic").value),
            "duration_s": self.duration_s,
            "settle_duration_s": self.settle_duration_s,
            "publish_rate_hz": 1.0 / self.publish_period_s,
            "wheel_debug_topic": str(self.get_parameter("wheel_debug_topic").value),
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_distance_m": self.wheel_distance_m,
            "wheel_debug_messages": len(self.wheel_debug_history),
            "tests": results,
        }


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ActuationSmokeTest()
    try:
        results = node.run_tests()
        node.output_path.parent.mkdir(parents=True, exist_ok=True)
        with node.output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        node.get_logger().info(f"Wrote actuation smoke test results to {node.output_path}")
        return 0
    except Exception as exc:
        node.get_logger().error(str(exc))
        return 1
    finally:
        try:
            node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

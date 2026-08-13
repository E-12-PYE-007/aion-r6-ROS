#!/usr/bin/env python3
"""Record rollout topic diagnostics beside collected sim data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def yaw_from_quaternion(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rate_hz(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    duration = times[-1] - times[0]
    if duration <= 1e-9:
        return 0.0
    return (len(times) - 1) / duration


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


class RolloutDiagnosticsNode(Node):
    def __init__(self) -> None:
        super().__init__("rollout_diagnostics")
        self.declare_parameter("output_dir", Parameter.Type.STRING)
        self.declare_parameter("odom_topic", "/sim_odom")
        self.declare_parameter("camera_topic", "/vla/cam")
        self.declare_parameter("action_chunk_topic", "/vla/action_chunk")
        self.declare_parameter("expert_cmd_vel_topic", "/expert/cmd_vel")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("sample_frequency_hz", 5.0)

        output_dir = str(self.get_parameter("output_dir").value)
        if not output_dir:
            raise RuntimeError("output_dir parameter is required.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "diagnostics.csv"
        self.summary_path = self.output_dir / "diagnostics_summary.json"

        self.start_time = time.monotonic()
        self.finalized = False

        self.times: dict[str, list[float]] = {
            "odom": [],
            "camera": [],
            "action_chunk": [],
            "expert_cmd_vel": [],
            "cmd_vel": [],
        }
        self.latest: dict[str, Any] = {
            "odom": {},
            "action_chunk": {},
            "expert_cmd_vel": {},
            "cmd_vel": {},
        }
        self.odom_distance_m = 0.0
        self.previous_odom_xy: tuple[float, float] | None = None
        self.expert_linear_x_values: list[float] = []
        self.expert_angular_z_values: list[float] = []
        self.cmd_linear_x_values: list[float] = []
        self.cmd_angular_z_values: list[float] = []
        self.action_first_x_values: list[float] = []
        self.action_last_x_values: list[float] = []
        self.action_negative_x_count = 0
        self.action_pose_count = 0
        self.max_abs_action_y = 0.0
        self.max_abs_action_theta = 0.0

        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields())
        self.csv_writer.writeheader()

        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("camera_topic").value),
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ActionChunk,
            str(self.get_parameter("action_chunk_topic").value),
            self.action_chunk_callback,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("expert_cmd_vel_topic").value),
            self.expert_cmd_vel_callback,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            self.cmd_vel_callback,
            10,
        )

        sample_frequency_hz = float(self.get_parameter("sample_frequency_hz").value)
        period_s = 1.0 / sample_frequency_hz if sample_frequency_hz > 0.0 else 0.2
        self.create_timer(period_s, self.write_row)
        self.get_logger().info(f"Writing rollout diagnostics to {self.output_dir}")

    @staticmethod
    def csv_fields() -> list[str]:
        return [
            "elapsed_s",
            "odom_x",
            "odom_y",
            "odom_yaw",
            "odom_linear_x",
            "odom_linear_y",
            "odom_yaw_rate",
            "odom_distance_m",
            "action_seq_num",
            "action_first_x",
            "action_first_y",
            "action_first_theta",
            "action_last_x",
            "action_last_y",
            "action_last_theta",
            "action_min_x",
            "action_max_x",
            "action_max_abs_y",
            "action_max_abs_theta",
            "action_negative_x_count",
            "expert_linear_x",
            "expert_angular_z",
            "cmd_linear_x",
            "cmd_angular_z",
            "odom_messages",
            "camera_messages",
            "action_chunk_messages",
            "expert_cmd_messages",
            "cmd_vel_messages",
        ]

    def now(self) -> float:
        return time.monotonic()

    def odom_callback(self, msg: Odometry) -> None:
        now = self.now()
        self.times["odom"].append(now)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if self.previous_odom_xy is not None:
            self.odom_distance_m += math.hypot(x - self.previous_odom_xy[0], y - self.previous_odom_xy[1])
        self.previous_odom_xy = (x, y)
        self.latest["odom"] = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "linear_x": float(msg.twist.twist.linear.x),
            "linear_y": float(msg.twist.twist.linear.y),
            "yaw_rate": float(msg.twist.twist.angular.z),
        }

    def camera_callback(self, _msg: Image) -> None:
        self.times["camera"].append(self.now())

    def action_chunk_callback(self, msg: ActionChunk) -> None:
        self.times["action_chunk"].append(self.now())
        poses = list(msg.relative_poses)
        if poses:
            xs = [float(pose.x) for pose in poses]
            ys = [float(pose.y) for pose in poses]
            thetas = [float(pose.theta) for pose in poses]
            self.action_first_x_values.append(xs[0])
            self.action_last_x_values.append(xs[-1])
            self.action_negative_x_count += sum(1 for value in xs if value < 0.0)
            self.action_pose_count += len(xs)
            self.max_abs_action_y = max(self.max_abs_action_y, max(abs(value) for value in ys))
            self.max_abs_action_theta = max(self.max_abs_action_theta, max(abs(value) for value in thetas))
            self.latest["action_chunk"] = {
                "seq_num": int(msg.seq_num),
                "first": [xs[0], ys[0], thetas[0]],
                "last": [xs[-1], ys[-1], thetas[-1]],
                "min_x": min(xs),
                "max_x": max(xs),
                "max_abs_y": max(abs(value) for value in ys),
                "max_abs_theta": max(abs(value) for value in thetas),
                "negative_x_count": sum(1 for value in xs if value < 0.0),
            }

    def expert_cmd_vel_callback(self, msg: Twist) -> None:
        self.times["expert_cmd_vel"].append(self.now())
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        self.expert_linear_x_values.append(linear_x)
        self.expert_angular_z_values.append(angular_z)
        self.latest["expert_cmd_vel"] = {"linear_x": linear_x, "angular_z": angular_z}

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.times["cmd_vel"].append(self.now())
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        self.cmd_linear_x_values.append(linear_x)
        self.cmd_angular_z_values.append(angular_z)
        self.latest["cmd_vel"] = {"linear_x": linear_x, "angular_z": angular_z}

    def write_row(self) -> None:
        odom = self.latest.get("odom") or {}
        action = self.latest.get("action_chunk") or {}
        first = action.get("first") or [None, None, None]
        last = action.get("last") or [None, None, None]
        expert = self.latest.get("expert_cmd_vel") or {}
        cmd = self.latest.get("cmd_vel") or {}
        row = {
            "elapsed_s": self.now() - self.start_time,
            "odom_x": finite(odom.get("x")),
            "odom_y": finite(odom.get("y")),
            "odom_yaw": finite(odom.get("yaw")),
            "odom_linear_x": finite(odom.get("linear_x")),
            "odom_linear_y": finite(odom.get("linear_y")),
            "odom_yaw_rate": finite(odom.get("yaw_rate")),
            "odom_distance_m": self.odom_distance_m,
            "action_seq_num": action.get("seq_num"),
            "action_first_x": finite(first[0]),
            "action_first_y": finite(first[1]),
            "action_first_theta": finite(first[2]),
            "action_last_x": finite(last[0]),
            "action_last_y": finite(last[1]),
            "action_last_theta": finite(last[2]),
            "action_min_x": finite(action.get("min_x")),
            "action_max_x": finite(action.get("max_x")),
            "action_max_abs_y": finite(action.get("max_abs_y")),
            "action_max_abs_theta": finite(action.get("max_abs_theta")),
            "action_negative_x_count": action.get("negative_x_count"),
            "expert_linear_x": finite(expert.get("linear_x")),
            "expert_angular_z": finite(expert.get("angular_z")),
            "cmd_linear_x": finite(cmd.get("linear_x")),
            "cmd_angular_z": finite(cmd.get("angular_z")),
            "odom_messages": len(self.times["odom"]),
            "camera_messages": len(self.times["camera"]),
            "action_chunk_messages": len(self.times["action_chunk"]),
            "expert_cmd_messages": len(self.times["expert_cmd_vel"]),
            "cmd_vel_messages": len(self.times["cmd_vel"]),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def summary(self) -> dict[str, Any]:
        duration_s = max(self.now() - self.start_time, 0.0)
        action_negative_fraction = (
            self.action_negative_x_count / self.action_pose_count if self.action_pose_count else 0.0
        )
        return {
            "duration_s": duration_s,
            "diagnostics_csv": self.csv_path.as_posix(),
            "messages": {
                "odom": len(self.times["odom"]),
                "camera": len(self.times["camera"]),
                "action_chunk": len(self.times["action_chunk"]),
                "expert_cmd_vel": len(self.times["expert_cmd_vel"]),
                "cmd_vel": len(self.times["cmd_vel"]),
            },
            "rates_hz": {
                "odom": rate_hz(self.times["odom"]),
                "camera": rate_hz(self.times["camera"]),
                "action_chunk": rate_hz(self.times["action_chunk"]),
                "expert_cmd_vel": rate_hz(self.times["expert_cmd_vel"]),
                "cmd_vel": rate_hz(self.times["cmd_vel"]),
            },
            "odom": {
                "distance_travelled_m": self.odom_distance_m,
                **(self.latest.get("odom") or {}),
            },
            "commands": {
                "mean_expert_linear_x": mean(self.expert_linear_x_values),
                "max_expert_linear_x": max(self.expert_linear_x_values) if self.expert_linear_x_values else 0.0,
                "mean_abs_expert_angular_z": mean([abs(value) for value in self.expert_angular_z_values]),
                "max_abs_expert_angular_z": max([abs(value) for value in self.expert_angular_z_values], default=0.0),
                "mean_cmd_linear_x": mean(self.cmd_linear_x_values),
                "max_cmd_linear_x": max(self.cmd_linear_x_values) if self.cmd_linear_x_values else 0.0,
                "mean_abs_cmd_angular_z": mean([abs(value) for value in self.cmd_angular_z_values]),
                "max_abs_cmd_angular_z": max([abs(value) for value in self.cmd_angular_z_values], default=0.0),
            },
            "action_chunk": {
                "first_x_mean": mean(self.action_first_x_values),
                "last_x_mean": mean(self.action_last_x_values),
                "negative_x_fraction": action_negative_fraction,
                "max_abs_y": self.max_abs_action_y,
                "max_abs_theta": self.max_abs_action_theta,
                **(self.latest.get("action_chunk") or {}),
            },
        }

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        try:
            self.write_row()
        except Exception:
            pass
        self.csv_file.close()
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, sort_keys=True)
        self.get_logger().info(f"Wrote diagnostics summary to {self.summary_path}")

    def destroy_node(self) -> bool:
        self.finalize()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RolloutDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

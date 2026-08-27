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
from std_msgs.msg import String


def yaw_from_quaternion(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
        self.declare_parameter("frame_debug_topic", "/expert/frame_debug")
        self.declare_parameter("isaac_pose_debug_topic", "/isaac/scene_pose_debug")
        self.declare_parameter("sample_frequency_hz", 5.0)

        output_dir = str(self.get_parameter("output_dir").value)
        if not output_dir:
            raise RuntimeError("output_dir parameter is required.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "diagnostics.csv"
        self.summary_path = self.output_dir / "diagnostics_summary.json"

        self.start_time = time.monotonic()
        self.initial_odom_stamp_s: float | None = None
        self.latest_odom_stamp_s: float | None = None
        self.finalized = False

        self.times: dict[str, list[float]] = {
            "odom": [],
            "camera": [],
            "action_chunk": [],
            "expert_cmd_vel": [],
            "cmd_vel": [],
            "frame_debug": [],
            "isaac_pose_debug": [],
        }
        self.stamps: dict[str, list[float]] = {
            "odom": [],
            "camera": [],
            "action_chunk": [],
        }
        self.latest: dict[str, Any] = {
            "odom": {},
            "action_chunk": {},
            "expert_cmd_vel": {},
            "cmd_vel": {},
            "frame_debug": {},
            "isaac_pose_debug": {},
        }
        self.odom_distance_m = 0.0
        self.previous_odom_xy: tuple[float, float] | None = None
        self.expert_linear_x_values: list[float] = []
        self.expert_angular_z_values: list[float] = []
        self.cmd_linear_x_values: list[float] = []
        self.cmd_angular_z_values: list[float] = []
        self.action_first_x_values: list[float] = []
        self.action_first_y_values: list[float] = []
        self.action_last_x_values: list[float] = []
        self.action_chunk_age_values: list[float] = []
        self.action_negative_x_count = 0
        self.action_pose_count = 0
        self.max_abs_action_y = 0.0
        self.max_abs_action_theta = 0.0
        self.isaac_min_fence_distance_m: float | None = None
        self.max_odom_isaac_position_error_m = 0.0

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
        self.create_subscription(
            String,
            str(self.get_parameter("frame_debug_topic").value),
            self.frame_debug_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("isaac_pose_debug_topic").value),
            self.isaac_pose_debug_callback,
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
            "wall_elapsed_s",
            "odom_stamp_s",
            "sim_elapsed_s",
            "real_time_factor",
            "camera_stamp_s",
            "action_stamp_s",
            "action_chunk_age_s",
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
            "pose_source",
            "raw_odom_x",
            "raw_odom_y",
            "raw_odom_yaw",
            "flip_isaac_y",
            "flip_scene_y",
            "flip_runtime_odom_y",
            "flip_runtime_odom_yaw",
            "local_x_after_flip",
            "local_y_after_flip",
            "local_yaw_after_flip",
            "world_x",
            "world_y",
            "world_yaw",
            "isaac_camera_initial_x",
            "isaac_camera_initial_y",
            "isaac_camera_initial_yaw",
            "isaac_camera_delta_x",
            "isaac_camera_delta_y",
            "current_world_x",
            "current_world_y",
            "current_world_yaw",
            "target_world_x",
            "target_world_y",
            "target_world_yaw",
            "target_path_distance_m",
            "target_source",
            "current_path_progress_m",
            "tracking_error_m",
            "off_path",
            "delta_world_x",
            "delta_world_y",
            "relative_x",
            "relative_y",
            "relative_theta",
            "isaac_rover_x",
            "isaac_rover_y",
            "isaac_rover_z",
            "isaac_rover_yaw",
            "isaac_planner_x",
            "isaac_planner_y",
            "isaac_planner_yaw",
            "isaac_camera_x",
            "isaac_camera_y",
            "isaac_camera_z",
            "isaac_camera_yaw",
            "isaac_rover_min_fence_distance_m",
            "odom_isaac_position_error_m",
            "odom_messages",
            "camera_messages",
            "action_chunk_messages",
            "expert_cmd_messages",
            "cmd_vel_messages",
            "frame_debug_messages",
            "isaac_pose_debug_messages",
        ]

    def now(self) -> float:
        return time.monotonic()

    def odom_callback(self, msg: Odometry) -> None:
        now = self.now()
        self.times["odom"].append(now)
        odom_stamp_s = stamp_to_seconds(msg.header.stamp)
        if odom_stamp_s > 0.0:
            self.stamps["odom"].append(odom_stamp_s)
            self.latest_odom_stamp_s = odom_stamp_s
            if self.initial_odom_stamp_s is None:
                self.initial_odom_stamp_s = odom_stamp_s
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
        stamp_s = stamp_to_seconds(_msg.header.stamp)
        if stamp_s > 0.0:
            self.stamps["camera"].append(stamp_s)

    def action_chunk_callback(self, msg: ActionChunk) -> None:
        self.times["action_chunk"].append(self.now())
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s > 0.0:
            self.stamps["action_chunk"].append(stamp_s)
        poses = list(msg.relative_poses)
        if poses:
            xs = [float(pose.x) for pose in poses]
            ys = [float(pose.y) for pose in poses]
            thetas = [float(pose.theta) for pose in poses]
            self.action_first_x_values.append(xs[0])
            self.action_first_y_values.append(ys[0])
            self.action_last_x_values.append(xs[-1])
            self.action_negative_x_count += sum(1 for value in xs if value < 0.0)
            self.action_pose_count += len(xs)
            self.max_abs_action_y = max(self.max_abs_action_y, max(abs(value) for value in ys))
            self.max_abs_action_theta = max(self.max_abs_action_theta, max(abs(value) for value in thetas))
            self.latest["action_chunk"] = {
                "stamp_s": stamp_s if stamp_s > 0.0 else None,
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

    def frame_debug_callback(self, msg: String) -> None:
        self.times["frame_debug"].append(self.now())
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring malformed /expert/frame_debug JSON")
            return
        if isinstance(data, dict):
            self.latest["frame_debug"] = data

    def isaac_pose_debug_callback(self, msg: String) -> None:
        self.times["isaac_pose_debug"].append(self.now())
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring malformed /isaac/scene_pose_debug JSON")
            return
        if not isinstance(data, dict):
            return
        self.latest["isaac_pose_debug"] = data
        fence_distance = data.get("rover_min_fence_distance_m")
        if isinstance(fence_distance, (float, int)) and math.isfinite(float(fence_distance)):
            value = float(fence_distance)
            self.isaac_min_fence_distance_m = (
                value
                if self.isaac_min_fence_distance_m is None
                else min(self.isaac_min_fence_distance_m, value)
            )

    def write_row(self) -> None:
        wall_elapsed_s = self.now() - self.start_time
        sim_elapsed_s = None
        if self.initial_odom_stamp_s is not None and self.latest_odom_stamp_s is not None:
            sim_elapsed_s = max(0.0, self.latest_odom_stamp_s - self.initial_odom_stamp_s)
        real_time_factor = (
            sim_elapsed_s / wall_elapsed_s
            if sim_elapsed_s is not None and wall_elapsed_s > 1e-9
            else None
        )
        odom = self.latest.get("odom") or {}
        action = self.latest.get("action_chunk") or {}
        first = action.get("first") or [None, None, None]
        last = action.get("last") or [None, None, None]
        expert = self.latest.get("expert_cmd_vel") or {}
        cmd = self.latest.get("cmd_vel") or {}
        frame = self.latest.get("frame_debug") or {}
        isaac_debug = self.latest.get("isaac_pose_debug") or {}
        isaac_rover = isaac_debug.get("rover_world_pose") or {}
        isaac_planner = isaac_debug.get("rover_planner_pose_flip_y") or {}
        isaac_camera = isaac_debug.get("camera_world_pose") or {}
        odom_isaac_position_error_m = None
        if (
            isinstance(isaac_planner, dict)
            and frame.get("world_x") is not None
            and frame.get("world_y") is not None
            and isaac_planner.get("x") is not None
            and isaac_planner.get("y") is not None
        ):
            odom_isaac_position_error_m = math.hypot(
                float(frame["world_x"]) - float(isaac_planner["x"]),
                float(frame["world_y"]) - float(isaac_planner["y"]),
            )
            self.max_odom_isaac_position_error_m = max(
                self.max_odom_isaac_position_error_m,
                odom_isaac_position_error_m,
            )
        camera_stamp_s = self.stamps["camera"][-1] if self.stamps["camera"] else None
        action_stamp_s = action.get("stamp_s")
        action_chunk_age_s = None
        if camera_stamp_s is not None and action_stamp_s is not None:
            action_chunk_age_s = float(camera_stamp_s) - float(action_stamp_s)
            self.action_chunk_age_values.append(action_chunk_age_s)
        row = {
            "elapsed_s": wall_elapsed_s,
            "wall_elapsed_s": wall_elapsed_s,
            "odom_stamp_s": self.latest_odom_stamp_s,
            "sim_elapsed_s": sim_elapsed_s,
            "real_time_factor": real_time_factor,
            "camera_stamp_s": camera_stamp_s,
            "action_stamp_s": action_stamp_s,
            "action_chunk_age_s": finite(action_chunk_age_s),
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
            "pose_source": frame.get("pose_source"),
            "raw_odom_x": finite(frame.get("raw_odom_x")),
            "raw_odom_y": finite(frame.get("raw_odom_y")),
            "raw_odom_yaw": finite(frame.get("raw_odom_yaw")),
            "flip_isaac_y": frame.get("flip_isaac_y"),
            "flip_scene_y": frame.get("flip_scene_y"),
            "flip_runtime_odom_y": frame.get("flip_runtime_odom_y"),
            "flip_runtime_odom_yaw": frame.get("flip_runtime_odom_yaw"),
            "local_x_after_flip": finite(frame.get("local_x_after_flip")),
            "local_y_after_flip": finite(frame.get("local_y_after_flip")),
            "local_yaw_after_flip": finite(frame.get("local_yaw_after_flip")),
            "world_x": finite(frame.get("world_x")),
            "world_y": finite(frame.get("world_y")),
            "world_yaw": finite(frame.get("world_yaw")),
            "isaac_camera_initial_x": finite(frame.get("isaac_camera_initial_x")),
            "isaac_camera_initial_y": finite(frame.get("isaac_camera_initial_y")),
            "isaac_camera_initial_yaw": finite(frame.get("isaac_camera_initial_yaw")),
            "isaac_camera_delta_x": finite(frame.get("isaac_camera_delta_x")),
            "isaac_camera_delta_y": finite(frame.get("isaac_camera_delta_y")),
            "current_world_x": finite(frame.get("current_world_x")),
            "current_world_y": finite(frame.get("current_world_y")),
            "current_world_yaw": finite(frame.get("current_world_yaw")),
            "target_world_x": finite(frame.get("target_world_x")),
            "target_world_y": finite(frame.get("target_world_y")),
            "target_world_yaw": finite(frame.get("target_world_yaw")),
            "target_path_distance_m": finite(frame.get("target_path_distance_m")),
            "target_source": frame.get("target_source"),
            "current_path_progress_m": finite(frame.get("current_path_progress_m")),
            "tracking_error_m": finite(frame.get("tracking_error_m")),
            "off_path": frame.get("off_path"),
            "delta_world_x": finite(frame.get("delta_world_x")),
            "delta_world_y": finite(frame.get("delta_world_y")),
            "relative_x": finite(frame.get("relative_x")),
            "relative_y": finite(frame.get("relative_y")),
            "relative_theta": finite(frame.get("relative_theta")),
            "isaac_rover_x": finite(isaac_rover.get("x") if isinstance(isaac_rover, dict) else None),
            "isaac_rover_y": finite(isaac_rover.get("y") if isinstance(isaac_rover, dict) else None),
            "isaac_rover_z": finite(isaac_rover.get("z") if isinstance(isaac_rover, dict) else None),
            "isaac_rover_yaw": finite(isaac_rover.get("yaw") if isinstance(isaac_rover, dict) else None),
            "isaac_planner_x": finite(isaac_planner.get("x") if isinstance(isaac_planner, dict) else None),
            "isaac_planner_y": finite(isaac_planner.get("y") if isinstance(isaac_planner, dict) else None),
            "isaac_planner_yaw": finite(isaac_planner.get("yaw") if isinstance(isaac_planner, dict) else None),
            "isaac_camera_x": finite(isaac_camera.get("x") if isinstance(isaac_camera, dict) else None),
            "isaac_camera_y": finite(isaac_camera.get("y") if isinstance(isaac_camera, dict) else None),
            "isaac_camera_z": finite(isaac_camera.get("z") if isinstance(isaac_camera, dict) else None),
            "isaac_camera_yaw": finite(isaac_camera.get("yaw") if isinstance(isaac_camera, dict) else None),
            "isaac_rover_min_fence_distance_m": finite(isaac_debug.get("rover_min_fence_distance_m")),
            "odom_isaac_position_error_m": finite(odom_isaac_position_error_m),
            "odom_messages": len(self.times["odom"]),
            "camera_messages": len(self.times["camera"]),
            "action_chunk_messages": len(self.times["action_chunk"]),
            "expert_cmd_messages": len(self.times["expert_cmd_vel"]),
            "cmd_vel_messages": len(self.times["cmd_vel"]),
            "frame_debug_messages": len(self.times["frame_debug"]),
            "isaac_pose_debug_messages": len(self.times["isaac_pose_debug"]),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def summary(self) -> dict[str, Any]:
        duration_s = max(self.now() - self.start_time, 0.0)
        sim_elapsed_s = 0.0
        if self.initial_odom_stamp_s is not None and self.latest_odom_stamp_s is not None:
            sim_elapsed_s = max(0.0, self.latest_odom_stamp_s - self.initial_odom_stamp_s)
        real_time_factor = sim_elapsed_s / duration_s if duration_s > 1e-9 else 0.0
        action_negative_fraction = (
            self.action_negative_x_count / self.action_pose_count if self.action_pose_count else 0.0
        )
        return {
            "wall_duration_s": duration_s,
            "sim_duration_s": sim_elapsed_s,
            "real_time_factor": real_time_factor,
            "diagnostics_csv": self.csv_path.as_posix(),
            "messages": {
                "odom": len(self.times["odom"]),
                "camera": len(self.times["camera"]),
                "action_chunk": len(self.times["action_chunk"]),
                "expert_cmd_vel": len(self.times["expert_cmd_vel"]),
                "cmd_vel": len(self.times["cmd_vel"]),
                "frame_debug": len(self.times["frame_debug"]),
                "isaac_pose_debug": len(self.times["isaac_pose_debug"]),
            },
            "rates_hz": {
                "odom": rate_hz(self.times["odom"]),
                "camera": rate_hz(self.times["camera"]),
                "action_chunk": rate_hz(self.times["action_chunk"]),
                "expert_cmd_vel": rate_hz(self.times["expert_cmd_vel"]),
                "cmd_vel": rate_hz(self.times["cmd_vel"]),
                "frame_debug": rate_hz(self.times["frame_debug"]),
                "isaac_pose_debug": rate_hz(self.times["isaac_pose_debug"]),
            },
            "sim_time_rates_hz": {
                "odom": rate_hz(self.stamps["odom"]),
                "camera": rate_hz(self.stamps["camera"]),
                "action_chunk": rate_hz(self.stamps["action_chunk"]),
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
                "first_y_mean": mean(self.action_first_y_values),
                "mean_abs_first_y": mean([abs(value) for value in self.action_first_y_values]),
                "max_abs_first_y": max([abs(value) for value in self.action_first_y_values], default=0.0),
                "last_x_mean": mean(self.action_last_x_values),
                "age_s_mean": mean(self.action_chunk_age_values),
                "age_s_max": max(self.action_chunk_age_values, default=0.0),
                "negative_x_fraction": action_negative_fraction,
                "max_abs_y": self.max_abs_action_y,
                "max_abs_theta": self.max_abs_action_theta,
                **(self.latest.get("action_chunk") or {}),
            },
            "frame_debug": self.latest.get("frame_debug") or {},
            "isaac_pose_debug": {
                "min_rover_fence_distance_m": self.isaac_min_fence_distance_m,
                "max_odom_isaac_position_error_m": self.max_odom_isaac_position_error_m,
                **(self.latest.get("isaac_pose_debug") or {}),
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

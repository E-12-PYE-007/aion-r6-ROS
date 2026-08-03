#!/usr/bin/env python3
"""Publish expert fenceline ActionChunk messages for Isaac Sim rollouts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import yaml
from aion_msgs.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_pose(msg: Odometry, flip_isaac_y: bool) -> tuple[np.ndarray, float]:
    x = float(msg.pose.pose.position.x)
    y = float(msg.pose.pose.position.y)
    yaw = yaw_from_quaternion(msg.pose.pose.orientation)
    if flip_isaac_y:
        return np.asarray([x, -y], dtype=np.float64), -yaw
    return np.asarray([x, y], dtype=np.float64), yaw


def transform_point(point: list[float], flip_isaac_y: bool) -> np.ndarray:
    if flip_isaac_y:
        return np.asarray([float(point[0]), -float(point[1])], dtype=np.float64)
    return np.asarray([float(point[0]), float(point[1])], dtype=np.float64)


def load_fence_polyline(layout_yaml: Path, fence_id: str, flip_isaac_y: bool) -> list[np.ndarray]:
    with layout_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    fences = config.get("fences", [])
    if not fences:
        raise ValueError(f"{layout_yaml} does not contain any fences.")
    selected = [fence for fence in fences if not fence_id or fence.get("name") == fence_id]
    if not selected:
        names = [fence.get("name", "<unnamed>") for fence in fences]
        raise ValueError(f"Fence {fence_id!r} not found in {layout_yaml}. Available: {names}")

    points: list[np.ndarray] = []
    for fence in selected:
        start = transform_point(fence["start"], flip_isaac_y)
        end = transform_point(fence["end"], flip_isaac_y)
        if not points:
            points.append(start)
        elif np.linalg.norm(points[-1] - start) > 1e-6:
            points.append(start)
        points.append(end)
    return points


def segment_lengths(polyline: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.linalg.norm(polyline[i + 1] - polyline[i]) for i in range(len(polyline) - 1)], dtype=np.float64)


def project_progress(polyline: list[np.ndarray], point: np.ndarray) -> float:
    lengths = segment_lengths(polyline)
    best_progress = 0.0
    best_distance = float("inf")
    cumulative = 0.0
    for i, length in enumerate(lengths):
        if length <= 1e-9:
            continue
        start = polyline[i]
        end = polyline[i + 1]
        direction = (end - start) / length
        t = float(np.dot(point - start, direction))
        t = min(max(t, 0.0), float(length))
        closest = start + t * direction
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative + t
        cumulative += float(length)
    return best_progress


def sample_offset_pose(polyline: list[np.ndarray], progress: float, offset_m: float, side: str) -> tuple[np.ndarray, float]:
    lengths = segment_lengths(polyline)
    total = float(np.sum(lengths))
    progress = min(max(progress, 0.0), total)
    cumulative = 0.0
    for i, length in enumerate(lengths):
        if i == len(lengths) - 1 or progress <= cumulative + float(length):
            local = progress - cumulative
            start = polyline[i]
            end = polyline[i + 1]
            direction = (end - start) / max(float(length), 1e-9)
            base = start + local * direction
            left_normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
            normal = left_normal if side == "left" else -left_normal
            position = base + offset_m * normal
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            return position, yaw
        cumulative += float(length)
    direction = polyline[-1] - polyline[-2]
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    return polyline[-1], yaw


def world_to_robot(anchor_position: np.ndarray, anchor_yaw: float, point: np.ndarray, yaw: float) -> tuple[float, float, float]:
    delta = point - anchor_position
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    x_robot = cos_yaw * float(delta[0]) + sin_yaw * float(delta[1])
    y_robot = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
    return x_robot, y_robot, wrap_to_pi(yaw - anchor_yaw)


class FencelineActionChunkPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fenceline_action_chunk_publisher")

        self.declare_parameter("layout_yaml", Parameter.Type.STRING)
        self.declare_parameter("fence_id", "main_fence")
        self.declare_parameter("follow_side", "left")
        self.declare_parameter("travel_direction", "forward")
        self.declare_parameter("preferred_offset_m", 0.8)
        self.declare_parameter("waypoint_spacing_m", 0.18)
        self.declare_parameter("publish_rate_hz", 3.0)
        self.declare_parameter("odom_topic", "sim_odom")
        self.declare_parameter("action_chunk_topic", "/vla/action_chunk")
        self.declare_parameter("flip_isaac_y", True)

        layout_yaml = Path(str(self.get_parameter("layout_yaml").value))
        if not layout_yaml.exists():
            raise RuntimeError(f"layout_yaml parameter must point to an existing YAML file: {layout_yaml}")
        self.flip_isaac_y = bool(self.get_parameter("flip_isaac_y").value)
        self.fence_id = str(self.get_parameter("fence_id").value)
        self.follow_side = str(self.get_parameter("follow_side").value).lower()
        if self.follow_side not in {"left", "right"}:
            raise RuntimeError("follow_side must be 'left' or 'right'.")
        self.travel_direction = str(self.get_parameter("travel_direction").value).lower()
        if self.travel_direction not in {"forward", "reverse"}:
            raise RuntimeError("travel_direction must be 'forward' or 'reverse'.")
        self.preferred_offset_m = float(self.get_parameter("preferred_offset_m").value)
        self.waypoint_spacing_m = float(self.get_parameter("waypoint_spacing_m").value)

        self.polyline = load_fence_polyline(layout_yaml, self.fence_id, self.flip_isaac_y)
        if self.travel_direction == "reverse":
            self.polyline = list(reversed(self.polyline))
        self.path_length_m = float(np.sum(segment_lengths(self.polyline)))

        self.current_position: Optional[np.ndarray] = None
        self.current_yaw: Optional[float] = None
        self.seq_num = 1
        self.publisher = self.create_publisher(ActionChunk, str(self.get_parameter("action_chunk_topic").value), 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_chunk)
        self.get_logger().info(
            f"Publishing expert fenceline chunks from {layout_yaml} fence={self.fence_id} side={self.follow_side}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.current_position, self.current_yaw = odom_to_pose(msg, self.flip_isaac_y)

    def publish_chunk(self) -> None:
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn("No sim odom received yet; not publishing ActionChunk")
            return

        progress = project_progress(self.polyline, self.current_position)
        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.seq_num = self.seq_num

        for index in range(1, len(msg.relative_poses) + 1):
            target_progress = min(progress + index * self.waypoint_spacing_m, self.path_length_m)
            target_position, target_yaw = sample_offset_pose(
                self.polyline,
                target_progress,
                self.preferred_offset_m,
                self.follow_side,
            )
            x, y, theta = world_to_robot(self.current_position, self.current_yaw, target_position, target_yaw)
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(theta)
            msg.relative_poses[index - 1] = pose

        self.publisher.publish(msg)
        self.seq_num += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FencelineActionChunkPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

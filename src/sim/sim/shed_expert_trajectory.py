#!/usr/bin/env python3
"""Task-spec-driven expert trajectory publisher for shed scenes."""

from __future__ import annotations

import math

import numpy as np
import rclpy

from sim.expert_policy_node import ExpertPolicyNode
from sim.expert_trajectory_utils import get_asset_bbox, point2, rotate


class ShedExpertTrajectoryNode(ExpertPolicyNode):
    def __init__(self) -> None:
        self.preferred_offset_m = 0.8
        super().__init__("shed_expert_trajectory")

    def resolve_path(self) -> list[np.ndarray]:
        self.declare_parameter("preferred_offset_m", 0.8)
        self.preferred_offset_m = float(
            self.variant.get("preferred_offset_m", self.get_parameter("preferred_offset_m").value)
        )

        task_type = self.task["task_type"]
        if task_type == "follow_shed_side":
            if self.task.get("shed_side") == "perimeter":
                return self.shed_perimeter_path()
            return self.shed_side_path(self.task.get("shed_side", "nearest"))

        if task_type == "approach_target":
            return [self.start_pose_point(), self.shed_center()]

        if task_type == "stop_at_landmark":
            side = self.task.get("landmark", {}).get("side", self.task.get("shed_side", "nearest"))
            path = self.shed_side_path(side)
            return [path[0], path[len(path) // 2]]

        if task_type == "hold_position":
            start = self.start_pose_point()
            return [start, start + np.asarray([0.001, 0.0], dtype=np.float64)]

        raise RuntimeError(f"Unsupported shed task_type {task_type!r}")

    def shed_center(self) -> np.ndarray:
        shed = self.scene.get("shed", {})
        return point2(shed.get("position", [0.0, 0.0, 0.0]), self.flip_isaac_y)

    def shed_yaw(self) -> float:
        yaw = math.radians(float(self.scene.get("shed", {}).get("yaw", 0.0)))
        if self.flip_isaac_y:
            yaw = -yaw
        return yaw

    def shed_bbox(self) -> list[float]:
        bbox = get_asset_bbox(self.scene, "shed", "shed", self.scene_path)
        if bbox is None:
            raise RuntimeError("Could not resolve shed bbox_size from scene assets or asset library.")
        return bbox

    def shed_side_path(self, side: str) -> list[np.ndarray]:
        if side == "nearest":
            side = self.nearest_shed_side()
        bbox = self.shed_bbox()
        half_x = float(bbox[0]) * 0.5
        half_y = float(bbox[1]) * 0.5
        center = self.shed_center()
        yaw = self.shed_yaw()
        clearance = self.preferred_offset_m

        local_segments = {
            "north": [np.asarray([-half_x, half_y + clearance]), np.asarray([half_x, half_y + clearance])],
            "south": [np.asarray([half_x, -half_y - clearance]), np.asarray([-half_x, -half_y - clearance])],
            "east": [np.asarray([half_x + clearance, half_y]), np.asarray([half_x + clearance, -half_y])],
            "west": [np.asarray([-half_x - clearance, -half_y]), np.asarray([-half_x - clearance, half_y])],
        }
        if side not in local_segments:
            raise RuntimeError(f"shed_side must be north/south/east/west/nearest/perimeter, got {side!r}")
        return [center + rotate(point, yaw) for point in local_segments[side]]

    def shed_perimeter_path(self) -> list[np.ndarray]:
        bbox = self.shed_bbox()
        half_x = float(bbox[0]) * 0.5 + self.preferred_offset_m
        half_y = float(bbox[1]) * 0.5 + self.preferred_offset_m
        center = self.shed_center()
        yaw = self.shed_yaw()
        start = self.start_pose_point()
        local_start = rotate(start - center, -yaw)

        clockwise = [
            np.asarray([half_x, half_y], dtype=np.float64),
            np.asarray([half_x, -half_y], dtype=np.float64),
            np.asarray([-half_x, -half_y], dtype=np.float64),
            np.asarray([-half_x, half_y], dtype=np.float64),
            np.asarray([half_x, half_y], dtype=np.float64),
        ]
        counterclockwise = list(reversed(clockwise))

        def nearest_point_on_segment(point: np.ndarray, start_point: np.ndarray, end_point: np.ndarray) -> np.ndarray:
            segment = end_point - start_point
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-9:
                return start_point
            t = float(np.dot(point - start_point, segment) / length_sq)
            t = min(1.0, max(0.0, t))
            return start_point + segment * t

        def path_from_loop(loop: list[np.ndarray]) -> list[np.ndarray]:
            best_i = 0
            best_point = loop[0]
            best_distance = math.inf
            for i, (a, b) in enumerate(zip(loop, loop[1:])):
                candidate = nearest_point_on_segment(local_start, a, b)
                candidate_distance = float(np.linalg.norm(local_start - candidate))
                if candidate_distance < best_distance:
                    best_i = i
                    best_point = candidate
                    best_distance = candidate_distance
            local_path = [best_point]
            local_path.extend(loop[best_i + 1:])
            local_path.extend(loop[1:best_i + 1])
            local_path.append(best_point)
            world_path = [center + rotate(point, yaw) for point in local_path]
            deduped = [world_path[0]]
            for point in world_path[1:]:
                if float(np.linalg.norm(point - deduped[-1])) > 1e-6:
                    deduped.append(point)
            return deduped

        def heading_error(path: list[np.ndarray]) -> float:
            if len(path) < 2:
                return math.inf
            delta = path[1] - path[0]
            path_yaw = math.atan2(float(delta[1]), float(delta[0]))
            return abs(math.atan2(math.sin(path_yaw - self.world_start_yaw), math.cos(path_yaw - self.world_start_yaw)))

        direction = str(self.task.get("perimeter_direction", "auto"))
        if direction == "clockwise":
            return path_from_loop(clockwise)
        if direction == "counterclockwise":
            return path_from_loop(counterclockwise)

        candidates = [path_from_loop(clockwise), path_from_loop(counterclockwise)]
        return min(candidates, key=heading_error)

    def nearest_shed_side(self) -> str:
        start = self.start_pose_point()
        center = self.shed_center()
        yaw = self.shed_yaw()
        relative = rotate(start - center, -yaw)
        bbox = self.shed_bbox()
        half_x = float(bbox[0]) * 0.5
        half_y = float(bbox[1]) * 0.5
        distances = {
            "east": abs(relative[0] - half_x),
            "west": abs(relative[0] + half_x),
            "north": abs(relative[1] - half_y),
            "south": abs(relative[1] + half_y),
        }
        return min(distances, key=distances.get)

    def start_pose_point(self) -> np.ndarray:
        starts = self.scene.get("rover_poses")
        if isinstance(starts, dict) and self.task.get("start_pose") in starts:
            return point2(starts[self.task["start_pose"]]["position"], self.flip_isaac_y)
        if isinstance(self.scene.get("rover_pose"), dict):
            return point2(self.scene["rover_pose"]["position"], self.flip_isaac_y)
        raise RuntimeError("Scene does not define a usable rover start pose.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ShedExpertTrajectoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

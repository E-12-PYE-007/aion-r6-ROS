#!/usr/bin/env python3
"""Task-spec-driven expert trajectory publisher for road scenes."""

from __future__ import annotations

import numpy as np
import rclpy

from sim.expert_policy_node import ExpertPolicyNode
from sim.expert_trajectory_utils import concat_segments, point2, road_by_name


class RoadExpertTrajectoryNode(ExpertPolicyNode):
    def __init__(self) -> None:
        super().__init__("road_expert_trajectory")

    def resolve_path(self) -> list[np.ndarray]:
        task_type = self.task["task_type"]
        if task_type == "follow_road":
            road = road_by_name(self.scene, self.task["target_road"])
            return concat_segments([road], self.flip_isaac_y)

        if task_type == "follow_and_turn":
            roads = [road_by_name(self.scene, name) for name in self.task["target_roads"]]
            return concat_segments(roads, self.flip_isaac_y)

        if task_type == "approach_target":
            return [self.start_pose_point(), point2(self.task["target_point"], self.flip_isaac_y)]

        if task_type == "stop_at_landmark" and "target_road" in self.task:
            road = road_by_name(self.scene, self.task["target_road"])
            return concat_segments([road], self.flip_isaac_y)

        if task_type == "hold_position":
            start = self.start_pose_point()
            return [start, start + np.asarray([0.001, 0.0], dtype=np.float64)]

        raise RuntimeError(f"Unsupported road task_type {task_type!r}")

    def start_pose_point(self) -> np.ndarray:
        starts = self.scene.get("rover_poses")
        if isinstance(starts, dict) and self.task.get("start_pose") in starts:
            return point2(starts[self.task["start_pose"]]["position"], self.flip_isaac_y)
        if isinstance(self.scene.get("rover_pose"), dict):
            return point2(self.scene["rover_pose"]["position"], self.flip_isaac_y)
        raise RuntimeError("Scene does not define a usable rover start pose.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoadExpertTrajectoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

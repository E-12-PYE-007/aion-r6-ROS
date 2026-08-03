#!/usr/bin/env python3
"""Task-spec-driven expert trajectory publisher for fenceline scenes."""

from __future__ import annotations

import numpy as np
import rclpy

from sim.expert_policy_node import ExpertPolicyNode
from sim.expert_trajectory_utils import (
    concat_segments,
    fence_by_name,
    line_path_from_points,
    offset_polyline,
    point2,
)


class FencelineExpertTrajectoryNode(ExpertPolicyNode):
    def __init__(self) -> None:
        self.preferred_offset_m = 0.8
        super().__init__("fenceline_expert_trajectory")

    def resolve_path(self) -> list[np.ndarray]:
        self.declare_parameter("preferred_offset_m", 0.8)
        self.preferred_offset_m = float(
            self.variant.get("preferred_offset_m", self.get_parameter("preferred_offset_m").value)
        )

        task_type = self.task["task_type"]
        if task_type == "follow_fence":
            fence = fence_by_name(self.scene, self.task["target_fence"])
            base_path = concat_segments([fence], self.flip_isaac_y)
            return offset_polyline(base_path, self.preferred_offset_m, self.task.get("follow_side", "left"))

        if task_type == "follow_and_turn":
            fences = [fence_by_name(self.scene, name) for name in self.task["target_fences"]]
            base_path = concat_segments(fences, self.flip_isaac_y)
            return offset_polyline(base_path, self.preferred_offset_m, self.task.get("follow_side", "left"))

        if task_type == "follow_fence_sequence":
            fences = [fence_by_name(self.scene, name) for name in self.task["target_fences"]]
            base_path = concat_segments(fences, self.flip_isaac_y)
            return offset_polyline(base_path, self.preferred_offset_m, self.task.get("follow_side", "left"))

        if task_type == "follow_corridor":
            left, right = [fence_by_name(self.scene, name) for name in self.task["corridor_fences"]]
            left_path = concat_segments([left], self.flip_isaac_y)
            right_path = concat_segments([right], self.flip_isaac_y)
            return [
                (left_path[0] + right_path[0]) * 0.5,
                (left_path[-1] + right_path[-1]) * 0.5,
            ]

        if task_type in {"pass_through_gap", "stop_at_gap", "switch_sides"}:
            return self.resolve_gap_path(task_type)

        if task_type == "stop_at_landmark" and "target_fence" in self.task:
            fence = fence_by_name(self.scene, self.task["target_fence"])
            base_path = concat_segments([fence], self.flip_isaac_y)
            return offset_polyline(base_path, self.preferred_offset_m, self.task.get("follow_side", "left"))

        if task_type == "hold_position":
            start = self.start_pose_point()
            return [start, start + np.asarray([0.001, 0.0], dtype=np.float64)]

        raise RuntimeError(f"Unsupported fenceline task_type {task_type!r}")

    def resolve_gap_path(self, task_type: str) -> list[np.ndarray]:
        target_gap = self.task["target_gap"]
        before = fence_by_name(self.scene, target_gap["before_fence"])
        after = fence_by_name(self.scene, target_gap["after_fence"])
        side = self.task.get("follow_side") or self.task.get("entry_side", "left")
        before_path = offset_polyline(concat_segments([before], self.flip_isaac_y), self.preferred_offset_m, side)
        after_side = self.task.get("exit_side", side)
        after_path = offset_polyline(concat_segments([after], self.flip_isaac_y), self.preferred_offset_m, after_side)
        gap_center = point2(target_gap["approximate_center"], self.flip_isaac_y)

        if task_type == "stop_at_gap":
            return [before_path[0], before_path[-1]]
        if task_type == "switch_sides":
            return [before_path[0], before_path[-1], gap_center, after_path[0], after_path[-1]]
        return [before_path[0], before_path[-1], gap_center, after_path[0], after_path[-1]]

    def start_pose_point(self) -> np.ndarray:
        starts = self.scene.get("rover_poses")
        if isinstance(starts, dict) and self.task.get("start_pose") in starts:
            return point2(starts[self.task["start_pose"]]["position"], self.flip_isaac_y)
        if isinstance(self.scene.get("rover_pose"), dict):
            return point2(self.scene["rover_pose"]["position"], self.flip_isaac_y)
        return line_path_from_points([[0.0, 0.0, 0.0]], self.flip_isaac_y)[0]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FencelineExpertTrajectoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A compact Hybrid A* planner for skid-steer/differential-drive-like expert paths."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sim.collision_map import CollisionMap
from sim.expert_trajectory_utils import wrap_to_pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class SearchNode:
    pose: Pose
    cost: float
    parent: Optional[tuple[int, int, int]]
    primitive_points: list[np.ndarray]


class HybridAStarPlanner:
    def __init__(
        self,
        collision_map: CollisionMap,
        grid_resolution_m: float = 0.15,
        yaw_resolution_rad: float = math.radians(10.0),
        step_size_m: float = 0.2,
        min_turn_radius_m: float = 0.35,
        goal_tolerance_m: float = 0.35,
        yaw_tolerance_rad: float = math.radians(35.0),
        max_iterations: int = 20000,
        allow_reverse: bool = False,
        point_cost_fn: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.collision_map = collision_map
        self.grid_resolution_m = grid_resolution_m
        self.yaw_resolution_rad = yaw_resolution_rad
        self.step_size_m = step_size_m
        self.max_curvature = 1.0 / max(min_turn_radius_m, 1e-6)
        self.goal_tolerance_m = goal_tolerance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.max_iterations = max_iterations
        self.allow_reverse = allow_reverse
        self.point_cost_fn = point_cost_fn

    def plan(self, start: Pose, goal: Pose) -> list[np.ndarray] | None:
        if self.collision_map.is_collision(np.asarray([start.x, start.y], dtype=np.float64)):
            return None
        if self.collision_map.is_collision(np.asarray([goal.x, goal.y], dtype=np.float64)):
            return None

        open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        nodes: dict[tuple[int, int, int], SearchNode] = {}
        start_key = self.key(start)
        nodes[start_key] = SearchNode(
            pose=start,
            cost=0.0,
            parent=None,
            primitive_points=[
                np.asarray([start.x, start.y], dtype=np.float64)
            ],
        )
        counter = 0
        heapq.heappush(open_heap, (self.heuristic(start, goal), counter, start_key))
        closed: set[tuple[int, int, int]] = set()

        best_key = start_key
        best_goal_distance = self.distance(start, goal)
        iterations = 0
        while open_heap and iterations < self.max_iterations:
            iterations += 1
            _, _, key = heapq.heappop(open_heap)
            if key in closed:
                continue
            closed.add(key)
            current = nodes[key]

            goal_distance = self.distance(current.pose, goal)
            if goal_distance < best_goal_distance:
                best_goal_distance = goal_distance
                best_key = key
            if self.reached_goal(current.pose, goal):
                return self.reconstruct(nodes, key)

            for next_pose, motion_cost, primitive_points in self.expand(current.pose):
                next_point = np.asarray([next_pose.x, next_pose.y], dtype=np.float64)
                if self.collision_map.is_collision(next_point):
                    continue
                next_key = self.key(next_pose)
                if next_key in closed:
                    continue
                new_cost = current.cost + motion_cost
                if next_key not in nodes or new_cost < nodes[next_key].cost:
                    nodes[next_key] = SearchNode(
                        pose=next_pose,
                        cost=new_cost,
                        parent=key,
                        primitive_points=primitive_points,
                    )
                    counter += 1
                    priority = new_cost + self.heuristic(next_pose, goal)
                    heapq.heappush(open_heap, (priority, counter, next_key))

        return None

    def sample_primitive(
        self,
        start: Pose,
        direction: float,
        curvature: float,
        sample_spacing_m: float = 0.05,
    ) -> list[np.ndarray] | None:
        """
        Sample one Hybrid A* motion primitive.

        Returns the sampled points when collision-free.
        Returns None when any sampled point is in collision.
        """

        num_samples = max(
            2,
            int(math.ceil(self.step_size_m / sample_spacing_m)),
        )

        primitive_points: list[np.ndarray] = []

        for index in range(1, num_samples + 1):
            fraction = index / num_samples
            ds = self.step_size_m * direction * fraction

            if abs(curvature) < 1e-9:
                x = start.x + ds * math.cos(start.yaw)
                y = start.y + ds * math.sin(start.yaw)
            else:
                yaw = wrap_to_pi(start.yaw + ds * curvature)
                radius = 1.0 / curvature

                x = start.x + radius * (
                    math.sin(yaw) - math.sin(start.yaw)
                )
                y = start.y - radius * (
                    math.cos(yaw) - math.cos(start.yaw)
                )

            point = np.asarray([x, y], dtype=np.float64)

            if self.collision_map.is_collision(point):
                return None

            primitive_points.append(point)

        return primitive_points

    def expand(
        self,
        pose: Pose,
    ) -> list[tuple[Pose, float, list[np.ndarray]]]:
        directions = [1.0]

        if self.allow_reverse:
            directions.append(-1.0)

        curvatures = [
            -self.max_curvature,
            -self.max_curvature * 0.5,
            0.0,
            self.max_curvature * 0.5,
            self.max_curvature,
        ]

        successors: list[
            tuple[Pose, float, list[np.ndarray]]
        ] = []

        # Translational motion primitives.
        for direction in directions:
            for curvature in curvatures:
                primitive_points = self.sample_primitive(
                    start=pose,
                    direction=direction,
                    curvature=curvature,
                    sample_spacing_m=0.05,
                )

                if primitive_points is None:
                    continue

                ds = self.step_size_m * direction

                if abs(curvature) < 1e-9:
                    next_yaw = pose.yaw
                    next_x = pose.x + ds * math.cos(pose.yaw)
                    next_y = pose.y + ds * math.sin(pose.yaw)
                else:
                    d_yaw = ds * curvature
                    next_yaw = wrap_to_pi(
                        pose.yaw + d_yaw
                    )
                    radius = 1.0 / curvature

                    next_x = pose.x + radius * (
                        math.sin(next_yaw)
                        - math.sin(pose.yaw)
                    )

                    next_y = pose.y - radius * (
                        math.cos(next_yaw)
                        - math.cos(pose.yaw)
                    )

                turn_penalty = (
                    0.05
                    * abs(curvature)
                    / self.max_curvature
                )

                reverse_penalty = (
                    0.4 if direction < 0.0 else 0.0
                )

                successors.append(
                    (
                        Pose(
                            next_x,
                            next_y,
                            next_yaw,
                        ),
                        abs(ds)
                        + turn_penalty
                        + reverse_penalty
                        + self.primitive_soft_cost(primitive_points, abs(ds)),
                        primitive_points,
                    )
                )

        # Rotation-in-place primitives for a skid-steer rover.
        rotation_step = self.yaw_resolution_rad

        current_point = np.asarray(
            [pose.x, pose.y],
            dtype=np.float64,
        )

        if not self.collision_map.is_collision(
            current_point
        ):
            for yaw_change in (
                -rotation_step,
                rotation_step,
            ):
                next_yaw = wrap_to_pi(
                    pose.yaw + yaw_change
                )

                successors.append(
                    (
                        Pose(
                            x=pose.x,
                            y=pose.y,
                            yaw=next_yaw,
                        ),
                        0.10,
                        [current_point.copy()],
                    )
                )

        return successors

    def primitive_soft_cost(self, primitive_points: list[np.ndarray], travel_distance_m: float) -> float:
        if self.point_cost_fn is None or not primitive_points or travel_distance_m <= 0.0:
            return 0.0
        costs = [max(0.0, float(self.point_cost_fn(point))) for point in primitive_points]
        return float(sum(costs) / len(costs)) * travel_distance_m

    def key(self, pose: Pose) -> tuple[int, int, int]:
        return (
            int(round(pose.x / self.grid_resolution_m)),
            int(round(pose.y / self.grid_resolution_m)),
            int(round(wrap_to_pi(pose.yaw) / self.yaw_resolution_rad)),
        )

    def heuristic(self, pose: Pose, goal: Pose) -> float:
        return self.distance(pose, goal) + 0.1 * abs(wrap_to_pi(goal.yaw - pose.yaw))
    

    @staticmethod
    def distance(a: Pose, b: Pose) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def reached_goal(self, pose: Pose, goal: Pose) -> bool:
        return self.distance(pose, goal) <= self.goal_tolerance_m and abs(wrap_to_pi(goal.yaw - pose.yaw)) <= self.yaw_tolerance_rad

    @staticmethod
    def reconstruct(
        nodes: dict[tuple[int, int, int], SearchNode],
        key: tuple[int, int, int],
    ) -> list[np.ndarray]:
        """
        Reconstruct the exact sampled Hybrid A* primitives.
        """

        node_sequence: list[SearchNode] = []
        current_key: tuple[int, int, int] | None = key

        while current_key is not None:
            node = nodes[current_key]
            node_sequence.append(node)
            current_key = node.parent

        node_sequence.reverse()

        if not node_sequence:
            return []

        path: list[np.ndarray] = []

        for index, node in enumerate(node_sequence):
            if index == 0:
                path.extend(node.primitive_points)
            else:
                path.extend(node.primitive_points)

        return path

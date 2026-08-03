#!/usr/bin/env python3
"""A compact Hybrid A* planner for skid-steer/differential-drive-like expert paths."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

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


class HybridAStarPlanner:
    def __init__(
        self,
        collision_map: CollisionMap,
        grid_resolution_m: float = 0.25,
        yaw_resolution_rad: float = math.radians(15.0),
        step_size_m: float = 0.35,
        min_turn_radius_m: float = 0.75,
        goal_tolerance_m: float = 0.35,
        yaw_tolerance_rad: float = math.radians(35.0),
        max_iterations: int = 20000,
        allow_reverse: bool = False,
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

    def plan(self, start: Pose, goal: Pose) -> list[np.ndarray] | None:
        if self.collision_map.is_collision(np.asarray([start.x, start.y], dtype=np.float64)):
            return None
        if self.collision_map.is_collision(np.asarray([goal.x, goal.y], dtype=np.float64)):
            return None

        open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        nodes: dict[tuple[int, int, int], SearchNode] = {}
        start_key = self.key(start)
        nodes[start_key] = SearchNode(start, 0.0, None)
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

            for next_pose, motion_cost in self.expand(current.pose):
                next_point = np.asarray([next_pose.x, next_pose.y], dtype=np.float64)
                if self.collision_map.is_collision(next_point):
                    continue
                next_key = self.key(next_pose)
                if next_key in closed:
                    continue
                new_cost = current.cost + motion_cost
                if next_key not in nodes or new_cost < nodes[next_key].cost:
                    nodes[next_key] = SearchNode(next_pose, new_cost, key)
                    counter += 1
                    priority = new_cost + self.heuristic(next_pose, goal)
                    heapq.heappush(open_heap, (priority, counter, next_key))

        if best_goal_distance <= self.goal_tolerance_m * 2.0:
            return self.reconstruct(nodes, best_key)
        return None

    def expand(self, pose: Pose) -> list[tuple[Pose, float]]:
        directions = [1.0]
        if self.allow_reverse:
            directions.append(-1.0)
        curvatures = [-self.max_curvature, -self.max_curvature * 0.5, 0.0, self.max_curvature * 0.5, self.max_curvature]
        successors = []
        for direction in directions:
            for curvature in curvatures:
                ds = self.step_size_m * direction
                if abs(curvature) < 1e-9:
                    next_yaw = pose.yaw
                    next_x = pose.x + ds * math.cos(pose.yaw)
                    next_y = pose.y + ds * math.sin(pose.yaw)
                else:
                    d_yaw = ds * curvature
                    next_yaw = wrap_to_pi(pose.yaw + d_yaw)
                    radius = 1.0 / curvature
                    next_x = pose.x + radius * (math.sin(next_yaw) - math.sin(pose.yaw))
                    next_y = pose.y - radius * (math.cos(next_yaw) - math.cos(pose.yaw))
                turn_penalty = 0.05 * abs(curvature) / self.max_curvature
                reverse_penalty = 0.4 if direction < 0.0 else 0.0
                successors.append((Pose(next_x, next_y, next_yaw), abs(ds) + turn_penalty + reverse_penalty))
        return successors

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
    def reconstruct(nodes: dict[tuple[int, int, int], SearchNode], key: tuple[int, int, int]) -> list[np.ndarray]:
        poses = []
        current_key: tuple[int, int, int] | None = key
        while current_key is not None:
            node = nodes[current_key]
            poses.append(np.asarray([node.pose.x, node.pose.y], dtype=np.float64))
            current_key = node.parent
        poses.reverse()
        return poses

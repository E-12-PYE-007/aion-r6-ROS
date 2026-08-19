#!/usr/bin/env python3
"""Obstacle extraction and collision checks for generated Isaac layouts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.expert_trajectory_utils import point2


def rotate(point: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray([c * point[0] - s * point[1], s * point[0] + c * point[1]], dtype=np.float64)


class OrientedBoxObstacle:
    def __init__(
        self,
        center: np.ndarray,
        yaw: float,
        half_extents: np.ndarray,
        name: str,
        obstacle_type: str,
    ) -> None:
        self.center = center
        self.yaw = yaw
        self.half_extents = half_extents
        self.name = name
        self.obstacle_type = obstacle_type

    def contains(self, point: np.ndarray, inflation_m: float) -> bool:
        local = rotate(point - self.center, -self.yaw)
        extents = self.half_extents + inflation_m
        return abs(float(local[0])) <= float(extents[0]) and abs(float(local[1])) <= float(extents[1])

    def signed_distance(self, point: np.ndarray, inflation_m: float) -> float:
        local = rotate(point - self.center, -self.yaw)
        extents = self.half_extents + inflation_m
        delta = np.abs(local) - extents
        outside = np.maximum(delta, 0.0)
        outside_distance = float(np.linalg.norm(outside))
        inside_distance = min(max(float(delta[0]), float(delta[1])), 0.0)
        return outside_distance + inside_distance


class CollisionMap:
    def __init__(
        self,
        obstacles: list[OrientedBoxObstacle],
        bounds: tuple[float, float, float, float],
        inflation_m: float,
    ) -> None:
        self.obstacles = obstacles
        self.bounds = bounds
        self.inflation_m = inflation_m

    def in_bounds(self, point: np.ndarray) -> bool:
        min_x, max_x, min_y, max_y = self.bounds
        return min_x <= float(point[0]) <= max_x and min_y <= float(point[1]) <= max_y

    def is_collision(self, point: np.ndarray) -> bool:
        if not self.in_bounds(point):
            return True
        return any(obstacle.contains(point, self.inflation_m) for obstacle in self.obstacles)

    def obstacle_clearance(self, point: np.ndarray, include_fences: bool = False) -> float:
        distances = [
            obstacle.signed_distance(point, self.inflation_m)
            for obstacle in self.obstacles
            if include_fences or obstacle.obstacle_type != "fence"
        ]
        if not distances:
            return math.inf
        return min(distances)

    @classmethod
    def from_scene(
        cls,
        scene: dict[str, Any],
        scene_path: Path,
        reference_points: list[np.ndarray],
        flip_isaac_y: bool,
        robot_radius_m: float,
        obstacle_padding_m: float,
        bounds_margin_m: float = 4.0,
    ) -> "CollisionMap":
        obstacles = []
        for index, obstacle in enumerate(scene.get("obstacles") or []):
            if not isinstance(obstacle, dict) or "position" not in obstacle:
                continue
            obstacle_type = str(obstacle.get("type", "obstacle"))
            name = str(obstacle.get("name", f"{obstacle_type}_{index:02d}"))
            bbox = lookup_obstacle_footprint(scene, obstacle_type, name)
            if bbox is None:
                bbox = [0.6, 0.6, 0.3]
            center = point2(obstacle["position"], flip_isaac_y)
            yaw = float(obstacle.get("yaw", 0.0))
            # Generated Isaac layouts may store yaw in degrees.
            if abs(yaw) > math.tau:
                yaw = math.radians(yaw)

            if flip_isaac_y:
                yaw = -yaw
            # Isaac asset_library.yaml stores full XYZ bounding-box sizes in metres.
            # The planner represents obstacles as oriented XY boxes, so store half extents.
            half_extents = np.asarray(
                [0.5 * float(bbox[0]), 0.5 * float(bbox[1])],
                dtype=np.float64,
            )
            obstacles.append(OrientedBoxObstacle(center, yaw, half_extents, name, obstacle_type))

        for index, fence in enumerate(scene.get("fences") or []):
            if not isinstance(fence, dict) or "start" not in fence or "end" not in fence:
                continue
            start = point2(fence["start"], flip_isaac_y)
            end = point2(fence["end"], flip_isaac_y)
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue
            center = 0.5 * (start + end)
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            thickness_m = float(fence.get("collision_thickness_m", 0.08))
            half_extents = np.asarray([0.5 * length, 0.5 * thickness_m], dtype=np.float64)
            obstacles.append(
                OrientedBoxObstacle(
                    center,
                    yaw,
                    half_extents,
                    str(fence.get("name", f"fence_{index:02d}")),
                    "fence",
                )
            )

        all_points = list(reference_points) + [obstacle.center for obstacle in obstacles]
        if not all_points:
            all_points = [np.asarray([0.0, 0.0], dtype=np.float64)]
        points = np.asarray(all_points, dtype=np.float64)
        min_x = float(np.min(points[:, 0]) - bounds_margin_m)
        max_x = float(np.max(points[:, 0]) + bounds_margin_m)
        min_y = float(np.min(points[:, 1]) - bounds_margin_m)
        max_y = float(np.max(points[:, 1]) + bounds_margin_m)
        return cls(obstacles, (min_x, max_x, min_y, max_y), robot_radius_m + obstacle_padding_m)


def lookup_obstacle_footprint(scene: dict[str, Any], obstacle_type: str, name: str) -> list[float] | None:
    assets = scene.get("assets", {})
    candidates = []
    if obstacle_type == "plant":
        candidates.extend([("plants", name), ("plant", name)])
    if obstacle_type == "miscellaneous":
        candidates.extend([("obstacles", name), ("miscellaneous", name)])
    candidates.extend([(obstacle_type, name), ("obstacles", name), ("plants", name), ("miscellaneous", name)])

    for group, asset_name in candidates:
        group_data = assets.get(group)
        if isinstance(group_data, dict):
            entry = group_data.get(asset_name)
            if isinstance(entry, dict):
                footprint = entry.get("navigation_footprint_m")
                if is_valid_xy_size(footprint):
                    return footprint
                if "bbox_size" in entry:
                    return entry["bbox_size"]
    return None


def is_valid_xy_size(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        return float(value[0]) > 0.0 and float(value[1]) > 0.0
    except (TypeError, ValueError):
        return False

#!/usr/bin/env python3
"""Plot task scene geometry, reference path, subgoals, and Hybrid A* plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.collision_map import CollisionMap, OrientedBoxObstacle
from sim.expert_trajectory_utils import find_task, find_variant, load_yaml, path_length
from sim.hybrid_astar import Pose
from sim.validate_scene_task_specs import (
    build_planner,
    plan_through_subgoals,
    reference_path_for_task,
    reference_subgoals,
    shifted_start_pose,
)


def obstacle_corners(obstacle: OrientedBoxObstacle) -> np.ndarray:
    hx, hy = obstacle.half_extents + obstacle.inflation_m
    local = np.asarray(
        [
            [-hx, -hy],
            [hx, -hy],
            [hx, hy],
            [-hx, hy],
            [-hx, -hy],
        ],
        dtype=np.float64,
    )
    c = np.cos(obstacle.yaw)
    s = np.sin(obstacle.yaw)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + obstacle.center


def plan_path(
    collision_map: CollisionMap,
    settings: dict,
    start_position: np.ndarray,
    start_yaw: float,
    subgoals: list[tuple[np.ndarray, float]],
) -> list[np.ndarray] | None:
    planner = build_planner(collision_map, settings)
    start_pose = Pose(float(start_position[0]), float(start_position[1]), float(start_yaw))
    planned_path: list[np.ndarray] = []
    for goal_position, goal_yaw in subgoals:
        segment = planner.plan(
            start_pose,
            Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw)),
        )
        if segment is None:
            return None
        if planned_path:
            planned_path.extend(segment[1:])
        else:
            planned_path.extend(segment)
        start_pose = Pose(float(goal_position[0]), float(goal_position[1]), float(goal_yaw))
    return planned_path


def plot_polyline(ax, points: list[np.ndarray], *args, **kwargs) -> None:
    if not points:
        return
    xy = np.asarray(points)
    ax.plot(xy[:, 0], xy[:, 1], *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_spec", type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant-id", default="nominal")
    parser.add_argument("--output", type=Path, default=Path("task_plan.png"))
    parser.add_argument("--no-flip-isaac-y", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flip_isaac_y = not args.no_flip_isaac_y
    task_spec = load_yaml(args.task_spec)
    scene_yaml = Path(task_spec["scene"]["source_yaml"])
    scene = load_yaml(scene_yaml)
    task = find_task(task_spec, args.task_id)
    variant = find_variant(task, args.variant_id)
    settings = variant.get("planner_settings") or {}

    reference_path = reference_path_for_task(scene, scene_yaml, task, variant, flip_isaac_y)
    start_position, start_yaw = shifted_start_pose(scene, task, variant, flip_isaac_y)
    subgoals = reference_subgoals(reference_path, settings)
    collision_map = CollisionMap.from_scene(
        scene,
        scene_yaml,
        [start_position] + [position for position, _ in subgoals] + reference_path,
        flip_isaac_y,
        robot_radius_m=float(settings.get("robot_radius_m", 0.35)),
        obstacle_padding_m=float(settings.get("obstacle_padding_m", 0.25)),
    )
    planned_ok, note = plan_through_subgoals(
        build_planner(collision_map, settings),
        start_position,
        start_yaw,
        subgoals,
        collision_map,
        settings,
    )
    planned_path = plan_path(collision_map, settings, start_position, start_yaw, subgoals)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"{args.task_id} / {args.variant_id}\nplanner_valid={planned_ok} {note or ''}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    for fence in scene.get("fences") or []:
        start = np.asarray([float(fence["start"][0]), float(fence["start"][1])])
        end = np.asarray([float(fence["end"][0]), float(fence["end"][1])])
        if flip_isaac_y:
            start[1] *= -1.0
            end[1] *= -1.0
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=3, label="fence")

    for obstacle in collision_map.obstacles:
        corners = obstacle_corners(obstacle)
        ax.fill(corners[:, 0], corners[:, 1], color="tab:red", alpha=0.2)
        ax.plot(corners[:, 0], corners[:, 1], color="tab:red", linewidth=0.8)
        ax.text(obstacle.center[0], obstacle.center[1], obstacle.name, fontsize=7, ha="center")

    plot_polyline(ax, reference_path, "--", color="tab:blue", linewidth=2, label="reference")
    if planned_path:
        plot_polyline(ax, planned_path, "-", color="tab:green", linewidth=2, label="Hybrid A*")
    subgoal_points = np.asarray([position for position, _ in subgoals])
    if len(subgoal_points):
        ax.scatter(subgoal_points[:, 0], subgoal_points[:, 1], marker="x", color="tab:purple", s=60, label="subgoals")
    ax.scatter([start_position[0]], [start_position[1]], marker="o", color="tab:orange", s=80, label="start")
    ax.arrow(
        float(start_position[0]),
        float(start_position[1]),
        0.6 * np.cos(start_yaw),
        0.6 * np.sin(start_yaw),
        color="tab:orange",
        head_width=0.15,
        length_includes_head=True,
    )
    ax.legend(loc="best")
    ax.set_xlabel("planner x (m)")
    ax.set_ylabel("planner y (m)")
    ax.text(
        0.01,
        0.01,
        f"path_length={path_length(reference_path):.2f}m\nsource={scene_yaml}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

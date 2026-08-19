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
    candidate_preserves_side,
    fence_offset_cost_for_task,
    reference_path_for_task,
    reference_subgoals,
    shifted_subgoal_candidates,
    shifted_start_pose,
    side_constraint_segments_for_task,
)


def obstacle_corners(
    obstacle: OrientedBoxObstacle,
    inflation_m: float,
) -> np.ndarray:
    hx, hy = obstacle.half_extents + inflation_m

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

    rotation = np.asarray(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    return local @ rotation.T + obstacle.center


def plan_path(
    collision_map: CollisionMap,
    settings: dict,
    start_position: np.ndarray,
    start_yaw: float,
    subgoals: list[tuple[np.ndarray, float]],
    side_constraint_segments: list[tuple[np.ndarray, np.ndarray]],
    point_cost_fn=None,
) -> tuple[list[np.ndarray] | None, list[tuple[np.ndarray, float]], int]:
    planner = build_planner(collision_map, settings, point_cost_fn=point_cost_fn)
    start_pose = Pose(float(start_position[0]), float(start_position[1]), float(start_yaw))
    planned_path: list[np.ndarray] = []
    selected_subgoals: list[tuple[np.ndarray, float]] = []
    nudged_count = 0
    max_lateral_m = float(settings.get("planner_subgoal_lateral_search_m", 2.0))
    max_longitudinal_m = float(settings.get("planner_subgoal_longitudinal_search_m", 2.0))
    search_step_m = float(settings.get("planner_subgoal_search_step_m", 0.5))
    max_candidates = int(settings.get("planner_subgoal_max_candidates", 48))
    for goal_position, goal_yaw in subgoals:
        selected_position = None
        selected_segment = None
        attempted_candidates = 0
        for candidate in shifted_subgoal_candidates(
            goal_position,
            goal_yaw,
            max_lateral_m,
            max_longitudinal_m,
            step_m=search_step_m,
        ):
            if collision_map.is_collision(candidate):
                continue
            if not candidate_preserves_side(goal_position, candidate, side_constraint_segments):
                continue
            if attempted_candidates >= max_candidates:
                continue
            attempted_candidates += 1
            segment = planner.plan(
                start_pose,
                Pose(float(candidate[0]), float(candidate[1]), float(goal_yaw)),
            )
            if segment is not None:
                selected_position = candidate
                selected_segment = segment
                break
        if selected_position is None or selected_segment is None:
            return None, selected_subgoals, nudged_count
        if float(np.linalg.norm(selected_position - goal_position)) > 1e-6:
            nudged_count += 1
        selected_subgoals.append((selected_position, goal_yaw))
        if planned_path:
            planned_path.extend(selected_segment[1:])
        else:
            planned_path.extend(selected_segment)
        start_pose = Pose(float(selected_position[0]), float(selected_position[1]), float(goal_yaw))
    return planned_path, selected_subgoals, nudged_count


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
        robot_radius_m=float(settings.get("robot_radius_m", 0.32)),
        obstacle_padding_m=float(settings.get("obstacle_padding_m", 0.08)),
    )
    planned_path, selected_subgoals, nudged_count = plan_path(
        collision_map,
        settings,
        start_position,
        start_yaw,
        subgoals,
        side_constraint_segments_for_task(scene, task, flip_isaac_y),
        fence_offset_cost_for_task(scene, task, variant, settings, flip_isaac_y),
    )
    planned_ok = planned_path is not None
    note = f"nudged {nudged_count} reference subgoals to nearby reachable points" if nudged_count else ""
    print(f"collision_inflation_m={collision_map.inflation_m:.2f}")
    for obstacle in collision_map.obstacles:
        physical_size = 2.0 * obstacle.half_extents
        inflated_size = 2.0 * (obstacle.half_extents + collision_map.inflation_m)
        print(
            f"{obstacle.name}: type={obstacle.obstacle_type} "
            f"physical=[{physical_size[0]:.2f}, {physical_size[1]:.2f}] "
            f"inflated=[{inflated_size[0]:.2f}, {inflated_size[1]:.2f}]"
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"{args.task_id} / {args.variant_id}\nplanner_valid={planned_ok} {note or ''}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    fence_label_used = False
    for fence in scene.get("fences") or []:
        start = np.asarray([float(fence["start"][0]), float(fence["start"][1])])
        end = np.asarray([float(fence["end"][0]), float(fence["end"][1])])
        if flip_isaac_y:
            start[1] *= -1.0
            end[1] *= -1.0
        label = "fence centerline" if not fence_label_used else None
        fence_label_used = True
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=3, label=label)

    inflated_label_used = False
    physical_label_used = False
    for obstacle in collision_map.obstacles:
        inflated_corners = obstacle_corners(
            obstacle,
            collision_map.inflation_m,
        )
        physical_corners = obstacle_corners(obstacle, 0.0)
        if obstacle.obstacle_type == "fence":
            inflated_color = "0.35"
            physical_color = "black"
        else:
            inflated_color = "tab:red"
            physical_color = "tab:red"
        inflated_label = "inflated collision footprint" if not inflated_label_used else None
        physical_label = "physical footprint" if not physical_label_used else None
        inflated_label_used = True
        physical_label_used = True
        ax.fill(inflated_corners[:, 0], inflated_corners[:, 1], color=inflated_color, alpha=0.12, label=inflated_label)
        ax.plot(inflated_corners[:, 0], inflated_corners[:, 1], color=inflated_color, linewidth=0.8)
        ax.plot(
            physical_corners[:, 0],
            physical_corners[:, 1],
            color=physical_color,
            linewidth=1.2,
            linestyle="-",
            label=physical_label,
        )
        ax.text(obstacle.center[0], obstacle.center[1], obstacle.name, fontsize=7, ha="center")

    plot_polyline(ax, reference_path, "--", color="tab:blue", linewidth=2, label="reference")
    if planned_path:
        plot_polyline(ax, planned_path, "-", color="tab:green", linewidth=2, label="Hybrid A*")
    subgoal_points = np.asarray([position for position, _ in subgoals])
    if len(subgoal_points):
        ax.scatter(subgoal_points[:, 0], subgoal_points[:, 1], marker="x", color="tab:purple", s=60, label="subgoals")
    selected_points = np.asarray([position for position, _ in selected_subgoals])
    if len(selected_points):
        ax.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            marker="+",
            color="tab:green",
            s=80,
            label="reachable subgoals",
        )
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
    ax.text(
        0.99,
        0.01,
        f"collision inflation={collision_map.inflation_m:.2f}m",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

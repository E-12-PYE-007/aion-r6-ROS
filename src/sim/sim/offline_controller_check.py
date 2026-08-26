#!/usr/bin/env python3
"""Offline sanity check for planned trajectories and the expert path tracker."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sim.collision_map import CollisionMap, OrientedBoxObstacle
from sim.expert_trajectory_utils import (
    find_task,
    find_variant,
    load_yaml,
    path_length,
    project_progress_near,
    sample_path_pose,
    world_to_robot,
    wrap_to_pi,
)
from sim.trajectory_profile import build_timed_trajectory, resample_path
from sim.validate_scene_task_specs import (
    build_planner,
    candidate_preserves_side,
    combined_point_cost_fn,
    distance_to_nearest_segment,
    fence_offset_cost_for_task,
    obstacle_clearance_cost_for_map,
    reference_path_for_task,
    reference_subgoals,
    shifted_start_pose,
    shifted_subgoal_candidates,
    side_constraint_segments_for_task,
)
from sim.hybrid_astar import Pose


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
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + obstacle.center


def plan_path(
    scene: dict[str, Any],
    scene_yaml: Path,
    task: dict[str, Any],
    variant: dict[str, Any],
    flip_scene_y: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    reference_path = reference_path_for_task(scene, scene_yaml, task, variant, flip_scene_y)
    start_position, start_yaw = shifted_start_pose(scene, task, variant, flip_scene_y)
    settings = variant.get("planner_settings") or {}
    subgoals = reference_subgoals(reference_path, settings)
    collision_map = CollisionMap.from_scene(
        scene,
        scene_yaml,
        [start_position] + [position for position, _ in subgoals] + reference_path,
        flip_scene_y,
        robot_radius_m=float(settings.get("robot_radius_m", 0.32)),
        obstacle_padding_m=float(settings.get("obstacle_padding_m", 0.08)),
    )
    planner = build_planner(
        collision_map,
        settings,
        point_cost_fn=combined_point_cost_fn(
            fence_offset_cost_for_task(scene, task, variant, settings, flip_scene_y),
            obstacle_clearance_cost_for_map(collision_map, settings),
        ),
    )

    planned_path: list[np.ndarray] = [start_position]
    start_pose = Pose(float(start_position[0]), float(start_position[1]), float(start_yaw))
    max_lateral_m = float(settings.get("planner_subgoal_lateral_search_m", 2.0))
    max_longitudinal_m = float(settings.get("planner_subgoal_longitudinal_search_m", 2.0))
    search_step_m = float(settings.get("planner_subgoal_search_step_m", 0.5))
    max_candidates = int(settings.get("planner_subgoal_max_candidates", 48))
    min_clearance_m = float(settings.get("planner_subgoal_min_clearance_m", 0.15))
    min_fence_clearance_m = float(settings.get("planner_fence_min_clearance_m", 0.65))
    side_segments = side_constraint_segments_for_task(scene, task, flip_scene_y)
    nudged = 0

    for index, (goal_position, goal_yaw) in enumerate(subgoals):
        selected_segment = None
        selected_position = None
        attempted = 0
        for candidate in shifted_subgoal_candidates(
            goal_position,
            goal_yaw,
            max_lateral_m,
            max_longitudinal_m,
            step_m=search_step_m,
        ):
            if collision_map.is_collision(candidate):
                continue
            if collision_map.obstacle_clearance(candidate, include_fences=False) < min_clearance_m:
                continue
            if (
                min_fence_clearance_m > 0.0
                and side_segments
                and distance_to_nearest_segment(candidate, side_segments) < min_fence_clearance_m
            ):
                continue
            if not candidate_preserves_side(goal_position, candidate, side_segments):
                continue
            if attempted >= max_candidates:
                continue
            attempted += 1
            segment = planner.plan(
                start_pose,
                Pose(float(candidate[0]), float(candidate[1]), float(goal_yaw)),
            )
            if segment is not None:
                selected_segment = segment
                selected_position = candidate
                break
        if selected_segment is None or selected_position is None:
            raise RuntimeError(f"Hybrid A* failed to reach subgoal {index} near {goal_position.tolist()}")
        if float(np.linalg.norm(selected_position - goal_position)) > 1e-6:
            nudged += 1
        planned_path.extend(selected_segment[1:] if len(selected_segment) > 1 else selected_segment)
        start_pose = Pose(float(selected_position[0]), float(selected_position[1]), float(goal_yaw))

    planned_path = resample_path(planned_path, 0.1)
    return reference_path, planned_path, {
        "start_x": float(start_position[0]),
        "start_y": float(start_position[1]),
        "start_yaw": float(start_yaw),
        "nudged_subgoals": nudged,
        "subgoal_count": len(subgoals),
    }


def select_tracking_target(
    path: list[np.ndarray],
    progress_m: float,
    position: np.ndarray,
    yaw: float,
    lookahead_m: float,
) -> tuple[np.ndarray, float, float]:
    total = path_length(path)
    start_distance = min(total, progress_m + lookahead_m)
    upper_distance = min(total, progress_m + max(5.0, lookahead_m * 5.0))
    best: tuple[float, np.ndarray, float, float] | None = None
    if start_distance >= total:
        target, target_yaw = sample_path_pose(path, total)
        return target, target_yaw, total
    steps = max(1, int(math.ceil((upper_distance - start_distance) / 0.15)))
    for index in range(steps + 1):
        distance = min(upper_distance, start_distance + index * 0.15)
        target, target_yaw = sample_path_pose(path, distance)
        x, y, theta = world_to_robot(position, yaw, target, target_yaw)
        if x <= 0.10 or abs(y) > 3.0:
            continue
        score = 0.7 * abs(y) + 0.25 * abs(x - lookahead_m) + 0.10 * abs(theta) + 0.02 * (distance - start_distance)
        if best is None or score < best[0]:
            best = (score, target, target_yaw, distance)
    if best is not None:
        _, target, target_yaw, distance = best
        return target, target_yaw, distance
    target, target_yaw = sample_path_pose(path, start_distance)
    return target, target_yaw, start_distance


def controller_command(
    path: list[np.ndarray],
    position: np.ndarray,
    yaw: float,
    progress_m: float,
    max_speed_mps: float,
    max_yaw_rate_radps: float,
    min_speed_mps: float,
    lookahead_m: float,
) -> tuple[float, float, dict[str, float]]:
    target, target_yaw, target_distance = select_tracking_target(path, progress_m, position, yaw, lookahead_m)
    x, y, theta = world_to_robot(position, yaw, target, target_yaw)
    heading_error = math.atan2(float(y), max(float(x), 0.05))
    steering_error = wrap_to_pi(1.05 * heading_error + 0.35 * theta)
    distance_sq = max(float(x * x + y * y), 1e-5)
    curvature = 2.0 * float(y) / distance_sq
    speed = float(max_speed_mps)
    if abs(curvature) > 1e-6:
        speed = min(speed, max(float(min_speed_mps), float(max_yaw_rate_radps) / abs(curvature)))
    if abs(steering_error) > 1.8:
        speed = min(speed, float(min_speed_mps))
    yaw_rate = max(-max_yaw_rate_radps, min(max_yaw_rate_radps, 0.9 * steering_error))
    return speed, yaw_rate, {
        "target_distance_m": float(target_distance),
        "relative_x_m": float(x),
        "relative_y_m": float(y),
        "relative_theta_rad": float(theta),
        "steering_error_rad": float(steering_error),
    }


def simulate(
    path: list[np.ndarray],
    start_x: float,
    start_y: float,
    start_yaw: float,
    duration_s: float,
    dt_s: float,
    max_speed_mps: float,
    max_yaw_rate_radps: float,
    min_speed_mps: float,
    lookahead_m: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    position = np.asarray([start_x, start_y], dtype=np.float64)
    yaw = float(start_yaw)
    progress_m = 0.0
    total = path_length(path)
    rows: list[dict[str, float]] = []
    max_tracking_error = 0.0
    tracking_errors = []
    for step in range(int(duration_s / dt_s) + 1):
        projected = project_progress_near(path, position, progress_m, max_backward_m=0.2, max_forward_m=2.0)
        progress_m = max(progress_m, projected)
        nearest, _ = sample_path_pose(path, progress_m)
        tracking_error = float(np.linalg.norm(position - nearest))
        max_tracking_error = max(max_tracking_error, tracking_error)
        tracking_errors.append(tracking_error)
        speed, yaw_rate, debug = controller_command(
            path,
            position,
            yaw,
            progress_m,
            max_speed_mps,
            max_yaw_rate_radps,
            min_speed_mps,
            lookahead_m,
        )
        rows.append({
            "t": step * dt_s,
            "x": float(position[0]),
            "y": float(position[1]),
            "yaw": float(yaw),
            "progress_m": float(progress_m),
            "tracking_error_m": tracking_error,
            "cmd_linear_x": float(speed),
            "cmd_angular_z_internal": float(yaw_rate),
            **debug,
        })
        if total - progress_m <= 0.25:
            break
        position = position + np.asarray([math.cos(yaw), math.sin(yaw)]) * speed * dt_s
        yaw = wrap_to_pi(yaw + yaw_rate * dt_s)
    metrics = {
        "success": bool(total - progress_m <= 0.25),
        "progress_m": float(progress_m),
        "path_length_m": float(total),
        "progress_fraction": float(progress_m / max(total, 1e-9)),
        "sim_time_s": float(rows[-1]["t"] if rows else 0.0),
        "mean_tracking_error_m": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        "max_tracking_error_m": float(max_tracking_error),
        "final_target_remaining_m": float(total - progress_m),
    }
    return rows, metrics


def plot_result(reference_path: list[np.ndarray], planned_path: list[np.ndarray], rows: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ref = np.asarray(reference_path)
    plan = np.asarray(planned_path)
    sim = np.asarray([[row["x"], row["y"]] for row in rows])
    ax.plot(ref[:, 0], ref[:, 1], "--", label="reference")
    ax.plot(plan[:, 0], plan[:, 1], "-", label="planned")
    ax.plot(sim[:, 0], sim[:, 1], "-", label="offline controller")
    ax.scatter(sim[0, 0], sim[0, 1], label="start")
    ax.scatter(sim[-1, 0], sim[-1, 1], marker="s", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("planner x (m)")
    ax.set_ylabel("planner y (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_result_with_scene(
    scene: dict[str, Any],
    collision_map: CollisionMap,
    reference_path: list[np.ndarray],
    planned_path: list[np.ndarray],
    rows: list[dict[str, float]],
    output: Path,
    flip_scene_y: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title("Offline Controller Check")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fence_label_used = False
    for fence in scene.get("fences") or []:
        start = np.asarray([float(fence["start"][0]), float(fence["start"][1])], dtype=np.float64)
        end = np.asarray([float(fence["end"][0]), float(fence["end"][1])], dtype=np.float64)
        if flip_scene_y:
            start[1] *= -1.0
            end[1] *= -1.0
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="black",
            linewidth=3,
            label="fence centerline" if not fence_label_used else None,
        )
        fence_label_used = True

    inflated_label_used = False
    physical_label_used = False
    for obstacle in collision_map.obstacles:
        inflated_corners = obstacle_corners(obstacle, collision_map.inflation_m)
        physical_corners = obstacle_corners(obstacle, 0.0)
        if obstacle.obstacle_type == "fence":
            inflated_color = "0.35"
            physical_color = "black"
        else:
            inflated_color = "tab:red"
            physical_color = "tab:red"
        ax.fill(
            inflated_corners[:, 0],
            inflated_corners[:, 1],
            color=inflated_color,
            alpha=0.12,
            label="inflated collision footprint" if not inflated_label_used else None,
        )
        ax.plot(inflated_corners[:, 0], inflated_corners[:, 1], color=inflated_color, linewidth=0.8)
        ax.plot(
            physical_corners[:, 0],
            physical_corners[:, 1],
            color=physical_color,
            linewidth=1.1,
            label="physical footprint" if not physical_label_used else None,
        )
        if obstacle.obstacle_type != "fence":
            ax.text(obstacle.center[0], obstacle.center[1], obstacle.name, fontsize=7, ha="center", va="center")
        inflated_label_used = True
        physical_label_used = True

    ref = np.asarray(reference_path)
    plan = np.asarray(planned_path)
    sim = np.asarray([[row["x"], row["y"]] for row in rows])
    ax.plot(ref[:, 0], ref[:, 1], "--", color="tab:blue", linewidth=2, label="reference")
    ax.plot(plan[:, 0], plan[:, 1], "-", color="tab:orange", linewidth=2, label="planned")
    ax.plot(sim[:, 0], sim[:, 1], "-", color="tab:green", linewidth=2, label="offline controller")
    ax.scatter(sim[0, 0], sim[0, 1], color="tab:blue", label="start", zorder=5)
    ax.scatter(sim[-1, 0], sim[-1, 1], color="tab:orange", marker="s", label="end", zorder=5)
    ax.set_xlabel("planner x (m)")
    ax.set_ylabel("planner y (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_spec", type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant-id", default="nominal")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/offline_controller_check"))
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument("--lookahead-m", type=float, default=0.9)
    parser.add_argument("--max-speed-mps", type=float, default=0.3)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=0.3)
    parser.add_argument("--min-speed-mps", type=float, default=0.16)
    parser.add_argument("--flip-scene-y", action="store_true")
    parser.add_argument(
        "--scene-yaml",
        type=Path,
        default=None,
        help="Override the scene YAML path stored in the task spec. Useful for downloaded specs from another machine.",
    )
    parser.add_argument(
        "--use-reference-path",
        action="store_true",
        help="Skip Hybrid A* replanning and test the controller against the task reference path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading task spec: {args.task_spec}", flush=True)
    spec = load_yaml(args.task_spec)
    scene_yaml = args.scene_yaml or Path(spec["scene"]["source_yaml"])
    print(f"Loading scene layout: {scene_yaml}", flush=True)
    scene = load_yaml(scene_yaml)
    task = find_task(spec, args.task_id)
    variant = find_variant(task, args.variant_id)
    if args.use_reference_path:
        print("Using reference path directly; skipping Hybrid A* replanning.", flush=True)
        reference_path = reference_path_for_task(scene, scene_yaml, task, variant, args.flip_scene_y)
        start_position, start_yaw = shifted_start_pose(scene, task, variant, args.flip_scene_y)
        planned_path = resample_path(reference_path, 0.1)
        plan_metrics = {
            "start_x": float(start_position[0]),
            "start_y": float(start_position[1]),
            "start_yaw": float(start_yaw),
            "nudged_subgoals": 0,
            "subgoal_count": 0,
            "used_reference_path": True,
        }
    else:
        print("Planning Hybrid A* path. This can be slow for obstacle-heavy scenes.", flush=True)
        reference_path, planned_path, plan_metrics = plan_path(scene, scene_yaml, task, variant, args.flip_scene_y)
        plan_metrics["used_reference_path"] = False
    print(
        f"Simulating controller: path_length={path_length(planned_path):.2f}m, "
        f"duration={args.duration_s:.1f}s",
        flush=True,
    )
    rows, metrics = simulate(
        planned_path,
        plan_metrics["start_x"],
        plan_metrics["start_y"],
        plan_metrics["start_yaw"],
        args.duration_s,
        args.dt_s,
        args.max_speed_mps,
        args.max_yaw_rate_radps,
        args.min_speed_mps,
        args.lookahead_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = variant.get("planner_settings") or {}
    start_position = np.asarray([plan_metrics["start_x"], plan_metrics["start_y"]], dtype=np.float64)
    collision_map = CollisionMap.from_scene(
        scene,
        scene_yaml,
        [start_position] + reference_path + planned_path,
        args.flip_scene_y,
        robot_radius_m=float(settings.get("robot_radius_m", 0.32)),
        obstacle_padding_m=float(settings.get("obstacle_padding_m", 0.08)),
    )
    report = {
        "task_spec": args.task_spec.as_posix(),
        "scene_yaml": scene_yaml.as_posix(),
        "task_id": args.task_id,
        "variant_id": args.variant_id,
        "plan": plan_metrics,
        "offline_controller": metrics,
    }
    (args.output_dir / "offline_controller_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_result_with_scene(
        scene,
        collision_map,
        reference_path,
        planned_path,
        rows,
        args.output_dir / "offline_controller_check.png",
        args.flip_scene_y,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output_dir / 'offline_controller_check.png'}")
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

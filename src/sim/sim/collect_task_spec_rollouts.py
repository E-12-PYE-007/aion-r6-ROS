#!/usr/bin/env python3
"""Orchestrate sim rollouts for every selected task/trajectory variant."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERT_BY_CONFIG_TYPE = {
    "fenceline": "fenceline_expert_trajectory",
    "road": "road_expert_trajectory",
    "shedline": "shed_expert_trajectory",
}

EXPERT_BY_TASK_TYPE = {
    "follow_fence": "fenceline_expert_trajectory",
    "follow_and_turn": "fenceline_expert_trajectory",
    "follow_fence_sequence": "fenceline_expert_trajectory",
    "follow_corridor": "fenceline_expert_trajectory",
    "pass_through_gap": "fenceline_expert_trajectory",
    "stop_at_gap": "fenceline_expert_trajectory",
    "switch_sides": "fenceline_expert_trajectory",
    "follow_road": "road_expert_trajectory",
    "follow_shed_side": "shed_expert_trajectory",
}


@dataclass
class Rollout:
    task: dict[str, Any]
    variant: dict[str, Any]
    trajectory_name: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unnamed"


def json_param(value: Any) -> str:
    return json.dumps(value or {}, separators=(",", ":"))


def ros_param_args(params: dict[str, Any]) -> list[str]:
    args = ["--ros-args"]

    for key, value in params.items():
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, bool):
            encoded = "true" if value else "false"

        elif isinstance(value, (dict, list)):
            # First convert the object to compact JSON, then encode that
            # JSON as a quoted YAML/ROS string parameter.
            json_text = json_param(value)
            encoded = json.dumps(json_text)

        elif isinstance(value, str):
            # JSON string quoting is also valid YAML string quoting.
            # This protects spaces, punctuation, colons and nested JSON.
            encoded = json.dumps(value)

        else:
            encoded = str(value)

        args.extend(["-p", f"{key}:={encoded}"])

    return args


def variant_is_valid(variant: dict[str, Any], only_planner_valid: bool) -> bool:
    if not only_planner_valid:
        return True
    validation = variant.get("planner_validation")
    return not isinstance(validation, dict) or bool(validation.get("valid", False))


def selected_rollouts(
    task_spec: dict[str, Any],
    task_ids: set[str] | None,
    variant_ids: set[str] | None,
    only_planner_valid: bool,
    limit: int | None,
) -> list[Rollout]:
    rollouts: list[Rollout] = []
    suite_id = normalize_name(str(task_spec.get("suite_id") or task_spec.get("scene", {}).get("scene_id") or "suite"))
    for task in task_spec.get("tasks", []):
        task_id = str(task.get("task_id", ""))
        if task_ids is not None and task_id not in task_ids:
            continue
        variants = task.get("trajectory_variants") or [{"variant_id": "nominal", "variant_type": "nominal"}]
        for variant in variants:
            variant_id = str(variant.get("variant_id", "nominal"))
            if variant_ids is not None and variant_id not in variant_ids:
                continue
            if not variant_is_valid(variant, only_planner_valid):
                continue
            trajectory_name = "__".join(
                [
                    suite_id,
                    f"task_{normalize_name(task_id)}",
                    f"variant_{normalize_name(variant_id)}",
                ]
            )
            rollouts.append(Rollout(task=task, variant=variant, trajectory_name=trajectory_name))
            if limit is not None and len(rollouts) >= limit:
                return rollouts
    return rollouts


def expert_executable(task_spec: dict[str, Any], task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type", ""))
    if task_type == "stop_at_landmark":
        if "target_road" in task:
            return "road_expert_trajectory"
        if "target_fence" in task:
            return "fenceline_expert_trajectory"
        return "shed_expert_trajectory"
    if task_type == "approach_target":
        config_type = str(task_spec.get("scene", {}).get("config_type", ""))
        if config_type == "shedline":
            return "shed_expert_trajectory"
        return "road_expert_trajectory"
    if task_type == "hold_position":
        return EXPERT_BY_CONFIG_TYPE.get(str(task_spec.get("scene", {}).get("config_type", "")), "fenceline_expert_trajectory")
    return EXPERT_BY_TASK_TYPE.get(task_type, "fenceline_expert_trajectory")


def command_for_expert(
    task_spec_path: Path,
    task_spec: dict[str, Any],
    rollout: Rollout,
    collection: dict[str, Any],
    rollout_dir: Path,
    *,
    drive_cmd_vel_topic: str | None = None,
) -> list[str]:
    expert = task_spec.get("expert", {})
    params = {
        "task_spec": task_spec_path.as_posix(),
        "task_id": rollout.task["task_id"],
        "variant_id": rollout.variant.get("variant_id", "nominal"),
        "odom_topic": collection.get("odom_topic", "/sim_odom"),
        "action_chunk_topic": collection.get("action_chunk_topic", "/vla/action_chunk"),
        "expert_cmd_vel_topic": drive_cmd_vel_topic or collection.get("expert_cmd_vel_topic", "/expert/cmd_vel"),
        "frame_debug_topic": collection.get("frame_debug_topic", "/expert/frame_debug"),
        "isaac_pose_debug_topic": collection.get("isaac_pose_debug_topic", "/isaac/scene_pose_debug"),
        "use_isaac_camera_pose_debug": bool(collection.get("use_isaac_camera_pose_debug", False)),
        "runtime_planned_path_output": (rollout_dir / "runtime_planned_path.json").as_posix(),
        "waypoint_spacing_m": expert.get("waypoint_spacing_m", 0.18),
        "publish_rate_hz": expert.get("publish_rate_hz", 3.0),
        "flip_isaac_y": False,
        "flip_scene_y": False,
        "flip_runtime_odom_y": False,
        "flip_runtime_odom_yaw": True,
    }
    return ["ros2", "run", "sim", expert_executable(task_spec, rollout.task)] + ros_param_args(params)


def command_for_collector(
    task_spec_path: Path,
    rollout: Rollout,
    collection: dict[str, Any],
    base_dir: Path,
    dataset_name: str,
) -> list[str]:
    structured_task = {key: value for key, value in rollout.task.items() if key != "trajectory_variants"}
    structured_task["selected_variant"] = rollout.variant
    params = {
        "base_dir": base_dir.as_posix(),
        "dataset_name": dataset_name,
        "trajectory_name": rollout.trajectory_name,
        "task_spec": task_spec_path.as_posix(),
        "task_id": rollout.task["task_id"],
        "variant_id": rollout.variant.get("variant_id", "nominal"),
        "variant_type": rollout.variant.get("variant_type", "nominal"),
        "recovery_case": rollout.variant.get("recovery_case"),
        "language_instruction": rollout.task.get("instruction", ""),
        "structured_task_json": json_param(structured_task),
        "planner_settings_json": json_param(rollout.variant.get("planner_settings")),
        "speed_profile_json": json_param(rollout.variant.get("speed_profile")),
        "camera_topic": collection.get("camera_topic", "/vla/cam"),
        "odom_topic": collection.get("odom_topic", "/sim_odom"),
        "cmd_vel_topic": collection.get("cmd_vel_topic", "/cmd_vel"),
        "action_chunk_topic": collection.get("action_chunk_topic", "/vla/action_chunk"),
        "sample_frequency_hz": collection.get("sample_frequency_hz", 3.0),
        "flip_isaac_y": False,
        "flip_scene_y": False,
        "flip_runtime_odom_y": False,
        "flip_runtime_odom_yaw": True,
    }
    return ["ros2", "run", "sim", "sim_dataset_collector"] + ros_param_args(params)


def command_for_diagnostics(
    rollout_dir: Path,
    collection: dict[str, Any],
) -> list[str]:
    params = {
        "output_dir": rollout_dir.as_posix(),
        "camera_topic": collection.get("camera_topic", "/vla/cam"),
        "odom_topic": collection.get("odom_topic", "/sim_odom"),
        "cmd_vel_topic": collection.get("cmd_vel_topic", "/cmd_vel"),
        "expert_cmd_vel_topic": collection.get("expert_cmd_vel_topic", "/expert/cmd_vel"),
        "action_chunk_topic": collection.get("action_chunk_topic", "/vla/action_chunk"),
        "frame_debug_topic": collection.get("frame_debug_topic", "/expert/frame_debug"),
        "isaac_pose_debug_topic": collection.get("isaac_pose_debug_topic", "/isaac/scene_pose_debug"),
        "sample_frequency_hz": max(float(collection.get("sample_frequency_hz", 3.0)), 5.0),
    }
    return ["ros2", "run", "sim", "rollout_diagnostics"] + ros_param_args(params)


def command_for_tracker(enabled: bool) -> list[str] | None:
    if not enabled:
        return None
    return ["ros2", "run", "hw_interface", "sim_waypoint_tracking"]


def command_for_sim_duration_wait(
    collection: dict[str, Any],
    duration_s: float,
) -> list[str]:
    wall_timeout_s = max(duration_s * 20.0, duration_s + 120.0)
    params = {
        "duration_s": duration_s,
        "odom_topic": collection.get("odom_topic", "/sim_odom"),
        "wall_timeout_s": wall_timeout_s,
    }
    return ["ros2", "run", "sim", "wait_for_sim_duration"] + ros_param_args(params)


def command_for_task_success_wait(
    task_spec_path: Path,
    rollout: Rollout,
    collection: dict[str, Any],
    max_duration_s: float,
    rollout_dir: Path,
) -> list[str]:
    wall_timeout_s = max(max_duration_s * 20.0, max_duration_s + 120.0)
    params = {
        "task_spec": task_spec_path.as_posix(),
        "task_id": rollout.task["task_id"],
        "variant_id": rollout.variant.get("variant_id", "nominal"),
        "odom_topic": collection.get("odom_topic", "/sim_odom"),
        "isaac_pose_debug_topic": collection.get("isaac_pose_debug_topic", "/isaac/scene_pose_debug"),
        "use_isaac_camera_pose_debug": bool(collection.get("use_isaac_camera_pose_debug", False)),
        "max_duration_s": max_duration_s,
        "fallback_duration_s": max_duration_s,
        "wall_timeout_s": wall_timeout_s,
        "summary_path": (rollout_dir / "task_success_wait_summary.json").as_posix(),
        "flip_scene_y": False,
        "flip_runtime_odom_y": False,
        "flip_runtime_odom_yaw": True,
    }
    return ["ros2", "run", "sim", "wait_for_task_success"] + ros_param_args(params)


def render_prepare_command(template: str, task_spec_path: Path, task_spec: dict[str, Any], rollout: Rollout) -> str:
    scene = task_spec.get("scene", {})
    return template.format(
        task_spec=task_spec_path.as_posix(),
        task_id=rollout.task["task_id"],
        variant_id=rollout.variant.get("variant_id", "nominal"),
        scene_yaml=scene.get("source_yaml", ""),
        generated_layout_yaml=scene.get("generated_layout_yaml", scene.get("source_yaml", "")),
        generated_usd=scene.get("generated_usd", ""),
        trajectory_name=rollout.trajectory_name,
    )


def default_bridge_prepare_command(
    args: argparse.Namespace,
    task_spec_path: Path,
    task_spec: dict[str, Any],
    rollout: Rollout,
) -> list[str]:
    scene = task_spec.get("scene", {})
    collection = task_spec.get("collection", {})
    command = [
        "ros2",
        "run",
        "sim",
        "prepare_isaac_rollout",
        "--task-spec",
        task_spec_path.as_posix(),
        "--task-id",
        rollout.task["task_id"],
        "--variant-id",
        rollout.variant.get("variant_id", "nominal"),
        "--generated-usd",
        str(scene.get("generated_usd", "")),
        "--layout-yaml",
        str(scene.get("generated_layout_yaml", scene.get("source_yaml", ""))),
        "--timeout-s",
        str(float(args.prepare_timeout_s)),
        "--topic-timeout-s",
        str(float(args.prepare_topic_timeout_s)),
        "--camera-topic",
        str(collection.get("camera_topic", "/vla/cam")),
        "--odom-topic",
        str(collection.get("odom_topic", "/sim_odom")),
    ]
    if args.isaac_root:
        command.extend(["--isaac-root", args.isaac_root])
    if args.bridge_command_file:
        command.extend(["--command-file", args.bridge_command_file])
    if args.bridge_status_file:
        command.extend(["--status-file", args.bridge_status_file])
    if args.skip_prepare_topic_wait:
        command.append("--skip-topic-wait")
    return command


def start_process(command: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )


def stop_process(process: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def run_rollout(
    task_spec_path: Path,
    task_spec: dict[str, Any],
    rollout: Rollout,
    base_dir: Path,
    dataset_name: str,
    duration_s: float,
    startup_wait_s: float,
    stop_wait_s: float,
    logs_dir: Path,
    use_tracker: bool,
    prepare_scene_command: str | None,
    dry_run: bool,
    bridge_prepare_command: list[str] | None,
    wait_for_task_success: bool = True,
) -> bool:
    collection = task_spec.get("collection", {})
    rollout_dir = base_dir / rollout.trajectory_name
    commands: list[tuple[str, list[str]]] = []
    tracker_command = command_for_tracker(use_tracker)
    if tracker_command is not None:
        commands.append(("tracker", tracker_command))
    drive_cmd_vel_topic = None if use_tracker else str(collection.get("cmd_vel_topic", "/cmd_vel"))
    diagnostics_collection = dict(collection)
    if drive_cmd_vel_topic is not None:
        diagnostics_collection["expert_cmd_vel_topic"] = drive_cmd_vel_topic
    commands.append((
        "expert",
        command_for_expert(
            task_spec_path,
            task_spec,
            rollout,
            collection,
            rollout_dir,
            drive_cmd_vel_topic=drive_cmd_vel_topic,
        ),
    ))
    commands.append(("diagnostics", command_for_diagnostics(rollout_dir, diagnostics_collection)))
    commands.append(("collector", command_for_collector(task_spec_path, rollout, collection, base_dir, dataset_name)))
    wait_command = (
        command_for_task_success_wait(task_spec_path, rollout, collection, duration_s, rollout_dir)
        if wait_for_task_success
        else command_for_sim_duration_wait(collection, duration_s)
    )

    print(f"\n=== {rollout.trajectory_name} ===")
    if prepare_scene_command:
        rendered = render_prepare_command(prepare_scene_command, task_spec_path, task_spec, rollout)
        print(f"prepare: {rendered}")
    if bridge_prepare_command is not None:
        print(f"prepare: {' '.join(bridge_prepare_command)}")
    for name, command in commands:
        print(f"{name}: {' '.join(command)}")
    print(f"wait: {' '.join(wait_command)}")
    print(f"max_duration_s: {duration_s:.1f}")
    if dry_run:
        return True

    if prepare_scene_command:
        subprocess.run(
            render_prepare_command(prepare_scene_command, task_spec_path, task_spec, rollout),
            shell=True,
            check=True,
        )
    if bridge_prepare_command is not None:
        subprocess.run(bridge_prepare_command, check=True)

    running: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, command in commands:
            process = start_process(command, logs_dir / rollout.trajectory_name / f"{name}.log")
            running.append((name, process))
            time.sleep(startup_wait_s)

        wait_process = start_process(
            wait_command,
            logs_dir / rollout.trajectory_name / "wait.log",
        )
        while wait_process.poll() is None:
            for name, process in running:
                if process.poll() is not None:
                    print(f"{name} exited early with code {process.returncode}")
                    stop_process(wait_process, timeout_s=stop_wait_s)
                    return False
            time.sleep(0.5)
        if wait_process.returncode != 0:
            print(f"wait exited with code {wait_process.returncode}")
            return False
        return True
    finally:
        for _, process in reversed(running):
            stop_process(process, timeout_s=stop_wait_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_spec", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="sim_fenceline")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--fixed-duration-wait", action="store_true")
    parser.add_argument("--startup-wait-s", type=float, default=1.0)
    parser.add_argument("--stop-wait-s", type=float, default=5.0)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/sim_rollouts"))
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--variant-id", action="append", dest="variant_ids")
    parser.add_argument("--include-invalid-variants", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    tracker_group = parser.add_mutually_exclusive_group()
    tracker_group.add_argument(
        "--no-tracker",
        dest="no_tracker",
        action="store_true",
        default=True,
        help="Drive Isaac directly from the expert /cmd_vel publisher. This is the default.",
    )
    tracker_group.add_argument(
        "--use-tracker",
        dest="no_tracker",
        action="store_false",
        help="Use hw_interface sim_waypoint_tracking to drive from /vla/action_chunk.",
    )
    parser.add_argument(
        "--prepare-scene-command",
        default=None,
        help=(
            "Optional shell command run before each rollout. Placeholders: "
            "{scene_yaml}, {generated_layout_yaml}, {generated_usd}, {task_spec}, {task_id}, {variant_id}, {trajectory_name}."
        ),
    )
    parser.add_argument("--use-isaac-bridge", action="store_true")
    parser.add_argument("--isaac-root", default=None)
    parser.add_argument("--bridge-command-file", default=None)
    parser.add_argument("--bridge-status-file", default=None)
    parser.add_argument("--prepare-timeout-s", type=float, default=90.0)
    parser.add_argument("--prepare-topic-timeout-s", type=float, default=30.0)
    parser.add_argument("--skip-prepare-topic-wait", action="store_true")
    parser.add_argument(
        "--use-isaac-camera-pose-debug",
        action="store_true",
        help=(
            "Diagnostic mode: drive the expert from camera_world_pose in "
            "/isaac/scene_pose_debug instead of /sim_odom."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_spec_path = args.task_spec.resolve()
    task_spec = load_yaml(task_spec_path)
    collection = dict(task_spec.get("collection", {}))
    if args.use_isaac_camera_pose_debug:
        collection["use_isaac_camera_pose_debug"] = True
    task_spec["collection"] = collection
    base_dir = args.base_dir or Path(str(collection.get("base_dir", "sim_datasets/generated")))
    duration_s = float(args.duration_s if args.duration_s is not None else collection.get("duration_s", 20.0))
    rollouts = selected_rollouts(
        task_spec,
        set(args.task_ids) if args.task_ids else None,
        set(args.variant_ids) if args.variant_ids else None,
        only_planner_valid=not args.include_invalid_variants,
        limit=args.limit,
    )
    if not rollouts:
        print("No rollouts selected.", file=sys.stderr)
        return 1

    print(f"Selected {len(rollouts)} rollout(s).")
    failures = 0
    for rollout in rollouts:
        bridge_prepare_command = (
            default_bridge_prepare_command(args, task_spec_path, task_spec, rollout)
            if args.use_isaac_bridge
            else None
        )
        ok = run_rollout(
            task_spec_path=task_spec_path,
            task_spec=task_spec,
            rollout=rollout,
            base_dir=base_dir,
            dataset_name=args.dataset_name,
            duration_s=duration_s,
            startup_wait_s=float(args.startup_wait_s),
            stop_wait_s=float(args.stop_wait_s),
            logs_dir=args.logs_dir,
            use_tracker=not args.no_tracker,
            prepare_scene_command=args.prepare_scene_command,
            dry_run=bool(args.dry_run),
            bridge_prepare_command=bridge_prepare_command,
            wait_for_task_success=not bool(args.fixed_duration_wait),
        )
        if not ok:
            failures += 1
            print(f"FAILED: {rollout.trajectory_name}")
    print(f"Finished {len(rollouts) - failures}/{len(rollouts)} rollout(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

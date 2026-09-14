#!/usr/bin/env python3
"""Load an Isaac rollout scene, record images/poses, and drive with keyboard teleop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sim.manual_teleop_cmd_vel import main as teleop_main
from sim.prepare_isaac_rollout import resolve_path, rollout_paths


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def find_task(task_spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in task_spec.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    raise ValueError(f"task_id {task_id!r} not found in task spec")


def find_variant(task: dict[str, Any], variant_id: str) -> dict[str, Any]:
    for variant in task.get("trajectory_variants") or []:
        if variant.get("variant_id") == variant_id:
            return variant
    if variant_id == "nominal":
        return {"variant_id": "nominal", "variant_type": "nominal"}
    raise ValueError(f"variant_id {variant_id!r} not found in task")


def ros_param_args(params: dict[str, Any]) -> list[str]:
    result = ["--ros-args"]
    for key, value in params.items():
        if value is None:
            value = ""
        if isinstance(value, bool):
            value = "true" if value else "false"
        result.extend(["-p", f"{key}:={value}"])
    return result


def json_param(value: Any) -> str:
    return json.dumps(value or {}, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant-id", default="nominal")
    parser.add_argument("--generated-usd", default=None)
    parser.add_argument("--layout-yaml", default=None)
    parser.add_argument("--isaac-root", default=None)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--trajectory-name", default=None)
    parser.add_argument("--dataset-name", default="manual_sim_fenceline")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--camera-topic", default="/vla/cam")
    parser.add_argument("--odom-topic", default="/sim_odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--isaac-pose-debug-topic", default="/isaac/scene_pose_debug")
    parser.add_argument("--sample-frequency-hz", type=float, default=3.0)
    parser.add_argument("--prepare-timeout-s", type=float, default=120.0)
    parser.add_argument("--topic-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-speed-mps", type=float, default=0.30)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=0.45)
    parser.add_argument("--max-accel-mps2", type=float, default=0.25)
    parser.add_argument("--max-decel-mps2", type=float, default=0.35)
    parser.add_argument("--max-angular-accel-radps2", type=float, default=0.60)
    parser.add_argument("--auto-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-auto-speed-mps", type=float, default=0.10)
    parser.add_argument("--turn-slowdown", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_spec_path = args.task_spec.expanduser().resolve()
    task_spec = load_yaml(task_spec_path)
    task = find_task(task_spec, args.task_id)
    variant = find_variant(task, args.variant_id)
    isaac_root = Path(args.isaac_root).expanduser().resolve() if args.isaac_root else None
    usd_path, layout_path = rollout_paths(task_spec_path, task_spec, isaac_root, args.generated_usd, args.layout_yaml)

    print("Preparing Isaac scene...", flush=True)
    prepare_cmd = [
        "ros2",
        "run",
        "sim",
        "prepare_isaac_rollout",
        "--task-spec",
        task_spec_path.as_posix(),
        "--task-id",
        args.task_id,
        "--variant-id",
        args.variant_id,
        "--generated-usd",
        usd_path.as_posix(),
        "--layout-yaml",
        layout_path.as_posix(),
        "--timeout-s",
        str(args.prepare_timeout_s),
        "--topic-timeout-s",
        str(args.topic_timeout_s),
        "--camera-topic",
        args.camera_topic,
        "--odom-topic",
        args.odom_topic,
    ]
    subprocess.run(prepare_cmd, check=True)

    trajectory_name = args.trajectory_name or f"manual_{int(time.time())}_{Path(layout_path).stem}_{args.variant_id}"
    structured_task = {key: value for key, value in task.items() if key != "trajectory_variants"}
    structured_task["selected_variant"] = variant
    params = {
        "base_dir": args.base_dir.expanduser().as_posix(),
        "dataset_name": args.dataset_name,
        "trajectory_name": trajectory_name,
        "task_spec": task_spec_path.as_posix(),
        "task_id": args.task_id,
        "variant_id": args.variant_id,
        "variant_type": variant.get("variant_type", "manual"),
        "recovery_case": variant.get("recovery_case"),
        "language_instruction": args.instruction or task.get("instruction", "Follow the fence."),
        "structured_task_json": json_param(structured_task),
        "planner_settings_json": json_param(variant.get("planner_settings")),
        "speed_profile_json": json_param(
            variant.get(
                "speed_profile",
                {
                    "max_speed_mps": args.max_speed_mps,
                    "max_yaw_rate_radps": args.max_yaw_rate_radps,
                    "max_accel_mps2": args.max_accel_mps2,
                    "max_decel_mps2": args.max_decel_mps2,
                    "max_angular_accel_radps2": args.max_angular_accel_radps2,
                    "auto_speed": args.auto_speed,
                    "min_auto_speed_mps": args.min_auto_speed_mps,
                    "turn_slowdown": args.turn_slowdown,
                },
            )
        ),
        "camera_topic": args.camera_topic,
        "odom_topic": args.odom_topic,
        "cmd_vel_topic": args.cmd_vel_topic,
        "action_chunk_topic": "/unused_manual_action_chunk",
        "isaac_pose_debug_topic": args.isaac_pose_debug_topic,
        "use_isaac_camera_pose_debug": True,
        "sample_frequency_hz": args.sample_frequency_hz,
        "flip_isaac_y": False,
        "flip_scene_y": False,
        "flip_runtime_odom_y": False,
        "flip_runtime_odom_yaw": True,
    }
    collector_cmd = ["ros2", "run", "sim", "sim_dataset_collector", *ros_param_args(params)]
    print(f"Recording to {args.base_dir / trajectory_name}", flush=True)
    collector = subprocess.Popen(collector_cmd)
    try:
        return teleop_main(
            [
                "--cmd-vel-topic",
                args.cmd_vel_topic,
                "--max-speed-mps",
                str(args.max_speed_mps),
                "--max-yaw-rate-radps",
                str(args.max_yaw_rate_radps),
                "--max-accel-mps2",
                str(args.max_accel_mps2),
                "--max-decel-mps2",
                str(args.max_decel_mps2),
                "--max-angular-accel-radps2",
                str(args.max_angular_accel_radps2),
                "--auto-speed" if args.auto_speed else "--no-auto-speed",
                "--min-auto-speed-mps",
                str(args.min_auto_speed_mps),
                "--turn-slowdown",
                str(args.turn_slowdown),
            ]
        )
    finally:
        collector.terminate()
        try:
            collector.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            collector.kill()
            collector.wait()
        print("Manual recording stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Keyboard-driven manual recording session over a collection manifest."""

from __future__ import annotations

import argparse
import json
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
import yaml


HELP = """
Manual manifest recording

Scene / episode:
  n or Enter : load next manifest row
  r          : start/stop recording this scene
  c          : cancel/discard current recording, keep scene loaded
  z          : cancel/discard current recording and reload same scene
  k          : skip current row
  q          : quit

Driving:
  w          : resume auto-speed, or increase manual speed
  s          : decrease manual speed
  a/d        : increase/decrease yaw rate
  x          : zero yaw rate
  m          : toggle auto-speed/manual-speed
  space      : stop rover
"""


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def ramp(current: float, target: float, limit_per_s: float, dt: float) -> float:
    return current + clamp(target - current, -limit_per_s * dt, limit_per_s * dt)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def resolve_path(value: Any, isaac_root: Path | None, base: Path | None = None) -> Path:
    if value is None or str(value) == "":
        raise ValueError("expected non-empty path")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if isaac_root is not None:
        return isaac_root / path
    if base is not None:
        return base / path
    return path


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


def read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    key = sys.stdin.read(1)
    if key == "\r":
        return "\n"
    return key


class ManualManifestRecorder(Node):
    def __init__(self, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
        super().__init__("manual_manifest_recorder")
        self.args = args
        self.rows = rows
        self.index = -1
        self.current_row: dict[str, Any] | None = None
        self.current_trajectory_name: str | None = None
        self.collector: subprocess.Popen | None = None
        self.publisher = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.auto_speed = bool(args.auto_speed)
        self.auto_drive_enabled = False
        self.target_speed = float(args.max_speed_mps) if self.auto_speed else 0.0
        self.target_yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw = 0.0
        self.last_time = time.monotonic()
        self.session_records: list[dict[str, Any]] = []
        self.timer = self.create_timer(1.0 / max(float(args.publish_hz), 1.0), self.tick)

    def desired_speed(self) -> float:
        if not self.auto_speed:
            return self.target_speed
        if not self.auto_drive_enabled:
            return 0.0
        yaw_fraction = min(abs(self.target_yaw) / max(float(self.args.max_yaw_rate_radps), 1e-6), 1.0)
        speed = float(self.args.max_speed_mps) * (1.0 - float(self.args.turn_slowdown) * yaw_fraction)
        return clamp(speed, float(self.args.min_auto_speed_mps), float(self.args.max_speed_mps))

    def tick(self) -> None:
        now = time.monotonic()
        dt = max(now - self.last_time, 1e-3)
        self.last_time = now
        desired_speed = self.desired_speed()
        accel = self.args.max_accel_mps2 if abs(desired_speed) > abs(self.current_speed) else self.args.max_decel_mps2
        self.current_speed = ramp(self.current_speed, desired_speed, accel, dt)
        self.current_yaw = ramp(self.current_yaw, self.target_yaw, self.args.max_angular_accel_radps2, dt)
        msg = Twist()
        msg.linear.x = float(self.current_speed)
        msg.angular.z = float(self.current_yaw)
        self.publisher.publish(msg)

    def stop_rover(self) -> None:
        self.auto_drive_enabled = False
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw = 0.0
        self.publisher.publish(Twist())

    def selected_rows(self) -> list[dict[str, Any]]:
        return self.rows

    def next_row(self) -> None:
        if self.collector is not None:
            print("Stop recording with r before loading the next scene.", flush=True)
            return
        self.stop_rover()
        self.index += 1
        if self.index >= len(self.selected_rows()):
            print("No more manifest rows.", flush=True)
            self.index = len(self.selected_rows()) - 1
            return
        row = self.selected_rows()[self.index]
        self.current_row = row
        self.current_trajectory_name = None
        print(f"\n[{self.index + 1}/{len(self.selected_rows())}] Loading {row.get('rollout_id')}", flush=True)
        self.prepare_scene(row)

    def skip_row(self) -> None:
        if self.collector is not None:
            print("Stop recording before skipping.", flush=True)
            return
        if self.current_row is not None:
            print(f"Skipped {self.current_row.get('rollout_id')}", flush=True)
        self.next_row()

    def prepare_scene(self, row: dict[str, Any]) -> None:
        manifest_base = self.args.manifest.expanduser().resolve().parent
        isaac_root = self.args.isaac_root.expanduser().resolve() if self.args.isaac_root else None
        task_spec = resolve_path(row.get("task_spec"), None, manifest_base)
        generated_usd = resolve_path(row.get("visual_usd") or row.get("generated_usd"), isaac_root, manifest_base)
        layout_yaml = resolve_path(row.get("layout_yaml"), isaac_root, manifest_base)
        cmd = [
            "ros2",
            "run",
            "sim",
            "prepare_isaac_rollout",
            "--task-spec",
            task_spec.as_posix(),
            "--task-id",
            str(row.get("task_id")),
            "--variant-id",
            str(row.get("variant_id", "nominal")),
            "--generated-usd",
            generated_usd.as_posix(),
            "--layout-yaml",
            layout_yaml.as_posix(),
            "--isaac-root",
            isaac_root.as_posix() if isaac_root else "",
            "--timeout-s",
            str(self.args.prepare_timeout_s),
            "--topic-timeout-s",
            str(self.args.topic_timeout_s),
            "--camera-topic",
            self.args.camera_topic,
            "--odom-topic",
            self.args.odom_topic,
        ]
        subprocess.run([part for part in cmd if part != ""], check=True)
        print("Scene ready. Press r to start recording, then drive with WASD.", flush=True)

    def start_recording(self) -> None:
        if self.current_row is None:
            print("Load a row first with n.", flush=True)
            return
        if self.collector is not None:
            print("Already recording.", flush=True)
            return
        row = self.current_row
        task_spec = resolve_path(row.get("task_spec"), None, self.args.manifest.expanduser().resolve().parent)
        rollout_id = str(row.get("rollout_id") or f"row_{self.index + 1:04d}")
        trajectory_name = f"manual_{self.index + 1:04d}_{rollout_id}"[:220]
        self.current_trajectory_name = trajectory_name
        collection = row.get("collection") if isinstance(row.get("collection"), dict) else {}
        structured_task = {
            "task_id": row.get("task_id"),
            "task_type": row.get("task_type"),
            "scenario_tags": row.get("scenario_tags", []),
            "selected_variant": {
                "variant_id": row.get("variant_id", "manual"),
                "variant_type": row.get("variant_type", "manual"),
                "recovery_case": row.get("recovery_case"),
            },
            "manual_source_rollout_id": rollout_id,
            "layout_yaml": row.get("layout_yaml"),
            "visual_usd": row.get("visual_usd"),
            "visual_id": row.get("visual_id"),
        }
        params = {
            "base_dir": self.args.base_dir.expanduser().as_posix(),
            "dataset_name": self.args.dataset_name,
            "trajectory_name": trajectory_name,
            "task_spec": task_spec.as_posix(),
            "task_id": row.get("task_id"),
            "variant_id": row.get("variant_id", "manual"),
            "variant_type": row.get("variant_type", "manual"),
            "recovery_case": row.get("recovery_case"),
            "language_instruction": self.args.instruction or row.get("instruction") or "Follow the fence.",
            "structured_task_json": json_param(structured_task),
            "planner_settings_json": "{}",
            "speed_profile_json": json_param(
                {
                    "max_speed_mps": self.args.max_speed_mps,
                    "max_yaw_rate_radps": self.args.max_yaw_rate_radps,
                    "max_accel_mps2": self.args.max_accel_mps2,
                    "max_decel_mps2": self.args.max_decel_mps2,
                    "max_angular_accel_radps2": self.args.max_angular_accel_radps2,
                    "auto_speed": self.auto_speed,
                    "min_auto_speed_mps": self.args.min_auto_speed_mps,
                    "turn_slowdown": self.args.turn_slowdown,
                    "manual_control": True,
                }
            ),
            "camera_topic": collection.get("camera_topic", self.args.camera_topic),
            "odom_topic": collection.get("odom_topic", self.args.odom_topic),
            "cmd_vel_topic": self.args.cmd_vel_topic,
            "action_chunk_topic": "/unused_manual_action_chunk",
            "isaac_pose_debug_topic": collection.get("isaac_pose_debug_topic", self.args.isaac_pose_debug_topic),
            "use_isaac_camera_pose_debug": True,
            "sample_frequency_hz": self.args.sample_frequency_hz,
            "flip_isaac_y": False,
            "flip_scene_y": False,
            "flip_runtime_odom_y": False,
            "flip_runtime_odom_yaw": True,
        }
        cmd = ["ros2", "run", "sim", "sim_dataset_collector", *ros_param_args(params)]
        self.collector = subprocess.Popen(cmd)
        self.auto_drive_enabled = self.auto_speed
        if self.auto_speed:
            self.target_speed = float(self.args.max_speed_mps)
        print(f"Recording started: {self.args.base_dir / trajectory_name}", flush=True)

    def trajectory_dir(self) -> Path | None:
        if not self.current_trajectory_name:
            return None
        return self.args.base_dir.expanduser() / self.current_trajectory_name

    def remove_current_trajectory_dir(self) -> None:
        path = self.trajectory_dir()
        if path is None:
            return
        base_dir = self.args.base_dir.expanduser().resolve()
        resolved = path.resolve()
        if base_dir != resolved and base_dir not in resolved.parents:
            raise RuntimeError(f"refusing to remove trajectory outside base_dir: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
            print(f"Discarded recording folder: {resolved}", flush=True)

    def stop_recording(self, *, save_session_record: bool = True) -> None:
        if self.collector is None:
            print("Not recording.", flush=True)
            return
        self.stop_rover()
        self.collector.terminate()
        try:
            self.collector.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.collector.kill()
            self.collector.wait()
        self.collector = None
        if not save_session_record:
            self.remove_current_trajectory_dir()
            print(f"Recording cancelled: {self.current_trajectory_name}", flush=True)
            self.current_trajectory_name = None
            return
        record = {
            "manifest_index": self.index,
            "rollout_id": None if self.current_row is None else self.current_row.get("rollout_id"),
            "trajectory_name": self.current_trajectory_name,
            "trajectory_dir": (self.args.base_dir / str(self.current_trajectory_name)).as_posix()
            if self.current_trajectory_name
            else None,
            "stopped_at": time.time(),
        }
        self.session_records.append(record)
        self.write_session_index()
        print(f"Recording stopped: {self.current_trajectory_name}", flush=True)

    def cancel_recording(self) -> None:
        if self.collector is None:
            print("No active recording to cancel.", flush=True)
            return
        self.stop_recording(save_session_record=False)

    def restart_scene(self) -> None:
        if self.current_row is None:
            print("Load a row first with n.", flush=True)
            return
        if self.collector is not None:
            self.stop_recording(save_session_record=False)
        self.stop_rover()
        print(f"Reloading {self.current_row.get('rollout_id')}", flush=True)
        self.prepare_scene(self.current_row)

    def write_session_index(self) -> None:
        self.args.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.base_dir / "manual_session_index.json"
        path.write_text(json.dumps(self.session_records, indent=2), encoding="utf-8")

    def handle_key(self, key: str) -> bool:
        if key == "q":
            return False
        if key in {"n", "\n"}:
            self.next_row()
        elif key == "k":
            self.skip_row()
        elif key == "r":
            if self.collector is None:
                self.start_recording()
            else:
                self.stop_recording()
        elif key == "c":
            self.cancel_recording()
        elif key == "z":
            self.restart_scene()
        elif key == "w":
            if self.auto_speed:
                self.auto_drive_enabled = True
                self.target_speed = float(self.args.max_speed_mps)
            else:
                self.target_speed = clamp(self.target_speed + self.args.speed_step_mps, 0.0, self.args.max_speed_mps)
        elif key == "s":
            self.target_speed = clamp(self.target_speed - self.args.speed_step_mps, 0.0, self.args.max_speed_mps)
        elif key == "a":
            self.target_yaw = clamp(self.target_yaw + self.args.yaw_step_radps, -self.args.max_yaw_rate_radps, self.args.max_yaw_rate_radps)
        elif key == "d":
            self.target_yaw = clamp(self.target_yaw - self.args.yaw_step_radps, -self.args.max_yaw_rate_radps, self.args.max_yaw_rate_radps)
        elif key == "x":
            self.target_yaw = 0.0
        elif key == "m":
            self.auto_speed = not self.auto_speed
            self.auto_drive_enabled = self.auto_speed and self.collector is not None
            self.target_speed = float(self.args.max_speed_mps) if self.auto_speed else 0.0
            print(f"Speed mode: {'auto' if self.auto_speed else 'manual'}", flush=True)
        elif key == " ":
            self.stop_rover()
        return True

    def shutdown(self) -> None:
        if self.collector is not None:
            self.stop_recording()
        self.stop_rover()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--isaac-root", type=Path, default=Path("~/isaac_files"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="manual_sim_fenceline")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--camera-topic", default="/vla/cam")
    parser.add_argument("--odom-topic", default="/sim_odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--isaac-pose-debug-topic", default="/isaac/scene_pose_debug")
    parser.add_argument("--sample-frequency-hz", type=float, default=3.0)
    parser.add_argument("--prepare-timeout-s", type=float, default=120.0)
    parser.add_argument("--topic-timeout-s", type=float, default=30.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--max-speed-mps", type=float, default=0.30)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=0.45)
    parser.add_argument("--max-accel-mps2", type=float, default=0.25)
    parser.add_argument("--max-decel-mps2", type=float, default=0.35)
    parser.add_argument("--max-angular-accel-radps2", type=float, default=0.60)
    parser.add_argument("--auto-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-auto-speed-mps", type=float, default=0.10)
    parser.add_argument("--turn-slowdown", type=float, default=0.70)
    parser.add_argument("--speed-step-mps", type=float, default=0.05)
    parser.add_argument("--yaw-step-radps", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_yaml(args.manifest.expanduser().resolve())
    rows = [row for row in manifest.get("rollouts", []) if isinstance(row, dict)]
    if not rows:
        print(f"No rollout rows found in {args.manifest}", flush=True)
        return 1
    rclpy.init()
    node = ManualManifestRecorder(args, rows)
    old_settings = termios.tcgetattr(sys.stdin)
    print(HELP, flush=True)
    print(f"Loaded {len(rows)} manifest row(s). Press n to load the first scene.", flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.02)
            key = read_key(0.0)
            if key:
                running = node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare Isaac for one task-spec rollout through the file-based bridge."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


DEFAULT_COMMAND_FILE = Path(os.environ.get("ISAAC_ROLLOUT_COMMAND_FILE", "/tmp/isaac_rollout_request.json"))
DEFAULT_STATUS_FILE = Path(os.environ.get("ISAAC_ROLLOUT_STATUS_FILE", "/tmp/isaac_rollout_status.json"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def resolve_path(value: str | None, root: Path | None, base: Path | None = None) -> Path:
    if not value:
        raise ValueError("Expected a non-empty path.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if root is not None:
        return (root / path).resolve()
    if base is not None:
        return (base / path).resolve()
    return path.resolve()


def find_task(task_spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in task_spec.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    available = [task.get("task_id", "<missing>") for task in task_spec.get("tasks", [])]
    raise ValueError(f"task_id {task_id!r} not found. Available: {available}")


def find_variant(task: dict[str, Any], variant_id: str) -> dict[str, Any]:
    for variant in task.get("trajectory_variants") or []:
        if variant.get("variant_id") == variant_id:
            return variant
    if variant_id == "nominal":
        return {"variant_id": "nominal", "variant_type": "nominal"}
    available = [variant.get("variant_id", "<missing>") for variant in task.get("trajectory_variants") or []]
    raise ValueError(f"variant_id {variant_id!r} not found. Available: {available}")


def choose_isaac_root(raw_root: str | None) -> Path | None:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    env_root = os.environ.get("ISAAC_SCENE_ROOT") or os.environ.get("ISAAC_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return None


def rollout_paths(
    task_spec_path: Path,
    task_spec: dict[str, Any],
    isaac_root: Path | None,
    generated_usd: str | None,
    layout_yaml: str | None,
) -> tuple[Path, Path]:
    scene = task_spec.get("scene", {})
    usd_value = generated_usd or scene.get("generated_usd")
    layout_value = layout_yaml or scene.get("generated_layout_yaml") or scene.get("source_yaml")
    spec_base = task_spec_path.parent
    usd_path = resolve_path(str(usd_value), isaac_root, spec_base)
    layout_path = resolve_path(str(layout_value), isaac_root, spec_base)
    return usd_path, layout_path


def wait_for_bridge(status_file: Path, request_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_status = None
    while time.monotonic() < deadline:
        status = load_json(status_file)
        if status is not None:
            last_status = status
            if status.get("request_id") == request_id:
                state = status.get("state")
                if state == "ready":
                    return status
                if state == "error":
                    raise RuntimeError(f"Isaac bridge failed: {status.get('message', '<no message>')}")
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for Isaac bridge request {request_id}. Last status: {last_status}")


def ros_topics() -> set[str]:
    result = subprocess.run(
        ["ros2", "topic", "list"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def wait_for_topics(required_topics: list[str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    required = set(required_topics)
    seen: set[str] = set()
    while time.monotonic() < deadline:
        seen = ros_topics()
        missing = required - seen
        if not missing:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for ROS topics: {sorted(required - seen)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant-id", default="nominal")
    parser.add_argument("--generated-usd", default=None)
    parser.add_argument("--layout-yaml", default=None)
    parser.add_argument("--isaac-root", default=None)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND_FILE)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--topic-timeout-s", type=float, default=30.0)
    parser.add_argument("--camera-topic", default="/vla/cam")
    parser.add_argument("--odom-topic", default="/sim_odom")
    parser.add_argument("--skip-topic-wait", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_spec_path = args.task_spec.resolve()
    task_spec = load_yaml(task_spec_path)
    task = find_task(task_spec, args.task_id)
    variant = find_variant(task, args.variant_id)
    isaac_root = choose_isaac_root(args.isaac_root)
    usd_path, layout_path = rollout_paths(
        task_spec_path,
        task_spec,
        isaac_root,
        args.generated_usd,
        args.layout_yaml,
    )

    request_id = f"{int(time.time())}_{uuid.uuid4().hex[:10]}"
    request = {
        "request_id": request_id,
        "task_spec": str(task_spec_path),
        "task_id": args.task_id,
        "variant_id": args.variant_id,
        "variant_type": variant.get("variant_type", "nominal"),
        "recovery_case": variant.get("recovery_case"),
        "usd_path": str(usd_path),
        "layout_yaml": str(layout_path),
        "created_time": time.time(),
    }
    atomic_write_json(args.command_file, request)
    status = wait_for_bridge(args.status_file, request_id, args.timeout_s)

    if not args.skip_topic_wait:
        wait_for_topics([args.odom_topic, args.camera_topic], args.topic_timeout_s)

    print(
        "Isaac ready: "
        f"request_id={request_id} "
        f"usd={status.get('usd_path')} "
        f"layout={status.get('layout_yaml')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

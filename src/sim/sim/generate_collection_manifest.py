#!/usr/bin/env python3
"""Generate a rollout collection manifest from task spec YAML files."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unnamed"


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.yaml")))
            paths.extend(sorted(input_path.rglob("*.yml")))
        else:
            paths.append(input_path)
    return sorted(set(path.resolve() for path in paths))


def variant_is_planner_valid(variant: dict[str, Any]) -> bool:
    validation = variant.get("planner_validation")
    return not isinstance(validation, dict) or bool(validation.get("valid", False))


def has_nonzero_start_delta(variant: dict[str, Any]) -> bool:
    delta = variant.get("start_pose_delta")
    if not isinstance(delta, dict):
        return False
    return any(abs(float(delta.get(key, 0.0))) > 1e-9 for key in ("x_m", "y_m", "yaw_rad"))


def resolve_path(value: str | None, isaac_root: Path | None, base: Path | None = None) -> Path | None:
    if not value:
        return None
    if value.startswith("/"):
        return Path(value)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if isaac_root is not None:
        return (isaac_root / path).resolve()
    if base is not None:
        return (base / path).resolve()
    return path


def as_manifest_path(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def rollout_trajectory_name(suite_id: str, task_id: str, variant_id: str, visual_id: str | None = None) -> str:
    parts = [
        normalize_name(suite_id),
        f"task_{normalize_name(task_id)}",
        f"variant_{normalize_name(variant_id)}",
    ]
    if visual_id:
        parts.append(f"visual_{normalize_name(visual_id)}")
    return "__".join(parts)


def pose_variant_layout_name(scene: dict[str, Any], task_id: str, variant_id: str) -> str:
    output_name = str(scene.get("scene_id") or scene.get("generated_usd") or "layout")
    output_name = Path(output_name).stem.removesuffix(".usd")
    return "__".join(
        [
            output_name,
            f"task_{normalize_name(task_id)}",
            f"variant_{normalize_name(variant_id)}",
        ]
    )


def generated_usd_from_layout(layout_path: Path, isaac_root: Path | None) -> Path | None:
    if not layout_path.exists():
        return None
    layout = load_yaml(layout_path)
    output_dir = layout.get("output_dir")
    output_name = layout.get("output_name")
    if not output_dir or not output_name:
        return None
    name = str(output_name)
    if not name.endswith(".usd"):
        name = f"{name}.usd"
    return resolve_path(f"{output_dir}/{name}", isaac_root, layout_path.parent)


def find_pose_variant_layout(
    pose_variant_layout_dir: Path | None,
    scene: dict[str, Any],
    task_id: str,
    variant_id: str,
) -> Path | None:
    if pose_variant_layout_dir is None:
        return None
    candidate = pose_variant_layout_dir / f"{pose_variant_layout_name(scene, task_id, variant_id)}.yaml"
    return candidate.resolve() if candidate.exists() else None


def visual_usds_for(base_usd: Path | None, include_visual_variations: bool, include_base_usd: bool) -> list[tuple[str, Path | None]]:
    if base_usd is None:
        return [("base", None)]

    variants: list[tuple[str, Path | None]] = []
    if include_base_usd:
        variants.append(("base", base_usd))
    if not include_visual_variations:
        return variants or [("base", base_usd)]

    directory = base_usd.parent
    stem = base_usd.stem
    prefix = stem.removesuffix("_base")
    for usd_path in sorted(directory.glob(f"{prefix}_*.usd")):
        if usd_path.resolve() == base_usd.resolve():
            continue
        visual_id = usd_path.stem.removeprefix(f"{prefix}_")
        variants.append((visual_id, usd_path.resolve()))
    return variants or [("base", base_usd)]


def build_rollouts_for_spec(
    spec_path: Path,
    spec: dict[str, Any],
    isaac_root: Path | None,
    pose_variant_layout_dir: Path | None,
    include_invalid_variants: bool,
    include_visual_variations: bool,
    include_base_usd: bool,
) -> list[dict[str, Any]]:
    scene = spec.get("scene", {})
    collection = spec.get("collection", {})
    suite_id = str(spec.get("suite_id") or scene.get("scene_id") or spec_path.stem)
    base_layout = resolve_path(
        scene.get("generated_layout_yaml") or scene.get("source_yaml"),
        isaac_root,
        spec_path.parent,
    )
    base_usd = resolve_path(scene.get("generated_usd"), isaac_root, spec_path.parent)

    rows: list[dict[str, Any]] = []
    for task in spec.get("tasks", []):
        task_id = str(task.get("task_id", ""))
        if not task_id:
            continue
        variants = task.get("trajectory_variants") or [{"variant_id": "nominal", "variant_type": "nominal"}]
        for variant in variants:
            variant_id = str(variant.get("variant_id", "nominal"))
            if not include_invalid_variants and not variant_is_planner_valid(variant):
                continue

            requires_pose_variant = has_nonzero_start_delta(variant)
            layout_path = base_layout
            usd_path = base_usd
            pose_variant_layout = None
            if requires_pose_variant:
                pose_variant_layout = find_pose_variant_layout(pose_variant_layout_dir, scene, task_id, variant_id)
                if pose_variant_layout is not None:
                    layout_path = pose_variant_layout
                    usd_path = generated_usd_from_layout(pose_variant_layout, isaac_root) or usd_path

            for visual_id, visual_usd in visual_usds_for(usd_path, include_visual_variations, include_base_usd):
                trajectory_name = rollout_trajectory_name(suite_id, task_id, variant_id, visual_id)
                row = {
                    "rollout_id": trajectory_name,
                    "status": "pending",
                    "scene_id": scene.get("scene_id"),
                    "config_type": scene.get("config_type"),
                    "task_spec": spec_path.as_posix(),
                    "task_id": task_id,
                    "task_type": task.get("task_type"),
                    "variant_id": variant_id,
                    "variant_type": variant.get("variant_type", "nominal"),
                    "recovery_case": variant.get("recovery_case"),
                    "requires_pose_variant": requires_pose_variant,
                    "pose_variant_ready": (not requires_pose_variant) or pose_variant_layout is not None,
                    "layout_yaml": as_manifest_path(layout_path),
                    "generated_usd": as_manifest_path(usd_path),
                    "visual_id": visual_id,
                    "visual_usd": as_manifest_path(visual_usd),
                    "trajectory_name": trajectory_name,
                    "instruction": task.get("instruction", ""),
                    "collection": {
                        "base_dir": collection.get("base_dir"),
                        "duration_s": collection.get("duration_s"),
                        "camera_topic": collection.get("camera_topic", "/vla/cam"),
                        "odom_topic": collection.get("odom_topic", "/sim_odom"),
                        "cmd_vel_topic": collection.get("cmd_vel_topic", "/cmd_vel"),
                        "action_chunk_topic": collection.get("action_chunk_topic", "/vla/action_chunk"),
                        "sample_frequency_hz": collection.get("sample_frequency_hz", 3.0),
                    },
                }
                if requires_pose_variant and pose_variant_layout is None:
                    row["warning"] = "Pose-perturbed variant has no matching expanded layout YAML."
                rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Task spec YAML file(s) or directories.")
    parser.add_argument("--output", type=Path, default=Path("src/sim/config/collection_manifest.yaml"))
    parser.add_argument("--isaac-root", type=Path, default=None)
    parser.add_argument("--pose-variant-layout-dir", type=Path, default=None)
    parser.add_argument("--include-invalid-variants", action="store_true")
    parser.add_argument("--include-visual-variations", action="store_true")
    parser.add_argument("--no-base-usd", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_paths = expand_inputs(args.inputs)
    if not spec_paths:
        raise SystemExit("No task spec YAML files found.")

    isaac_root = args.isaac_root.resolve() if args.isaac_root else None
    pose_variant_layout_dir = args.pose_variant_layout_dir.resolve() if args.pose_variant_layout_dir else None
    rollouts: list[dict[str, Any]] = []
    for spec_path in spec_paths:
        spec = load_yaml(spec_path)
        if not isinstance(spec.get("scene"), dict) or not isinstance(spec.get("tasks"), list):
            continue
        rollouts.extend(
            build_rollouts_for_spec(
                spec_path=spec_path,
                spec=spec,
                isaac_root=isaac_root,
                pose_variant_layout_dir=pose_variant_layout_dir,
                include_invalid_variants=bool(args.include_invalid_variants),
                include_visual_variations=bool(args.include_visual_variations),
                include_base_usd=not bool(args.no_base_usd),
            )
        )

    manifest = {
        "manifest_version": 0.1,
        "created_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_specs": [path.as_posix() for path in spec_paths],
        "isaac_root": as_manifest_path(isaac_root),
        "pose_variant_layout_dir": as_manifest_path(pose_variant_layout_dir),
        "include_visual_variations": bool(args.include_visual_variations),
        "include_base_usd": not bool(args.no_base_usd),
        "counts": {
            "specs": len(spec_paths),
            "rollouts": len(rollouts),
            "pose_variant_rollouts": sum(1 for row in rollouts if row["requires_pose_variant"]),
            "missing_pose_variant_rollouts": sum(
                1 for row in rollouts if row["requires_pose_variant"] and not row["pose_variant_ready"]
            ),
        },
        "rollouts": rollouts,
    }
    write_yaml(args.output, manifest)
    print(f"Wrote {args.output} ({len(rollouts)} rollout rows)")
    if args.summary:
        counts = manifest["counts"]
        print(
            f"specs={counts['specs']} rollouts={counts['rollouts']} "
            f"pose_variants={counts['pose_variant_rollouts']} "
            f"missing_pose_variants={counts['missing_pose_variant_rollouts']}"
        )


if __name__ == "__main__":
    main()

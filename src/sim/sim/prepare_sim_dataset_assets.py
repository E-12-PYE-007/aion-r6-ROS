#!/usr/bin/env python3
"""Prepare sim task specs, valid specs, recovery layouts, and recovery USDs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from sim.expand_pose_variants import expand_layout


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def expand_inputs(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_dir():
        paths = [*input_path.rglob("*.yaml"), *input_path.rglob("*.yml")]
        return sorted(path.resolve() for path in paths)
    return [input_path.resolve()]


def run_command(command: list[str], label: str, cwd: Path | None = None) -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"Stage: {label}", flush=True)
    print("=" * 80, flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)
    print(f"[OK] {label}", flush=True)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def spec_paths(path: Path) -> list[Path]:
    return expand_inputs(path)


def task_ids_for_spec(spec: dict[str, Any]) -> list[str]:
    task_ids: list[str] = []
    for task in spec.get("tasks") or []:
        task_id = task.get("task_id")
        if task_id:
            task_ids.append(str(task_id))
    return task_ids


def expand_pose_variant_layouts(valid_spec_dir: Path, pose_variant_dir: Path) -> list[Path]:
    written: list[Path] = []
    for spec_path in spec_paths(valid_spec_dir):
        spec = load_yaml(spec_path)
        scene = spec.get("scene") or {}
        layout_value = scene.get("generated_layout_yaml") or scene.get("source_yaml")
        if not layout_value:
            print(f"Skipping {spec_path}: no generated layout YAML in scene metadata", flush=True)
            continue
        layout_path = Path(str(layout_value)).expanduser().resolve()
        if not layout_path.exists():
            print(f"Skipping {spec_path}: layout YAML does not exist: {layout_path}", flush=True)
            continue
        task_ids = task_ids_for_spec(spec)
        if not task_ids:
            print(f"Skipping {spec_path}: no tasks", flush=True)
            continue
        written.extend(
            expand_layout(
                layout_path=layout_path,
                task_spec_path=spec_path.resolve(),
                output_dir=pose_variant_dir,
                task_ids=task_ids,
                variant_ids=None,
            )
        )
    return written


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def base_usds_for_layout(layout_yaml: Path, isaac_root: Path) -> list[Path]:
    layout = load_yaml(layout_yaml)
    output_dir = (isaac_root / str(layout["output_dir"])).resolve()
    output_name = str(layout["output_name"])
    config_type = str(layout.get("config_type", ""))
    if config_type == "road":
        names = []
        for surface_name in (layout.get("road_surfaces") or {}).keys():
            road_name = f"{output_name.removesuffix('_base')}_{surface_name}_base.usd"
            names.append(output_dir / road_name)
        return names
    if not output_name.endswith(".usd"):
        output_name = f"{output_name}.usd"
    return [output_dir / output_name]


def generate_pose_variant_usds(
    pose_variant_dir: Path,
    isaac_root: Path,
    variation_config: Path | None,
    skip_variations: bool,
) -> int:
    layout_paths = spec_paths(pose_variant_dir)
    generated_count = 0
    for layout_yaml in layout_paths:
        layout = load_yaml(layout_yaml)
        config_type = str(layout.get("config_type", ""))
        if config_type == "road":
            road_surfaces = list((layout.get("road_surfaces") or {}).keys())
            if not road_surfaces:
                raise ValueError(f"{layout_yaml} is a road layout with no road_surfaces.")
            for surface_name in road_surfaces:
                run_command(
                    [
                        sys.executable,
                        "scripts/isaac_scene_generator.py",
                        relative_or_absolute(layout_yaml, isaac_root),
                        str(surface_name),
                    ],
                    label=f"Generate pose-variant road USD: {layout_yaml.name} / {surface_name}",
                    cwd=isaac_root,
                )
        else:
            run_command(
                [
                    sys.executable,
                    "scripts/isaac_scene_generator.py",
                    relative_or_absolute(layout_yaml, isaac_root),
                ],
                label=f"Generate pose-variant USD: {layout_yaml.name}",
                cwd=isaac_root,
            )

        base_usds = base_usds_for_layout(layout_yaml, isaac_root)
        generated_count += len(base_usds)
        if skip_variations:
            continue
        if variation_config is None:
            raise ValueError("--variation-config is required unless --skip-visual-variations is set.")
        for base_usd in base_usds:
            if not base_usd.exists():
                raise FileNotFoundError(f"Expected base USD was not generated: {base_usd}")
            run_command(
                [
                    sys.executable,
                    "scripts/variation_generator.py",
                    relative_or_absolute(base_usd, isaac_root),
                    relative_or_absolute(variation_config, isaac_root),
                ],
                label=f"Generate pose-variant visual USDs: {base_usd.name}",
                cwd=isaac_root,
            )
    return generated_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-dir",
        type=Path,
        required=True,
        help="Directory containing generated Isaac layout YAMLs.",
    )
    parser.add_argument(
        "--task-spec-dir",
        type=Path,
        default=Path("src/sim/config/generated_task_specs"),
        help="Output directory for generated task specs.",
    )
    parser.add_argument(
        "--valid-task-spec-dir",
        type=Path,
        default=Path("src/sim/config/scene_valid_task_specs"),
        help="Output directory for planner-filtered valid task specs.",
    )
    parser.add_argument(
        "--pose-variant-dir",
        type=Path,
        required=True,
        help="Output directory for expanded recovery pose variant layout YAMLs.",
    )
    parser.add_argument(
        "--isaac-root",
        type=Path,
        required=True,
        help="Isaac scene-generation repository root.",
    )
    parser.add_argument(
        "--variation-config",
        type=Path,
        default=None,
        help="Isaac variation config YAML used to generate visual USDs for pose variants.",
    )
    parser.add_argument(
        "--task-family",
        choices=("following_only", "all_supported"),
        default="following_only",
    )
    parser.add_argument("--recovery-jitter-count", type=int, default=0)
    parser.add_argument("--recovery-jitter-seed", type=int, default=17)
    parser.add_argument("--max-start-distance-m", type=float, default=2.0)
    parser.add_argument("--skip-visual-variations", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete output dirs before running.")
    parser.add_argument(
        "--skip-pose-usd-generation",
        action="store_true",
        help="Stop after expanding pose-variant layout YAMLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout_dir = args.layout_dir.expanduser().resolve()
    task_spec_dir = args.task_spec_dir.expanduser().resolve()
    valid_task_spec_dir = args.valid_task_spec_dir.expanduser().resolve()
    pose_variant_dir = args.pose_variant_dir.expanduser().resolve()
    isaac_root = args.isaac_root.expanduser().resolve()
    variation_config = args.variation_config.expanduser().resolve() if args.variation_config else None

    if args.clean:
        clean_dir(task_spec_dir)
        clean_dir(valid_task_spec_dir)
        clean_dir(pose_variant_dir)
    else:
        task_spec_dir.mkdir(parents=True, exist_ok=True)
        valid_task_spec_dir.mkdir(parents=True, exist_ok=True)
        pose_variant_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "generate_scene_task_specs",
            layout_dir.as_posix(),
            "--output-dir",
            task_spec_dir.as_posix(),
            "--task-family",
            str(args.task_family),
            "--recovery-jitter-count",
            str(max(0, int(args.recovery_jitter_count))),
            "--recovery-jitter-seed",
            str(int(args.recovery_jitter_seed)),
            "--summary",
        ],
        label="Generate task specs",
    )

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "validate_scene_task_specs",
            task_spec_dir.as_posix(),
            "--check-planner",
            "--allow-invalid",
            "--verbose",
            "--max-start-distance-m",
            str(float(args.max_start_distance_m)),
            "--write-valid-output-dir",
            valid_task_spec_dir.as_posix(),
        ],
        label="Validate task specs and write valid specs",
    )

    print("\n" + "=" * 80, flush=True)
    print("Stage: Expand recovery pose-variant layout YAMLs", flush=True)
    print("=" * 80, flush=True)
    written = expand_pose_variant_layouts(valid_task_spec_dir, pose_variant_dir)
    print(f"[OK] Generated {len(written)} pose-variant layout YAML(s).", flush=True)

    generated_usd_count = 0
    if not args.skip_pose_usd_generation:
        generated_usd_count = generate_pose_variant_usds(
            pose_variant_dir=pose_variant_dir,
            isaac_root=isaac_root,
            variation_config=variation_config,
            skip_variations=bool(args.skip_visual_variations),
        )

    print("\n" + "=" * 80, flush=True)
    print("Sim dataset asset prep complete.", flush=True)
    print("=" * 80, flush=True)
    print(f"layout_dir: {layout_dir}", flush=True)
    print(f"task_spec_dir: {task_spec_dir}", flush=True)
    print(f"valid_task_spec_dir: {valid_task_spec_dir}", flush=True)
    print(f"pose_variant_dir: {pose_variant_dir}", flush=True)
    print(f"pose_variant_layouts: {len(written)}", flush=True)
    print(f"pose_variant_base_usds: {generated_usd_count}", flush=True)


if __name__ == "__main__":
    main()

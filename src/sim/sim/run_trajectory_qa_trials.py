#!/usr/bin/env python3
"""Run a one-task-per-layout trajectory QA batch and plot planned vs executed paths."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


PREFERRED_TASK_TYPES = [
    "follow_fence_sequence",
    "follow_fence",
    "follow_road",
    "follow_shed_side",
]


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


def run_command(command: list[str], label: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("\n" + "=" * 80, flush=True)
    print(f"Stage: {label}", flush=True)
    print("=" * 80, flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, check=check)
    if result.returncode == 0:
        print(f"[OK] {label}", flush=True)
    else:
        print(f"[WARN] {label} exited with code {result.returncode}", flush=True)
    return result


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def infer_config_type(layout_path: Path) -> str:
    try:
        data = load_yaml(layout_path)
    except Exception:
        return "unknown"
    config_type = data.get("config_type")
    return str(config_type) if config_type else "unknown"


def layout_family(layout_dir: Path, layout_path: Path) -> str:
    rel = layout_path.resolve().relative_to(layout_dir.resolve())
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = layout_path.stem
    for suffix in ("_seed", "_roverstart", "_roverapproach"):
        if suffix in stem:
            return stem.split(suffix, 1)[0]
    return stem


def layout_sort_key(layout_path: Path) -> tuple[int, str]:
    name = layout_path.as_posix()
    rank = 0
    if "seed41" not in name:
        rank += 10
    if "roverstart_left" not in name and "roverapproach_centre" not in name:
        rank += 1
    return rank, name


def select_layouts(
    layout_dir: Path,
    exclude_regex: str | None,
    limit: int | None,
    one_per_family: bool,
) -> list[Path]:
    pattern = re.compile(exclude_regex) if exclude_regex else None
    eligible: list[Path] = []
    for path in sorted([*layout_dir.rglob("*.yaml"), *layout_dir.rglob("*.yml")]):
        if not path.is_file():
            continue
        rel = path.resolve().relative_to(layout_dir.resolve()).as_posix()
        if pattern and pattern.search(rel):
            continue
        config_type = infer_config_type(path)
        if config_type not in {"fenceline", "road", "shedline"}:
            continue
        eligible.append(path.resolve())

    if not one_per_family:
        selected = sorted(eligible, key=layout_sort_key)
        return selected[:limit] if limit is not None else selected

    grouped: dict[str, list[Path]] = {}
    for path in eligible:
        family = layout_family(layout_dir, path)
        grouped.setdefault(family, []).append(path)

    selected = []
    for family in sorted(grouped):
        selected.append(sorted(grouped[family], key=layout_sort_key)[0])
        if limit is not None and len(selected) >= limit:
            break
    return selected


def copy_selected_layouts(selected: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for layout in selected:
        destination = output_dir / layout.name
        shutil.copy2(layout, destination)
        print(f"Selected layout: {layout} -> {destination}", flush=True)


def task_type_rank(row: dict[str, Any]) -> int:
    task_type = str(row.get("task_type", ""))
    try:
        return PREFERRED_TASK_TYPES.index(task_type)
    except ValueError:
        return len(PREFERRED_TASK_TYPES)


def row_rank(row: dict[str, Any]) -> tuple[int, int, str]:
    variant_rank = 0 if row.get("variant_id") == "nominal" else 1
    return task_type_rank(row), variant_rank, str(row.get("rollout_id", ""))


def filter_manifest_one_row_per_scene(input_manifest: Path, output_manifest: Path) -> dict[str, Any]:
    manifest = load_yaml(input_manifest)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.get("rollouts") or []:
        if not isinstance(row, dict):
            continue
        if row.get("status", "pending") != "pending":
            continue
        if row.get("requires_pose_variant"):
            continue
        scene_id = str(row.get("scene_id") or row.get("rollout_id") or "unknown")
        grouped.setdefault(scene_id, []).append(row)

    selected_rows = []
    for scene_id in sorted(grouped):
        rows = sorted(grouped[scene_id], key=row_rank)
        selected_rows.append(rows[0])

    filtered = dict(manifest)
    filtered["source_manifest"] = input_manifest.as_posix()
    filtered["selection_rule"] = "one pending non-pose-variant row per scene_id, preferring sequence/nominal rows"
    filtered["rollouts"] = selected_rows
    counts = dict(filtered.get("counts", {}))
    counts.update(
        {
            "rollouts": len(selected_rows),
            "pending": len(selected_rows),
            "running": 0,
            "complete": 0,
            "failed": 0,
            "skipped": 0,
        }
    )
    filtered["counts"] = counts
    write_yaml(output_manifest, filtered)
    return filtered


def metadata_field(metadata: dict[str, Any], *keys: str) -> Any:
    value: Any = metadata
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def plot_rollouts(rollout_dir: Path, plots_dir: Path) -> int:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for metadata_path in sorted(rollout_dir.rglob("metadata.json")):
        rollout = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        summary_path = rollout / "task_success_wait_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        task_spec = (
            summary.get("task_spec")
            or metadata.get("task_spec")
            or metadata.get("task_spec_path")
            or metadata.get("source_task_spec")
            or metadata_field(metadata, "manifest_row", "source_task_spec")
        )
        task_id = metadata.get("task_id") or metadata_field(metadata, "structured_task", "task_id")
        variant_id = metadata.get("variant_id") or metadata_field(metadata, "selected_variant", "variant_id") or "nominal"
        if not task_spec or not task_id:
            print(f"Skipping plot for {rollout}: missing task spec or task id", flush=True)
            continue
        output = plots_dir / f"{rollout.name}.png"
        result = run_command(
            [
                "ros2",
                "run",
                "sim",
                "plot_task_plan",
                str(task_spec),
                "--task-id",
                str(task_id),
                "--variant-id",
                str(variant_id),
                "--rollout-dir",
                rollout.as_posix(),
                "--output",
                output.as_posix(),
            ],
            label=f"Plot {rollout.name}",
            check=False,
        )
        if result.returncode == 0:
            plotted += 1
    return plotted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-dir", type=Path, default=Path("~/isaac_files/configs/generated_layouts"))
    parser.add_argument("--isaac-root", type=Path, default=Path("~/isaac_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("~/sim_datasets/most_recent_trials"))
    parser.add_argument("--limit-layouts", type=int, default=None)
    parser.add_argument(
        "--exclude-layout-regex",
        default=r"(^|/)pose_variants/|(^|/)jetbot/|corridor",
        help="Regex applied to layout paths relative to --layout-dir.",
    )
    parser.add_argument("--task-family", choices=("following_only", "all_supported"), default="following_only")
    parser.add_argument("--include-visual-variations", action="store_true")
    parser.add_argument("--max-visuals-per-task-variant", type=int, default=1)
    parser.add_argument("--visual-sample-seed", type=int, default=23)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--prepare-timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run every eligible layout in --layout-dir instead of selecting one representative per layout family.",
    )
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    parser.add_argument("--dry-run", action="store_true", help="Prepare specs/manifest/plots setup but do not run Isaac rollouts.")
    parser.add_argument("--skip-run", action="store_true", help="Stop after writing the filtered manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout_dir = args.layout_dir.expanduser().resolve()
    isaac_root = args.isaac_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_layout_dir = output_dir / "selected_layouts"
    specs_dir = output_dir / "task_specs"
    valid_specs_dir = output_dir / "valid_task_specs"
    rollouts_dir = output_dir / "rollouts"
    plots_dir = output_dir / "plots"
    manifest_all = output_dir / "manifest_all.yaml"
    manifest = output_dir / "manifest.yaml"

    if args.clean:
        clean_dir(output_dir)
    for path in (selected_layout_dir, specs_dir, valid_specs_dir, rollouts_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    selected = select_layouts(
        layout_dir,
        args.exclude_layout_regex,
        args.limit_layouts,
        one_per_family=not bool(args.all_layouts),
    )
    if not selected:
        raise SystemExit(f"No generated layout YAMLs selected from {layout_dir}")
    copy_selected_layouts(selected, selected_layout_dir)

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "generate_scene_task_specs",
            selected_layout_dir.as_posix(),
            "--output-dir",
            specs_dir.as_posix(),
            "--task-family",
            args.task_family,
            "--summary",
        ],
        label="Generate one-layout QA task specs",
    )

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "validate_scene_task_specs",
            specs_dir.as_posix(),
            "--check-planner",
            "--allow-invalid",
            "--verbose",
            "--write-valid-output-dir",
            valid_specs_dir.as_posix(),
        ],
        label="Validate one-layout QA task specs",
    )

    manifest_command = [
        "ros2",
        "run",
        "sim",
        "generate_collection_manifest",
        valid_specs_dir.as_posix(),
        "--output",
        manifest_all.as_posix(),
        "--isaac-root",
        isaac_root.as_posix(),
        "--task-family",
        args.task_family,
        "--max-visuals-per-task-variant",
        str(args.max_visuals_per_task_variant),
        "--visual-sample-seed",
        str(args.visual_sample_seed),
        "--summary",
    ]
    if args.include_visual_variations:
        manifest_command.append("--include-visual-variations")
    run_command(manifest_command, label="Generate full QA manifest")

    filtered = filter_manifest_one_row_per_scene(manifest_all, manifest)
    print(f"Wrote filtered QA manifest: {manifest}", flush=True)
    print(f"Selected {len(filtered.get('rollouts') or [])} rollout(s).", flush=True)
    for row in filtered.get("rollouts") or []:
        print(
            f"  - {row.get('scene_id')} | {row.get('task_id')} | "
            f"{row.get('variant_id')} | {row.get('visual_id')}",
            flush=True,
        )

    if args.skip_run:
        return 0

    rollout_result = run_command(
        [
            "ros2",
            "run",
            "sim",
            "run_collection_manifest",
            manifest.as_posix(),
            "--retry-failed",
            "--no-tracker",
            "--use-isaac-bridge",
            "--isaac-root",
            isaac_root.as_posix(),
            "--base-dir",
            rollouts_dir.as_posix(),
            "--duration-s",
            str(args.duration_s),
            "--prepare-timeout-s",
            str(args.prepare_timeout_s),
            *(["--dry-run"] if args.dry_run else []),
        ],
        label="Run QA manifest rollouts",
        check=False,
    )
    if rollout_result.returncode != 0:
        print(
            "QA rollout command reported one or more failed rollouts; continuing to plot collected outputs.",
            flush=True,
        )

    if not args.dry_run:
        plotted = plot_rollouts(rollouts_dir, plots_dir)
        print(f"Plotted {plotted} rollout(s) into {plots_dir}", flush=True)

    print("\nMost recent trials complete.", flush=True)
    print(f"Output folder: {output_dir}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Rollouts: {rollouts_dir}", flush=True)
    print(f"Plots: {plots_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

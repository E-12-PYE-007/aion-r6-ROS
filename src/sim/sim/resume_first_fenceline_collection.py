#!/usr/bin/env python3
"""Resume the first fenceline collection after task-spec validation.

Expected existing output directory contents:

* selected_layouts/
* task_specs/
* valid_task_specs/

This resumes from valid_task_specs and runs:

* selected nominal/recovery policy
* recovery pose expansion
* pose-variant USD + sampled visual USD generation
* manifest generation/filtering
* rollout collection
* plan-vs-actual plotting
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from sim.run_first_fenceline_collection import (
    build_selection,
    expand_selected_recovery_variants,
    filter_manifest,
    generate_pose_variant_usds,
    plot_rollouts,
    prepare_manifest_input_specs,
    run_command,
    write_yaml,
)


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("~/sim_datasets/first_fenceline_collection"))
    parser.add_argument("--isaac-root", type=Path, default=Path("~/isaac_files"))
    parser.add_argument("--isaac-python", default=None)
    parser.add_argument("--variant-sample-seed", type=int, default=11)
    parser.add_argument("--visual-sample-seed", type=int, default=23)
    parser.add_argument("--variation-config", default="configs/variation_configs/variation.yaml")
    parser.add_argument("--max-visuals-per-pose-variant", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--gap-duration-s", type=float, default=120.0)
    parser.add_argument("--corner-duration-s", type=float, default=120.0)
    parser.add_argument("--rectangle-duration-s", type=float, default=180.0)
    parser.add_argument("--large-perimeter-duration-s", type=float, default=260.0)
    parser.add_argument("--prepare-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--include-unvalidated-nominal",
        action="store_true",
        help="Use task_specs nominal rows for scenes that do not yet have a valid_task_specs file.",
    )
    parser.add_argument("--skip-pose-expansion", action="store_true")
    parser.add_argument("--skip-pose-usd-generation", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--clean-downstream",
        action="store_true",
        help="Delete pose_variants, rollouts, plots, and manifests before resuming.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    isaac_root = args.isaac_root.expanduser().resolve()
    isaac_python = args.isaac_python or sys.executable

    valid_specs_dir = output_dir / "valid_task_specs"
    task_specs_dir = output_dir / "task_specs"
    pose_variants_dir = output_dir / "pose_variants"
    rollouts_dir = output_dir / "rollouts"
    plots_dir = output_dir / "plots"
    manifest_all = output_dir / "manifest_all.yaml"
    manifest = output_dir / "manifest.yaml"
    manifest_input_specs_dir = output_dir / "manifest_input_specs"
    selection_path = output_dir / "selected_tasks_and_variants.yaml"
    duration_policy = {
        "straight": float(args.duration_s),
        "gap": float(args.gap_duration_s),
        "corner": float(args.corner_duration_s),
        "rectangle": float(args.rectangle_duration_s),
        "large_perimeter": float(args.large_perimeter_duration_s),
    }

    if not valid_specs_dir.exists():
        raise SystemExit(f"Missing validated specs directory: {valid_specs_dir}")
    if args.include_unvalidated_nominal and not task_specs_dir.exists():
        raise SystemExit(f"Missing raw task specs directory: {task_specs_dir}")

    if args.clean_downstream:
        for path in (pose_variants_dir, rollouts_dir, plots_dir, manifest_input_specs_dir):
            clean_directory(path)
        for path in (manifest_all, manifest, selection_path):
            if path.exists():
                path.unlink()
    else:
        for path in (pose_variants_dir, rollouts_dir, plots_dir, manifest_input_specs_dir):
            path.mkdir(parents=True, exist_ok=True)

    selection = build_selection(
        valid_specs_dir,
        int(args.variant_sample_seed),
        raw_specs_dir=task_specs_dir,
        include_unvalidated_nominal=bool(args.include_unvalidated_nominal),
    )
    if not selection:
        raise SystemExit("No valid fenceline follow tasks selected from valid_task_specs.")
    write_yaml(selection_path, {"selection": selection})
    print(f"Wrote selected task/variant policy: {selection_path}", flush=True)
    print(f"Selected {len(selection)} task(s).", flush=True)
    print(
        "Selection validation status: "
        f"{sum(1 for item in selection.values() if item.get('planner_validation_status') == 'validated')} validated, "
        f"{sum(1 for item in selection.values() if item.get('planner_validation_status') == 'unvalidated')} unvalidated nominal fallback",
        flush=True,
    )
    manifest_input_specs = prepare_manifest_input_specs(selection, manifest_input_specs_dir)

    if not args.skip_pose_expansion:
        expand_selected_recovery_variants(selection, pose_variants_dir)

    if not args.skip_pose_usd_generation:
        generate_pose_variant_usds(
            pose_variants_dir=pose_variants_dir,
            isaac_root=isaac_root,
            isaac_python=isaac_python,
            variation_config=args.variation_config,
            visual_sample_seed=int(args.visual_sample_seed),
            max_visuals_per_pose=int(args.max_visuals_per_pose_variant),
        )

    run_command(
        [
            "ros2",
            "run",
            "sim",
            "generate_collection_manifest",
            manifest_input_specs.as_posix(),
            "--output",
            manifest_all.as_posix(),
            "--isaac-root",
            isaac_root.as_posix(),
            "--pose-variant-layout-dir",
            pose_variants_dir.as_posix(),
            "--include-visual-variations",
            "--no-base-usd",
            "--max-visuals-per-task-variant",
            "1",
            "--visual-sample-seed",
            str(args.visual_sample_seed),
            "--task-family",
            "following_only",
            "--summary",
        ],
        label="Generate full first-pass manifest from existing valid specs",
    )

    filtered = filter_manifest(manifest_all, manifest, selection, duration_policy)
    print(f"Wrote filtered first-pass manifest: {manifest}", flush=True)
    print(f"Selected {len(filtered.get('rollouts') or [])} rollout(s).", flush=True)
    print(yaml.safe_dump(filtered.get("counts", {}), sort_keys=True), flush=True)
    all_rows = load_yaml(manifest_all).get("rollouts") or []
    filtered_rows = filtered.get("rollouts") or []
    all_recovery = sum(1 for row in all_rows if str(row.get("variant_id") or "").startswith("recovery_"))
    filtered_recovery = sum(1 for row in filtered_rows if str(row.get("variant_id") or "").startswith("recovery_"))
    bad_visuals = sum("__variant_recovery" in str(row.get("visual_id") or "") for row in filtered_rows)
    print(
        "Manifest sanity: "
        f"manifest_all_recovery={all_recovery} filtered_recovery={filtered_recovery} "
        f"bad_recovery_visual_ids={bad_visuals}",
        flush=True,
    )
    if all_recovery > 0 and filtered_recovery == 0:
        raise RuntimeError("manifest_all.yaml contains recovery rows, but manifest.yaml filtered all of them out.")
    if bad_visuals > 0:
        raise RuntimeError("manifest.yaml contains recovery pose USDs misclassified as visual_id rows.")

    run_command(
        ["ros2", "run", "sim", "summarize_collection_manifest", manifest.as_posix()],
        label="Summarize filtered first-pass manifest",
        check=False,
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
            "--prepare-timeout-s",
            str(args.prepare_timeout_s),
            *(["--dry-run"] if args.dry_run else []),
        ],
        label="Run resumed first-pass fenceline rollouts",
        check=False,
    )
    if rollout_result.returncode != 0:
        print("Collection reported one or more failed rollouts; plotting any collected outputs.", flush=True)

    if not args.dry_run:
        plotted = plot_rollouts(rollouts_dir, plots_dir)
        print(f"Plotted {plotted} rollout(s) into {plots_dir}", flush=True)

    print("\nResumed first fenceline collection complete.", flush=True)
    print(f"Output folder: {output_dir}", flush=True)
    print(f"Selection: {selection_path}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Rollouts: {rollouts_dir}", flush=True)
    print(f"Plots: {plots_dir}", flush=True)
    return 1 if rollout_result.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

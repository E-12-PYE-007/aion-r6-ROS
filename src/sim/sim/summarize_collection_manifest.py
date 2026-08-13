#!/usr/bin/env python3
"""Print coverage summaries for one or more rollout collection manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


DEFAULT_FIELDS = (
    "config_type",
    "scene_id",
    "task_type",
    "variant_type",
    "data_category",
    "recovery_case",
    "visual_id",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping.")
    return data


def row_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return "<none>"
    return str(value)


def pose_state(row: dict[str, Any]) -> str:
    if not row.get("requires_pose_variant"):
        return "not_required"
    if row.get("pose_variant_ready", True):
        return "ready"
    return "missing"


def count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row_value(row, field) for row in rows).items()))


def summarize_rows(rows: list[dict[str, Any]], fields: tuple[str, ...], max_tags: int) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "pending")) for row in rows)
    tag_counts: Counter[str] = Counter()
    for row in rows:
        tag_counts.update(str(tag) for tag in row.get("scenario_tags", []) if tag is not None)

    return {
        "rollouts": len(rows),
        "status": dict(sorted(status_counts.items())),
        "pose_variants": dict(sorted(Counter(pose_state(row) for row in rows).items())),
        "fields": {field: count_field(rows, field) for field in fields},
        "scenario_tags": dict(tag_counts.most_common(max_tags)),
    }


def print_counts(title: str, counts: dict[str, int], max_rows: int) -> None:
    print(f"{title}:")
    if not counts:
        print("  <none>: 0")
        return
    for index, (key, count) in enumerate(counts.items()):
        if index >= max_rows:
            remaining = len(counts) - max_rows
            print(f"  ... {remaining} more")
            break
        print(f"  {key}: {count}")


def print_summary(path: Path, summary: dict[str, Any], max_rows: int) -> None:
    print(f"\n{path}")
    print(f"rollouts: {summary['rollouts']}")
    print_counts("status", summary["status"], max_rows)
    print_counts("pose_variants", summary["pose_variants"], max_rows)
    for field, counts in summary["fields"].items():
        print_counts(field, counts, max_rows)
    print_counts("scenario_tags", summary["scenario_tags"], max_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument(
        "--field",
        action="append",
        choices=DEFAULT_FIELDS,
        help="Manifest row field to summarize. Repeatable. Defaults to the standard coverage fields.",
    )
    parser.add_argument("--max-rows", type=int, default=30, help="Maximum printed values per field.")
    parser.add_argument("--max-tags", type=int, default=30, help="Maximum scenario tags to include.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()
    if args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1")
    if args.max_tags < 1:
        raise SystemExit("--max-tags must be >= 1")
    args.field = tuple(args.field or DEFAULT_FIELDS)
    return args


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    for manifest_path in args.manifest:
        manifest = load_yaml(manifest_path)
        rows = [row for row in manifest.get("rollouts", []) if isinstance(row, dict)]
        summaries[manifest_path.as_posix()] = summarize_rows(rows, args.field, args.max_tags)

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    for raw_path, summary in summaries.items():
        print_summary(Path(raw_path), summary, args.max_rows)


if __name__ == "__main__":
    main()

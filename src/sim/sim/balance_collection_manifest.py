#!/usr/bin/env python3
"""Sample a balanced rollout collection manifest from a larger manifest."""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


DEFAULT_FOLLOWING_TASK_WEIGHTS = {
    "follow_fence": 0.40,
    "follow_and_turn": 0.25,
    "follow_fence_sequence": 0.20,
    "follow_road": 0.10,
    "follow_shed_side": 0.05,
}

DEFAULT_VARIANT_WEIGHTS = {
    "nominal": 0.60,
    "clearance": 0.20,
    "recovery": 0.20,
}


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


def now_string() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return "<none>"
    return str(value)


def group_key(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    if not fields:
        return ("all",)
    return tuple(group_value(row, field) for field in fields)


def parse_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("weights must use KEY=WEIGHT, for example follow_fence=0.4")
    key, raw_weight = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("weight key cannot be empty")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid weight {raw_weight!r}") from exc
    if weight < 0.0:
        raise argparse.ArgumentTypeError("weights must be non-negative")
    return key, weight


def parse_cap(value: str) -> tuple[str, str, int]:
    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError("caps must use FIELD=VALUE:COUNT, for example scene_id=fence_gap_01:20")
    field, rest = value.split("=", 1)
    row_value, raw_count = rest.rsplit(":", 1)
    field = field.strip()
    row_value = row_value.strip()
    if not field or not row_value:
        raise argparse.ArgumentTypeError("cap field and value cannot be empty")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid cap count {raw_count!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("cap count must be >= 0")
    return field, row_value, count


def row_matches_status(row: dict[str, Any], statuses: set[str] | None) -> bool:
    if statuses is None:
        return True
    return str(row.get("status", "pending")) in statuses


def row_matches_tag(row: dict[str, Any], required_tags: set[str]) -> bool:
    if not required_tags:
        return True
    tags = {str(tag) for tag in row.get("scenario_tags", []) if tag is not None}
    return required_tags.issubset(tags)


def row_allowed_by_caps(row: dict[str, Any], selected: list[dict[str, Any]], caps: list[tuple[str, str, int]]) -> bool:
    for field, value, cap in caps:
        if group_value(row, field) != value:
            continue
        current = sum(1 for candidate in selected if group_value(candidate, field) == value)
        if current >= cap:
            return False
    return True


def eligible_rows(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    statuses = set(args.status) if args.status else None
    required_tags = set(args.scenario_tag or [])
    rows = []
    for row in manifest.get("rollouts", []):
        if not isinstance(row, dict):
            continue
        if args.only_pose_ready and not bool(row.get("pose_variant_ready", True)):
            continue
        if args.task_type and row.get("task_type") not in set(args.task_type):
            continue
        if args.variant_type and row.get("variant_type") not in set(args.variant_type):
            continue
        if args.scene_id and row.get("scene_id") not in set(args.scene_id):
            continue
        if args.data_category and row.get("data_category") not in set(args.data_category):
            continue
        if not row_matches_status(row, statuses):
            continue
        if not row_matches_tag(row, required_tags):
            continue
        rows.append(copy.deepcopy(row))
    return rows


def weights_for(args: argparse.Namespace, keys: list[tuple[str, ...]]) -> dict[tuple[str, ...], float]:
    explicit = dict(args.weight or [])
    if not explicit and args.preset == "following_first_pass" and args.group_by == ["task_type"]:
        explicit = DEFAULT_FOLLOWING_TASK_WEIGHTS
    elif not explicit and args.preset == "following_first_pass" and args.group_by == ["variant_type"]:
        explicit = DEFAULT_VARIANT_WEIGHTS

    weights: dict[tuple[str, ...], float] = {}
    for key in keys:
        if len(key) == 1 and key[0] in explicit:
            weights[key] = explicit[key[0]]
        else:
            joined = "|".join(key)
            weights[key] = explicit.get(joined, 1.0)
    return weights


def quotas_for(keys: list[tuple[str, ...]], weights: dict[tuple[str, ...], float], total: int) -> dict[tuple[str, ...], int]:
    if total <= 0 or not keys:
        return {}
    positive = {key: max(0.0, weights.get(key, 0.0)) for key in keys}
    weight_sum = sum(positive.values())
    if weight_sum <= 0.0:
        positive = {key: 1.0 for key in keys}
        weight_sum = float(len(keys))

    raw = {key: total * positive[key] / weight_sum for key in keys}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    for key in sorted(keys, key=lambda item: raw[item] - quotas[item], reverse=True):
        if remaining <= 0:
            break
        quotas[key] += 1
        remaining -= 1
    return quotas


def sample_balanced(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, args.group_by)].append(row)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)

    target_total = min(args.max_rollouts or len(rows), len(rows))
    keys = sorted(grouped)
    quotas = quotas_for(keys, weights_for(args, keys), target_total)
    caps = args.max_per or []

    selected: list[dict[str, Any]] = []
    for key in keys:
        for row in grouped[key][: quotas.get(key, 0)]:
            if row_allowed_by_caps(row, selected, caps):
                selected.append(row)

    if len(selected) < target_total:
        selected_ids = {row.get("rollout_id") for row in selected}
        leftovers = [row for key in keys for row in grouped[key] if row.get("rollout_id") not in selected_ids]
        rng.shuffle(leftovers)
        for row in leftovers:
            if len(selected) >= target_total:
                break
            if row_allowed_by_caps(row, selected, caps):
                selected.append(row)

    rng.shuffle(selected)
    if args.reset_status:
        for row in selected:
            row["status"] = "pending"
            for transient in ("worker_id", "claimed_time", "started_time", "finished_time", "error", "validation"):
                row.pop(transient, None)
    return selected


def counts_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rollouts": len(rows),
        "pose_variant_rollouts": sum(1 for row in rows if row.get("requires_pose_variant")),
        "missing_pose_variant_rollouts": sum(
            1 for row in rows if row.get("requires_pose_variant") and not row.get("pose_variant_ready", True)
        ),
        "pending": sum(1 for row in rows if row.get("status", "pending") == "pending"),
        "running": sum(1 for row in rows if row.get("status") == "running"),
        "complete": sum(1 for row in rows if row.get("status") == "complete"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=("following_first_pass", "none"),
        default="following_first_pass",
        help="Default weights to use when no explicit --weight values are provided.",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=None,
        choices=("task_type", "variant_type", "scene_id", "config_type", "data_category", "visual_id", "recovery_case"),
        help="Field to balance across. Repeat to balance across combinations. Default: task_type.",
    )
    parser.add_argument("--weight", action="append", type=parse_weight, help="Group weight KEY=WEIGHT.")
    parser.add_argument("--max-per", action="append", type=parse_cap, help="Cap rows by FIELD=VALUE:COUNT.")
    parser.add_argument("--task-type", action="append")
    parser.add_argument("--variant-type", action="append")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--data-category", action="append")
    parser.add_argument("--scenario-tag", action="append")
    parser.add_argument("--status", action="append", help="Eligible row status. Repeatable. Default: all statuses.")
    parser.add_argument("--only-pose-ready", action="store_true", default=True)
    parser.add_argument("--include-pose-not-ready", action="store_false", dest="only_pose_ready")
    parser.add_argument("--reset-status", action="store_true", default=True)
    parser.add_argument("--keep-status", action="store_false", dest="reset_status")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    args.group_by = args.group_by or ["task_type"]
    if args.max_rollouts is not None and args.max_rollouts < 0:
        raise SystemExit("--max-rollouts must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest)
    rows = eligible_rows(manifest, args)
    selected = sample_balanced(rows, args)

    output = copy.deepcopy(manifest)
    output["source_manifest"] = args.manifest.resolve().as_posix()
    output["balanced_time"] = now_string()
    output["balance"] = {
        "group_by": args.group_by,
        "preset": args.preset,
        "max_rollouts": args.max_rollouts,
        "seed": args.seed,
        "eligible_rollouts": len(rows),
        "selected_rollouts": len(selected),
        "weights": {key: weight for key, weight in (args.weight or [])},
        "max_per": [f"{field}={value}:{count}" for field, value, count in (args.max_per or [])],
    }
    output["rollouts"] = selected
    output["counts"] = {**dict(output.get("counts", {})), **counts_for(selected)}
    write_yaml(args.output, output)

    print(f"Wrote {args.output} ({len(selected)} rollout rows from {len(rows)} eligible)")
    if args.summary:
        for field in args.group_by:
            counts: dict[str, int] = defaultdict(int)
            for row in selected:
                counts[group_value(row, field)] += 1
            print(f"{field}:")
            for key, count in sorted(counts.items()):
                print(f"  {key}: {count}")


if __name__ == "__main__":
    main()

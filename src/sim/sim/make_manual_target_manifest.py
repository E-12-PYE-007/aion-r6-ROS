#!/usr/bin/env python3
"""Create a smaller manual-driving manifest from a generated collection manifest."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROLE_ORDER = [
    "toward_fence_avoidance",
    "toward_fence",
    "away_from_fence",
    "brief_occlusion",
    "medium_occlusion",
    "clear",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def role_from_row(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("scene_id", "rollout_id", "layout_yaml", "task_spec")
    )
    for role in ROLE_ORDER:
        if role in text:
            return role
    return "other"


def kind_from_row(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("scene_id", "rollout_id", "layout_yaml", "task_id")
    )
    if "_corner_" in text or "corner" in text or str(row.get("task_type")) == "follow_fence_sequence":
        return "corner"
    if "_side_" in text or str(row.get("task_type")) == "follow_fence":
        return "side"
    return "other"


def follow_from_row(row: dict[str, Any]) -> str:
    task_id = str(row.get("task_id", ""))
    text = " ".join(str(row.get(key, "")) for key in ("scene_id", "rollout_id", "layout_yaml", "task_spec"))
    if "_left_" in text or "_left_" in task_id or task_id.endswith("_left_from_scene_rover_pose"):
        return "left"
    if "_right_" in text or "_right_" in task_id or task_id.endswith("_right_from_scene_rover_pose"):
        return "right"
    return "unknown"


def variant_bucket(row: dict[str, Any]) -> str:
    variant_id = str(row.get("variant_id", ""))
    recovery_case = str(row.get("recovery_case", ""))
    text = f"{variant_id} {recovery_case}"
    if variant_id == "nominal":
        return "nominal"
    if "heading" in text:
        return "heading"
    if any(token in text for token in ("offset", "too_close", "too_far", "start")):
        return "start_pose"
    return "other"


def turn_from_row(row: dict[str, Any]) -> str:
    follow = follow_from_row(row)
    if kind_from_row(row) != "corner":
        return "none"
    return follow if follow in {"left", "right"} else "unknown"


def usable(row: dict[str, Any]) -> bool:
    if row.get("visual_id") == "base":
        return False
    if not row.get("visual_usd"):
        return False
    if row.get("requires_pose_variant") and not row.get("pose_variant_ready"):
        return False
    if role_from_row(row) == "other":
        return False
    if kind_from_row(row) not in {"side", "corner"}:
        return False
    if follow_from_row(row) not in {"left", "right"}:
        return False
    if variant_bucket(row) not in {"nominal", "heading", "start_pose"}:
        return False
    return True


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (kind_from_row(row), role_from_row(row), follow_from_row(row), turn_from_row(row))


def row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    bucket_priority = {"heading": 0, "start_pose": 1, "nominal": 2}
    return (bucket_priority.get(variant_bucket(row), 9), str(row.get("rollout_id", "")))


def select_rows(rows: list[dict[str, Any]], per_group: int, max_rows: int | None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if usable(row):
            groups[group_key(row)].append(row)
    for key in groups:
        groups[key].sort(key=row_sort_key)

    selected: list[dict[str, Any]] = []
    role_priority = ["toward_fence_avoidance", "away_from_fence", "brief_occlusion", "medium_occlusion", "toward_fence", "clear"]
    ordered_keys = sorted(
        groups,
        key=lambda key: (
            0 if key[0] == "corner" else 1,
            role_priority.index(key[1]) if key[1] in role_priority else 99,
            key[2],
            key[3],
        ),
    )
    for key in ordered_keys:
        selected.extend(groups[key][:per_group])

    if max_rows is not None and len(selected) > max_rows:
        # Preserve variety by taking one pass through groups at a time.
        trimmed: list[dict[str, Any]] = []
        for index in range(per_group):
            for key in ordered_keys:
                if index < len(groups[key]):
                    trimmed.append(groups[key][index])
                    if len(trimmed) >= max_rows:
                        return trimmed
        return trimmed
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rollouts": len(rows),
        "kind": dict(Counter(kind_from_row(row) for row in rows)),
        "role": dict(Counter(role_from_row(row) for row in rows)),
        "follow_side": dict(Counter(follow_from_row(row) for row in rows)),
        "turn_direction": dict(Counter(turn_from_row(row) for row in rows if kind_from_row(row) == "corner")),
        "variant_bucket": dict(Counter(variant_bucket(row) for row in rows)),
        "visual_id": dict(Counter(str(row.get("visual_id")) for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=72)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_yaml(args.input_manifest.expanduser().resolve())
    rows = manifest.get("rollouts")
    if not isinstance(rows, list):
        raise ValueError("input manifest has no rollouts list")
    selected = [copy.deepcopy(row) for row in select_rows(rows, args.per_group, args.max_rows)]
    for row in selected:
        row["status"] = "pending"
        row.pop("worker_id", None)
        row.pop("claimed_at", None)
        row.pop("error", None)
        row.pop("failure_reason", None)
        row.pop("last_error", None)
        row["manual_target"] = {
            "kind": kind_from_row(row),
            "role": role_from_row(row),
            "follow_side": follow_from_row(row),
            "turn_direction": turn_from_row(row),
            "variant_bucket": variant_bucket(row),
        }

    output = copy.deepcopy(manifest)
    output["source_manifest"] = args.input_manifest.expanduser().as_posix()
    output["selection_rule"] = (
        "manual driving target set: textured non-base snippet rows, balanced by side/corner, "
        "obstacle/occlusion role, fence side, and nominal/heading/start-pose variants where available"
    )
    output["rollouts"] = selected
    output["counts"] = {
        "rollouts": len(selected),
        "pending": len(selected),
        "running": 0,
        "complete": 0,
        "failed": 0,
        "skipped": 0,
    }
    output["manual_target_summary"] = summarize(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(f"wrote: {args.output}")
    print(yaml.safe_dump(output["manual_target_summary"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

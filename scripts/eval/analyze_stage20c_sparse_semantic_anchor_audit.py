#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _find_files(root: Path, name: str) -> List[Path]:
    if root.is_file() and root.name == name:
        return [root]
    if root.is_dir():
        return sorted(root.rglob(name))
    return []


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _basic_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _nested_counts(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        source = str(row.get("anchor_source") or "unknown")
        operation = str(row.get(key) or "unknown")
        grouped[source][operation] += 1
    return {source: dict(ops) for source, ops in grouped.items()}


def analyze(root: Path) -> Dict[str, Any]:
    memory_event_files = _find_files(root, "memory_events.jsonl")
    summary_files = _find_files(root, "memory_episode_summary.jsonl")

    anchor_events: List[Dict[str, Any]] = []
    semantic_events = 0
    episode_summaries: List[Dict[str, Any]] = []

    for path in memory_event_files:
        for row in _iter_jsonl(path):
            event_type = row.get("event_type")
            if event_type == "occ_memory_semantic":
                semantic_events += 1
            elif event_type == "occ_memory_semantic_anchor":
                item = dict(row)
                item["_file"] = str(path)
                anchor_events.append(item)

    for path in summary_files:
        for row in _iter_jsonl(path):
            episode_summaries.append(dict(row))

    operation_counts = Counter(str(row.get("anchor_operation") or "unknown") for row in anchor_events)
    kind_counts = Counter(str(row.get("semantic_kind") or "unknown") for row in anchor_events)
    source_counts = Counter(str(row.get("anchor_source") or "unknown") for row in anchor_events)
    term_counts = Counter(str(row.get("semantic_top_match") or "unknown") for row in anchor_events)
    state_counts = Counter(str(row.get("goal_state") or "unknown") for row in anchor_events)
    direction_counts = Counter(str(row.get("direction_bucket") or "unknown") for row in anchor_events)
    source_operation_counts = _nested_counts(anchor_events, "anchor_operation")

    unique_anchor_ids = {
        (str(row.get("scene_id")), str(row.get("episode_id")), str(row.get("anchor_id")))
        for row in anchor_events
        if row.get("anchor_id") is not None
    }
    unique_term_grid_pairs = {
        (
            str(row.get("scene_id")),
            str(row.get("episode_id")),
            str(row.get("semantic_top_match")),
            tuple(row.get("grid") or []),
        )
        for row in anchor_events
    }

    stats = {
        "source_offset_x_px_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_offset_x_px")) for row in anchor_events) if v is not None]
        ),
        "source_offset_y_px_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_offset_y_px")) for row in anchor_events) if v is not None]
        ),
        "source_center_distance_px_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_center_distance_px")) for row in anchor_events) if v is not None]
        ),
        "source_ray_norm_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_ray_norm")) for row in anchor_events) if v is not None]
        ),
        "source_ray_yaw_deg_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_ray_yaw_deg")) for row in anchor_events) if v is not None]
        ),
        "source_ray_pitch_deg_stats": _basic_stats(
            [v for v in (_safe_float(row.get("source_ray_pitch_deg")) for row in anchor_events) if v is not None]
        ),
        "global_bearing_deg_stats": _basic_stats(
            [v for v in (_safe_float(row.get("global_bearing_deg")) for row in anchor_events) if v is not None]
        ),
        "relative_bearing_deg_stats": _basic_stats(
            [v for v in (_safe_float(row.get("relative_bearing_deg")) for row in anchor_events) if v is not None]
        ),
        "pose_origin_distance_m_stats": _basic_stats(
            [v for v in (_safe_float(row.get("pose_origin_distance_m")) for row in anchor_events) if v is not None]
        ),
        "pose_step_distance_m_stats": _basic_stats(
            [v for v in (_safe_float(row.get("pose_step_distance_m")) for row in anchor_events) if v is not None]
        ),
        "pose_step_dyaw_deg_stats": _basic_stats(
            [v for v in (_safe_float(row.get("pose_step_dyaw_deg")) for row in anchor_events) if v is not None]
        ),
    }

    episode_anchor_counts = [
        int(row.get("semantic_anchor_count") or 0)
        for row in episode_summaries
        if row.get("semantic_anchor_enabled")
    ]
    episode_merge_rates = [
        _safe_float(row.get("semantic_anchor_merge_rate"))
        for row in episode_summaries
        if row.get("semantic_anchor_enabled")
    ]
    episode_source_ops = [
        row.get("semantic_anchor_source_operation_counts")
        for row in episode_summaries
        if row.get("semantic_anchor_enabled")
    ]

    merge_rate = 0.0
    if operation_counts["added"] + operation_counts["merged"] > 0:
        merge_rate = float(operation_counts["merged"]) / float(
            operation_counts["added"] + operation_counts["merged"]
        )

    return {
        "task": "stage20c_sparse_semantic_anchor_audit",
        "root": str(root),
        "memory_event_files": [str(path) for path in memory_event_files],
        "summary_files": [str(path) for path in summary_files],
        "semantic_event_count": int(semantic_events),
        "anchor_event_count": int(len(anchor_events)),
        "unique_anchor_id_count": int(len(unique_anchor_ids)),
        "unique_term_grid_pair_count": int(len(unique_term_grid_pairs)),
        "anchor_uniqueness_rate": (
            float(len(unique_anchor_ids)) / float(len(anchor_events)) if anchor_events else None
        ),
        "anchor_duplicate_rate": (
            float(len(anchor_events) - len(unique_anchor_ids)) / float(len(anchor_events))
            if anchor_events
            else None
        ),
        "anchor_operation_counts": dict(operation_counts),
        "anchor_merge_rate_from_events": float(merge_rate),
        "semantic_kind_counts": dict(kind_counts),
        "anchor_source_counts": dict(source_counts),
        "anchor_source_operation_counts": source_operation_counts,
        "goal_state_counts": dict(state_counts),
        "direction_counts": dict(direction_counts),
        "top_terms": dict(term_counts.most_common(25)),
        "stats": stats,
        "episode_summary_count": int(len(episode_summaries)),
        "episode_anchor_count_stats": _basic_stats([float(v) for v in episode_anchor_counts]),
        "episode_merge_rate_stats": _basic_stats(
            [v for v in episode_merge_rates if v is not None]
        ),
        "episode_source_operation_examples": [item for item in episode_source_ops[:5] if item],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage20c sparse semantic anchor audit logs.")
    parser.add_argument("run_root", type=Path, help="Run directory or vlmap_safety_debug directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <run_root>/stage20c_sparse_semantic_anchor_audit_summary.json.",
    )
    args = parser.parse_args()

    summary = analyze(args.run_root)
    output_path = args.output or args.run_root / "stage20c_sparse_semantic_anchor_audit_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

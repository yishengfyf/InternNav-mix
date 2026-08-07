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


def analyze(root: Path) -> Dict[str, Any]:
    memory_event_files = _find_files(root, "memory_events.jsonl")
    summary_files = _find_files(root, "memory_episode_summary.jsonl")

    anchor_events: List[Dict[str, Any]] = []
    semantic_events = 0
    for path in memory_event_files:
        for row in _iter_jsonl(path):
            event_type = row.get("event_type")
            if event_type == "occ_memory_semantic":
                semantic_events += 1
            elif event_type == "occ_memory_semantic_anchor":
                item = dict(row)
                item["_file"] = str(path)
                anchor_events.append(item)

    episode_summaries: List[Dict[str, Any]] = []
    for path in summary_files:
        episode_summaries.extend(_iter_jsonl(path))

    operation_counts = Counter(str(row.get("anchor_operation") or "unknown") for row in anchor_events)
    kind_counts = Counter(str(row.get("semantic_kind") or "unknown") for row in anchor_events)
    source_counts = Counter(str(row.get("anchor_source") or "unknown") for row in anchor_events)
    term_counts = Counter(str(row.get("semantic_top_match") or "unknown") for row in anchor_events)
    state_counts = Counter(str(row.get("goal_state") or "unknown") for row in anchor_events)
    direction_counts = Counter(str(row.get("direction_bucket") or "unknown") for row in anchor_events)

    open_scores = [
        value
        for value in (_safe_float(row.get("open_score")) for row in anchor_events)
        if value is not None
    ]
    occupied_ratios = [
        value
        for value in (_safe_float(row.get("local_occupied_ratio")) for row in anchor_events)
        if value is not None
    ]
    distances = [
        value
        for value in (_safe_float(row.get("distance_m")) for row in anchor_events)
        if value is not None
    ]
    depths = [
        value
        for value in (_safe_float(row.get("depth_m")) for row in anchor_events)
        if value is not None
    ]

    episodes_with_anchors = {
        (str(row.get("scene_id")), str(row.get("episode_id")))
        for row in anchor_events
        if row.get("episode_id") is not None
    }
    anchor_count_by_episode = Counter(
        (str(row.get("scene_id")), str(row.get("episode_id")))
        for row in anchor_events
        if row.get("episode_id") is not None
    )

    examples_by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in anchor_events:
        kind = str(row.get("semantic_kind") or "unknown")
        if len(examples_by_kind[kind]) >= 5:
            continue
        examples_by_kind[kind].append(
            {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "anchor_id": row.get("anchor_id"),
                "anchor_source": row.get("anchor_source"),
                "term": row.get("semantic_top_match"),
                "score": row.get("semantic_top_score"),
                "grid": row.get("grid"),
                "goal_state": row.get("goal_state"),
                "direction_bucket": row.get("direction_bucket"),
                "distance_m": row.get("distance_m"),
                "open_score": row.get("open_score"),
                "local_occupied_ratio": row.get("local_occupied_ratio"),
            }
        )

    summary_anchor_counts = [
        int(row.get("semantic_anchor_count") or 0)
        for row in episode_summaries
        if row.get("semantic_anchor_enabled")
    ]

    return {
        "task": "stage20a_sparse_semantic_anchors",
        "root": str(root),
        "memory_event_files": [str(path) for path in memory_event_files],
        "summary_files": [str(path) for path in summary_files],
        "semantic_event_count": int(semantic_events),
        "anchor_event_count": int(len(anchor_events)),
        "episode_summary_count": int(len(episode_summaries)),
        "episodes_with_anchors": int(len(episodes_with_anchors)),
        "anchor_operation_counts": dict(operation_counts),
        "semantic_kind_counts": dict(kind_counts),
        "anchor_source_counts": dict(source_counts),
        "goal_state_counts": dict(state_counts),
        "direction_counts": dict(direction_counts),
        "top_terms": dict(term_counts.most_common(25)),
        "open_score_stats": _basic_stats(open_scores),
        "local_occupied_ratio_stats": _basic_stats(occupied_ratios),
        "distance_m_stats": _basic_stats(distances),
        "depth_m_stats": _basic_stats(depths),
        "summary_anchor_count_stats": _basic_stats([float(v) for v in summary_anchor_counts]),
        "top_anchor_episode_counts": [
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "anchor_events": int(value),
            }
            for key, value in anchor_count_by_episode.most_common(10)
        ],
        "examples_by_kind": dict(examples_by_kind),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage20a sparse semantic anchor logs.")
    parser.add_argument("run_root", type=Path, help="Run directory or vlmap_safety_debug directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <run_root>/stage20a_sparse_semantic_anchor_summary.json.",
    )
    args = parser.parse_args()

    summary = analyze(args.run_root)
    output_path = args.output or args.run_root / "stage20a_sparse_semantic_anchor_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

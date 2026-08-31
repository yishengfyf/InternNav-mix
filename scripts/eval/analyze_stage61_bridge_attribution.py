#!/usr/bin/env python3
"""Offline attribution of Stage60 post-turn RGB-D bridge failures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _path_metrics(goal: list[Any], path: list[list[Any]], cell_size: float) -> tuple[float, float]:
    distances = [
        math.hypot(float(goal[0]) - float(cell[0]), float(goal[1]) - float(cell[1]))
        * cell_size
        for cell in path
    ]
    index = min(range(len(distances)), key=lambda value: distances[value])
    return float(distances[index]), float(index * cell_size)


def analyze(source: Path, *, cell_size: float = 0.05) -> dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    events = data.get("events") or []
    counts: Counter[str] = Counter()
    near_nonfree: Counter[str] = Counter()
    headroom: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for event in events:
        anchor = next(
            (row for row in event.get("anchors") or [] if row.get("anchor") == "last_productive_pre_loop"),
            None,
        )
        if not anchor:
            continue
        post = anchor.get("post_turn_counterfactual") or {}
        bridge = post.get("bridge") or {}
        path = bridge.get("path") or []
        event_counts: Counter[str] = Counter()
        for probe in bridge.get("probe_records") or []:
            if not probe.get("projection_valid") or not probe.get("goal_grid"):
                reason = "invalid_pixel_projection"
            elif probe.get("goal_state") != "free":
                reason = "goal_not_free"
            else:
                deviation, progress = _path_metrics(probe["goal_grid"], path, cell_size)
                if deviation > 0.35:
                    reason = "outside_path_corridor"
                elif progress < 0.25:
                    reason = "insufficient_path_progress"
                else:
                    reason = "path_eligible_or_other"
            counts[reason] += 1
            event_counts[reason] += 1
            if probe.get("goal_grid") and path:
                deviation, progress = _path_metrics(probe["goal_grid"], path, cell_size)
                if deviation <= 0.35 and progress >= 0.25 and probe.get("goal_state") != "free":
                    near_nonfree[str(probe.get("goal_state"))] += 1
        graph = next(
            (row.get("graph") or {} for row in (anchor.get("stage58_support_policy") or {}).get("arms") or []
             if row.get("policy") == "known_free_floor_frames2"),
            {},
        )
        for step in graph.get("step_records") or []:
            blocked = int(step.get("headroom_blocked_count", 0) or 0)
            if blocked:
                headroom["steps_with_headroom_block"] += 1
                headroom["blocked_sample_count"] += blocked
        records.append({
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "trigger_step": event.get("trigger_step"),
            "post_turn_reason": post.get("reason"),
            "post_turn_bridge_reason": post.get("bridge_reason"),
            "first_edge_visible": bool(post.get("first_edge_visible_projection")),
            "event_probe_counts": dict(event_counts),
        })
    return {
        "task": "stage61_bridge_attribution",
        "source": str(source),
        "event_count": len(records),
        "probe_reason_counts": dict(counts),
        "near_path_nonfree_counts": dict(near_nonfree),
        "headroom_counts": dict(headroom),
        "post_turn_visible_event_count": sum(int(row["first_edge_visible"]) for row in records),
        "records": records,
        "online_navigation_changed": False,
        "gt_used_for_navigation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

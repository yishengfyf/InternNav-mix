#!/usr/bin/env python3
"""Select causal executed-rotation thresholds on fixed Stage25 ledgers."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.stage25_event_gt_review import (
    evaluate_detector_against_gt_lite, scene_split,
)
from scripts.eval.analyze_stage25_gt_detector import (
    discover_episodes, episode_key, jsonl, loop_events_by_episode, mine_events,
    progress_by_episode,
)


def parse_numbers(values: Sequence[str], cast: Any) -> List[Any]:
    return [cast(value) for value in values]


def split_rows(rows: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if scene_split(row.get("scene_id")) == split]


def evaluation_for_split(
    events: Sequence[Mapping[str, Any]],
    annotated_gt: Sequence[Mapping[str, Any]],
    split: str,
) -> Dict[str, Any]:
    """Evaluate one explicit split so holdout cannot influence dev selection."""
    return evaluate_detector_against_gt_lite(
        split_rows(events, split),
        [dict(row) for row in annotated_gt if row.get("split") == split],
    )["all"]


def variant_rank(variant: Mapping[str, Any], baseline: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Conservative dev-only ordering; lower tuple is better."""
    metrics = variant["dev"]
    baseline_protection = float(baseline.get("wrong_way_protection_rate") or 0.0)
    protection = float(metrics.get("wrong_way_protection_rate") or 0.0)
    recall = float(metrics.get("true_trap_recall") or 0.0)
    protects_baseline = protection + 1e-12 >= baseline_protection
    return (
        0 if protects_baseline else 1,
        -recall,
        -protection,
        int(metrics.get("detector_event_count") or 0),
        float(variant.get("median_confirmation_delay_steps") or float("inf")),
        -float(variant["minimum_degrees"]),
        -int(variant["minimum_turn_actions"]),
        float(variant["maximum_displacement_m"]),
        int(variant["window_steps"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", default=["24", "32", "40"])
    parser.add_argument("--degrees", nargs="+", default=["270", "300", "330", "345"])
    parser.add_argument("--turn-actions", nargs="+", default=["18", "20", "22", "24"])
    parser.add_argument("--max-displacements", nargs="+", default=["0.20", "0.25", "0.35"])
    args = parser.parse_args()

    annotated_gt = json.loads(args.gt_manifest.read_text(encoding="utf-8"))
    loops = loop_events_by_episode(args.run_root)
    progress = progress_by_episode(args.run_root)
    episodes = []
    for episode_dir in discover_episodes(args.run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        key = episode_key(meta)
        episodes.append((meta, jsonl(episode_dir / "observations.jsonl"), loops.get(key, [])))

    baseline_events: List[Dict[str, Any]] = []
    for meta, observations, episode_loops in episodes:
        result = mine_events(observations, episode_loops, [])
        outcome = progress.get(episode_key(meta), {})
        for event in result["D2"]:
            baseline_events.append({
                **event,
                "scene_id": meta.get("scene_id"),
                "episode_id": meta.get("episode_id"),
                "outcome": {"success": outcome.get("success"), "spl": outcome.get("spl")},
            })
    baseline_dev = evaluation_for_split(baseline_events, annotated_gt, "dev")

    grid = itertools.product(
        parse_numbers(args.windows, int),
        parse_numbers(args.degrees, float),
        parse_numbers(args.turn_actions, int),
        parse_numbers(args.max_displacements, float),
    )
    variants: List[Dict[str, Any]] = []
    selected_events_by_key: Dict[Tuple[int, float, int, float], List[Dict[str, Any]]] = {}
    for window, degrees, turn_actions, max_displacement in grid:
        events: List[Dict[str, Any]] = []
        rotation_delays: List[int] = []
        for meta, observations, episode_loops in episodes:
            result = mine_events(
                observations, episode_loops, [],
                executed_rotation_window=window,
                executed_rotation_min_degrees=degrees,
                executed_rotation_min_turn_actions=turn_actions,
                executed_rotation_max_forward_actions=0,
                executed_rotation_max_displacement_m=max_displacement,
            )
            outcome = progress.get(episode_key(meta), {})
            for event in result["D2_executed_rotation"]:
                row = {
                    **event,
                    "scene_id": meta.get("scene_id"),
                    "episode_id": meta.get("episode_id"),
                    "outcome": {"success": outcome.get("success"), "spl": outcome.get("spl")},
                }
                events.append(row)
                if event["event_family"] == "G2_executed_rotation_loop":
                    rotation_delays.append(int(event["confirmation_delay_steps"]))
        key = (window, degrees, turn_actions, max_displacement)
        selected_events_by_key[key] = events
        variants.append({
            "window_steps": window,
            "minimum_degrees": degrees,
            "minimum_turn_actions": turn_actions,
            "maximum_forward_actions": 0,
            "maximum_displacement_m": max_displacement,
            "event_count": len(events),
            "episode_count": len({episode_key(event) for event in events}),
            "rotation_event_count": sum(
                event["event_family"] == "G2_executed_rotation_loop" for event in events
            ),
            "median_confirmation_delay_steps": (
                statistics.median(rotation_delays) if rotation_delays else None
            ),
            "dev": evaluation_for_split(events, annotated_gt, "dev"),
        })

    variants.sort(key=lambda row: variant_rank(row, baseline_dev))
    selected = variants[0]
    selected_key = (
        int(selected["window_steps"]), float(selected["minimum_degrees"]),
        int(selected["minimum_turn_actions"]), float(selected["maximum_displacement_m"]),
    )
    # Holdout is opened once, only after the dev-only ordering is frozen.
    selected["holdout"] = evaluation_for_split(
        selected_events_by_key[selected_key], annotated_gt, "holdout"
    )
    baseline_holdout = evaluation_for_split(baseline_events, annotated_gt, "holdout")
    report = {
        "task": "stage25_executed_rotation_gt_lite_selection",
        "causal_detector": True,
        "future_used_by_detector": False,
        "shadow_only": True,
        "selection_protocol": "all_variants_dev_only_then_selected_variant_holdout_once",
        "selection_order": [
            "wrong_way_protection_not_below_D2", "max_true_trap_recall",
            "max_wrong_way_protection", "min_detector_events", "min_confirmation_delay",
        ],
        "episode_count": len(episodes),
        "gt_manifest_count": len(annotated_gt),
        "variant_count": len(variants),
        "baseline_D2": {"dev": baseline_dev, "holdout": baseline_holdout},
        "selected": selected,
        "selected_detector_events": selected_events_by_key[selected_key],
        "dev_variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        **{
            key: value for key, value in report.items()
            if key not in {"dev_variants", "selected_detector_events"}
        },
        "top_dev_variants": variants[:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

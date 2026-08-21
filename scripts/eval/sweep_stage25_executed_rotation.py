#!/usr/bin/env python3
"""Sweep causal executed-rotation thresholds on fixed Stage25 ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.analyze_stage25_gt_detector import (
    discover_episodes, episode_key, jsonl, loop_events_by_episode, mine_events,
    progress_by_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configs", nargs="+", default=["270:18", "300:20", "330:22", "345:20"],
        help="Pairs of minimum degrees and minimum turn actions.",
    )
    args = parser.parse_args()
    configs = []
    for value in args.configs:
        degrees, turn_actions = value.split(":", 1)
        configs.append((float(degrees), int(turn_actions)))
    loops = loop_events_by_episode(args.run_root)
    progress = progress_by_episode(args.run_root)
    episodes = []
    for episode_dir in discover_episodes(args.run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        key = episode_key(meta)
        episodes.append((meta, jsonl(episode_dir / "observations.jsonl"), loops.get(key, [])))
    variants = []
    for degrees, turn_actions in configs:
        events = []
        for meta, observations, episode_loops in episodes:
            result = mine_events(
                observations, episode_loops, [],
                executed_rotation_min_degrees=degrees,
                executed_rotation_min_turn_actions=turn_actions,
                executed_rotation_max_forward_actions=0,
            )
            outcome = progress.get(episode_key(meta), {})
            for event in result["D2_executed_rotation"]:
                if event["event_family"] != "G2_executed_rotation_loop":
                    continue
                events.append({
                    **event,
                    "scene_id": meta.get("scene_id"),
                    "episode_id": meta.get("episode_id"),
                    "outcome": {
                        "success": outcome.get("success"),
                        "spl": outcome.get("spl"),
                    },
                })
        variants.append({
            "minimum_degrees": degrees,
            "minimum_turn_actions": turn_actions,
            "event_count": len(events),
            "episode_count": len({episode_key(event) for event in events}),
            "successful_episode_count": len({
                episode_key(event) for event in events
                if float((event.get("outcome") or {}).get("success") or 0.0) > 0.0
            }),
            "median_confirmation_delay_steps": (
                sorted(event["confirmation_delay_steps"] for event in events)[len(events) // 2]
                if events else None
            ),
            "events": events,
        })
    report = {
        "task": "stage25_executed_rotation_threshold_sweep",
        "causal_detector": True,
        "future_used_by_detector": False,
        "shadow_only": True,
        "episode_count": len(episodes),
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        **{key: value for key, value in report.items() if key != "variants"},
        "variants": [
            {key: value for key, value in variant.items() if key != "events"}
            for variant in variants
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

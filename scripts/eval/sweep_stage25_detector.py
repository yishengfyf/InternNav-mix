#!/usr/bin/env python3
"""Run offline Stage25 route-confirmation sensitivity on one fixed ledger."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.analyze_stage25_gt_detector import (
    discover_episodes, episode_key, jsonl, loop_events_by_episode,
    lseg_events, mine_events, progress_by_episode,
)


def sweep(run_root: Path) -> Dict[str, Any]:
    progress = progress_by_episode(run_root)
    loops = loop_events_by_episode(run_root)
    episodes = []
    for episode_dir in discover_episodes(run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        key = episode_key(meta)
        episodes.append({
            "key": key, "observations": jsonl(episode_dir / "observations.jsonl"),
            "loops": loops.get(key, []), "semantic": lseg_events(episode_dir),
            "success": progress.get(key, {}).get("success"),
        })

    settings = itertools.product(
        (0.20, 0.25, 0.35), (0.75, 1.50, 2.00),
        (8, 12, 16), (256, 512, 1024),
    )
    rows: List[Dict[str, Any]] = []
    for radius, min_path, confirm_steps, max_growth in settings:
        events = []
        for episode in episodes:
            variants = mine_events(
                episode["observations"], episode["loops"], episode["semantic"],
                route_radius_m=radius, route_min_path_m=min_path,
                route_confirm_min_steps=confirm_steps,
                route_confirm_max_steps=max(16, confirm_steps),
                route_max_unique_occ_growth=max_growth,
            )
            for event in variants["D2"]:
                events.append({**event, "episode_key": episode["key"], "success": episode["success"]})
        route = [event for event in events if event["event_family"] == "G3_route_topology"]
        persistent = [event for event in events if event["recoverability_proxy"] == "persistent_episode"]
        rows.append({
            "route_radius_m": radius, "route_min_path_m": min_path,
            "route_confirm_steps": confirm_steps, "route_max_unique_occ_growth": max_growth,
            "event_count": len(events), "route_event_count": len(route),
            "route_episode_count": len({event["episode_key"] for event in route}),
            "route_success_episode_count": len({event["episode_key"] for event in route if event["success"]}),
            "persistent_count": len(persistent),
            "persistent_episode_count": len({event["episode_key"] for event in persistent}),
            "route_recovery_counts": dict(Counter(event["recoverability_proxy"] for event in route)),
            "route_events": [
                {
                    "episode_key": event["episode_key"], "signal_step": event["signal_step"],
                    "step_id": event["step_id"], "recoverability": event["recoverability_proxy"],
                }
                for event in route
            ],
        })
    return {
        "episode_count": len(episodes), "configuration_count": len(rows),
        "fixed_s2": True, "future_used_by_detector": False, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sweep(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

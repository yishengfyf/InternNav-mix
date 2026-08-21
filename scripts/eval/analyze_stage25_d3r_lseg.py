#!/usr/bin/env python3
"""Aggregate Stage25 D3R causal LSeg confirmation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.root.glob("rank*/*/d3r_episode.json"))
    ]
    events = [event for episode in episodes for event in episode["events"]]
    supported = [event for event in events if event["d3r"]["supports_existing_suspicion"]]
    d3q = [event for event in events if event["d3q_supported"]]
    incremental = [event for event in supported if not event["d3q_supported"]]
    report = {
        "task": "stage25_d3r_causal_lseg_confirmation",
        "episode_count": len(episodes), "event_count": len(events),
        "d3q_supported_count": len(d3q),
        "d3r_supported_count": len(supported),
        "d3r_incremental_over_d3q_count": len(incremental),
        "d3r_supported_recoverability": dict(Counter(
            event["recoverability_proxy"] for event in supported
        )),
        "d3r_supported_family": dict(Counter(event["event_family"] for event in supported)),
        "unique_lseg_call_count": sum(
            int(episode["unique_lseg_call_count"]) for episode in episodes
        ),
        "inference_seconds_total": sum(
            float(episode["inference_seconds_total"]) for episode in episodes
        ),
        "future_used_by_detector": False,
        "semantic_can_create_event": False,
        "decision_status": "confirmation_audit_only_pending_manual_event_gt",
        "incremental_events": incremental,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a broad, detector-independent Stage25 human review pool."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.stage25_event_gt_review import (
    action_interval_summary, mine_review_windows,
)
from scripts.eval.analyze_stage25_gt_detector import (
    canonical_observations, discover_episodes, jsonl, lseg_events,
    progress_by_episode, render_event_evidence,
)


def key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def overlaps_detector(
    candidate: Mapping[str, Any], events: List[Mapping[str, Any]], tolerance: int = 8,
) -> bool:
    start, end = int(candidate["onset_step"]), int(candidate["end_step"])
    return any(
        int(event.get("step_id", 0)) <= end + tolerance
        and int(event.get("end_step", event.get("step_id", 0))) >= start - tolerance
        for event in events
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--detector-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = json.loads(args.detector_candidates.read_text(encoding="utf-8"))["D2"]
    progress = progress_by_episode(args.run_root)
    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for event in detector:
        by_episode.setdefault(key(event), []).append(event)
    review: List[Dict[str, Any]] = []
    evidence_dir = args.output / "event_evidence"
    for episode_dir in discover_episodes(args.run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        observations = jsonl(episode_dir / "observations.jsonl")
        actions = jsonl(episode_dir / "actions.jsonl")
        rows = canonical_observations(observations)
        episode_events = by_episode.get(key(meta), [])
        outcome = progress.get(key(meta), {})
        semantic = lseg_events(episode_dir)
        for index, candidate in enumerate(mine_review_windows(rows)):
            candidate.update({
                "scene_id": meta.get("scene_id"),
                "episode_id": meta.get("episode_id"),
                "overlaps_d2": overlaps_detector(candidate, episode_events),
                "outcome": {
                    "success": outcome.get("success"),
                    "spl": outcome.get("spl"),
                    "steps": outcome.get("steps"),
                },
                "offline_action_audit": action_interval_summary(
                    rows, actions,
                    onset_step=int(candidate["onset_step"]),
                    end_step=int(candidate["end_step"]),
                ),
                "annotation": {
                    "state": None, "type": None, "onset_step": None,
                    "end_step": None, "recoverability": None,
                    "failure_link": None, "intervention_likely_needed": None,
                    "confidence": None, "notes": "",
                },
            })
            name = (
                f"{meta.get('scene_id')}_{meta.get('episode_id')}_review{index:03d}_"
                f"{candidate['review_family']}.png"
            )
            evidence_event = {
                "step_id": candidate["step_id"],
                "signal_step": candidate["onset_step"],
                "event_family": candidate["review_family"],
                "evidence": [candidate["review_family"]],
            }
            render_event_evidence(
                episode_dir, observations, semantic, evidence_event,
                evidence_dir / name, status="event_gt_review_only",
            )
            candidate["evidence_image"] = str(Path("event_evidence") / name)
            review.append(candidate)
    report = {
        "task": "stage25_broad_event_gt_review_pool",
        "uses_future_for_review_only": True,
        "future_used_by_detector": False,
        "detector_candidate_count": len(detector),
        "review_candidate_count": len(review),
        "review_family_counts": dict(Counter(row["review_family"] for row in review)),
        "review_episode_count": len({key(row) for row in review}),
        "review_overlapping_d2_count": sum(row["overlaps_d2"] for row in review),
        "review_without_d2_count": sum(not row["overlaps_d2"] for row in review),
        "review_without_d2_episode_count": len({
            key(row) for row in review if not row["overlaps_d2"]
        }),
        "event_gt_status": "pending_manual_review",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stage25_event_gt_review_manifest.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "stage25_event_gt_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

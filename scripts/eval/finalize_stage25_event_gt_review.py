#!/usr/bin/env python3
"""Finalize Stage25 GT-lite with explicit visual adjudications."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.stage25_event_gt_review import intervals_overlap, scene_split


def same_review_event(event: Mapping[str, Any], review: Mapping[str, Any]) -> bool:
    return (
        str(event.get("scene_id")) == str(review.get("scene_id"))
        and str(event.get("episode_id")) == str(review.get("episode_id"))
        and str(event.get("event_family")) == str(review.get("event_family"))
        and int(event.get("step_id", -1)) == int(review.get("step_id", -2))
    )


def evaluate_final(
    events: Sequence[Mapping[str, Any]],
    objective_gt: Sequence[Mapping[str, Any]],
    visual_reviews: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    objective_true = [
        row for row in objective_gt
        if (row.get("annotation") or {}).get("auto_status") == "objective_confirmed"
        and (row.get("annotation") or {}).get("state") == "true_trap"
    ]
    objective_wrong = [
        row for row in objective_gt
        if (row.get("annotation") or {}).get("auto_status") == "objective_confirmed"
        and (row.get("annotation") or {}).get("state") == "wrong_way_progress"
    ]
    visual_true = [row for row in visual_reviews if row.get("state") == "true_trap"]
    visual_negative = [row for row in visual_reviews if row.get("state") != "true_trap"]

    def subset(split: Optional[str]) -> Dict[str, Any]:
        split_events = [
            event for event in events
            if split is None or scene_split(event.get("scene_id")) == split
        ]
        gt_true = [row for row in objective_true if split is None or row.get("split") == split]
        gt_wrong = [row for row in objective_wrong if split is None or row.get("split") == split]
        manual_true = [
            row for row in visual_true
            if split is None or scene_split(row.get("scene_id")) == split
        ]
        manual_negative = [
            row for row in visual_negative
            if split is None or scene_split(row.get("scene_id")) == split
        ]
        categories: Counter[str] = Counter()
        for event in split_events:
            if any(intervals_overlap(event, row) for row in gt_true):
                categories["objective_true_trap"] += 1
            elif any(same_review_event(event, row) for row in manual_true):
                categories["visual_true_trap"] += 1
            elif any(intervals_overlap(event, row) for row in gt_wrong):
                categories["wrong_way_overlap"] += 1
            elif any(same_review_event(event, row) for row in manual_negative):
                categories["visual_hesitation"] += 1
            else:
                categories["unadjudicated"] += 1
        detected_objective = sum(
            any(intervals_overlap(row, event) for event in split_events) for row in gt_true
        )
        detected_visual = sum(
            any(same_review_event(event, row) for event in split_events) for row in manual_true
        )
        true_events = categories["objective_true_trap"] + categories["visual_true_trap"]
        adjudicated_events = len(split_events) - categories["unadjudicated"]
        combined_gt_count = len(gt_true) + len(manual_true)
        return {
            "detector_event_count": len(split_events),
            "event_adjudication_counts": dict(categories),
            "adjudicated_event_count": adjudicated_events,
            "event_precision_on_adjudicated": (
                true_events / adjudicated_events if adjudicated_events else None
            ),
            "event_precision_status": (
                "complete_for_this_detector"
                if categories["unadjudicated"] == 0
                else "partial_unadjudicated_events_remain"
            ),
            "detector_independent_true_trap_count": len(gt_true),
            "detector_independent_true_trap_detected": detected_objective,
            "detector_independent_recall": (
                detected_objective / len(gt_true) if gt_true else None
            ),
            "visual_supplement_true_trap_count": len(manual_true),
            "visual_supplement_true_trap_detected": detected_visual,
            "combined_confirmed_true_trap_count": combined_gt_count,
            "combined_confirmed_true_trap_detected": detected_objective + detected_visual,
            "combined_confirmed_recall": (
                (detected_objective + detected_visual) / combined_gt_count
                if combined_gt_count else None
            ),
            "wrong_way_window_count": len(gt_wrong),
            "wrong_way_window_protected": sum(
                not any(intervals_overlap(row, event) for event in split_events)
                for row in gt_wrong
            ),
            "visual_hesitation_count": len(manual_negative),
        }

    return {"all": subset(None), "dev": subset("dev"), "holdout": subset("holdout")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-gt", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--detector-candidates", type=Path, required=True)
    parser.add_argument("--rotation-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    objective = json.loads(args.objective_gt.read_text(encoding="utf-8"))
    visual = json.loads(args.visual_review.read_text(encoding="utf-8"))
    candidates = json.loads(args.detector_candidates.read_text(encoding="utf-8"))
    rotation = json.loads(args.rotation_selection.read_text(encoding="utf-8"))
    variants = {
        "D0": candidates["D0"],
        "D1": candidates["D1"],
        "D2": candidates["D2"],
        "D2_ER_default": candidates["D2_executed_rotation"],
        "D2_ER_selected": rotation["selected_detector_events"],
    }
    report = {
        "task": "stage25_event_gt_lite_final_visual_audit",
        "objective_gt_source": "detector_independent_executed_future_trajectory",
        "visual_supplement_source": "detector_sourced_evidence_manual_visual_review",
        "visual_supplement_is_detector_independent": False,
        "outcome_is_event_gt": False,
        "future_used_by_detector": False,
        "gt_scope": "local_stagnation_wrong_way_and_adjudicated_D2_events_only",
        "objective_manifest_count": len(objective),
        "visual_review_count": len(visual),
        "visual_review_state_counts": dict(Counter(row.get("state") for row in visual)),
        "detectors": {
            name: evaluate_final(events, objective, visual)
            for name, events in variants.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

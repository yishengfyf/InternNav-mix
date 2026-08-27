#!/usr/bin/env python3
"""Aggregate Stage45 offline candidate rejection truth events."""

import argparse
import json
from pathlib import Path


def _load_events(run_root: Path):
    events = []
    for path in sorted(run_root.rglob("stage27_m3_candidate_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def analyze(run_root: Path):
    events = _load_events(run_root)
    audited = [
        event for event in events
        if isinstance(event.get("offline_rejection_truth_audit"), dict)
    ]
    candidate_audits = [
        item
        for event in audited
        for item in event["offline_rejection_truth_audit"].get("audits", [])
    ]
    valid = [item for item in candidate_audits if item.get("valid")]
    checks = {
        "events_present": bool(audited),
        "all_events_shadow_only": all(
            event.get("shadow_only") is True
            and event.get("action_applied") is False
            for event in audited
        ),
        "candidate_features_use_no_gt": all(
            not event.get("candidate_feature_gt_fields_used")
            and not event.get("gt_fields_used")
            for event in audited
        ),
        "offline_gt_never_used_for_navigation": all(
            event["offline_rejection_truth_audit"].get(
                "gt_used_for_navigation"
            ) is False
            and all(item.get("gt_used_for_navigation") is False for item in event["offline_rejection_truth_audit"].get("audits", []))
            for event in audited
        ),
        "unknown_is_not_free": all(
            event.get("unknown_is_free") is False
            and event["offline_rejection_truth_audit"].get("unknown_is_free") is False
            for event in audited
        ),
        "all_candidate_audits_valid": bool(candidate_audits)
        and len(valid) == len(candidate_audits),
    }
    return {
        "task": "stage45_candidate_rejection_truth_audit",
        "event_count": len(audited),
        "candidate_audit_count": len(candidate_audits),
        "valid_candidate_audit_count": len(valid),
        "complete_gt_safe_corridor_count": sum(
            bool(item.get("complete_gt_safe_corridor")) for item in valid
        ),
        "route_occ_false_block_candidate_count": sum(
            bool(item.get("route_occ_false_block_candidate")) for item in valid
        ),
        "floor_footprint_false_block_candidate_count": sum(
            bool(item.get("floor_footprint_false_block_candidate"))
            for item in valid
        ),
        "sparse_2d_false_free_cell_count": sum(
            int(item.get("sparse_2d_false_free_cell_count", 0)) for item in valid
        ),
        "floor_footprint_false_free_cell_count": sum(
            int(item.get("floor_footprint_false_free_cell_count", 0))
            for item in valid
        ),
        "integrity_checks": checks,
        "integrity_passed": all(checks.values()),
        "stage46_certificate_gate": bool(
            valid
            and any(
                item.get("complete_gt_safe_corridor")
                and (
                    item.get("route_occ_false_block_candidate")
                    or item.get("floor_footprint_false_block_candidate")
                )
                and not int(item.get("sparse_2d_false_free_cell_count", 0))
                and not int(
                    item.get("floor_footprint_false_free_cell_count", 0)
                )
                for item in valid
            )
        ),
        "active_recovery_enabled": False,
        "ranker_trained": False,
        "unknown_is_free": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

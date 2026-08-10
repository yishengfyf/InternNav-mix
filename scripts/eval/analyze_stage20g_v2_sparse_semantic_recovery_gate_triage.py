"""Analyze Stage20g-v2 multi-evidence semantic-recovery gate logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_stage20g_sparse_semantic_recovery_gate import (
    _active_event_files,
    _episode_key,
    _progress_files,
    _read_json_records,
    analyze as _analyze_stage20g,
)


def analyze(paths):
    summary = _analyze_stage20g(paths)
    summary["task"] = "stage20g_v2_sparse_semantic_recovery_gate_triage"

    progress = {}
    for path in _progress_files(paths):
        for row in _read_json_records(path):
            if row.get("episode_id") is not None:
                progress[_episode_key(row)] = row

    events = []
    for path in _active_event_files(paths):
        for row in _read_json_records(path):
            if row.get("event_type") == "stage19_semantic_resilience_active":
                events.append(row)

    tier_counts = Counter()
    tier_outcome_counts = defaultdict(Counter)
    tier_episode_keys = defaultdict(set)
    tier_failure_type_counts = defaultdict(Counter)
    tier_reason_counts = defaultdict(Counter)
    tier_rejection_counts = defaultdict(Counter)
    strict_pass_events = []
    adapter_events = []
    abstain_events = []

    for row in events:
        gate = row.get("v2_evidence_gate") or {}
        tier = str(row.get("v2_evidence_tier") or gate.get("tier") or "missing")
        key = _episode_key(row)
        progress_row = progress.get(key, {})
        outcome = "success" if float(progress_row.get("success", 0.0) or 0.0) > 0.0 else "failed"
        progress_failure_type = str(
            progress_row.get("stage19_semantic_resilience_episode_failure_type") or "missing"
        )

        tier_counts[tier] += 1
        tier_outcome_counts[tier][outcome] += 1
        tier_episode_keys[tier].add(key)
        tier_failure_type_counts[tier][progress_failure_type] += 1
        tier_reason_counts[tier][str(row.get("reason") or "unknown")] += 1
        for reason in list(gate.get("hard_abstain_reasons") or []):
            tier_rejection_counts[tier][str(reason)] += 1

        if tier == "strict_intervention":
            strict_pass_events.append(row)
        elif tier == "adapter_candidate":
            adapter_events.append(row)
        elif tier == "abstain":
            abstain_events.append(row)

    summary["v2_triage_summary"] = {
        "event_count": len(events),
        "tier_counts": dict(tier_counts),
        "tier_episode_counts": {tier: len(keys) for tier, keys in tier_episode_keys.items()},
        "tier_outcome_counts": {
            tier: dict(counts) for tier, counts in tier_outcome_counts.items()
        },
        "tier_progress_failure_type_counts": {
            tier: dict(counts) for tier, counts in tier_failure_type_counts.items()
        },
        "tier_reason_counts": {tier: dict(counts) for tier, counts in tier_reason_counts.items()},
        "tier_hard_abstain_reason_counts": {
            tier: dict(counts) for tier, counts in tier_rejection_counts.items()
        },
        "strict_intervention_event_count": len(strict_pass_events),
        "adapter_candidate_event_count": len(adapter_events),
        "abstain_event_count": len(abstain_events),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run root, vlmap_safety_debug dir, or active events JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    args = parser.parse_args()

    summary = analyze(args.paths)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

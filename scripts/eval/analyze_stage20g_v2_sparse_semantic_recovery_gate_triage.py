"""Analyze or replay Stage20g-v2 multi-evidence recovery triage logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internnav.semantic_recovery_triage import (
    DEFAULT_SEMANTIC_RECOVERY_TRIAGE_CONFIG,
    classify_semantic_recovery_triage,
)

from analyze_stage20g_sparse_semantic_recovery_gate import (
    _active_event_files,
    _episode_key,
    _progress_files,
    _read_json_records,
    analyze as _analyze_stage20g,
)


EVIDENCE_FIELDS = (
    "s2_policy_conflict",
    "obstacle_context",
    "spatial_constriction",
    "persistence",
    "geometry_safe",
    "active_gate_safe",
    "frontier_like_anchor",
    "escape_anchor_safe",
    "execution_window_safe",
)


def _safe_ratio(numerator, denominator):
    return None if not denominator else float(numerator) / float(denominator)


def _replay_gate(row, triage_config):
    return classify_semantic_recovery_triage(
        row.get("candidate"),
        triage_config,
        failure_type=str(row.get("failure_type") or "unknown"),
        recommended_primitive=str(row.get("recommended_primitive") or "hold_s2"),
        trigger_reasons=row.get("trigger_reasons") or [],
        context_tags=row.get("recovery_context_tags") or row.get("context_tags") or [],
        step_id=row.get("step_id"),
    )


def _compact_candidate_record(row, gate, progress_row, source):
    candidate = row.get("candidate") or {}
    return {
        "scene_episode": _episode_key(row),
        "step_id": row.get("step_id"),
        "outcome": "success"
        if float(progress_row.get("success", 0.0) or 0.0) > 0.0
        else "failed",
        "progress_failure_type": str(
            progress_row.get("stage19_semantic_resilience_episode_failure_type") or "missing"
        ),
        "event_failure_type": str(row.get("failure_type") or "unknown"),
        "recommended_primitive": str(row.get("recommended_primitive") or "unknown"),
        "triage_source": source,
        "tier": str(gate.get("tier") or "missing"),
        "triage_reason": str(gate.get("reason") or "unknown"),
        "online_gate_reason": str(row.get("reason") or "unknown"),
        "trigger_reasons": list(row.get("trigger_reasons") or []),
        "context_tags": list(row.get("recovery_context_tags") or row.get("context_tags") or []),
        "evidence_vote_count": int(gate.get("evidence_vote_count", 0) or 0),
        "evidence": {name: bool(gate.get(name)) for name in EVIDENCE_FIELDS},
        "candidate": {
            "candidate_type": candidate.get("candidate_type"),
            "direction_bucket": candidate.get("direction_bucket"),
            "geometry_safe": bool(candidate.get("geometry_safe")),
            "active_gate_safe": bool(candidate.get("active_gate_safe")),
            "open_score": candidate.get("semantic_resilience_open_score"),
            "recovery_score": candidate.get("semantic_resilience_score"),
            "target_frontier_score": candidate.get("target_frontier_score"),
            "target_frontier_intent_safe": bool(candidate.get("target_frontier_intent_safe")),
            "completed_landmark_penalty": candidate.get("completed_landmark_penalty"),
            "backtrack_distance_m": candidate.get("semantic_resilience_backtrack_distance_m"),
            "step_gap": candidate.get("semantic_resilience_step_gap"),
            "nearby_visit_count": candidate.get("nearby_visit_count"),
        },
    }


def analyze(paths, *, replay_missing=True, triage_config=None):
    summary = _analyze_stage20g(paths)
    summary["task"] = "stage20g_v2_sparse_semantic_recovery_gate_triage"

    replay_config = dict(DEFAULT_SEMANTIC_RECOVERY_TRIAGE_CONFIG)
    replay_config.update(dict(triage_config or {}))

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
    tier_episode_outcome_keys = defaultdict(lambda: defaultdict(set))
    tier_failure_type_counts = defaultdict(Counter)
    tier_failure_type_keys = defaultdict(lambda: defaultdict(set))
    tier_triage_reason_counts = defaultdict(Counter)
    tier_online_reason_counts = defaultdict(Counter)
    tier_rejection_counts = defaultdict(Counter)
    tier_evidence_counts = defaultdict(Counter)
    tier_evidence_pattern_counts = defaultdict(Counter)
    source_counts = Counter()
    candidate_records = {"strict_intervention": [], "adapter_candidate": []}
    embedded_replay_compared_count = 0
    embedded_replay_agreement_count = 0
    embedded_replay_mismatches = []
    applied_tier_counts = Counter()
    applied_records = []
    non_strict_applied_records = []

    for row in events:
        embedded_gate = dict(row.get("v2_evidence_gate") or {})
        embedded_tier = str(
            row.get("v2_evidence_tier") or embedded_gate.get("tier") or ""
        )
        replayed_gate = _replay_gate(row, replay_config)
        replayed_tier = str(replayed_gate.get("tier") or "missing")
        if embedded_tier:
            gate = embedded_gate
            tier = embedded_tier
            source = "embedded"
            embedded_replay_compared_count += 1
            if embedded_tier == replayed_tier:
                embedded_replay_agreement_count += 1
            else:
                embedded_replay_mismatches.append(
                    {
                        "scene_episode": _episode_key(row),
                        "step_id": row.get("step_id"),
                        "embedded_tier": embedded_tier,
                        "replayed_tier": replayed_tier,
                        "embedded_reason": embedded_gate.get("reason"),
                        "replayed_reason": replayed_gate.get("reason"),
                    }
                )
        elif replay_missing:
            gate = replayed_gate
            tier = replayed_tier
            source = "replayed"
        else:
            gate = {}
            tier = "missing"
            source = "missing"

        key = _episode_key(row)
        progress_row = progress.get(key, {})
        outcome = "success" if float(progress_row.get("success", 0.0) or 0.0) > 0.0 else "failed"
        progress_failure_type = str(
            progress_row.get("stage19_semantic_resilience_episode_failure_type") or "missing"
        )

        source_counts[source] += 1
        tier_counts[tier] += 1
        tier_outcome_counts[tier][outcome] += 1
        tier_episode_keys[tier].add(key)
        tier_episode_outcome_keys[tier][outcome].add(key)
        tier_failure_type_counts[tier][progress_failure_type] += 1
        tier_failure_type_keys[tier][progress_failure_type].add(key)
        tier_triage_reason_counts[tier][str(gate.get("reason") or "unknown")] += 1
        tier_online_reason_counts[tier][str(row.get("reason") or "unknown")] += 1
        for reason in list(gate.get("hard_abstain_reasons") or []):
            tier_rejection_counts[tier][str(reason)] += 1
        enabled_evidence = []
        for name in EVIDENCE_FIELDS:
            if bool(gate.get(name)):
                tier_evidence_counts[tier][name] += 1
                enabled_evidence.append(name)
        tier_evidence_pattern_counts[tier]["+".join(enabled_evidence) or "none"] += 1

        if tier in candidate_records:
            candidate_records[tier].append(
                _compact_candidate_record(row, gate, progress_row, source)
            )
        if bool(row.get("applied")):
            applied_record = _compact_candidate_record(
                row, gate, progress_row, source
            )
            applied_record.update(
                {
                    "actions": list(row.get("actions") or []),
                    "action_count": len(list(row.get("actions") or [])),
                    "action_plan": dict(row.get("action_plan") or {}),
                    "shadow_only": bool(row.get("shadow_only")),
                }
            )
            applied_tier_counts[tier] += 1
            applied_records.append(applied_record)
            if tier != "strict_intervention":
                non_strict_applied_records.append(applied_record)

    final_episode_keys = set(progress)
    failed_stuck_keys = {
        key
        for key, row in progress.items()
        if float(row.get("success", 0.0) or 0.0) <= 0.0
        and str(row.get("stage19_semantic_resilience_episode_failure_type") or "missing")
        == "stuck_collision"
    }
    strict_keys = tier_episode_keys.get("strict_intervention", set())
    adapter_keys = tier_episode_keys.get("adapter_candidate", set())
    strict_failed_keys = tier_episode_outcome_keys["strict_intervention"].get("failed", set())
    strict_success_keys = tier_episode_outcome_keys["strict_intervention"].get("success", set())
    adapter_stuck_keys = tier_failure_type_keys["adapter_candidate"].get("stuck_collision", set())
    strict_stuck_keys = tier_failure_type_keys["strict_intervention"].get("stuck_collision", set())

    summary["v2_triage_summary"] = {
        "event_count": len(events),
        "final_episode_count": len(final_episode_keys),
        "triage_source_counts": dict(source_counts),
        "replay_missing_enabled": bool(replay_missing),
        "replay_config": replay_config,
        "embedded_replay_comparison": {
            "compared_event_count": embedded_replay_compared_count,
            "agreement_event_count": embedded_replay_agreement_count,
            "mismatch_event_count": len(embedded_replay_mismatches),
            "agreement_rate": _safe_ratio(
                embedded_replay_agreement_count, embedded_replay_compared_count
            ),
            "mismatches": embedded_replay_mismatches,
        },
        "active_safety_audit": {
            "applied_event_count": len(applied_records),
            "applied_tier_counts": dict(applied_tier_counts),
            "strict_applied_event_count": applied_tier_counts.get(
                "strict_intervention", 0
            ),
            "non_strict_applied_event_count": len(non_strict_applied_records),
            "all_applied_events_are_strict": not non_strict_applied_records,
            "applied_records": applied_records,
            "violations": non_strict_applied_records,
        },
        "tier_counts": dict(tier_counts),
        "tier_episode_counts": {tier: len(keys) for tier, keys in tier_episode_keys.items()},
        "tier_outcome_counts": {
            tier: dict(counts) for tier, counts in tier_outcome_counts.items()
        },
        "tier_episode_outcome_counts": {
            tier: {outcome: len(keys) for outcome, keys in outcome_keys.items()}
            for tier, outcome_keys in tier_episode_outcome_keys.items()
        },
        "tier_progress_failure_type_counts": {
            tier: dict(counts) for tier, counts in tier_failure_type_counts.items()
        },
        "tier_progress_failure_type_episode_counts": {
            tier: {failure_type: len(keys) for failure_type, keys in failure_keys.items()}
            for tier, failure_keys in tier_failure_type_keys.items()
        },
        "tier_triage_reason_counts": {
            tier: dict(counts) for tier, counts in tier_triage_reason_counts.items()
        },
        "tier_online_gate_reason_counts": {
            tier: dict(counts) for tier, counts in tier_online_reason_counts.items()
        },
        "tier_hard_abstain_reason_counts": {
            tier: dict(counts) for tier, counts in tier_rejection_counts.items()
        },
        "tier_evidence_true_counts": {
            tier: dict(counts) for tier, counts in tier_evidence_counts.items()
        },
        "tier_evidence_pattern_counts": {
            tier: dict(counts) for tier, counts in tier_evidence_pattern_counts.items()
        },
        "strict_intervention_event_count": tier_counts.get("strict_intervention", 0),
        "adapter_candidate_event_count": tier_counts.get("adapter_candidate", 0),
        "abstain_event_count": tier_counts.get("abstain", 0),
        "decision_metrics": {
            "failed_stuck_episode_count": len(failed_stuck_keys),
            "strict_episode_count": len(strict_keys),
            "strict_failed_episode_count": len(strict_failed_keys),
            "strict_success_episode_count": len(strict_success_keys),
            "strict_failed_precision": _safe_ratio(len(strict_failed_keys), len(strict_keys)),
            "strict_success_risk": _safe_ratio(len(strict_success_keys), len(strict_keys)),
            "strict_stuck_coverage": _safe_ratio(
                len(strict_stuck_keys.intersection(failed_stuck_keys)), len(failed_stuck_keys)
            ),
            "adapter_stuck_density": _safe_ratio(
                len(adapter_stuck_keys.intersection(failed_stuck_keys)), len(adapter_keys)
            ),
            "strict_or_adapter_stuck_coverage": _safe_ratio(
                len((strict_keys | adapter_keys).intersection(failed_stuck_keys)),
                len(failed_stuck_keys),
            ),
        },
        "candidate_records": candidate_records,
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
    parser.add_argument(
        "--no-replay-missing",
        action="store_true",
        help="Leave legacy events without embedded V2 evidence in the missing tier.",
    )
    parser.add_argument(
        "--triage-config-json",
        type=Path,
        default=None,
        help="Optional JSON object overriding shared replay defaults.",
    )
    args = parser.parse_args()

    triage_config = None
    if args.triage_config_json:
        triage_config = json.loads(args.triage_config_json.read_text(encoding="utf-8"))
        if not isinstance(triage_config, dict):
            parser.error("--triage-config-json must contain a JSON object")

    summary = analyze(
        args.paths,
        replay_missing=not args.no_replay_missing,
        triage_config=triage_config,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

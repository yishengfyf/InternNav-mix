"""Audit Stage53 same-event control/look-down/context four-arm shadow."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ARMS = {"control", "lookdown_only", "context_only", "lookdown_context"}


def _read_jsonl(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(run_root: Path, expected_episodes: int):
    progress = _read_jsonl(run_root / "progress.json")
    event_paths = glob.glob(
        str(run_root / "vlmap_safety_debug" / "*" / "stage53_recovery_ab_events.jsonl")
    )
    events = [row for path in event_paths for row in _read_jsonl(Path(path))]
    arms = [arm for event in events for arm in list(event.get("arms") or [])]
    by_variant = defaultdict(list)
    for arm in arms:
        by_variant[str(arm.get("variant") or "missing")].append(arm)
    variant_summary = {}
    for variant, rows in sorted(by_variant.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        variant_summary[variant] = {
            "event_count": len(rows),
            "ok_count": len(ok),
            "error_count": len(rows) - len(ok),
            "protocol_valid_count": sum(
                bool(row.get("hinted_protocol_valid")) for row in ok
            ),
            "valid_pixel_count": sum(bool(row.get("hinted_valid")) for row in ok),
            "reobserve_count": sum(bool(row.get("hinted_reobserve")) for row in ok),
            "continues_repeated_error_direction_count": sum(
                bool(row.get("continues_repeated_error_direction")) for row in ok
            ),
            "change_type_counts": dict(
                Counter(str(row.get("change_type")) for row in ok)
            ),
        }
    geometry_reports = [
        (event.get("lookdown_geometry") or {}).get("report") or {}
        for event in events
    ]
    complete_event_count = sum(
        {str(arm.get("variant")) for arm in list(event.get("arms") or [])}
        == EXPECTED_ARMS
        for event in events
    )
    errors = [arm for arm in arms if arm.get("status") != "ok"]
    binding_violations = [
        arm for arm in arms if not bool(arm.get("prompt_image_binding_valid"))
    ]
    context_claim_violations = [
        arm for arm in arms if list(arm.get("forbidden_context_terms") or [])
    ]
    state_mutations = [event for event in events if event.get("official_memory_mutated")]
    action_violations = [event for event in events if event.get("action_applied")]
    gt_leakage = [event for event in events if list(event.get("gt_fields_used") or [])]
    integrity_passed = bool(
        len(progress) == expected_episodes
        and events
        and complete_event_count == len(events)
        and set(by_variant) == EXPECTED_ARMS
        and not errors
        and not binding_violations
        and not context_claim_violations
        and not state_mutations
        and not action_violations
        and not gt_leakage
        and all(
            bool((event.get("lookdown_geometry") or {}).get("observation_readable"))
            for event in events
        )
        and all(
            bool((event.get("lookdown_geometry") or {}).get("temporary_update_valid"))
            for event in events
        )
    )
    return {
        "task": "stage53_recovery_ab_shadow",
        "expected_episode_count": int(expected_episodes),
        "completed_episode_count": len(progress),
        "event_count": len(events),
        "complete_four_arm_event_count": int(complete_event_count),
        "variant_summary": variant_summary,
        "lookdown_geometry_valid_event_count": sum(bool(row.get("valid")) for row in geometry_reports),
        "lookdown_geometry_eligible_record_count": sum(
            int(row.get("eligible_count", 0) or 0) for row in geometry_reports
        ),
        "lookdown_geometry_in_bounds_record_count": sum(
            bool(item.get("lookahead_pixel_in_bounds"))
            for report in geometry_reports
            for item in list(report.get("lookahead_records") or [])
        ),
        "lookdown_geometry_reason_counts": dict(
            Counter(str(row.get("reason")) for row in geometry_reports)
        ),
        "counterfactual_error_count": len(errors),
        "prompt_image_binding_violation_count": len(binding_violations),
        "recovery_context_claim_violation_count": len(context_claim_violations),
        "official_memory_mutation_count": len(state_mutations),
        "action_applied_violation_count": len(action_violations),
        "gt_leakage_count": len(gt_leakage),
        "unknown_is_free": False,
        "shadow_only": True,
        "integrity_passed": integrity_passed,
        "error_records": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_root, args.expected_episodes)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

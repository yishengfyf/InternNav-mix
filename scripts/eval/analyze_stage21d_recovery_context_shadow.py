"""Audit Stage21d text-only vs text+event-images recovery-context shadow queries."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_IDENTITY = (
    "scene_id",
    "episode_id",
    "current_query_step",
    "trigger_step",
    "triage_tier",
    "failure_type",
)


def _read_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _progress(run_root: Path):
    return _read_jsonl(run_root / "progress.json")


def _events(run_root: Path):
    paths = glob.glob(
        str(run_root / "vlmap_safety_debug" / "*" / "s2_recovery_context_events.jsonl")
    )
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _ratio(value, total):
    return None if not total else float(value) / float(total)


def analyze(run_root: Path, expected_episodes: int):
    progress = _progress(run_root)
    events = _events(run_root)
    set_events = [row for row in events if row.get("event_type") == "s2_recovery_context_set"]
    cf_events = [
        row for row in events
        if row.get("event_type") == "s2_recovery_context_counterfactual"
    ]
    variants = defaultdict(list)
    for row in cf_events:
        variants[str(row.get("variant") or "missing")].append(row)

    variant_summary = {}
    for name, rows in sorted(variants.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        valid = [row for row in ok if bool(row.get("hinted_valid"))]
        continued = [row for row in ok if bool(row.get("continues_repeated_error_direction"))]
        latencies = sorted(float(row.get("latency_ms", 0.0) or 0.0) for row in ok)
        variant_summary[name] = {
            "event_count": len(rows),
            "ok_count": len(ok),
            "error_count": len(rows) - len(ok),
            "hinted_valid_count": len(valid),
            "hinted_valid_rate": _ratio(len(valid), len(ok)),
            "change_type_counts": dict(Counter(str(row.get("change_type")) for row in ok)),
            "direction_changed_count": sum(
                bool(row.get("direction_bucket_changed")) for row in ok
            ),
            "large_shift_40px_count": sum(
                bool(row.get("large_valid_pixel_shift_40px")) for row in ok
            ),
            "continues_repeated_error_direction_count": len(continued),
            "continues_repeated_error_direction_rate": _ratio(len(continued), len(ok)),
            "extra_image_event_count": sum(int(row.get("extra_image_count", 0) or 0) > 0 for row in ok),
            "latency_mean_ms": (
                None if not latencies else sum(latencies) / len(latencies)
            ),
            "latency_p95_ms": (
                None if not latencies else latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            ),
        }

    paired = defaultdict(dict)
    for row in cf_events:
        key = (
            row.get("scene_id"),
            row.get("episode_id"),
            row.get("trigger_step"),
            row.get("current_query_step"),
        )
        paired[key][str(row.get("variant") or "missing")] = row
    complete_pairs = [
        rows for rows in paired.values() if {"text_only", "text_images"}.issubset(rows)
    ]
    output_diff_count = sum(
        rows["text_only"].get("hinted_output") != rows["text_images"].get("hinted_output")
        for rows in complete_pairs
    )
    missing_identity = [
        row for row in cf_events if any(row.get(field) is None for field in REQUIRED_IDENTITY)
    ]
    error_events = [row for row in cf_events if row.get("status") != "ok"]
    action_violations = [row for row in events if bool(row.get("action_applied"))]
    action_violations.extend(
        row
        for row in progress
        if int(row.get("s2_loop_strict_active_applied_count", 0) or 0) > 0
        or int(row.get("stage19_semantic_resilience_active_applied_count", 0) or 0) > 0
    )
    gt_leakage = [row for row in events if list(row.get("gt_fields_used") or [])]
    variants_present = set(variants)
    image_variant_has_evidence = any(
        int(row.get("extra_image_count", 0) or 0) > 0
        for row in variants.get("text_images", [])
    )
    integrity_passed = bool(
        len(progress) == expected_episodes
        and set_events
        and cf_events
        and {"text_only", "text_images"}.issubset(variants_present)
        and complete_pairs
        and image_variant_has_evidence
        and not error_events
        and not missing_identity
        and not action_violations
        and not gt_leakage
    )
    return {
        "task": "stage21d_recovery_context_shadow_ab",
        "expected_episode_count": expected_episodes,
        "completed_episode_count": len(progress),
        "context_set_event_count": len(set_events),
        "counterfactual_event_count": len(cf_events),
        "context_episode_count": len(
            {(row.get("scene_id"), row.get("episode_id")) for row in set_events}
        ),
        "triage_tier_counts": dict(
            Counter(str(row.get("triage_tier")) for row in set_events)
        ),
        "variant_summary": variant_summary,
        "ab_pair_count": len(complete_pairs),
        "ab_output_difference_count": output_diff_count,
        "ab_output_difference_rate": _ratio(output_diff_count, len(complete_pairs)),
        "missing_identity_count": len(missing_identity),
        "counterfactual_error_count": len(error_events),
        "action_applied_violation_count": len(action_violations),
        "gt_leakage_count": len(gt_leakage),
        "text_images_has_event_image_evidence": image_variant_has_evidence,
        "integrity_passed": integrity_passed,
        "interpretation_guard": (
            "A valid or changed pixel is a steerability/data-chain signal, not causal "
            "navigation improvement. Active remains disabled in this audit."
        ),
        "error_records": error_events,
        "missing_identity_records": missing_identity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.run_root, args.expected_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_all and not summary["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


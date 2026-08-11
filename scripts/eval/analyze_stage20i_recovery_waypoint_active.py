"""Analyze Stage20i directional recovery-waypoint active probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_stage20g_sparse_semantic_recovery_gate import (
    _active_event_files,
    _episode_key,
    _progress_files,
    _read_json_records,
)
from analyze_stage20g_v2_sparse_semantic_recovery_gate_triage import (
    analyze as analyze_stage20g_v2,
)


ALLOWED_APPLIED_TIERS = {"strict_intervention", "adapter_candidate"}
METRICS = {
    "success": "success",
    "spl": "spl",
    "ne": "ne",
    "steps": "steps",
    "collision_count": "collision_count",
}


def _load_progress(paths):
    rows = {}
    for path in _progress_files(paths):
        for row in _read_json_records(path):
            if row.get("episode_id") is not None:
                rows[_episode_key(row)] = row
    return rows


def _mean(rows, field):
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return None if not values else sum(values) / len(values)


def _comparison_record(key, baseline, active, applied_record=None):
    record = {
        "scene_episode": key,
        "baseline_success": float(baseline.get("success", 0.0) or 0.0),
        "active_success": float(active.get("success", 0.0) or 0.0),
    }
    for output_name, field in METRICS.items():
        baseline_value = float(baseline.get(field, 0.0) or 0.0)
        active_value = float(active.get(field, 0.0) or 0.0)
        record[f"baseline_{output_name}"] = baseline_value
        record[f"active_{output_name}"] = active_value
        record[f"delta_{output_name}"] = active_value - baseline_value
    if applied_record:
        record.update(
            {
                "tier": applied_record.get("tier"),
                "step_id": applied_record.get("step_id"),
                "execution_mode": applied_record.get("execution_mode"),
                "pixel_goal_plan": applied_record.get("pixel_goal_plan"),
                "candidate": applied_record.get("candidate"),
            }
        )
    return record


def analyze(active_paths, *, baseline_progress=None):
    summary = analyze_stage20g_v2(active_paths)
    summary["task"] = "stage20i_recovery_waypoint_active_probe"
    active_progress = _load_progress(active_paths)
    execution_failure_records = []
    for path in _active_event_files(active_paths):
        for row in _read_json_records(path):
            if row.get("event_type") not in {
                "stage19_semantic_resilience_active",
                "stage19_semantic_resilience_execution",
            }:
                continue
            if str(row.get("reason") or "") not in {
                "directional_pixel_goal_execution_failed",
                "directional_replan_generation_failed",
            }:
                continue
            candidate = dict(row.get("candidate") or {})
            execution_failure_records.append(
                {
                    "scene_episode": _episode_key(row),
                    "step_id": row.get("step_id"),
                    "tier": str(
                        row.get("v2_evidence_tier")
                        or (row.get("v2_evidence_gate") or {}).get("tier")
                        or "missing"
                    ),
                    "direction": candidate.get("direction_bucket"),
                    "pixel_goal_plan": dict(row.get("pixel_goal_plan") or {}),
                    "execution_error_type": row.get("execution_error_type"),
                    "execution_error": row.get("execution_error")
                    or row.get("post_apply_execution_error"),
                }
            )
    active_audit = summary.get("v2_triage_summary", {}).get("active_safety_audit", {})
    applied_records = list(active_audit.get("applied_records") or [])
    applied_tiers = Counter(str(row.get("tier") or "missing") for row in applied_records)
    unexpected = [
        row for row in applied_records if str(row.get("tier") or "missing") not in ALLOWED_APPLIED_TIERS
    ]
    non_directional = [
        row
        for row in applied_records
        if str(row.get("execution_mode") or "") != "directional_pixel_goal"
    ]
    invalid_plans = [
        row
        for row in applied_records
        if not bool((row.get("pixel_goal_plan") or {}).get("valid"))
    ]
    summary["stage20i_execution_safety_audit"] = {
        "allowed_applied_tiers": sorted(ALLOWED_APPLIED_TIERS),
        "applied_event_count": len(applied_records),
        "applied_tier_counts": dict(applied_tiers),
        "unexpected_applied_tier_count": len(unexpected),
        "non_directional_execution_count": len(non_directional),
        "invalid_pixel_goal_plan_count": len(invalid_plans),
        "execution_failure_count": len(execution_failure_records),
        "execution_failure_records": execution_failure_records,
        "passed": (
            not unexpected
            and not non_directional
            and not invalid_plans
            and not execution_failure_records
        ),
    }

    if baseline_progress is None:
        summary["paired_comparison"] = {
            "available": False,
            "reason": "baseline_progress_not_provided",
        }
        return summary

    baseline_rows = _load_progress([baseline_progress])
    common_keys = sorted(set(active_progress).intersection(baseline_rows))
    applied_by_key = {
        str(row.get("scene_episode")): row
        for row in applied_records
        if row.get("scene_episode")
    }
    applied_keys = sorted(set(applied_by_key).intersection(common_keys))
    common_active = [active_progress[key] for key in common_keys]
    common_baseline = [baseline_rows[key] for key in common_keys]
    aggregate = {"episode_count": len(common_keys)}
    for output_name, field in METRICS.items():
        baseline_mean = _mean(common_baseline, field)
        active_mean = _mean(common_active, field)
        aggregate[f"baseline_{output_name}_mean"] = baseline_mean
        aggregate[f"active_{output_name}_mean"] = active_mean
        aggregate[f"delta_{output_name}_mean"] = (
            None
            if baseline_mean is None or active_mean is None
            else active_mean - baseline_mean
        )

    applied_comparisons = [
        _comparison_record(
            key,
            baseline_rows[key],
            active_progress[key],
            applied_by_key[key],
        )
        for key in applied_keys
    ]
    win_records = [
        row
        for row in applied_comparisons
        if row["baseline_success"] <= 0.0 and row["active_success"] > 0.0
    ]
    regression_records = [
        row
        for row in applied_comparisons
        if row["baseline_success"] > 0.0 and row["active_success"] <= 0.0
    ]
    summary["paired_comparison"] = {
        "available": True,
        "baseline_progress": str(baseline_progress),
        "active_episode_count": len(active_progress),
        "baseline_episode_count": len(baseline_rows),
        "common_episode_count": len(common_keys),
        "missing_active_episode_count": len(set(baseline_rows).difference(active_progress)),
        "missing_baseline_episode_count": len(set(active_progress).difference(baseline_rows)),
        "aggregate": aggregate,
        "applied_episode_count": len(applied_keys),
        "failed_to_success_count": len(win_records),
        "success_to_failed_count": len(regression_records),
        "net_success_flip": len(win_records) - len(regression_records),
        "failed_to_success_records": win_records,
        "success_to_failed_records": regression_records,
        "applied_episode_records": applied_comparisons,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Stage20i run root or vlmap_safety_debug directory.",
    )
    parser.add_argument(
        "--baseline-progress",
        type=Path,
        default=None,
        help="Optional paired Stage20g-v2 shadow progress.json.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze(args.paths, baseline_progress=args.baseline_progress)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

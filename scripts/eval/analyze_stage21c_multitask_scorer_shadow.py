#!/usr/bin/env python3
"""Audit Stage21c online frozen-scorer shadow events."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _distribution(values: Iterable[float]) -> Dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(items), "mean": mean(items), "min": min(items), "max": max(items)}


def build_audit(run_root: Path, expected_episodes: int, minimum_valid_rate: float,
                maximum_p95_latency_ms: float) -> Dict[str, Any]:
    progress = _rows(run_root / "progress.json")
    event_paths = sorted(run_root.glob("vlmap_safety_debug/**/occ_memory/memory_events.jsonl"))
    all_events = [row for path in event_paths for row in _rows(path)]
    candidate_events = [row for row in all_events if row.get("event_type") == "occ_memory_query_candidates"]
    shadow_events = [row for row in candidate_events if (row.get("stage21_multitask_shadow") or {}).get("enabled")]
    valid = [row for row in shadow_events if (row.get("stage21_multitask_shadow") or {}).get("valid")]
    errors = [row for row in shadow_events if (row.get("stage21_multitask_shadow") or {}).get("reason") in {"error", "not_initialized"}]
    applied = [row for row in shadow_events if (row.get("stage21_multitask_shadow") or {}).get("action_applied")]
    nonfinite = [row for row in valid if not _finite(row.get("stage21_multitask_shadow"))]
    unsafe_progress = [row for row in valid if (row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("valid") and not (row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("geometry_safe")]
    unsafe_recovery = [row for row in valid if (row.get("stage21_multitask_shadow") or {}).get("recovery_selected", {}).get("valid") and not (row.get("stage21_multitask_shadow") or {}).get("recovery_selected", {}).get("geometry_safe")]
    dimensions = Counter(int((row.get("stage21_multitask_shadow") or {}).get("feature_dim", -1)) for row in valid)
    schemas = Counter(str((row.get("stage21_multitask_shadow") or {}).get("schema_version") or "missing") for row in valid)
    latency = [float((row.get("stage21_multitask_shadow") or {}).get("inference_latency_ms", 0.0)) for row in valid]
    latency_sorted = sorted(latency)
    p95 = latency_sorted[min(len(latency_sorted) - 1, int(0.95 * len(latency_sorted)))] if latency_sorted else None
    missing = [float((row.get("stage21_multitask_shadow") or {}).get("missing_numeric_mean", 0.0)) for row in valid]
    progress_changes = sum(bool((row.get("stage21_multitask_shadow") or {}).get("progress_changes_candidate_score")) for row in valid)
    intent_changes = sum(bool((row.get("stage21_multitask_shadow") or {}).get("progress_changes_intent_alignment")) for row in valid)
    selected_types = Counter(str((row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("candidate_type") or "none") for row in valid)
    selected_directions = Counter(str((row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("direction_bucket") or "none") for row in valid)
    scores = {name: [] for name in ("progress", "safety", "geometry_safe_probability", "recovery", "recovery_promising")}
    for row in valid:
        for item in (row.get("stage21_multitask_shadow") or {}).get("scores") or []:
            for name in scores:
                if item.get(name) is not None:
                    scores[name].append(float(item[name]))
    valid_rate = len(valid) / max(1, len(shadow_events))
    checks = {
        "episode_count": len(progress) == expected_episodes,
        "has_shadow_events": bool(shadow_events),
        "minimum_valid_rate": valid_rate >= minimum_valid_rate,
        "zero_errors": not errors,
        "zero_action_applied": not applied,
        "finite_features_and_scores": not nonfinite,
        "feature_dim_353": bool(valid) and set(dimensions) == {353},
        "schema_v1": bool(valid) and set(schemas) == {"stage21_structured_online_v1"},
        "geometry_filter_progress": not unsafe_progress,
        "geometry_filter_recovery": not unsafe_recovery,
        "latency_p95": p95 is not None and p95 <= maximum_p95_latency_ms,
    }
    return {
        "task": "stage21c_multitask_scorer_online_shadow_audit",
        "run_root": str(run_root.resolve()), "expected_episode_count": expected_episodes,
        "progress_episode_count": len(progress), "memory_event_file_count": len(event_paths),
        "candidate_event_count": len(candidate_events), "shadow_event_count": len(shadow_events),
        "valid_shadow_event_count": len(valid), "valid_shadow_event_rate": valid_rate,
        "error_count": len(errors), "action_applied_count": len(applied), "nonfinite_count": len(nonfinite),
        "unsafe_progress_selected_count": len(unsafe_progress), "unsafe_recovery_selected_count": len(unsafe_recovery),
        "feature_dimensions": dict(dimensions), "schema_versions": dict(schemas),
        "progress_change_candidate_score_count": progress_changes,
        "progress_change_candidate_score_rate": progress_changes / max(1, len(valid)),
        "progress_change_intent_count": intent_changes,
        "progress_change_intent_rate": intent_changes / max(1, len(valid)),
        "progress_selected_type_counts": dict(selected_types),
        "progress_selected_direction_counts": dict(selected_directions),
        "missing_numeric": _distribution(missing), "latency_ms": {**_distribution(latency), "p95": p95},
        "score_distributions": {name: _distribution(values) for name, values in scores.items()},
        "thresholds": {"minimum_valid_rate": minimum_valid_rate, "maximum_p95_latency_ms": maximum_p95_latency_ms},
        "checks": checks, "passed": all(checks.values()),
        "scope": {"shadow_only": True, "frozen_s2_nextdit": True, "active_recovery": False},
        "sample_errors": [{"scene_id": row.get("scene_id"), "episode_id": row.get("episode_id"), "shadow": row.get("stage21_multitask_shadow")} for row in errors[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-valid-rate", type=float, default=0.95)
    parser.add_argument("--max-p95-latency-ms", type=float, default=250.0)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    result = build_audit(args.run_root, args.expected_episodes, args.min_valid_rate, args.max_p95_latency_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.require_all and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

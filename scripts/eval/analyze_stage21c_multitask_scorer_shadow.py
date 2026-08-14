#!/usr/bin/env python3
"""Audit Stage21c online frozen-scorer shadow events.

The report separates three questions that must not be conflated:
1. runtime/action invariance;
2. learned candidate-ranking diagnostics;
3. conservative recovery-triage timing and snapshot coverage.

No online outcome label is used and no recovery action is applied.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


RECOVERY_TYPES = {"resilience_backtrack", "backtrack_reobserve"}


def _rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


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


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    items = sorted(float(value) for value in values)
    if not items:
        return None
    index = min(len(items) - 1, max(0, int(float(percentile) * len(items))))
    return float(items[index])


def _event_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(row.get("scene_id") or ""),
        int(row.get("episode_id", -1) or -1),
        int(row.get("step_id", -1) or -1),
    )


def _score_margin(items: List[Dict[str, Any]], key: str, indices: Iterable[int]) -> Optional[float]:
    values = sorted(
        (float(items[index].get(key, 0.0) or 0.0) for index in indices),
        reverse=True,
    )
    return values[0] - values[1] if len(values) >= 2 else None


def _snapshot_exists(run_root: Path, row: Dict[str, Any]) -> bool:
    rgb_file = str(row.get("rgb_file") or "")
    if not rgb_file:
        return False
    direct = run_root / "vlmap_safety_debug" / rgb_file
    if direct.is_file():
        return True
    return any(path.is_file() for path in run_root.glob(f"vlmap_safety_debug/**/{Path(rgb_file).name}"))


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def build_audit(
    run_root: Path,
    expected_episodes: int,
    minimum_valid_rate: float,
    maximum_p95_latency_ms: float,
    minimum_loop_events: int = 0,
    maximum_success_trigger_rate: Optional[float] = None,
    maximum_strict_tier_rate: Optional[float] = None,
    strict_rate_minimum_events: int = 4,
) -> Dict[str, Any]:
    progress = _rows(run_root / "progress.json")
    navigation = _json_object(run_root / "result.json")
    event_paths = sorted(run_root.glob("vlmap_safety_debug/**/occ_memory/memory_events.jsonl"))
    all_events = [row for path in event_paths for row in _rows(path)]
    candidate_events = [row for row in all_events if row.get("event_type") == "occ_memory_query_candidates"]
    shadow_events = [
        row for row in candidate_events
        if (row.get("stage21_multitask_shadow") or {}).get("enabled")
    ]
    valid = [
        row for row in shadow_events
        if (row.get("stage21_multitask_shadow") or {}).get("valid")
    ]
    errors = [
        row for row in shadow_events
        if (row.get("stage21_multitask_shadow") or {}).get("reason") in {"error", "not_initialized"}
    ]
    applied = [
        row for row in shadow_events
        if (row.get("stage21_multitask_shadow") or {}).get("action_applied")
    ]
    nonfinite = [row for row in valid if not _finite(row.get("stage21_multitask_shadow"))]
    unsafe_progress = [
        row for row in valid
        if (row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("valid")
        and not (row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("geometry_safe")
    ]
    unsafe_recovery = [
        row for row in valid
        if (row.get("stage21_multitask_shadow") or {}).get("recovery_selected", {}).get("valid")
        and not (row.get("stage21_multitask_shadow") or {}).get("recovery_selected", {}).get("geometry_safe")
    ]
    dimensions = Counter(
        int((row.get("stage21_multitask_shadow") or {}).get("feature_dim", -1)) for row in valid
    )
    schemas = Counter(
        str((row.get("stage21_multitask_shadow") or {}).get("schema_version") or "missing")
        for row in valid
    )
    latency = [
        float((row.get("stage21_multitask_shadow") or {}).get("inference_latency_ms", 0.0))
        for row in valid
    ]
    p95 = _percentile(latency, 0.95)
    missing = [
        float((row.get("stage21_multitask_shadow") or {}).get("missing_numeric_mean", 0.0))
        for row in valid
    ]
    progress_changes = sum(
        bool((row.get("stage21_multitask_shadow") or {}).get("progress_changes_candidate_score"))
        for row in valid
    )
    intent_changes = sum(
        bool((row.get("stage21_multitask_shadow") or {}).get("progress_changes_intent_alignment"))
        for row in valid
    )
    selected_types = Counter(
        str((row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("candidate_type") or "none")
        for row in valid
    )
    selected_directions = Counter(
        str((row.get("stage21_multitask_shadow") or {}).get("progress_selected", {}).get("direction_bucket") or "none")
        for row in valid
    )

    scores = {
        name: []
        for name in ("progress", "safety", "geometry_safe_probability", "recovery", "recovery_promising")
    }
    candidate_types: Counter[str] = Counter()
    progress_margins: List[float] = []
    candidate_score_margins: List[float] = []
    intent_margins: List[float] = []
    recovery_eligible_count = 0
    recovery_selected_count = 0
    recovery_selected_types: Counter[str] = Counter()
    scored_candidate_count = 0
    for row in valid:
        shadow = row.get("stage21_multitask_shadow") or {}
        items = list(shadow.get("scores") or [])
        source_items = list(row.get("candidates") or [])
        scored_candidate_count += len(items)
        ordinary = [
            index for index, item in enumerate(items)
            if str(item.get("candidate_type") or "unknown") not in RECOVERY_TYPES
            and bool(item.get("geometry_safe"))
        ]
        recovery = [
            index for index, item in enumerate(items)
            if str(item.get("candidate_type") or "unknown") in RECOVERY_TYPES
            and bool(item.get("geometry_safe"))
        ]
        recovery_eligible_count += len(recovery)
        selected_recovery = shadow.get("recovery_selected") or {}
        if selected_recovery.get("valid"):
            recovery_selected_count += 1
            recovery_selected_types[str(selected_recovery.get("candidate_type") or "unknown")] += 1
        progress_margin = _score_margin(items, "progress", ordinary)
        if progress_margin is not None:
            progress_margins.append(progress_margin)
        source_ordinary = [index for index in ordinary if index < len(source_items)]
        candidate_margin = _score_margin(source_items, "score", source_ordinary)
        if candidate_margin is not None:
            candidate_score_margins.append(candidate_margin)
        intent_margin = _score_margin(source_items, "intent_alignment_score", source_ordinary)
        if intent_margin is not None:
            intent_margins.append(intent_margin)
        for item in items:
            candidate_types[str(item.get("candidate_type") or "unknown")] += 1
            for name in scores:
                if item.get(name) is not None:
                    scores[name].append(float(item[name]))

    loop_paths = sorted(run_root.glob("vlmap_safety_debug/**/s2_action_loop_events.jsonl"))
    loop_events = [row for path in loop_paths for row in _rows(path)]
    candidate_by_key = {_event_key(row): row for row in candidate_events}
    triage_tiers: Counter[str] = Counter()
    triage_reasons: Counter[str] = Counter()
    triage_join: Counter[str] = Counter()
    triage_evidence: Counter[str] = Counter()
    loop_steps: List[float] = []
    strict_steps: List[float] = []
    loop_streaks: List[float] = []
    loop_turns: List[float] = []
    loop_translations: List[float] = []
    gt_fields_nonempty = 0
    snapshot_expected_count = 0
    snapshot_present_count = 0
    strict_timing_unsafe_count = 0
    loop_samples: List[Dict[str, Any]] = []
    for loop in loop_events:
        triage = loop.get("triage") or {}
        tier = str(loop.get("triage_tier") or triage.get("tier") or "missing")
        reason = str(loop.get("triage_reason") or triage.get("reason") or "missing")
        triage_tiers[tier] += 1
        triage_reasons[reason] += 1
        for field in (
            "s2_policy_conflict", "obstacle_context", "spatial_constriction", "persistence",
            "escape_anchor_safe", "execution_window_safe", "intervention_time_safe",
            "backtrack_distance_safe", "anchor_fresh", "geometry_safe",
        ):
            if bool(triage.get(field)):
                triage_evidence[field] += 1
        step_id = float(loop.get("step_id", 0) or 0)
        loop_steps.append(step_id)
        if tier == "strict_intervention":
            strict_steps.append(step_id)
            if not (
                bool(triage.get("intervention_time_safe"))
                and bool(triage.get("execution_window_safe"))
                and bool(triage.get("geometry_safe"))
                and bool(triage.get("escape_anchor_safe"))
            ):
                strict_timing_unsafe_count += 1
        loop_streaks.append(float(loop.get("same_turn_generation_streak", 0) or 0))
        loop_turns.append(float(loop.get("cumulative_turn_actions", 0) or 0))
        loop_translations.append(float(loop.get("translation_m", 0.0) or 0.0))
        if loop.get("gt_fields_used"):
            gt_fields_nonempty += 1
        snapshot_expected = bool(loop.get("rgb_snapshot_expected"))
        if snapshot_expected:
            snapshot_expected_count += 1
            if _snapshot_exists(run_root, loop):
                snapshot_present_count += 1
        matched = candidate_by_key.get(_event_key(loop))
        stage_shadow = (matched or {}).get("stage21_multitask_shadow") or {}
        if matched is None:
            triage_join["missing_candidate_event"] += 1
        else:
            triage_join["candidate_event_found"] += 1
            if stage_shadow.get("valid"):
                triage_join["stage21_valid"] += 1
            if (stage_shadow.get("recovery_selected") or {}).get("valid"):
                triage_join["stage21_recovery_selected"] += 1
            else:
                triage_join["stage21_recovery_not_selected"] += 1
        if len(loop_samples) < 50:
            loop_samples.append({
                "scene_id": loop.get("scene_id"),
                "episode_id": loop.get("episode_id"),
                "step_id": loop.get("step_id"),
                "failure_type": loop.get("failure_type"),
                "triage_tier": tier,
                "triage_reason": reason,
                "rgb_file": loop.get("rgb_file"),
                "stage21_progress_selected": stage_shadow.get("progress_selected"),
                "stage21_recovery_selected": stage_shadow.get("recovery_selected"),
            })

    loop_snapshot_paths = sorted(run_root.glob("vlmap_safety_debug/**/s2_action_loop_snapshots/*.jpg"))
    stuck_snapshot_paths = sorted(run_root.glob("vlmap_safety_debug/**/stuck_snapshots/*.jpg"))
    progress_by_episode = {
        (str(row.get("scene_id") or ""), int(row.get("episode_id", -1) or -1)): row
        for row in progress
    }
    success_episode_count = sum(float(row.get("success", 0.0) or 0.0) > 0.5 for row in progress)
    loop_success_episodes = {
        (str(row.get("scene_id") or ""), int(row.get("episode_id", -1) or -1))
        for row in loop_events
        if float((progress_by_episode.get(
            (str(row.get("scene_id") or ""), int(row.get("episode_id", -1) or -1))
        ) or {}).get("success", 0.0) or 0.0) > 0.5
    }
    strict_success_episodes = {
        (str(row.get("scene_id") or ""), int(row.get("episode_id", -1) or -1))
        for row in loop_events
        if str(row.get("triage_tier") or (row.get("triage") or {}).get("tier") or "")
        == "strict_intervention"
        and float((progress_by_episode.get(
            (str(row.get("scene_id") or ""), int(row.get("episode_id", -1) or -1))
        ) or {}).get("success", 0.0) or 0.0) > 0.5
    }
    success_trigger_rate = len(loop_success_episodes) / max(1, success_episode_count)
    strict_success_trigger_rate = len(strict_success_episodes) / max(1, success_episode_count)
    strict_tier_rate = triage_tiers.get("strict_intervention", 0) / max(1, len(loop_events))
    unsupported_selected_count = selected_types.get("frontier", 0) + selected_types.get("open_floor", 0)
    low_progress_margin_count = sum(value < 0.5 for value in progress_margins)
    valid_rate = len(valid) / max(1, len(shadow_events))
    checks = {
        "episode_count": len(progress) == expected_episodes,
        "navigation_result_present": bool(navigation),
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
        "gt_leakage_free": gt_fields_nonempty == 0,
        "strict_intervention_timing_safe": strict_timing_unsafe_count == 0,
        "snapshot_coverage": snapshot_present_count == snapshot_expected_count,
        "minimum_loop_events": len(loop_events) >= int(minimum_loop_events),
        "success_trigger_rate": (
            maximum_success_trigger_rate is None
            or success_trigger_rate <= float(maximum_success_trigger_rate)
        ),
        "strict_tier_rate": (
            maximum_strict_tier_rate is None
            or len(loop_events) < int(strict_rate_minimum_events)
            or strict_tier_rate <= float(maximum_strict_tier_rate)
        ),
    }
    return {
        "task": "stage21c_multitask_scorer_online_shadow_audit",
        "run_root": str(run_root.resolve()),
        "expected_episode_count": expected_episodes,
        "progress_episode_count": len(progress),
        "frozen_s2_navigation_metrics": {
            "episodes": navigation.get("length"),
            "success_rate": navigation.get("sucs_all"),
            "spl": navigation.get("spls_all"),
            "oracle_success": navigation.get("oss_all"),
            "ne": navigation.get("nes_all"),
            "collision_count": navigation.get("collision_count_sum"),
            "collision_episode_rate": navigation.get("collision_episode_rate"),
            "collision_free_rate": navigation.get("collision_free_rate"),
            "collision_free_success_rate": navigation.get("cf_sucs_all"),
            "collision_free_spl": navigation.get("cf_spls_all"),
            "interpretation": "Frozen S2 baseline for these episodes; scorer never changed an action",
        },
        "memory_event_file_count": len(event_paths),
        "candidate_event_count": len(candidate_events),
        "shadow_event_count": len(shadow_events),
        "valid_shadow_event_count": len(valid),
        "valid_shadow_event_rate": valid_rate,
        "error_count": len(errors),
        "action_applied_count": len(applied),
        "nonfinite_count": len(nonfinite),
        "unsafe_progress_selected_count": len(unsafe_progress),
        "unsafe_recovery_selected_count": len(unsafe_recovery),
        "feature_dimensions": dict(dimensions),
        "schema_versions": dict(schemas),
        "progress_change_candidate_score_count": progress_changes,
        "progress_change_candidate_score_rate": progress_changes / max(1, len(valid)),
        "progress_change_intent_count": intent_changes,
        "progress_change_intent_rate": intent_changes / max(1, len(valid)),
        "progress_selected_type_counts": dict(selected_types),
        "progress_selected_direction_counts": dict(selected_directions),
        "candidate_selection_diagnostics": {
            "valid_events": len(valid),
            "scored_candidate_count": scored_candidate_count,
            "scored_candidate_type_counts": dict(candidate_types),
            "progress_margin": _distribution(progress_margins),
            "candidate_score_margin": _distribution(candidate_score_margins),
            "intent_alignment_margin": _distribution(intent_margins),
            "recovery_eligible_candidate_count": recovery_eligible_count,
            "recovery_selected_event_count": recovery_selected_count,
            "recovery_selected_rate_over_valid_events": recovery_selected_count / max(1, len(valid)),
            "recovery_selected_type_counts": dict(recovery_selected_types),
            "under_supported_selected_count": unsupported_selected_count,
            "under_supported_selected_rate": unsupported_selected_count / max(1, len(valid)),
            "under_supported_types": ["frontier", "open_floor"],
            "progress_margin_below_0_5_count": low_progress_margin_count,
            "progress_margin_below_0_5_rate": low_progress_margin_count / max(1, len(progress_margins)),
            "interpretation": "ranking/disagreement diagnostics only; no online outcome label or causal recovery claim",
        },
        "triage_timing_diagnostics": {
            "loop_event_count": len(loop_events),
            "tier_counts": dict(triage_tiers),
            "reason_counts": dict(triage_reasons),
            "evidence_true_counts": dict(triage_evidence),
            "step_id": _distribution(loop_steps),
            "strict_intervention_step_id": _distribution(strict_steps),
            "same_turn_generation_streak": _distribution(loop_streaks),
            "cumulative_turn_actions": _distribution(loop_turns),
            "translation_m": _distribution(loop_translations),
            "same_step_candidate_join": dict(triage_join),
            "strict_timing_unsafe_count": strict_timing_unsafe_count,
            "success_episode_count": success_episode_count,
            "loop_success_episode_count": len(loop_success_episodes),
            "loop_success_episode_rate": success_trigger_rate,
            "strict_success_episode_count": len(strict_success_episodes),
            "strict_success_episode_rate": strict_success_trigger_rate,
            "strict_tier_rate": strict_tier_rate,
            "samples": loop_samples,
            "interpretation": "triage timing/evidence audit; shadow-only and not an intervention success metric",
        },
        "snapshot_diagnostics": {
            "loop_event_count": len(loop_events),
            "snapshot_expected_count": snapshot_expected_count,
            "snapshot_present_count": snapshot_present_count,
            "snapshot_coverage_rate": snapshot_present_count / max(1, snapshot_expected_count),
            "loop_snapshot_jpg_count": len(loop_snapshot_paths),
            "stuck_snapshot_jpg_count": len(stuck_snapshot_paths),
        },
        "gt_leakage_audit": {
            "loop_events_with_nonempty_gt_fields_used": gt_fields_nonempty,
            "passed": gt_fields_nonempty == 0,
        },
        "missing_numeric": _distribution(missing),
        "latency_ms": {**_distribution(latency), "p95": p95},
        "score_distributions": {name: _distribution(values) for name, values in scores.items()},
        "thresholds": {
            "minimum_valid_rate": minimum_valid_rate,
            "maximum_p95_latency_ms": maximum_p95_latency_ms,
            "minimum_loop_events": minimum_loop_events,
            "maximum_success_trigger_rate": maximum_success_trigger_rate,
            "maximum_strict_tier_rate": maximum_strict_tier_rate,
            "strict_rate_minimum_events": strict_rate_minimum_events,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "scope": {"shadow_only": True, "frozen_s2_nextdit": True, "active_recovery": False},
        "sample_errors": [
            {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "shadow": row.get("stage21_multitask_shadow"),
            }
            for row in errors[:10]
        ],
    }


def render_summary(result: Dict[str, Any]) -> str:
    navigation = result["frozen_s2_navigation_metrics"]
    candidate = result["candidate_selection_diagnostics"]
    triage = result["triage_timing_diagnostics"]
    snapshots = result["snapshot_diagnostics"]
    latency = result["latency_ms"]
    missing = result["missing_numeric"]
    return "\n".join([
        "# Stage21c frozen scorer shadow key metrics",
        "",
        "## Scope",
        "",
        "- Frozen S2/NextDiT: true",
        "- Active recovery: false",
        "- Action applied by scorer: 0",
        "- Episode-time parameter update: false",
        "",
        "## Runtime integrity",
        "",
        f"- Episodes: {result['progress_episode_count']} / {result['expected_episode_count']}",
        f"- Valid scorer events: {result['valid_shadow_event_count']} / {result['shadow_event_count']} "
        f"({100.0 * result['valid_shadow_event_rate']:.2f}%)",
        f"- Errors / nonfinite / unsafe progress / unsafe recovery: "
        f"{result['error_count']} / {result['nonfinite_count']} / "
        f"{result['unsafe_progress_selected_count']} / {result['unsafe_recovery_selected_count']}",
        f"- Latency mean / p95: {_fmt(latency['mean'])} / {_fmt(latency['p95'])} ms",
        f"- Missing numeric mean: {_fmt(missing['mean'])} / 64",
        "",
        "## Frozen S2 navigation baseline",
        "",
        f"- SR / SPL / NE / oracle success: {_fmt(navigation['success_rate'], 4)} / "
        f"{_fmt(navigation['spl'], 4)} / {_fmt(navigation['ne'], 4)} / "
        f"{_fmt(navigation['oracle_success'], 4)}",
        f"- Collision count / episode rate: {_fmt(navigation['collision_count'], 0)} / "
        f"{_fmt(100.0 * float(navigation['collision_episode_rate']), 2) if navigation['collision_episode_rate'] is not None else 'n/a'}%",
        f"- Collision-free SR / SPL: {_fmt(navigation['collision_free_success_rate'], 4)} / "
        f"{_fmt(navigation['collision_free_spl'], 4)}",
        "- These are not scorer gains because action_applied remained zero.",
        "",
        "## Candidate selection diagnostics",
        "",
        f"- Candidate events / scored candidates: {candidate['valid_events']} / {candidate['scored_candidate_count']}",
        f"- Learned progress changes candidate-score choice: "
        f"{result['progress_change_candidate_score_count']} "
        f"({100.0 * result['progress_change_candidate_score_rate']:.2f}%)",
        f"- Learned progress changes intent-alignment choice: "
        f"{result['progress_change_intent_count']} ({100.0 * result['progress_change_intent_rate']:.2f}%)",
        f"- Learned progress selected types: {json.dumps(result['progress_selected_type_counts'], ensure_ascii=False)}",
        f"- Learned progress selected directions: {json.dumps(result['progress_selected_direction_counts'], ensure_ascii=False)}",
        f"- Progress top1-top2 margin mean: {candidate['progress_margin']['mean']}",
        f"- Low-margin choices (<0.5): {candidate['progress_margin_below_0_5_count']} "
        f"({100.0 * candidate['progress_margin_below_0_5_rate']:.2f}%)",
        f"- Under-supported frontier/open_floor selections: {candidate['under_supported_selected_count']} "
        f"({100.0 * candidate['under_supported_selected_rate']:.2f}%)",
        f"- Recovery eligible candidates / selected events: "
        f"{candidate['recovery_eligible_candidate_count']} / {candidate['recovery_selected_event_count']}",
        "- Meaning: these are ranking, coverage and disagreement diagnostics; shadow has no causal outcome label.",
        "",
        "## Intervention timing diagnostics",
        "",
        f"- S2 loop events: {triage['loop_event_count']}",
        f"- Triage tiers: {json.dumps(triage['tier_counts'], ensure_ascii=False)}",
        f"- Triage reasons: {json.dumps(triage['reason_counts'], ensure_ascii=False)}",
        f"- Strict events with unsafe timing/evidence: {triage['strict_timing_unsafe_count']}",
        f"- Successful episodes with any loop trigger: {triage['loop_success_episode_count']} / "
        f"{triage['success_episode_count']} ({100.0 * triage['loop_success_episode_rate']:.2f}%)",
        f"- Successful episodes with strict trigger: {triage['strict_success_episode_count']} / "
        f"{triage['success_episode_count']} ({100.0 * triage['strict_success_episode_rate']:.2f}%)",
        f"- Strict tier rate over loop events: {100.0 * triage['strict_tier_rate']:.2f}%",
        f"- Same-step scorer/candidate join: {json.dumps(triage['same_step_candidate_join'], ensure_ascii=False)}",
        "- Meaning: timing is judged by conservative online evidence only; it is not intervention success.",
        "",
        "## Visual evidence",
        "",
        f"- Expected/present loop snapshots: {snapshots['snapshot_expected_count']} / "
        f"{snapshots['snapshot_present_count']}",
        f"- Loop snapshot JPGs: {snapshots['loop_snapshot_jpg_count']}",
        f"- Stuck snapshot JPGs: {snapshots['stuck_snapshot_jpg_count']}",
        "",
        "## Gate decision",
        "",
        f"- Audit passed: {str(bool(result['passed'])).lower()}",
        "- Passing this shadow audit permits only a tiny strict-intervention active smoke; it does not prove SR/SPL gain.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-valid-rate", type=float, default=0.95)
    parser.add_argument("--max-p95-latency-ms", type=float, default=250.0)
    parser.add_argument("--min-loop-events", type=int, default=0)
    parser.add_argument("--max-success-trigger-rate", type=float)
    parser.add_argument("--max-strict-tier-rate", type=float)
    parser.add_argument("--strict-rate-min-events", type=int, default=4)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    result = build_audit(
        args.run_root,
        args.expected_episodes,
        args.min_valid_rate,
        args.max_p95_latency_ms,
        args.min_loop_events,
        args.max_success_trigger_rate,
        args.max_strict_tier_rate,
        args.strict_rate_min_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(render_summary(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.require_all and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Stage46 helpers for binding frozen M3 candidates to the active executor."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple


SAFE_STAGE = "route_occ_clearance_frontier"
ALLOWED_CANDIDATE_STAGES = {
    SAFE_STAGE,
    "route_occ",
    "route_only",
}
SCHEMA_VERSION = "stage46_m3_one_primitive_active_v1"


def _number(candidate: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = candidate.get(name, default)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _composite_key(candidate: Mapping[str, Any]) -> Tuple[float, ...]:
    """Frozen Stage40 composite ordering; lower keys rank first."""
    return (
        -_number(candidate, "route_support_edge_count", 0.0),
        -_number(candidate, "local_free_fraction", 0.0),
        _number(candidate, "path_length_m", 10**6),
        -_number(candidate, "source_step", -10**6),
    )


def _safe(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("floor_aligned_known_free")
        and _number(candidate, "unknown_fraction", 1.0) == 0.0
        and _number(candidate, "occupied_fraction", 1.0) == 0.0
        and not candidate.get("route_occ_conflict")
        and not candidate.get("gt_fields_used")
        and not candidate.get("action_applied")
        and candidate.get("shadow_only")
    )


def _frozen_record(candidate: Mapping[str, Any]) -> bool:
    return bool(
        not candidate.get("gt_fields_used")
        and not candidate.get("action_applied")
        and candidate.get("shadow_only")
    )


def _eligible(candidate: Mapping[str, Any], stage: str, safety_mode: str) -> bool:
    if safety_mode == "strict":
        return _safe(candidate)
    if safety_mode == "route_occ_turn_only":
        return bool(
            stage == "route_occ"
            and _frozen_record(candidate)
            and not candidate.get("route_occ_conflict")
            and _number(candidate, "unknown_fraction", 1.0) == 0.0
            and _number(candidate, "occupied_fraction", 1.0) == 0.0
        )
    if safety_mode == "route_only_turn_only":
        return bool(stage == "route_only" and _frozen_record(candidate))
    return False


def select_frozen_m3_candidate(
    event: Mapping[str, Any],
    *,
    candidate_stage: str = SAFE_STAGE,
    safety_mode: str = "strict",
) -> Tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Select one already-safe M3 candidate without changing its pool."""
    candidate_stage = str(candidate_stage or SAFE_STAGE)
    if candidate_stage not in ALLOWED_CANDIDATE_STAGES:
        candidate_stage = SAFE_STAGE
        safety_mode = "strict"
    pool = list(
        (event.get("ablation") or {})
        .get(candidate_stage, {})
        .get("candidates", [])
        or []
    )
    safe = [
        dict(candidate)
        for candidate in pool
        if _eligible(candidate, candidate_stage, safety_mode)
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_stage": candidate_stage,
        "safety_mode": safety_mode,
        "ranking_rule": "stage40_composite_fixed_heuristic",
        "pool_count": int(len(pool)),
        "safe_count": int(len(safe)),
        "unsafe_record_count": int(len(pool) - len(safe)),
        "selected_candidate_id": None,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "ranker_trained": False,
        "turn_only_relaxation": bool(safety_mode != "strict"),
        "translation_allowed": bool(safety_mode == "strict"),
    }
    if not safe:
        report["reason"] = "zero_safe_m3_candidate"
        return None, report
    selected = min(safe, key=_composite_key)
    report.update(
        {
            "reason": "selected",
            "selected_candidate_id": selected.get("candidate_id"),
        }
    )
    return selected, report


def bind_candidate_to_loop_event(
    loop_event: Mapping[str, Any],
    stage27_event: Mapping[str, Any],
    *,
    candidate_stage: str = SAFE_STAGE,
    safety_mode: str = "strict",
) -> dict[str, Any]:
    """Return an executor-facing copy; never mutates the frozen M3 record."""
    result = dict(loop_event)
    candidate, selection = select_frozen_m3_candidate(
        stage27_event,
        candidate_stage=candidate_stage,
        safety_mode=safety_mode,
    )
    result["stage46_candidate_selection"] = selection
    result["candidate_source"] = "stage27_frozen_m3"
    result["event_schema_version"] = SCHEMA_VERSION
    if candidate is None:
        result["candidate"] = None
        result["triage_tier"] = "hold"
        result["triage_reason"] = "zero_safe_m3_candidate"
        return result

    executor_candidate = dict(candidate)
    executor_candidate.update(
        {
            "geometry_safe": True,
            "active_gate_safe": True,
            "direction_bucket": "path",
            "stage46_safety_derivation": selection["candidate_stage"],
            "stage54_safety_mode": selection["safety_mode"],
            "stage54_turn_only_relaxation": bool(selection["turn_only_relaxation"]),
            "stage54_translation_allowed": bool(selection["translation_allowed"]),
        }
    )
    result.update(
        {
            "candidate": executor_candidate,
            "triage_tier": "strict_intervention",
            "triage_reason": "frozen_d0_plus_safe_m3_candidate",
        }
    )
    return result


def active_path_within_bound(path_m: Any, max_active_path_m: float) -> bool:
    """A zero bound disables the post-selection local-distance gate."""
    try:
        bound = float(max_active_path_m)
        distance = float(path_m)
    except (TypeError, ValueError):
        return True
    return bool(bound <= 0.0 or distance <= bound + 1e-9)


def iterative_reorientation_decision(
    previous_bearing_deg: Any,
    current_bearing_deg: Any,
    *,
    primitive_count: int,
    max_primitives: int,
    deadband_deg: float,
) -> dict[str, Any]:
    """Decide whether one more turn is justified after a fresh path re-audit."""
    result = {
        "continue_reorientation": False,
        "turn_direction": None,
        "reason": None,
        "previous_abs_bearing_deg": None,
        "current_abs_bearing_deg": None,
    }
    try:
        previous = float(previous_bearing_deg)
        current = float(current_bearing_deg)
    except (TypeError, ValueError):
        result["reason"] = "missing_reaudited_bearing"
        return result
    if not math.isfinite(previous) or not math.isfinite(current):
        result["reason"] = "nonfinite_reaudited_bearing"
        return result
    previous_abs = abs(previous)
    current_abs = abs(current)
    result.update(
        {
            "previous_abs_bearing_deg": previous_abs,
            "current_abs_bearing_deg": current_abs,
        }
    )
    if int(primitive_count) >= max(0, int(max_primitives)):
        result["reason"] = "iterative_reorient_budget_exhausted"
        return result
    if current_abs <= max(0.0, float(deadband_deg)):
        result["reason"] = "path_aligned_no_visible_proxy"
        return result
    if current_abs >= previous_abs - 1e-6:
        result["reason"] = "iterative_reorient_not_converging"
        return result
    result.update(
        {
            "continue_reorientation": True,
            "turn_direction": "left" if current > 0.0 else "right",
            "reason": "iterative_reorient_queued",
        }
    )
    return result

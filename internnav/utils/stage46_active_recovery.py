"""Stage46 helpers for binding frozen M3 candidates to the active executor."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple


SAFE_STAGE = "route_occ_clearance_frontier"
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


def select_frozen_m3_candidate(
    event: Mapping[str, Any],
) -> Tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Select one already-safe M3 candidate without changing its pool."""
    pool = list(
        (event.get("ablation") or {})
        .get(SAFE_STAGE, {})
        .get("candidates", [])
        or []
    )
    safe = [dict(candidate) for candidate in pool if _safe(candidate)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_stage": SAFE_STAGE,
        "ranking_rule": "stage40_composite_fixed_heuristic",
        "pool_count": int(len(pool)),
        "safe_count": int(len(safe)),
        "unsafe_record_count": int(len(pool) - len(safe)),
        "selected_candidate_id": None,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "ranker_trained": False,
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
    loop_event: Mapping[str, Any], stage27_event: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an executor-facing copy; never mutates the frozen M3 record."""
    result = dict(loop_event)
    candidate, selection = select_frozen_m3_candidate(stage27_event)
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
            "stage46_safety_derivation": SAFE_STAGE,
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

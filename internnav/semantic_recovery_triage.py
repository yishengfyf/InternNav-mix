"""Pure multi-evidence triage for conservative semantic recovery decisions.

The evaluator and offline analysis both call this module so a shadow-log replay
uses exactly the same decision rule as an online Stage20g-v2 run.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional


DEFAULT_SEMANTIC_RECOVERY_TRIAGE_CONFIG = {
    "v2_evidence_gate_enable": True,
    "v2_evidence_gate_min_open_score": 0.70,
    "v2_evidence_gate_min_doorway_score": 0.60,
    "v2_evidence_gate_min_target_frontier_score": 0.10,
    "v2_evidence_gate_min_step_gap": 20,
    "v2_evidence_gate_min_nearby_visits": 3,
    "max_completed_landmark_penalty": 0.0,
    "min_step": 35,
    "min_backtrack_m": 1.0,
    "max_backtrack_m": 4.0,
    "max_step_gap": 120,
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _string_set(values: Optional[Iterable[Any]]) -> set[str]:
    return {str(item) for item in (values or []) if str(item)}


def classify_semantic_recovery_triage(
    candidate: Optional[Mapping[str, Any]],
    cfg: Optional[Mapping[str, Any]],
    *,
    failure_type: str,
    recommended_primitive: str,
    trigger_reasons: Optional[Iterable[Any]] = None,
    context_tags: Optional[Iterable[Any]] = None,
    step_id: Optional[int] = None,
) -> dict[str, Any]:
    """Classify one recovery candidate into strict/adapter/abstain.

    The rule is intentionally structured around independent evidence groups:
    policy-memory conflict, hazard context, failure persistence, and escape
    anchor quality. Only agreement across every group reaches the strict tier.
    """

    config = dict(DEFAULT_SEMANTIC_RECOVERY_TRIAGE_CONFIG)
    config.update(dict(cfg or {}))
    reasons = _string_set(trigger_reasons)
    tags = _string_set(context_tags)
    candidate_dict = dict(candidate or {})

    if not bool(config.get("v2_evidence_gate_enable", False)):
        return {"enabled": False, "tier": "disabled", "reason": "disabled"}
    if not candidate_dict:
        return {
            "enabled": True,
            "tier": "abstain",
            "reason": "missing_candidate",
            "hard_abstain_reasons": ["missing_candidate"],
            "evidence_vote_count": 0,
        }

    geometry_safe = bool(candidate_dict.get("geometry_safe"))
    active_gate_safe = bool(candidate_dict.get("active_gate_safe"))
    target_frontier_intent_safe = bool(candidate_dict.get("target_frontier_intent_safe"))
    target_frontier_escape = bool(candidate_dict.get("target_frontier_escape_candidate"))
    open_score = _finite_float(candidate_dict.get("semantic_resilience_open_score"))
    target_frontier_score = _finite_float(candidate_dict.get("target_frontier_score"))
    doorway_score = _finite_float(candidate_dict.get("target_frontier_doorway_like_score"))
    completed_penalty = _finite_float(candidate_dict.get("completed_landmark_penalty"))
    step_gap_value = _optional_int(candidate_dict.get("semantic_resilience_step_gap"))
    backtrack_distance = _finite_float(
        candidate_dict.get("semantic_resilience_backtrack_distance_m", candidate_dict.get("distance_m")),
        default=-1.0,
    )
    step_id_value = _optional_int(step_id)
    nearby_visits = _optional_int(candidate_dict.get("nearby_visit_count")) or 0
    direction_bucket = str(candidate_dict.get("direction_bucket") or "unknown")

    s2_policy_conflict = bool(
        reasons.intersection({"current_waypoint_occupied", "current_waypoint_not_active_safe"})
        or "policy_memory_conflict" in tags
    )
    obstacle_context = bool(
        "semantic_obstacle_near_trap" in reasons
        or "semantic_obstacle_context" in tags
        or (_optional_int(candidate_dict.get("semantic_resilience_obstacle_term_count")) or 0) > 0
    )
    spatial_constriction = bool(
        "local_trap" in reasons
        or "spatial_constriction" in tags
    )
    semantic_only = bool(
        reasons.intersection({"semantic_dead_zone", "semantic_stagnation"})
        and not spatial_constriction
        and not obstacle_context
        and not s2_policy_conflict
    )
    persistence = bool(
        nearby_visits >= int(config["v2_evidence_gate_min_nearby_visits"])
        or (
            step_gap_value is not None
            and step_gap_value >= int(config["v2_evidence_gate_min_step_gap"])
        )
    )
    frontier_like_anchor = bool(
        target_frontier_intent_safe
        or target_frontier_escape
        or target_frontier_score >= float(config["v2_evidence_gate_min_target_frontier_score"])
        or doorway_score >= float(config["v2_evidence_gate_min_doorway_score"])
    )
    escape_anchor_safe = bool(
        geometry_safe
        and active_gate_safe
        and completed_penalty <= float(config["max_completed_landmark_penalty"])
        and open_score >= float(config["v2_evidence_gate_min_open_score"])
        and frontier_like_anchor
    )
    back_only_without_anchor = bool(direction_bucket == "back" and not frontier_like_anchor)
    intervention_time_safe = bool(
        step_id_value is None or step_id_value >= int(config["min_step"])
    )
    backtrack_distance_safe = bool(
        float(config["min_backtrack_m"])
        <= backtrack_distance
        <= float(config["max_backtrack_m"])
    )
    anchor_fresh = bool(
        step_gap_value is None or step_gap_value <= int(config["max_step_gap"])
    )
    execution_window_safe = bool(
        intervention_time_safe and backtrack_distance_safe and anchor_fresh
    )

    evidence = {
        "enabled": True,
        "failure_type_allowed": str(failure_type) == "stuck_collision",
        "primitive_allowed": str(recommended_primitive)
        in {"reorient_reobserve", "one_safe_forward_reobserve"},
        "s2_policy_conflict": s2_policy_conflict,
        "obstacle_context": obstacle_context,
        "spatial_constriction": spatial_constriction,
        "semantic_only": semantic_only,
        "persistence": persistence,
        "geometry_safe": geometry_safe,
        "active_gate_safe": active_gate_safe,
        "frontier_like_anchor": frontier_like_anchor,
        "escape_anchor_safe": escape_anchor_safe,
        "intervention_time_safe": intervention_time_safe,
        "backtrack_distance_safe": backtrack_distance_safe,
        "anchor_fresh": anchor_fresh,
        "execution_window_safe": execution_window_safe,
        "completed_landmark_penalty": completed_penalty,
        "back_only_without_anchor": back_only_without_anchor,
        "open_score": open_score,
        "target_frontier_score": target_frontier_score,
        "target_frontier_doorway_like_score": doorway_score,
        "semantic_resilience_step_gap": step_gap_value,
        "semantic_resilience_backtrack_distance_m": backtrack_distance,
        "step_id": step_id_value,
        "nearby_visit_count": nearby_visits,
    }

    hard_abstain_reasons = []
    if not evidence["failure_type_allowed"]:
        hard_abstain_reasons.append("failure_type_not_stuck_collision")
    if not evidence["primitive_allowed"]:
        hard_abstain_reasons.append("primitive_not_recovery")
    if not geometry_safe:
        hard_abstain_reasons.append("candidate_not_geometry_safe")
    if completed_penalty > float(config["max_completed_landmark_penalty"]):
        hard_abstain_reasons.append("completed_landmark_penalty")
    if semantic_only:
        hard_abstain_reasons.append("semantic_only_no_spatial_conflict")
    if back_only_without_anchor:
        hard_abstain_reasons.append("back_only_without_anchor")
    if not intervention_time_safe:
        hard_abstain_reasons.append("too_early")
    if not backtrack_distance_safe:
        hard_abstain_reasons.append("backtrack_distance_out_of_range")
    if not anchor_fresh:
        hard_abstain_reasons.append("stale_backtrack_anchor")

    strict_intervention = bool(
        not hard_abstain_reasons
        and s2_policy_conflict
        and obstacle_context
        and spatial_constriction
        and persistence
        and escape_anchor_safe
        and execution_window_safe
    )
    adapter_candidate = bool(
        not strict_intervention
        and not hard_abstain_reasons
        and s2_policy_conflict
        and obstacle_context
        and geometry_safe
        and open_score >= float(config["v2_evidence_gate_min_open_score"])
        and (active_gate_safe or frontier_like_anchor)
    )

    if strict_intervention:
        tier = "strict_intervention"
        reason = "multi_evidence_consistent"
    elif adapter_candidate:
        tier = "adapter_candidate"
        reason = "needs_ranker_or_more_context"
    else:
        tier = "abstain"
        reason = ",".join(hard_abstain_reasons) if hard_abstain_reasons else "insufficient_evidence"

    evidence["tier"] = tier
    evidence["reason"] = reason
    evidence["hard_abstain_reasons"] = hard_abstain_reasons
    evidence["evidence_vote_count"] = sum(
        bool(evidence[name])
        for name in (
            "s2_policy_conflict",
            "obstacle_context",
            "spatial_constriction",
            "persistence",
            "escape_anchor_safe",
            "execution_window_safe",
        )
    )
    return evidence

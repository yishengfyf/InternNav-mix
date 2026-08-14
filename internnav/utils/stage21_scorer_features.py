"""Torch-free Stage21 structured feature schema shared by training and runtime."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Tuple


NUMERIC_FIELDS: Tuple[str, ...] = (
    "distance_m", "frontier_distance_m", "topology_novelty_score", "nearby_visit_count", "revisit_risk",
    "direction_angle_deg", "angle_to_current_waypoint_deg", "intent_alignment_score",
    "distance_to_current_waypoint_m", "semantic_relevance_score", "semantic_novelty_score",
    "semantic_confidence_score", "semantic_bind_score", "next_landmark_relevance", "completed_landmark_penalty",
    "repeated_semantic_penalty", "target_frontier_cluster_count", "target_frontier_doorway_like_score",
    "target_frontier_corridor_continuation_score", "target_frontier_intent_deviation_penalty",
    "target_frontier_local_free_count", "target_frontier_local_occupied_count", "target_frontier_local_unknown_count",
    "target_frontier_transition_prior", "anchor_visible_free_ratio", "anchor_occupied_ratio_observed",
    "anchor_frontier_count", "anchor_branch_count", "anchor_executable_exit_count",
    "anchor_connected_component_count", "anchor_branch_depth_mean", "anchor_direction_entropy",
    "anchor_semantic_unique_count", "anchor_instruction_relevant_count", "anchor_high_conf_landmark_count",
    "anchor_next_landmark_count", "anchor_passage_semantic_count", "anchor_outgoing_trace_direction_count",
    "anchor_last_visit_step", "anchor_last_visit_age_steps", "anchor_recent_return_count",
    "anchor_recent_cycle_count", "anchor_revisit_interval_min_steps", "anchor_revisit_interval_mean_steps",
    "anchor_short_cycle_risk", "current_visible_free_ratio", "current_frontier_count", "current_branch_count",
    "current_executable_exit_count", "current_connected_component_count", "current_branch_depth_mean",
    "current_direction_entropy", "current_to_anchor_free_ratio_gain", "current_to_anchor_frontier_gain",
    "current_to_anchor_branch_gain", "current_to_anchor_direction_entropy_gain", "anchor_semantic_top_score",
    "semantic_resilience_backtrack_distance_m", "semantic_resilience_local_trap",
    "semantic_resilience_nearest_obstacle_distance_m", "semantic_resilience_nearest_passage_distance_m",
    "semantic_resilience_obstacle_term_count", "semantic_resilience_passage_term_count",
    "semantic_resilience_step_gap", "semantic_resilience_source_step_id",
)

BOOLEAN_FIELDS: Tuple[str, ...] = (
    "aligned_with_current_waypoint", "semanticized_candidate", "instruction_relevant",
    "points_to_revisited_region", "target_frontier_candidate", "target_frontier_escape_candidate",
    "target_frontier_intent_safe", "goal_progress_enabled", "target_frontier_enabled",
    "semantic_resilience_candidate", "semantic_resilience_local_trap", "semantic_resilience_recovery_trigger",
    "anchor_source_is_keyframe", "anchor_high_conf_semantic",
)

CANDIDATE_TYPES: Tuple[str, ...] = (
    "semantic_frontier", "semantic_keyframe", "open_floor", "resilience_backtrack",
    "backtrack_reobserve", "frontier", "unknown",
)
DIRECTION_BUCKETS: Tuple[str, ...] = ("front", "left", "right", "back", "unknown")


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _present_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _candidate_and_current(row: Mapping[str, Any]):
    inputs = row.get("online_inputs") or {}
    return inputs.get("candidate") or {}, inputs.get("current_policy_candidate") or {}


def feature_names() -> List[str]:
    names: List[str] = []
    for field in NUMERIC_FIELDS:
        names.extend((f"candidate::{field}", f"current_s2::{field}", f"delta::{field}"))
        names.extend((f"candidate_present::{field}", f"current_s2_present::{field}"))
    names.extend(f"candidate_bool::{field}" for field in BOOLEAN_FIELDS)
    names.extend(f"candidate_type={value}" for value in CANDIDATE_TYPES)
    names.extend(f"direction_bucket={value}" for value in DIRECTION_BUCKETS)
    names.extend(("direction_sin", "direction_cos"))
    return names


def encode_row(row: Mapping[str, Any]) -> List[float]:
    candidate, current = _candidate_and_current(row)
    values: List[float] = []
    for field in NUMERIC_FIELDS:
        c_present = _present_float(candidate.get(field))
        s_present = _present_float(current.get(field))
        c_value = _safe_float(candidate.get(field))
        s_value = _safe_float(current.get(field))
        values.extend((c_value, s_value, c_value - s_value, float(c_present), float(s_present)))
    values.extend(float(bool(candidate.get(field))) for field in BOOLEAN_FIELDS)
    candidate_type = str(candidate.get("candidate_type") or candidate.get("source") or "unknown")
    values.extend(float(candidate_type == value) for value in CANDIDATE_TYPES)
    direction = str(candidate.get("direction_bucket") or "unknown")
    values.extend(float(direction == value) for value in DIRECTION_BUCKETS)
    angle = math.radians(_safe_float(candidate.get("direction_angle_deg")))
    values.extend((math.sin(angle), math.cos(angle)))
    if len(values) != len(feature_names()):
        raise AssertionError("Stage21 feature schema dimension mismatch")
    return values

"""Dependency-light shared code for the offline Stage17 progress ranker.

This module intentionally lives beside the training tools instead of under the
``internnav`` package: importing the full navigation package loads simulator
configuration dependencies that the small offline data audit does not need.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


CANDIDATE_TYPES = ("frontier", "semantic_frontier", "semantic_keyframe", "open_floor")
DIRECTION_BUCKETS = ("front", "left", "right", "back")

_NUMERIC_FIELDS = (
    "distance_m",
    "frontier_distance_m",
    "frontier_progress_score",
    "topology_novelty_score",
    "nearby_visit_count",
    "revisit_risk",
    "angle_to_current_waypoint_deg",
    "intent_alignment_score",
    "distance_to_current_waypoint_m",
    "semantic_relevance_score",
    "semantic_novelty_score",
    "semantic_confidence_score",
    "semantic_bind_score",
    "next_landmark_relevance",
    "completed_landmark_penalty",
    "repeated_semantic_penalty",
    "semantic_progress_score",
    "unknown_target_frontier_bonus",
    "goal_progress_score",
    "target_frontier_score",
    "target_frontier_doorway_like_score",
    "target_frontier_corridor_continuation_score",
    "target_frontier_intent_deviation_penalty",
    "score",
)

_BOOLEAN_FIELDS = (
    "geometry_safe",
    "active_gate_safe",
    "aligned_with_current_waypoint",
    "semanticized_candidate",
    "instruction_relevant",
    "points_to_revisited_region",
    "target_frontier_candidate",
    "target_frontier_escape_candidate",
    "target_frontier_intent_safe",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def feature_names() -> List[str]:
    """Return stable online-only feature names in encoding order."""

    names = list(_NUMERIC_FIELDS)
    names.extend(_BOOLEAN_FIELDS)
    names.extend(f"candidate_type={name}" for name in CANDIDATE_TYPES)
    names.extend(f"direction_bucket={name}" for name in DIRECTION_BUCKETS)
    names.extend(("direction_sin", "direction_cos", "is_completed_landmark"))
    return names


def encode_candidate(candidate: Dict[str, Any]) -> List[float]:
    """Encode one candidate without consuming route labels or other GT fields."""

    values = [_safe_float(candidate.get(name)) for name in _NUMERIC_FIELDS]
    values.extend(float(bool(candidate.get(name))) for name in _BOOLEAN_FIELDS)
    candidate_type = str(candidate.get("candidate_type") or "")
    values.extend(float(candidate_type == name) for name in CANDIDATE_TYPES)
    direction = str(candidate.get("direction_bucket") or "")
    values.extend(float(direction == name) for name in DIRECTION_BUCKETS)
    angle_rad = math.radians(_safe_float(candidate.get("direction_angle_deg")))
    values.extend((math.sin(angle_rad), math.cos(angle_rad)))
    values.append(float(str(candidate.get("landmark_status") or "").lower() == "completed"))
    return values

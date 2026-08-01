"""Runtime-only Stage17 progress-ranker shadow scoring.

The training scripts live under ``scripts/train`` to avoid importing Habitat
during cheap offline experiments.  This module mirrors the small feature schema
and MLP in the package runtime so Habitat eval can load a trained checkpoint
without depending on script-relative imports.

The scorer is shadow-only: it returns hypothetical selections and never changes
navigation actions.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


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


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def feature_names() -> List[str]:
    names = list(_NUMERIC_FIELDS)
    names.extend(_BOOLEAN_FIELDS)
    names.extend(f"candidate_type={name}" for name in CANDIDATE_TYPES)
    names.extend(f"direction_bucket={name}" for name in DIRECTION_BUCKETS)
    names.extend(("direction_sin", "direction_cos", "is_completed_landmark"))
    return names


def encode_candidate(candidate: Mapping[str, Any]) -> List[float]:
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


class _ProgressCandidateRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.scorer(candidate_features).squeeze(-1)


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper - lower <= 1e-8:
        return [0.5 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def _resilience_prior(candidate: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    geometry_safe = float(bool(candidate.get("geometry_safe")))
    active_gate_safe = float(bool(candidate.get("active_gate_safe")))
    escape = float(bool(candidate.get("target_frontier_escape_candidate")))
    target_frontier = float(bool(candidate.get("target_frontier_candidate")))
    not_revisited = 1.0 - float(bool(candidate.get("points_to_revisited_region")))
    not_completed = 1.0 - float(str(candidate.get("landmark_status") or "").lower() == "completed")
    novelty = (
        _clamp01(candidate.get("topology_novelty_score"))
        + _clamp01(candidate.get("semantic_novelty_score"))
        + _clamp01(candidate.get("unknown_target_frontier_bonus"))
    ) / 3.0
    low_revisit_risk = 1.0 - _clamp01(candidate.get("revisit_risk"))
    future_observability = 0.60 * escape + 0.25 * target_frontier + 0.15 * novelty
    recoverability = (
        0.25 * geometry_safe
        + 0.20 * active_gate_safe
        + 0.20 * not_revisited
        + 0.15 * not_completed
        + 0.20 * low_revisit_risk
    )
    prior = 0.55 * future_observability + 0.45 * recoverability
    return prior, {
        "future_observability_proxy": float(future_observability),
        "recoverability_proxy": float(recoverability),
        "geometry_safe": float(geometry_safe),
        "active_gate_safe": float(active_gate_safe),
        "target_frontier_escape": float(escape),
        "target_frontier": float(target_frontier),
        "points_to_revisited_region": float(1.0 - not_revisited),
        "completed_landmark": float(1.0 - not_completed),
    }


def _select_by_score(scores: Sequence[float]) -> Optional[int]:
    if not scores:
        return None
    return max(range(len(scores)), key=lambda index: float(scores[index]))


def _candidate_summary(candidate: Optional[Mapping[str, Any]], index: Optional[int]) -> Dict[str, Any]:
    if candidate is None or index is None:
        return {"index": None, "candidate_id": None, "valid": False}
    prior, proxy = _resilience_prior(candidate)
    return {
        "index": int(index),
        "candidate_id": candidate.get("candidate_id"),
        "valid": True,
        "score": _safe_float(candidate.get("score")),
        "target_frontier_score": _safe_float(candidate.get("target_frontier_score")),
        "resilience_prior": float(prior),
        "future_observability_proxy": proxy["future_observability_proxy"],
        "recoverability_proxy": proxy["recoverability_proxy"],
        "geometry_safe": bool(candidate.get("geometry_safe")),
        "active_gate_safe": bool(candidate.get("active_gate_safe")),
        "target_frontier_candidate": bool(candidate.get("target_frontier_candidate")),
        "target_frontier_escape_candidate": bool(candidate.get("target_frontier_escape_candidate")),
        "points_to_revisited_region": bool(candidate.get("points_to_revisited_region")),
        "landmark_status": candidate.get("landmark_status"),
        "completed_landmark": bool(proxy["completed_landmark"] > 0.5),
        "repeated_semantic": _safe_float(candidate.get("repeated_semantic_penalty")) > 0.0,
        "candidate_type": candidate.get("candidate_type"),
        "direction_bucket": candidate.get("direction_bucket"),
        "xy": candidate.get("xy"),
        "grid": candidate.get("grid"),
    }


class ProgressRankerShadowScorer:
    """Load a Stage17 ranker checkpoint and score OccMem candidates online."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "cpu",
        resilience_weight: float = 0.20,
    ):
        self.checkpoint_path = os.path.expanduser(str(checkpoint_path or ""))
        self.device_name = str(device or "cpu")
        self.resilience_weight = max(0.0, float(resilience_weight))
        if not self.checkpoint_path:
            raise ValueError("progress ranker checkpoint path is empty")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"progress ranker checkpoint not found: {self.checkpoint_path}")
        device_obj = torch.device(self.device_name)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for progress ranker shadow but unavailable")
        checkpoint = torch.load(self.checkpoint_path, map_location=device_obj)
        feature_dim = int(checkpoint["feature_dim"])
        expected_dim = len(feature_names())
        if feature_dim != expected_dim:
            raise ValueError(
                f"checkpoint feature_dim={feature_dim} does not match runtime schema={expected_dim}"
            )
        self.model = _ProgressCandidateRanker(
            feature_dim,
            int(checkpoint.get("hidden_dim", 128)),
            float(checkpoint.get("dropout", 0.10)),
        ).to(device_obj)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.device = device_obj
        self.checkpoint_metrics = checkpoint.get("metrics")

    @torch.no_grad()
    def score_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        items = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        if not items:
            return {"enabled": True, "valid": False, "reason": "no_candidates"}
        features = [encode_candidate(candidate) for candidate in items]
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        ranker_scores = [float(value) for value in self.model(tensor).detach().cpu().tolist()]
        normalized_ranker_scores = _minmax(ranker_scores)
        resilience_priors = []
        future_observability = []
        recoverability = []
        for candidate in items:
            prior, proxy = _resilience_prior(candidate)
            resilience_priors.append(float(prior))
            future_observability.append(float(proxy["future_observability_proxy"]))
            recoverability.append(float(proxy["recoverability_proxy"]))
        ranker_resilience_scores = [
            float(base + self.resilience_weight * prior)
            for base, prior in zip(normalized_ranker_scores, resilience_priors)
        ]
        candidate_scores = [_safe_float(candidate.get("score")) for candidate in items]
        target_frontier_scores = [_safe_float(candidate.get("target_frontier_score")) for candidate in items]

        ranker_idx = _select_by_score(ranker_scores)
        resilience_idx = _select_by_score(ranker_resilience_scores)
        candidate_score_idx = _select_by_score(candidate_scores)
        target_frontier_idx = _select_by_score(target_frontier_scores)

        def item_at(index: Optional[int]) -> Optional[Mapping[str, Any]]:
            if index is None or index < 0 or index >= len(items):
                return None
            return items[index]

        return {
            "enabled": True,
            "valid": True,
            "reason": "ok",
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "resilience_weight": float(self.resilience_weight),
            "feature_dim": len(feature_names()),
            "ranker_scores": ranker_scores,
            "ranker_normalized_scores": normalized_ranker_scores,
            "resilience_priors": resilience_priors,
            "future_observability_proxies": future_observability,
            "recoverability_proxies": recoverability,
            "ranker_resilience_scores": ranker_resilience_scores,
            "candidate_score_selected": _candidate_summary(
                item_at(candidate_score_idx), candidate_score_idx
            ),
            "target_frontier_selected": _candidate_summary(
                item_at(target_frontier_idx), target_frontier_idx
            ),
            "ranker_selected": _candidate_summary(item_at(ranker_idx), ranker_idx),
            "ranker_resilience_selected": _candidate_summary(
                item_at(resilience_idx), resilience_idx
            ),
            "ranker_changes_target_frontier": bool(
                ranker_idx is not None and target_frontier_idx is not None and ranker_idx != target_frontier_idx
            ),
            "ranker_resilience_changes_target_frontier": bool(
                resilience_idx is not None
                and target_frontier_idx is not None
                and resilience_idx != target_frontier_idx
            ),
            "ranker_resilience_changes_candidate_score": bool(
                resilience_idx is not None
                and candidate_score_idx is not None
                and resilience_idx != candidate_score_idx
            ),
        }

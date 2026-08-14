"""Runtime Stage21b frozen multi-head scorer shadow.

The scorer only computes diagnostics.  It never selects or applies a Habitat
action.  Errors are handled by the caller as shadow-only fallbacks.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from internnav.utils.stage21_scorer_features import (
    CANDIDATE_TYPES,
    NUMERIC_FIELDS,
    encode_row,
    feature_names,
)


RECOVERY_TYPES = {"resilience_backtrack", "backtrack_reobserve"}


class _Stage21MultiHeadScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.safety_head = nn.Linear(hidden_dim, 1)
        self.safety_geometry_head = nn.Linear(hidden_dim, 1)
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.recovery_promising_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "progress": self.progress_head(hidden).squeeze(-1),
            "safety": self.safety_head(hidden).squeeze(-1),
            "safety_geometry": self.safety_geometry_head(hidden).squeeze(-1),
            "recovery": self.recovery_head(hidden).squeeze(-1),
            "recovery_promising": self.recovery_promising_head(hidden).squeeze(-1),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _candidate_summary(candidate: Optional[Mapping[str, Any]], index: Optional[int], score: Optional[float]) -> Dict[str, Any]:
    if candidate is None or index is None:
        return {"index": None, "candidate_id": None, "valid": False}
    return {
        "index": int(index), "candidate_id": candidate.get("candidate_id"), "valid": True,
        "candidate_type": candidate.get("candidate_type") or candidate.get("source") or "unknown",
        "direction_bucket": candidate.get("direction_bucket") or "unknown",
        "geometry_safe": bool(candidate.get("geometry_safe")),
        "active_gate_safe": bool(candidate.get("active_gate_safe")),
        "score": _safe_float(score), "candidate_score": _safe_float(candidate.get("score")),
    }


class Stage21MultiTaskShadowScorer:
    """Load a Stage21b checkpoint and compute no-op online diagnostics."""

    def __init__(self, *, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = os.path.expanduser(str(checkpoint_path or ""))
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Stage21 scorer checkpoint not found: {self.checkpoint_path}")
        requested = torch.device(str(device or "cpu"))
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for Stage21 scorer shadow but unavailable")
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=requested, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=requested)
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint.get("feature_schema"):
            schema = checkpoint["feature_schema"]
        else:
            with open(os.path.join(checkpoint_dir, "feature_schema.json"), encoding="utf-8") as handle:
                schema = json.load(handle)
        runtime_names = feature_names()
        if list(schema.get("feature_names") or []) != runtime_names:
            raise ValueError("Stage21 scorer feature schema mismatch")
        if checkpoint.get("normalizer"):
            normalizer = checkpoint["normalizer"]
        else:
            with open(os.path.join(checkpoint_dir, "normalizer.json"), encoding="utf-8") as handle:
                normalizer = json.load(handle)
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.std = np.asarray(normalizer["std"], dtype=np.float32)
        if self.mean.shape != self.std.shape or self.mean.size != len(runtime_names) or np.any(self.std <= 0.0):
            raise ValueError("Invalid Stage21 scorer normalizer")
        config = checkpoint.get("training_config") or {}
        self.model = _Stage21MultiHeadScorer(
            len(runtime_names), int(config.get("hidden_dim", 128)), float(config.get("dropout", 0.10))
        ).to(requested)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.device = requested
        self.feature_dim = len(runtime_names)
        self.schema_version = str(schema.get("schema_version") or "unknown")

    @staticmethod
    def _select(indices: Sequence[int], scores: Sequence[float]) -> Optional[int]:
        return max(indices, key=lambda index: float(scores[index])) if indices else None

    @torch.no_grad()
    def score_candidates(
        self, candidates: Sequence[Mapping[str, Any]], current_policy_candidate: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        items = [dict(item) for item in candidates if isinstance(item, Mapping)]
        if not items:
            return {"enabled": True, "valid": False, "reason": "no_candidates", "action_applied": False}
        current = dict(current_policy_candidate or {})
        rows = [{"online_inputs": {"candidate": item, "current_policy_candidate": current}} for item in items]
        matrix = np.asarray([encode_row(row) for row in rows], dtype=np.float32)
        matrix = (matrix - self.mean) / self.std
        if not np.isfinite(matrix).all():
            raise ValueError("non_finite_stage21_features")
        output = self.model(torch.as_tensor(matrix, dtype=torch.float32, device=self.device))
        progress = output["progress"].detach().cpu().numpy().astype(np.float64)
        safety = torch.sigmoid(output["safety"]).detach().cpu().numpy().astype(np.float64)
        geometry = torch.sigmoid(output["safety_geometry"]).detach().cpu().numpy().astype(np.float64)
        recovery = torch.sigmoid(output["recovery"]).detach().cpu().numpy().astype(np.float64)
        promising = torch.sigmoid(output["recovery_promising"]).detach().cpu().numpy().astype(np.float64)
        arrays = (progress, safety, geometry, recovery, promising)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("non_finite_stage21_scores")
        ordinary = [
            index for index, item in enumerate(items)
            if str(item.get("candidate_type") or item.get("source") or "unknown") not in RECOVERY_TYPES
            and bool(item.get("geometry_safe"))
        ]
        recovery_indices = [
            index for index, item in enumerate(items)
            if str(item.get("candidate_type") or item.get("source") or "unknown") in RECOVERY_TYPES
            and bool(item.get("geometry_safe"))
        ]
        candidate_scores = [_safe_float(item.get("score")) for item in items]
        intent_scores = [_safe_float(item.get("intent_alignment_score")) for item in items]
        selected_progress = self._select(ordinary, progress.tolist())
        selected_candidate = self._select(ordinary, candidate_scores)
        selected_intent = self._select(ordinary, intent_scores)
        selected_recovery = self._select(recovery_indices, recovery.tolist())
        missing = []
        for item in items:
            missing.append(sum(item.get(field) is None for field in NUMERIC_FIELDS))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "enabled": True, "valid": True, "reason": "ok", "action_applied": False,
            "checkpoint_path": self.checkpoint_path, "schema_version": self.schema_version,
            "feature_dim": self.feature_dim, "device": str(self.device),
            "candidate_count": len(items), "ordinary_eligible_count": len(ordinary),
            "recovery_eligible_count": len(recovery_indices),
            "missing_numeric_mean": float(np.mean(missing)) if missing else 0.0,
            "inference_latency_ms": float(elapsed_ms),
            "scores": [
                {"index": index, "candidate_id": item.get("candidate_id"),
                 "candidate_type": item.get("candidate_type") or item.get("source") or "unknown",
                 "direction_bucket": item.get("direction_bucket") or "unknown",
                 "geometry_safe": bool(item.get("geometry_safe")),
                 "progress": float(progress[index]), "safety": float(safety[index]),
                 "geometry_safe_probability": float(geometry[index]),
                 "recovery": float(recovery[index]), "recovery_promising": float(promising[index])}
                for index, item in enumerate(items)
            ],
            "progress_selected": _candidate_summary(items[selected_progress] if selected_progress is not None else None, selected_progress, progress[selected_progress] if selected_progress is not None else None),
            "candidate_score_selected": _candidate_summary(items[selected_candidate] if selected_candidate is not None else None, selected_candidate, candidate_scores[selected_candidate] if selected_candidate is not None else None),
            "intent_alignment_selected": _candidate_summary(items[selected_intent] if selected_intent is not None else None, selected_intent, intent_scores[selected_intent] if selected_intent is not None else None),
            "recovery_selected": _candidate_summary(items[selected_recovery] if selected_recovery is not None else None, selected_recovery, recovery[selected_recovery] if selected_recovery is not None else None),
            "progress_changes_candidate_score": bool(selected_progress is not None and selected_candidate is not None and selected_progress != selected_candidate),
            "progress_changes_intent_alignment": bool(selected_progress is not None and selected_intent is not None and selected_progress != selected_intent),
            "recovery_would_be_diagnostic_only": True,
        }

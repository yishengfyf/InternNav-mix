"""Causal frame selection and audit gates for replayed LSeg frequency tests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np


def _scalar(value: Any, default: float = 0.0) -> float:
    array = np.asarray(value if value is not None else [default], dtype=np.float32)
    return float(array.reshape(-1)[0]) if array.size else float(default)


def _angle_delta(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def select_causal_keyframes(
    observations: Sequence[Mapping[str, Any]],
    query_observation_keys: Iterable[str],
    visual_descriptors: Optional[Mapping[int, np.ndarray]] = None,
    *,
    translation_m: float = 0.50,
    rotation_deg: float = 30.0,
    height_m: float = 0.20,
    pitch_deg: float = 15.0,
    visual_change: float = 0.12,
    min_gap: int = 2,
    max_gap: int = 4,
) -> Dict[int, List[str]]:
    """Select query and event frames without consulting LSeg predictions or GT."""
    query_keys: Set[str] = {str(value) for value in query_observation_keys}
    descriptors = visual_descriptors or {}
    selected: Dict[int, List[str]] = {}
    last: Optional[Mapping[str, Any]] = None
    last_index: Optional[int] = None

    for observation in observations:
        index = int(observation["record_index"])
        key = str(observation.get("observation_key", ""))
        reasons: List[str] = []
        if key in query_keys:
            reasons.append("s2_query")
        if last is None:
            reasons.append("episode_start")

        gap = index - last_index if last_index is not None else 0
        if last is not None and not reasons and gap >= max(1, min_gap):
            pose = observation.get("pose") or {}
            last_pose = last.get("pose") or {}
            position = np.asarray(pose.get("gps") or [0.0, 0.0], dtype=np.float32)
            last_position = np.asarray(
                last_pose.get("gps") or [0.0, 0.0], dtype=np.float32
            )
            if float(np.linalg.norm(position - last_position)) >= translation_m:
                reasons.append("translation")
            heading = _scalar(pose.get("compass"))
            last_heading = _scalar(last_pose.get("compass"))
            if math.degrees(_angle_delta(heading, last_heading)) >= rotation_deg:
                reasons.append("rotation")
            current_height = _scalar(pose.get("stage23a_gt_relative_height_m"))
            last_height = _scalar(last_pose.get("stage23a_gt_relative_height_m"))
            if abs(current_height - last_height) >= height_m:
                reasons.append("height")
            current_pitch = _scalar(observation.get("camera_pitch_deg"))
            last_pitch = _scalar(last.get("camera_pitch_deg"))
            if abs(current_pitch - last_pitch) >= pitch_deg:
                reasons.append("pitch")
            current_descriptor = descriptors.get(index)
            last_descriptor = descriptors.get(int(last["record_index"]))
            if current_descriptor is not None and last_descriptor is not None:
                change = float(np.mean(np.abs(
                    np.asarray(current_descriptor, dtype=np.float32)
                    - np.asarray(last_descriptor, dtype=np.float32)
                )))
                if change >= visual_change:
                    reasons.append("visual_change")
            if gap >= max_gap:
                reasons.append("max_gap")

        if reasons:
            selected[index] = sorted(set(reasons))
            last = observation
            last_index = index

    return selected


def short_lived_labels(
    frame_class_counts: Sequence[Mapping[str, int]], max_frames: int = 2
) -> Set[str]:
    frequency = Counter()
    for counts in frame_class_counts:
        frequency.update(str(label) for label, count in counts.items() if int(count) > 0)
    return {label for label, count in frequency.items() if count <= max_frames}


def evaluate_frequency_gate(
    q: Mapping[str, Any], qk: Mapping[str, Any], all_frames: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply the predeclared quality/cost gate to aggregate variant metrics."""
    all_calls = max(1, int(all_frames.get("call_count", 0)))
    all_gt = float(all_frames.get("gt_hit_rate") or 0.0)
    all_classes = max(1, int(all_frames.get("class_count", 0)))
    all_short = int(all_frames.get("short_lived_class_count", 0))
    q_conflict = float(q.get("conflicts_per_100_nodes") or 0.0)
    checks = {
        "gt_at_least_95pct_of_all": float(qk.get("gt_hit_rate") or 0.0)
        >= 0.95 * all_gt,
        "class_coverage_at_least_95pct_of_all": int(qk.get("class_count", 0))
        / all_classes >= 0.95,
        "short_lived_coverage_at_least_95pct_of_all": (
            all_short == 0 or int(qk.get("short_lived_class_count", 0))
            / all_short >= 0.95
        ),
        "calls_at_most_65pct_of_all": int(qk.get("call_count", 0))
        / all_calls <= 0.65,
        "conflict_rate_not_above_q": float(
            qk.get("conflicts_per_100_nodes") or 0.0
        ) <= q_conflict + 1e-9,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision": (
            "approve_q_plus_k_for_shadow_downstream"
            if all(checks.values()) else "retain_audit_only_and_tune_keyframes"
        ),
    }

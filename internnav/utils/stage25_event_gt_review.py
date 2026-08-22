"""Broad offline-only windows for human Stage25 event-GT review."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


ACTION_NAMES = {
    0: "stop", 1: "forward", 2: "left", 3: "right", 4: "lookup", 5: "lookdown",
}


def episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def scene_split(scene_id: Any, *, holdout_fraction: float = 0.30) -> str:
    """Assign whole scenes deterministically so related episodes never cross splits."""
    threshold = max(0, min(10, int(round(float(holdout_fraction) * 10))))
    bucket = int(hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "holdout" if bucket >= 10 - threshold else "dev"


def intervals_overlap(
    first: Mapping[str, Any], second: Mapping[str, Any], *, tolerance: int = 8,
) -> bool:
    if episode_key(first) != episode_key(second):
        return False
    first_start = int(first.get("onset_step", first.get("signal_step", first.get("step_id", 0))))
    first_end = int(first.get("end_step", first.get("step_id", first_start)))
    second_start = int(second.get("onset_step", second.get("signal_step", second.get("step_id", 0))))
    second_end = int(second.get("end_step", second.get("step_id", second_start)))
    return first_start <= second_end + tolerance and second_start <= first_end + tolerance


def objective_review_annotation(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Conservatively confirm kinematic GT-lite labels and abstain on ambiguity.

    These labels use future trajectory context for offline evaluation only. They are
    deliberately detector-independent and cannot be consumed as online features.
    """
    family = str(candidate.get("review_family") or "")
    audit = candidate.get("offline_action_audit") or {}
    counts = audit.get("action_counts") or {}
    outcome = candidate.get("outcome") or {}
    duration = int(candidate.get("duration_steps") or 0)
    applied = int(audit.get("applied_action_count") or 0)
    collision_delta = float(audit.get("collision_delta") or 0.0)
    displacement = float(candidate.get("displacement_m") or 0.0)
    route_m = float(candidate.get("path_length_m") or 0.0)
    turn_count = int(counts.get("left") or 0) + int(counts.get("right") or 0)
    turn_ratio = float(audit.get("turn_only_ratio") or 0.0)
    turn_degrees = float(audit.get("total_abs_turn_deg") or 0.0)
    forward_count = int(counts.get("forward") or 0)
    base = {
        "auto_status": "abstain",
        "state": None,
        "type": None,
        "onset_step": None,
        "end_step": None,
        "recoverability": None,
        "failure_link": None,
        "intervention_likely_needed": None,
        "confidence": "insufficient",
        "objective_reasons": [],
        "uses_future_for_review_only": True,
        "detector_feature_eligible": False,
        "notes": "Ambiguous objective evidence; preserve for visual/manual review.",
    }
    if family == "offline_wrong_way_progress":
        goal_increase = float(candidate.get("goal_distance_increase_m") or 0.0)
        if route_m >= 1.50 and goal_increase >= 1.00 and applied >= 16:
            base.update({
                "auto_status": "objective_confirmed",
                "state": "wrong_way_progress",
                "type": "W1_geodesic_regression",
                "onset_step": int(candidate["onset_step"]),
                "end_step": int(candidate["end_step"]),
                "recoverability": "not_a_local_trap",
                "failure_link": "outcome_association_only",
                "intervention_likely_needed": None,
                "confidence": "high",
                "objective_reasons": [
                    "executed_path_at_least_1.5m",
                    "geodesic_distance_increased_at_least_1.0m",
                ],
                "notes": "Confirmed progress regression, not a local-stuck positive.",
            })
        return base
    if family != "offline_local_stagnation":
        return base
    reasons: List[str] = []
    if collision_delta >= 2.0:
        reasons.append("collision_burst_during_low_motion")
    if forward_count >= 3 and displacement <= 0.15:
        reasons.append("repeated_forward_without_realized_displacement")
    if turn_count >= 12 and turn_ratio >= 0.80 and turn_degrees >= 360.0:
        reasons.append("full_rotation_during_local_stagnation")
    if duration < 16 or applied < 16 or route_m > 0.50 or displacement > 0.25 or not reasons:
        return base
    event_type = (
        "G1_geometry_execution"
        if any(reason != "full_rotation_during_local_stagnation" for reason in reasons)
        else "G2_local_rotation_loop"
    )
    episode_steps = int(outcome.get("steps") or 0)
    reaches_episode_end = episode_steps > 0 and int(candidate["end_step"]) >= episode_steps - 1
    if reaches_episode_end:
        recoverability = "persistent" if duration >= 32 else "episode_ended"
    else:
        recoverability = "self_recovered_quick" if duration <= 32 else "self_recovered_delayed"
    success = outcome.get("success")
    if success == 1 or success == 1.0:
        failure_link = "successful_episode_efficiency_opportunity"
    elif reaches_episode_end:
        failure_link = "contributing_proxy_pending_causal_review"
    else:
        failure_link = "outcome_association_only"
    base.update({
        "auto_status": "objective_confirmed",
        "state": "true_trap",
        "type": event_type,
        "onset_step": int(candidate["onset_step"]),
        "end_step": int(candidate["end_step"]),
        "recoverability": recoverability,
        "failure_link": failure_link,
        "intervention_likely_needed": bool(duration >= 32 or collision_delta >= 2.0),
        "confidence": "high",
        "objective_reasons": reasons,
        "notes": "GT-lite confirmation from executed future trajectory; no detector signal used.",
    })
    return base


def annotate_review_candidates(
    candidates: Sequence[Mapping[str, Any]], *, holdout_fraction: float = 0.30,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["split"] = scene_split(candidate.get("scene_id"), holdout_fraction=holdout_fraction)
        row["annotation"] = objective_review_annotation(candidate)
        annotated.append(row)
    return annotated


def evaluate_detector_against_gt_lite(
    detector_events: Sequence[Mapping[str, Any]],
    annotated_review: Sequence[Mapping[str, Any]], *, tolerance: int = 8,
) -> Dict[str, Any]:
    """Report recall/protection on adjudicated windows without inventing precision."""
    positives = [
        row for row in annotated_review
        if (row.get("annotation") or {}).get("auto_status") == "objective_confirmed"
        and (row.get("annotation") or {}).get("state") == "true_trap"
    ]
    wrong_way = [
        row for row in annotated_review
        if (row.get("annotation") or {}).get("auto_status") == "objective_confirmed"
        and (row.get("annotation") or {}).get("state") == "wrong_way_progress"
    ]

    def subset_report(split: Optional[str]) -> Dict[str, Any]:
        gt = [row for row in positives if split is None or row.get("split") == split]
        wrong = [row for row in wrong_way if split is None or row.get("split") == split]
        events = [
            event for event in detector_events
            if split is None or scene_split(event.get("scene_id")) == split
        ]
        detected_gt = [row for row in gt if any(intervals_overlap(row, event, tolerance=tolerance) for event in events)]
        wrong_overlap = [row for row in wrong if any(intervals_overlap(row, event, tolerance=tolerance) for event in events)]
        matched_events = [event for event in events if any(intervals_overlap(event, row, tolerance=tolerance) for row in gt)]
        wrong_only_events = [
            event for event in events
            if not any(intervals_overlap(event, row, tolerance=tolerance) for row in gt)
            and any(intervals_overlap(event, row, tolerance=tolerance) for row in wrong)
        ]
        adjudicated_event_ids = {id(event) for event in matched_events + wrong_only_events}
        return {
            "objective_true_trap_count": len(gt),
            "detected_true_trap_count": len(detected_gt),
            "true_trap_recall": len(detected_gt) / len(gt) if gt else None,
            "objective_wrong_way_count": len(wrong),
            "wrong_way_overlap_count": len(wrong_overlap),
            "wrong_way_protection_rate": 1.0 - len(wrong_overlap) / len(wrong) if wrong else None,
            "detector_event_count": len(events),
            "confirmed_detector_event_count": len(matched_events),
            "wrong_way_only_detector_event_count": len(wrong_only_events),
            "unadjudicated_detector_event_count": len(events) - len(adjudicated_event_ids),
            "precision_status": "not_computed_unadjudicated_detector_events_remain",
        }

    return {
        "all": subset_report(None),
        "dev": subset_report("dev"),
        "holdout": subset_report("holdout"),
    }


def distance(a: Any, b: Any) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def path_length(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(distance(a.get("gps"), b.get("gps")) for a, b in zip(rows, rows[1:]))


def _compass_radians(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("compass")
    try:
        return float(value[0] if isinstance(value, (list, tuple)) else value)
    except (TypeError, ValueError, IndexError):
        return None


def action_interval_summary(
    observations: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]],
    *, onset_step: int, end_step: int,
) -> Dict[str, Any]:
    """Summarize executed actions for review; these are never detector features."""
    interval_actions = [
        row for row in actions
        if onset_step <= int(row.get("step_id", -1)) <= end_step
        and row.get("action_applied") is not False
    ]
    counts = {name: 0 for name in ACTION_NAMES.values()}
    sources: Dict[str, int] = defaultdict(int)
    collision_delta = 0.0
    for row in interval_actions:
        try:
            action = int(row.get("action"))
        except (TypeError, ValueError):
            continue
        counts[ACTION_NAMES.get(action, f"other_{action}")] = (
            counts.get(ACTION_NAMES.get(action, f"other_{action}"), 0) + 1
        )
        sources[str(row.get("action_source") or "unknown")] += 1
        collision_delta += float((row.get("audit_metrics") or {}).get("collision_delta") or 0.0)
    interval_observations = [
        row for row in observations
        if onset_step <= int(row.get("step_id", -1)) <= end_step
    ]
    turn_degrees = 0.0
    for previous, current in zip(interval_observations, interval_observations[1:]):
        previous_yaw = _compass_radians(previous)
        current_yaw = _compass_radians(current)
        if previous_yaw is None or current_yaw is None:
            continue
        delta = math.atan2(
            math.sin(current_yaw - previous_yaw),
            math.cos(current_yaw - previous_yaw),
        )
        turn_degrees += abs(math.degrees(delta))
    goal_delta = None
    if interval_observations:
        try:
            goal_delta = float(interval_observations[-1]["distance_to_goal"]) - float(
                interval_observations[0]["distance_to_goal"]
            )
        except (TypeError, ValueError):
            pass
    locomotion_count = counts["forward"] + counts["left"] + counts["right"]
    return {
        "applied_action_count": len(interval_actions),
        "action_counts": counts,
        "action_source_counts": dict(sorted(sources.items())),
        "collision_delta": collision_delta,
        "total_abs_turn_deg": turn_degrees,
        "turn_only_ratio": (
            (counts["left"] + counts["right"]) / max(1, locomotion_count)
        ),
        "goal_distance_delta_m": goal_delta,
        "uses_future_for_review_only": True,
    }


def merge_windows(
    windows: Sequence[Mapping[str, Any]], *, max_gap_steps: int = 4,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for window in windows:
        by_family[str(window["review_family"])].append(window)
    for family_windows in by_family.values():
        family_merged: List[Dict[str, Any]] = []
        for window in sorted(family_windows, key=lambda item: int(item["onset_step"])):
            current = dict(window)
            if (
                family_merged
                and int(current["onset_step"])
                <= int(family_merged[-1]["end_step"]) + max_gap_steps
            ):
                family_merged[-1]["end_step"] = max(
                    int(family_merged[-1]["end_step"]), int(current["end_step"])
                )
                family_merged[-1]["step_id"] = int(family_merged[-1]["end_step"])
                family_merged[-1]["support_count"] = (
                    int(family_merged[-1]["support_count"]) + 1
                )
                for field in (
                    "displacement_m", "path_length_m", "goal_distance_increase_m",
                ):
                    if field in current:
                        family_merged[-1][field] = max(
                            float(family_merged[-1].get(field, 0.0)),
                            float(current[field]),
                        )
            else:
                current["support_count"] = 1
                family_merged.append(current)
        merged.extend(family_merged)
    merged.sort(key=lambda item: (int(item["onset_step"]), item["review_family"]))
    for window in merged:
        window["duration_steps"] = (
            int(window["end_step"]) - int(window["onset_step"]) + 1
        )
    return merged


def mine_review_windows(
    rows: Sequence[Mapping[str, Any]], *, stall_window: int = 16,
    stall_max_displacement_m: float = 0.25, stall_max_path_m: float = 0.50,
    regression_window: int = 32, regression_min_path_m: float = 1.50,
    regression_min_goal_increase_m: float = 1.00,
) -> List[Dict[str, Any]]:
    """Use future context only to propose review windows, never detector inputs."""
    candidates: List[Dict[str, Any]] = []
    for end in range(stall_window - 1, len(rows)):
        window = rows[end - stall_window + 1:end + 1]
        displacement = distance(window[0].get("gps"), window[-1].get("gps"))
        route_m = path_length(window)
        if displacement <= stall_max_displacement_m and route_m <= stall_max_path_m:
            candidates.append({
                "review_family": "offline_local_stagnation",
                "onset_step": int(window[0]["step_id"]),
                "end_step": int(window[-1]["step_id"]),
                "step_id": int(window[-1]["step_id"]),
                "displacement_m": displacement,
                "path_length_m": route_m,
                "uses_future_for_review_only": True,
            })
    for end in range(regression_window - 1, len(rows)):
        window = rows[end - regression_window + 1:end + 1]
        route_m = path_length(window)
        try:
            goal_increase = float(window[-1]["distance_to_goal"]) - float(
                window[0]["distance_to_goal"]
            )
        except (TypeError, ValueError):
            continue
        if route_m >= regression_min_path_m and goal_increase >= regression_min_goal_increase_m:
            candidates.append({
                "review_family": "offline_wrong_way_progress",
                "onset_step": int(window[0]["step_id"]),
                "end_step": int(window[-1]["step_id"]),
                "step_id": int(window[-1]["step_id"]),
                "path_length_m": route_m,
                "goal_distance_increase_m": goal_increase,
                "uses_future_for_review_only": True,
            })
    return merge_windows(candidates)

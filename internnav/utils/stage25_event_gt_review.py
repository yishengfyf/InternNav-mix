"""Broad offline-only windows for human Stage25 event-GT review."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


ACTION_NAMES = {
    0: "stop", 1: "forward", 2: "left", 3: "right", 4: "lookup", 5: "lookdown",
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

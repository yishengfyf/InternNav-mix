"""Broad offline-only windows for human Stage25 event-GT review."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence


def distance(a: Any, b: Any) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def path_length(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(distance(a.get("gps"), b.get("gps")) for a, b in zip(rows, rows[1:]))


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

"""Read-only local elevation support graph for Stage57 diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


def _ordered_path(values: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    result = []
    seen = set()
    for value in values:
        try:
            cell = (int(value[0]), int(value[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if cell not in seen:
            seen.add(cell)
            result.append(cell)
    return result


def _profiles(memory: Any, cells: set[tuple[int, int]]) -> dict[tuple[int, int], list[dict]]:
    occ_counts = getattr(memory, "occ_counts", {}) or {}
    frame_counts = getattr(memory, "occ3d_frame_counts", {}) or {}
    cell_profiles: dict[tuple[int, int], list[dict]] = defaultdict(list)
    cs = max(float(getattr(memory, "cs", 0.05)), 1e-6)
    for key, hits in occ_counts.items():
        try:
            row, col, height = int(key[0]), int(key[1]), int(key[2])
        except (TypeError, ValueError, IndexError):
            continue
        cell = (row, col)
        if cell not in cells or int(hits or 0) <= 0:
            continue
        cell_profiles[cell].append(
            {
                "height_index": height,
                "z_m": float(height * cs),
                "occupied_hits": int(hits),
                "support_frame_count": int(frame_counts.get(key, 0) or 0),
            }
        )
    for values in cell_profiles.values():
        values.sort(key=lambda item: (item["z_m"], -item["support_frame_count"]))
    return cell_profiles


def _rasterize_path(values: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in zip(values, values[1:]):
        row0, col0 = start
        row1, col1 = end
        steps = max(abs(row1 - row0), abs(col1 - col0), 1)
        for index in range(steps):
            alpha = float(index) / float(steps)
            cell = (
                int(round(row0 + alpha * (row1 - row0))),
                int(round(col0 + alpha * (col1 - col0))),
            )
            if not result or cell != result[-1]:
                result.append(cell)
    if values and (not result or result[-1] != values[-1]):
        result.append(values[-1])
    return result


def _nearby_values(
    profiles: Mapping[tuple[int, int], list[dict]],
    cell: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
    cs: float,
) -> list[dict]:
    values = []
    for dr, dc in offsets:
        for item in profiles.get((cell[0] + dr, cell[1] + dc), []):
            values.append(
                {
                    **item,
                    "source_cell": [cell[0] + dr, cell[1] + dc],
                    "source_distance_m": float(math.hypot(dr, dc) * cs),
                }
            )
    return values


def _choose_support(
    values: Sequence[Mapping[str, Any]],
    previous_z: float,
    *,
    min_support_frames: int,
    max_step_up_m: float,
    max_step_down_m: float,
    initial: bool = False,
) -> Mapping[str, Any] | None:
    candidates = [
        item
        for item in values
        if int(item.get("support_frame_count", 0) or 0) >= int(min_support_frames)
        and (
            initial
            or previous_z - float(max_step_down_m) - 1e-9
            <= float(item["z_m"])
            <= previous_z + float(max_step_up_m) + 1e-9
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(float(item["z_m"]) - previous_z),
            float(item.get("source_distance_m", 0.0) or 0.0),
            float(item["z_m"]),
            -int(item.get("support_frame_count", 0) or 0),
            -int(item.get("occupied_hits", 0) or 0),
        ),
    )


def audit_local_elevation_support(
    memory: Any,
    path_cells: Sequence[Sequence[int]],
    *,
    initial_floor_z_m: float = 0.0,
    footprint_radius_m: float = 0.18,
    min_support_frames: int = 2,
    max_step_up_m: float = 0.20,
    max_step_down_m: float = 0.20,
    headroom_m: float = 1.50,
    support_search_radius_m: float = 0.10,
    minimum_safe_segment_m: float = 0.25,
    max_records: int = 256,
) -> dict[str, Any]:
    """Build a conservative support graph without changing navigation state."""
    sparse_path = _ordered_path(path_cells)
    path = _rasterize_path(sparse_path)
    result: dict[str, Any] = {
        "schema_version": "stage57_local_elevation_support_v2",
        "enabled": True,
        "shadow_only": True,
        "decision_applied": False,
        "unknown_is_free": False,
        "pixel_translation_allowed": False,
        "reason": None,
        "path_cell_count": len(path),
        "sparse_path_cell_count": len(sparse_path),
        "footprint_radius_m": float(footprint_radius_m),
        "initial_floor_z_m": float(initial_floor_z_m),
        "support_frame_threshold": int(min_support_frames),
        "max_step_up_m": float(max_step_up_m),
        "max_step_down_m": float(max_step_down_m),
        "headroom_m": float(headroom_m),
        "support_search_radius_m": float(support_search_radius_m),
        "minimum_safe_segment_m": float(minimum_safe_segment_m),
    }
    if not path:
        result["reason"] = "missing_path"
        return result

    cs = max(float(getattr(memory, "cs", 0.05)), 1e-6)
    radius = max(0.0, float(footprint_radius_m))
    radius_cells = int(math.ceil(radius / cs))
    offsets = [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if math.hypot(dr, dc) * cs <= radius + 1e-9
    ]
    search_radius_cells = int(math.ceil(max(0.0, support_search_radius_m) / cs))
    search_offsets = [
        (dr, dc)
        for dr in range(-search_radius_cells, search_radius_cells + 1)
        for dc in range(-search_radius_cells, search_radius_cells + 1)
        if math.hypot(dr, dc) * cs <= float(support_search_radius_m) + 1e-9
    ]
    corridor = []
    seen = set()
    for row, col in path:
        for dr, dc in offsets:
            cell = (row + dr, col + dc)
            if cell not in seen:
                seen.add(cell)
                corridor.append(cell)
    profile_cells = {
        (row + dr, col + dc)
        for row, col in corridor
        for dr, dc in search_offsets
    }
    profiles = _profiles(memory, profile_cells)

    previous_z = float(initial_floor_z_m)
    center_records = []
    discontinuities = 0
    for index, cell in enumerate(path):
        support = _choose_support(
            _nearby_values(profiles, cell, search_offsets, cs),
            previous_z,
            min_support_frames=min_support_frames,
            max_step_up_m=max_step_up_m,
            max_step_down_m=max_step_down_m,
            initial=False,
        )
        if support is None:
            center_records.append({"cell": list(cell), "support_known": False})
            continue
        support_z = float(support["z_m"])
        delta = support_z - previous_z
        if index > 0 and (
            delta > float(max_step_up_m) + 1e-9
            or delta < -float(max_step_down_m) - 1e-9
        ):
            discontinuities += 1
        previous_z = support_z
        center_records.append(
            {
                "cell": list(cell),
                "support_known": True,
                "support_z_m": support_z,
                "delta_from_previous_m": float(delta),
                "support_frame_count": int(support.get("support_frame_count", 0) or 0),
                "support_source_cell": support.get("source_cell"),
                "support_source_distance_m": support.get("source_distance_m"),
            }
        )

    corridor_records = []
    support_known = 0
    headroom_blocked = 0
    safe_step_count = 0
    longest_safe_steps = 0
    current_safe_steps = 0
    leading_safe_steps = 0
    leading_safe_open = True
    step_records = []
    for center, center_record in zip(path, center_records):
        center_z = center_record.get("support_z_m")
        footprint_known = 0
        footprint_headroom_blocked = 0
        for dr, dc in offsets:
            cell = (center[0] + dr, center[1] + dc)
            values = _nearby_values(profiles, cell, search_offsets, cs)
            support = None if center_z is None else _choose_support(
                values,
                float(center_z),
                min_support_frames=min_support_frames,
                max_step_up_m=max_step_up_m,
                max_step_down_m=max_step_down_m,
            )
            if support is None:
                record = {"cell": list(cell), "support_known": False}
            else:
                support_z = float(support["z_m"])
                overhead = [
                    item
                    for item in values
                    if support_z + 0.15 < float(item["z_m"]) <= support_z + float(headroom_m)
                    and int(item.get("support_frame_count", 0) or 0) >= int(min_support_frames)
                ]
                footprint_known += 1
                support_known += 1
                footprint_headroom_blocked += int(bool(overhead))
                headroom_blocked += int(bool(overhead))
                record = {
                    "cell": list(cell),
                    "support_known": True,
                    "support_z_m": support_z,
                    "support_frame_count": int(support.get("support_frame_count", 0) or 0),
                    "support_source_distance_m": support.get("source_distance_m"),
                    "headroom_blocked": bool(overhead),
                    "height_record_count": len(values),
                }
            corridor_records.append(record)
        step_safe = bool(
            center_z is not None
            and footprint_known == len(offsets)
            and footprint_headroom_blocked == 0
        )
        safe_step_count += int(step_safe)
        current_safe_steps = current_safe_steps + 1 if step_safe else 0
        longest_safe_steps = max(longest_safe_steps, current_safe_steps)
        if leading_safe_open and step_safe:
            leading_safe_steps += 1
        else:
            leading_safe_open = False
        step_records.append(
            {
                "center_cell": list(center),
                "center_support_known": center_z is not None,
                "footprint_cell_count": len(offsets),
                "footprint_support_known_count": footprint_known,
                "headroom_blocked_count": footprint_headroom_blocked,
                "full_footprint_safe": step_safe,
            }
        )

    center_known = sum(bool(item.get("support_known")) for item in center_records)
    corridor_sample_count = len(path) * len(offsets)
    corridor_coverage = support_known / max(1, corridor_sample_count)
    center_coverage = center_known / max(1, len(path))
    longest_safe_segment_m = float(max(0, longest_safe_steps - 1) * cs)
    leading_safe_segment_m = float(max(0, leading_safe_steps - 1) * cs)
    result.update(
        {
            "reason": "ok",
            "corridor_cell_count": len(corridor),
            "corridor_footprint_sample_count": corridor_sample_count,
            "corridor_support_known_count": int(support_known),
            "corridor_support_coverage": float(corridor_coverage),
            "centerline_support_known_count": int(center_known),
            "centerline_support_coverage": float(center_coverage),
            "height_discontinuity_count": int(discontinuities),
            "headroom_blocked_count": int(headroom_blocked),
            "continuous_support_centerline": bool(
                center_coverage >= 1.0 and discontinuities == 0
            ),
            "full_footprint_safe_step_count": int(safe_step_count),
            "longest_full_footprint_safe_step_count": int(longest_safe_steps),
            "longest_full_footprint_safe_segment_m": longest_safe_segment_m,
            "leading_full_footprint_safe_step_count": int(leading_safe_steps),
            "leading_full_footprint_safe_segment_m": leading_safe_segment_m,
            "leading_full_footprint_safe_corridor": bool(
                leading_safe_segment_m + 1e-9 >= float(minimum_safe_segment_m)
                and discontinuities == 0
            ),
            "full_footprint_support": bool(path and safe_step_count == len(path)),
            "full_footprint_safe_corridor": bool(
                longest_safe_segment_m + 1e-9 >= float(minimum_safe_segment_m)
                and discontinuities == 0
            ),
            "eligible_corridor": False,
            "records_are_bounded_sample": len(corridor_records) > int(max_records),
            "centerline_records": center_records[: max(1, int(max_records))],
            "step_records": step_records[: max(1, int(max_records))],
            "corridor_records": corridor_records[: max(1, int(max_records))],
        }
    )
    return result

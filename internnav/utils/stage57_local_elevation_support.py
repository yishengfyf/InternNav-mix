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
    max_records: int = 256,
) -> dict[str, Any]:
    """Build a conservative support graph without changing navigation state."""
    path = _ordered_path(path_cells)
    result: dict[str, Any] = {
        "schema_version": "stage57_local_elevation_support_v1",
        "enabled": True,
        "shadow_only": True,
        "decision_applied": False,
        "unknown_is_free": False,
        "pixel_translation_allowed": False,
        "reason": None,
        "path_cell_count": len(path),
        "footprint_radius_m": float(footprint_radius_m),
        "initial_floor_z_m": float(initial_floor_z_m),
        "support_frame_threshold": int(min_support_frames),
        "max_step_up_m": float(max_step_up_m),
        "max_step_down_m": float(max_step_down_m),
        "headroom_m": float(headroom_m),
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
    corridor = []
    seen = set()
    for row, col in path:
        for dr, dc in offsets:
            cell = (row + dr, col + dc)
            if cell not in seen:
                seen.add(cell)
                corridor.append(cell)
    profiles = _profiles(memory, set(corridor))

    previous_z = float(initial_floor_z_m)
    center_records = []
    discontinuities = 0
    for index, cell in enumerate(path):
        support = _choose_support(
            profiles.get(cell, []),
            previous_z,
            min_support_frames=min_support_frames,
            max_step_up_m=max_step_up_m,
            max_step_down_m=max_step_down_m,
            initial=index == 0,
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
            }
        )

    corridor_records = []
    support_known = 0
    headroom_blocked = 0
    for cell in corridor:
        values = profiles.get(cell, [])
        support = _choose_support(
            values,
            float(initial_floor_z_m),
            min_support_frames=min_support_frames,
            max_step_up_m=max_step_up_m,
            max_step_down_m=max_step_down_m,
            initial=True,
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
            support_known += 1
            headroom_blocked += int(bool(overhead))
            record = {
                "cell": list(cell),
                "support_known": True,
                "support_z_m": support_z,
                "support_frame_count": int(support.get("support_frame_count", 0) or 0),
                "headroom_blocked": bool(overhead),
                "height_record_count": len(values),
            }
        corridor_records.append(record)

    center_known = sum(bool(item.get("support_known")) for item in center_records)
    corridor_coverage = support_known / max(1, len(corridor))
    center_coverage = center_known / max(1, len(path))
    result.update(
        {
            "reason": "ok",
            "corridor_cell_count": len(corridor),
            "corridor_support_known_count": int(support_known),
            "corridor_support_coverage": float(corridor_coverage),
            "centerline_support_known_count": int(center_known),
            "centerline_support_coverage": float(center_coverage),
            "height_discontinuity_count": int(discontinuities),
            "headroom_blocked_count": int(headroom_blocked),
            "continuous_support_centerline": bool(
                center_coverage >= 1.0 and discontinuities == 0
            ),
            "full_footprint_support": bool(corridor and corridor_coverage >= 1.0),
            "full_footprint_safe_corridor": bool(
                corridor
                and corridor_coverage >= 1.0
                and discontinuities == 0
                and headroom_blocked == 0
            ),
            "eligible_corridor": False,
            "records_are_bounded_sample": len(corridor_records) > int(max_records),
            "centerline_records": center_records[: max(1, int(max_records))],
            "corridor_records": corridor_records[: max(1, int(max_records))],
        }
    )
    return result

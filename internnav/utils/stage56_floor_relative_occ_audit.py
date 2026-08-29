"""Read-only floor-relative independent-frame OCC audit."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


def _ordered_unique(values: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
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


def _sample(values: Sequence[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    if len(values) <= limit:
        return list(values)
    indices = {
        int(round(index * (len(values) - 1) / max(1, limit - 1)))
        for index in range(max(1, int(limit)))
    }
    return [values[index] for index in sorted(indices)]


def audit_candidate_floor_relative_frames(
    memory: Any,
    candidate: Mapping[str, Any],
    *,
    max_cell_records: int = 256,
    min_occupied_frames: int = 2,
    min_free_frames: int = 2,
) -> dict:
    """Compare legacy any-hit with same-band frame consensus without mutation."""
    path = _ordered_unique(candidate.get("path_cells") or [])
    result = {
        "schema_version": "stage56_candidate_floor_relative_frames_v1",
        "enabled": True,
        "candidate_id": candidate.get("candidate_id"),
        "decision_applied": False,
        "unknown_is_free": False,
        "pixel_translation_allowed": False,
        "reason": None,
    }
    if not path:
        result["reason"] = "missing_candidate_path"
        return result
    if not hasattr(memory, "floor_relative_frame_cell_evidence"):
        result["reason"] = "floor_relative_frame_api_unavailable"
        return result

    cs = max(1e-6, float(getattr(memory, "cs", 0.05)))
    radius_m = float(candidate.get("footprint_radius_m", 0.18) or 0.18)
    radius_cells = int(math.ceil(radius_m / cs))
    offsets = [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if math.hypot(dr, dc) * cs <= radius_m + 1e-9
    ]
    corridor = []
    seen = set()
    for row, col in path:
        for dr, dc in offsets:
            cell = (row + dr, col + dc)
            if cell not in seen:
                seen.add(cell)
                corridor.append(cell)

    floor_z = float(candidate.get("floor_z_m", 0.0) or 0.0)
    height_max = float(candidate.get("floor_aligned_height_max_m", 1.5) or 1.5)
    trigger = candidate.get("trigger_grid")
    trigger_cell = None
    try:
        trigger_cell = (int(trigger[0]), int(trigger[1]))
    except (TypeError, ValueError, IndexError):
        pass
    sampled = set(_sample(corridor, max(1, int(max_cell_records))))
    path_set = set(path)
    legacy_counts = Counter()
    consensus_counts = Counter()
    transitions = Counter()
    records = []
    frame_masks_available = True
    for row, col in corridor:
        legacy = memory.validation_floor_aligned_cell_evidence(
            row, col, floor_z, height_max_m=height_max
        )
        consensus = memory.floor_relative_frame_cell_evidence(
            row,
            col,
            floor_z,
            height_max_m=height_max,
            min_occupied_frames=min_occupied_frames,
            min_free_frames=min_free_frames,
        )
        legacy_state = str(legacy.get("state") or "unknown")
        consensus_state = str(consensus.get("state") or "unknown")
        legacy_counts[legacy_state] += 1
        consensus_counts[consensus_state] += 1
        transitions[f"{legacy_state}_to_{consensus_state}"] += 1
        frame_masks_available = frame_masks_available and bool(
            consensus.get("frame_masks_available")
        )
        if (row, col) not in sampled:
            continue
        in_current_footprint = bool(
            trigger_cell is not None
            and math.hypot(row - trigger_cell[0], col - trigger_cell[1]) * cs
            <= radius_m + 1e-9
        )
        records.append(
            {
                "cell": [row, col],
                "on_centerline": (row, col) in path_set,
                "in_side_footprint": (row, col) not in path_set,
                "in_current_agent_footprint": in_current_footprint,
                "legacy_state": legacy_state,
                "consensus_state": consensus_state,
                "occupied_hits": int(legacy.get("occupied_hits", 0) or 0),
                "free_hits": int(legacy.get("free_hits", 0) or 0),
                "occupied_frame_count": int(
                    consensus.get("occupied_frame_count", 0) or 0
                ),
                "free_frame_count": int(
                    consensus.get("free_frame_count", 0) or 0
                ),
                "occupied_age_observations": consensus.get(
                    "occupied_age_observations"
                ),
                "free_age_observations": consensus.get("free_age_observations"),
                "current_frame_occupied": bool(
                    consensus.get("current_frame_occupied")
                ),
                "last_occupied_metadata": consensus.get(
                    "last_occupied_metadata"
                ),
                "last_free_metadata": consensus.get("last_free_metadata"),
                "occupied_height_records": list(
                    consensus.get("occupied_height_records") or []
                )[:32],
            }
        )

    return {
        **result,
        "reason": "ok" if frame_masks_available else "frame_masks_unavailable",
        "frame_masks_available": bool(frame_masks_available),
        "floor_z_m": floor_z,
        "height_max_m": height_max,
        "footprint_radius_m": radius_m,
        "path_cell_count": len(path),
        "corridor_cell_count": len(corridor),
        "cell_record_count": len(records),
        "corridor_summary_is_complete": True,
        "cell_records_are_bounded_sample": len(records) < len(corridor),
        "legacy_state_scope": "floor_relative_height_band",
        "frame_consensus_scope": "same_floor_relative_height_band",
        "legacy_state_counts": dict(legacy_counts),
        "frame_consensus_state_counts": dict(consensus_counts),
        "state_transition_counts": dict(transitions),
        "complete_corridor_legacy_free": bool(
            corridor and legacy_counts["free"] == len(corridor)
        ),
        "complete_corridor_frame_consensus_free": bool(
            corridor and consensus_counts["free"] == len(corridor)
        ),
        "cell_records": records,
    }

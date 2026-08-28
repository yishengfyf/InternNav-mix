"""Read-only frame-aware OCC and local 2.5D candidate-path audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Optional, Sequence


def should_post_turn_collision_guard(
    *,
    enabled: bool,
    armed: bool,
    previous_action: int,
    forward_action: int,
    previous_action_source: str,
    collision_delta: float,
    guard_age_steps: Optional[int],
    horizon_steps: int,
    requery_count: int,
    requery_budget: int,
) -> bool:
    """Return whether a collided post-turn S2 queue must be discarded."""
    return bool(
        enabled
        and armed
        and int(previous_action) == int(forward_action)
        and str(previous_action_source) == "system2_action_queue"
        and float(collision_delta) > 0.0
        and (
            int(horizon_steps) <= 0
            or (
                guard_age_steps is not None
                and int(guard_age_steps) <= int(horizon_steps)
            )
        )
        and int(requery_count) < int(requery_budget)
    )


def _sample_ordered(values: Sequence[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    ordered = []
    seen = set()
    for value in values:
        cell = (int(value[0]), int(value[1]))
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    if len(ordered) <= limit:
        return ordered
    indices = {
        int(round(index * (len(ordered) - 1) / max(1, limit - 1)))
        for index in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def _height_profile(memory: Any, cells: set[tuple[int, int]]) -> dict:
    profile = defaultdict(list)
    occ_frames = getattr(memory, "occ3d_frame_counts", {}) or {}
    for key, hits in (getattr(memory, "occ_counts", {}) or {}).items():
        row, col, height = int(key[0]), int(key[1]), int(key[2])
        cell = (row, col)
        if cell in cells and int(hits or 0) > 0:
            profile[cell].append(
                {
                    "height_index": height,
                    "z_m": float(height * float(memory.cs)),
                    "occupied_hits": int(hits),
                    "occupied_frame_count": int(occ_frames.get(key, 0) or 0),
                }
            )
    for values in profile.values():
        values.sort(key=lambda item: item["height_index"])
    return profile


def _support_sequence(
    path: Sequence[tuple[int, int]],
    profiles: Mapping[tuple[int, int], list[dict]],
    *,
    initial_floor_z_m: float,
    min_support_frames: int,
    max_step_up_m: float,
    max_step_down_m: float,
    headroom_m: float,
) -> dict:
    tracked = float(initial_floor_z_m)
    known = 0
    discontinuities = 0
    headroom_blocked = 0
    max_abs_delta = 0.0
    records = []
    for cell in path:
        values = list(profiles.get(cell, []))
        candidates = [
            item
            for item in values
            if int(item["occupied_frame_count"]) >= int(min_support_frames)
            and tracked - float(max_step_down_m) - 1e-9
            <= float(item["z_m"])
            <= tracked + float(max_step_up_m) + 1e-9
        ]
        support = min(
            candidates,
            key=lambda item: (
                abs(float(item["z_m"]) - tracked),
                -int(item["occupied_frame_count"]),
                -int(item["occupied_hits"]),
            ),
            default=None,
        )
        if support is None:
            records.append({"cell": list(cell), "support_known": False})
            continue
        support_z = float(support["z_m"])
        delta = support_z - tracked
        max_abs_delta = max(max_abs_delta, abs(delta))
        if delta > float(max_step_up_m) + 1e-9 or delta < -float(max_step_down_m) - 1e-9:
            discontinuities += 1
        overhead = [
            item
            for item in values
            if support_z + 0.15 < float(item["z_m"]) <= support_z + float(headroom_m)
            and int(item["occupied_frame_count"]) >= int(min_support_frames)
        ]
        if overhead:
            headroom_blocked += 1
        known += 1
        tracked = support_z
        records.append(
            {
                "cell": list(cell),
                "support_known": True,
                "support_z_m": support_z,
                "delta_from_previous_m": float(delta),
                "support_frame_count": int(support["occupied_frame_count"]),
                "headroom_blocked": bool(overhead),
            }
        )
    coverage = float(known / max(1, len(path)))
    return {
        "path_cell_count": len(path),
        "support_known_count": known,
        "support_coverage": coverage,
        "height_discontinuity_count": discontinuities,
        "headroom_blocked_count": headroom_blocked,
        "max_abs_support_step_m": max_abs_delta,
        "continuous_support_shadow": bool(
            path and coverage >= 0.60 and discontinuities == 0 and headroom_blocked == 0
        ),
        "records": records,
    }


def audit_candidate_occ_2p5d(
    memory: Any,
    candidate: Mapping[str, Any],
    *,
    max_path_cells: int = 64,
    max_corridor_cells: int = 256,
    min_support_frames: int = 2,
    max_step_up_m: float = 0.20,
    max_step_down_m: float = 0.20,
    headroom_m: float = 1.50,
) -> dict:
    """Describe alternative OCC evidence without changing navigation state."""
    raw_path = list(candidate.get("path_cells") or [])
    path = _sample_ordered(raw_path, max(1, int(max_path_cells)))
    result = {
        "schema_version": "stage55_occ_2p5d_audit_v1",
        "enabled": True,
        "candidate_id": candidate.get("candidate_id"),
        "path_cell_count_raw": len(raw_path),
        "path_cell_count_audited": len(path),
        "decision_applied": False,
        "unknown_is_free": False,
        "reason": None,
    }
    if not path:
        result["reason"] = "missing_candidate_path"
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
    corridor_records = _sample_ordered(
        corridor, max(1, int(max_corridor_cells))
    )
    corridor_record_set = set(corridor_records)
    corridor_set = set(corridor)
    floor_z = float(candidate.get("floor_z_m", 0.0) or 0.0)
    height_max = float(candidate.get("floor_aligned_height_max_m", 1.5) or 1.5)
    profiles = _height_profile(memory, corridor_set | set(path))

    legacy_counts = Counter()
    consensus_counts = Counter()
    blocked_consensus_free = 0
    blocked_consensus_unknown = 0
    cell_records = []
    for row, col in corridor:
        evidence = memory.validation_floor_aligned_cell_evidence(
            row, col, floor_z, height_max_m=height_max
        )
        frame = dict(evidence.get("frame_aware_cell_evidence") or {})
        legacy_counts[str(evidence.get("state") or "unknown")] += 1
        consensus = str(frame.get("frame_consensus_state") or "unknown")
        consensus_counts[consensus] += 1
        if evidence.get("state") == "blocked" and consensus == "free":
            blocked_consensus_free += 1
        if evidence.get("state") == "blocked" and consensus == "unknown":
            blocked_consensus_unknown += 1
        if (row, col) in corridor_record_set:
            cell_records.append(
                {
                    "cell": [row, col],
                    "on_centerline": (row, col) in set(path),
                    "legacy_floor_state": evidence.get("state"),
                    "occupied_hits": int(evidence.get("occupied_hits", 0) or 0),
                    "free_hits": int(evidence.get("free_hits", 0) or 0),
                    "frame_consensus_state": consensus,
                    "occupied_frame_count": int(
                        frame.get("occupied_frame_count", 0) or 0
                    ),
                    "free_frame_count": int(
                        frame.get("free_frame_count", 0) or 0
                    ),
                    "last_occupied_observation": frame.get(
                        "last_occupied_observation"
                    ),
                    "last_free_observation": frame.get(
                        "last_free_observation"
                    ),
                    "current_frame_occupied": bool(
                        frame.get("current_frame_occupied")
                    ),
                    "height_profile": list(profiles.get((row, col), []))[:16],
                }
            )

    support = _support_sequence(
        path,
        profiles,
        initial_floor_z_m=floor_z,
        min_support_frames=max(1, int(min_support_frames)),
        max_step_up_m=float(max_step_up_m),
        max_step_down_m=float(max_step_down_m),
        headroom_m=float(headroom_m),
    )
    result.update(
        {
            "reason": "ok",
            "floor_z_m": floor_z,
            "footprint_radius_m": radius_m,
            "corridor_cell_count": len(corridor),
            "corridor_cell_record_count": len(cell_records),
            "corridor_summary_is_complete": True,
            "cell_records_are_bounded_sample": len(cell_records) < len(corridor),
            "legacy_floor_state_scope": "floor_aligned_height_band",
            "frame_consensus_scope": "all_height_2d_cell",
            "legacy_floor_state_counts": dict(legacy_counts),
            "frame_consensus_state_counts": dict(consensus_counts),
            "legacy_blocked_frame_consensus_free_count": blocked_consensus_free,
            "legacy_blocked_frame_consensus_unknown_count": blocked_consensus_unknown,
            "support_2p5d": support,
            "cell_records": cell_records,
        }
    )
    return result

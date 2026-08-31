"""Stage58.1 read-only support-policy counterfactuals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from internnav.utils.stage57_local_elevation_support import (
    audit_local_elevation_support,
)


SCHEMA_VERSION = "stage58_support_policy_v1"


def _known_free_floor_cells(memory: Any) -> set[tuple[int, int]]:
    free = {
        (int(key[0]), int(key[1]))
        for key, value in (getattr(memory, "free2d_counts", {}) or {}).items()
        if int(value or 0) > 0
    }
    occupied = {
        (int(key[0]), int(key[1]))
        for key, value in (getattr(memory, "occ2d_counts", {}) or {}).items()
        if int(value or 0) > 0
    }
    return free - occupied


def audit_support_policy_sweep(
    memory: Any,
    path_cells: Sequence[Sequence[int]],
    *,
    runtime_contract: Mapping[str, Any],
    offline_primitive_truth: Mapping[str, Any],
    footprint_radius_m: float = 0.10,
    initial_floor_z_m: float = 0.0,
    max_step_m: float = 0.20,
    headroom_m: float = 1.50,
    minimum_safe_segment_m: float = 0.25,
) -> dict[str, Any]:
    """Compare bounded floor-support policies without mutating SparseOcc."""
    truth = dict(offline_primitive_truth or {})
    truth_valid = bool(truth.get("valid"))
    truth_safe = bool(truth.get("primitive_safe")) if truth_valid else None
    known_free = _known_free_floor_cells(memory)
    policies = (
        ("observed_frames2", 2, False),
        ("observed_frames1", 1, False),
        ("known_free_floor_frames2", 2, True),
        ("known_free_floor_frames1", 1, True),
    )
    arms = []
    for name, min_frames, floor_fallback in policies:
        graph = audit_local_elevation_support(
            memory,
            path_cells,
            initial_floor_z_m=float(initial_floor_z_m),
            footprint_radius_m=float(footprint_radius_m),
            min_support_frames=int(min_frames),
            max_step_up_m=float(max_step_m),
            max_step_down_m=float(max_step_m),
            headroom_m=float(headroom_m),
            minimum_safe_segment_m=float(minimum_safe_segment_m),
            floor_support_cells=known_free if floor_fallback else None,
            floor_support_source=(
                "known_free_2d_without_occupied_2d" if floor_fallback else None
            ),
        )
        predicted_safe = bool(graph.get("leading_full_footprint_safe_corridor"))
        arms.append(
            {
                "policy": name,
                "minimum_support_frames": int(min_frames),
                "known_free_floor_fallback": bool(floor_fallback),
                "predicted_first_primitive_safe": predicted_safe,
                "direct_path_primitive_candidate": predicted_safe,
                "offline_truth_valid": truth_valid,
                "offline_truth_safe": truth_safe,
                "false_safe": bool(truth_valid and predicted_safe and not truth_safe),
                "false_block": bool(truth_valid and not predicted_safe and truth_safe),
                "graph": graph,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "shadow_only": True,
        "audit_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "direct_path_primitive_executed": False,
        "unknown_is_free": False,
        "known_free_floor_fallback_mutates_memory": False,
        "gt_used_for_navigation": False,
        "runtime_contract": dict(runtime_contract or {}),
        "offline_geometry_truth": truth,
        "path_cell_count": int(len(path_cells or [])),
        "known_free_floor_cell_count": int(len(known_free)),
        "footprint_radius_m": float(footprint_radius_m),
        "minimum_safe_segment_m": float(minimum_safe_segment_m),
        "arms": arms,
    }

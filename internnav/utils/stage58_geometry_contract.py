"""Read-only Habitat geometry contract audit for recovery paths."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from internnav.utils.stage57_local_elevation_support import (
    audit_local_elevation_support,
)


SCHEMA_VERSION = "stage58_geometry_contract_v1"


def audit_geometry_radius_sweep(
    memory: Any,
    path_cells: Sequence[Sequence[int]],
    *,
    footprint_radii_m: Sequence[float],
    runtime_contract: Mapping[str, Any],
    offline_primitive_truth: Mapping[str, Any],
    initial_floor_z_m: float = 0.0,
    min_support_frames: int = 2,
    max_step_m: float = 0.20,
    headroom_m: float = 1.50,
    minimum_safe_segment_m: float = 0.25,
) -> dict[str, Any]:
    """Compare footprint margins without changing navigation or map state."""
    radii = sorted({max(0.0, float(value)) for value in footprint_radii_m})
    primitive_truth = dict(offline_primitive_truth or {})
    truth_valid = bool(primitive_truth.get("valid"))
    truth_safe = bool(primitive_truth.get("primitive_safe")) if truth_valid else None
    arms = []
    for radius_m in radii:
        graph = audit_local_elevation_support(
            memory,
            path_cells,
            initial_floor_z_m=float(initial_floor_z_m),
            footprint_radius_m=float(radius_m),
            min_support_frames=int(min_support_frames),
            max_step_up_m=float(max_step_m),
            max_step_down_m=float(max_step_m),
            headroom_m=float(headroom_m),
            minimum_safe_segment_m=float(minimum_safe_segment_m),
        )
        predicted_safe = bool(graph.get("leading_full_footprint_safe_corridor"))
        arms.append(
            {
                "footprint_radius_m": float(radius_m),
                "predicted_first_primitive_safe": predicted_safe,
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
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "runtime_contract": dict(runtime_contract or {}),
        "offline_geometry_truth": primitive_truth,
        "path_cell_count": int(len(path_cells or [])),
        "minimum_safe_segment_m": float(minimum_safe_segment_m),
        "arms": arms,
    }

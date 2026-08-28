"""Read-only recovery anchor and local BEV digest contracts.

The records are causal evidence for bounded re-observation.  They never turn
an old route node into a currently safe node and never emit an action.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage38_recovery_anchor_v1"


def _event_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": str(row.get("scene_id")),
        "episode_id": int(row.get("episode_id", -1)),
        "step_id": int(row.get("step_id", -1)),
    }


def _geometry_safety(candidate: Mapping[str, Any]) -> tuple[bool, str]:
    if bool(candidate.get("route_occ_conflict")):
        return False, "route_occ_conflict"
    if float(candidate.get("unknown_fraction", 0.0) or 0.0) != 0.0:
        return False, "unknown_path_evidence"
    if float(candidate.get("occupied_fraction", 0.0) or 0.0) != 0.0:
        return False, "occupied_path_evidence"
    if not bool(candidate.get("floor_aligned_known_free")):
        return False, "clearance_or_floor_not_safe"
    return True, "current_sparseocc_safe"


def build_recovery_anchor(
    event: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    capture_view: Mapping[str, Any] | None = None,
    capture_semantic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture evidence without granting a persistent safety label."""
    safe, reason = _geometry_safety(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "anchor_id": str(candidate.get("candidate_id") or "unknown"),
        "event_key": _event_key(event),
        "capture": {
            "source_type": candidate.get("source_type"),
            "source_step": candidate.get("source_step"),
            "route_support": candidate.get("route_support"),
            "route_support_edge_count": candidate.get("route_support_edge_count"),
            "path_cells": list(candidate.get("path_cells") or []),
            "pose": dict(event.get("capture_pose") or {}),
            "geometry": {
                "floor_z_m": candidate.get("floor_z_m"),
                "floor_z_source": candidate.get("floor_z_source"),
                "footprint_radius_m": candidate.get("footprint_radius_m"),
                "clearance_height_m": candidate.get("floor_aligned_height_max_m"),
                "capture_safe": bool(safe),
                "capture_safety_reason": reason,
            },
            "semantic": dict(capture_semantic or {}),
            "view": dict(capture_view or {}),
        },
        "trigger_reaudit_required": True,
        "current_safety": {
            "status": "not_a_persistent_label",
            "revalidated": False,
            "unknown_is_free": False,
            "safety_authority": "SparseOcc_current_reaudit",
        },
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
    }


def attach_current_reaudit(
    anchor: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Attach current SparseOcc result; semantic evidence cannot override it."""
    result = dict(anchor)
    current = dict(result.get("current_safety") or {})
    if candidate is None:
        current.update({"revalidated": False, "status": "no_current_candidate", "reason": "abstain"})
    else:
        safe, reason = _geometry_safety(candidate)
        current.update({
            "revalidated": True,
            "status": "safe" if safe else "rejected",
            "reason": reason,
            "unknown_is_free": False,
        })
    result["current_safety"] = current
    result["action_applied"] = False
    result["shadow_only"] = True
    return result


def build_recovery_bev_digest(
    anchor: Mapping[str, Any],
    *,
    channels: Mapping[str, Any] | None = None,
    semantic_relevance: float | None = None,
    reobserve_gain: float | None = None,
) -> dict[str, Any]:
    """Summarize BEV channels for ranking/diagnostics only."""
    current = dict(anchor.get("current_safety") or {})
    return {
        "schema_version": "stage38_recovery_bev_digest_v1",
        "anchor_id": anchor.get("anchor_id"),
        "event_key": dict(anchor.get("event_key") or {}),
        "channels": dict(channels or {}),
        "semantic_relevance": semantic_relevance,
        "reobserve_gain": reobserve_gain,
        "safety_authority": "current_sparseocc_reaudit",
        "safety_status": current.get("status", "not_revalidated"),
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "stair_height_is_inferred": False,
        "recommended_use": "ranking_or_diagnostic_only",
        "action_applied": False,
        "shadow_only": True,
        "gt_fields_used": [],
    }


def build_recovery_bev_spatial_snapshot(
    *, center_grid: Sequence[int], free_cells: Iterable[Sequence[int]],
    occupied_cells: Iterable[Sequence[int]],
    pose_trace: Iterable[Mapping[str, Any] | Sequence[int]] = (),
    semantic_cells: Iterable[Sequence[int]] = (), radius_cells: int = 24,
    semantic_nodes: Iterable[Mapping[str, Any]] = (),
    current_pose: Mapping[str, Any] | None = None,
    hfov_deg: float | None = None,
    depth_endpoints: Iterable[Mapping[str, Any]] = (),
    candidate_path: Iterable[Sequence[int]] = (),
    footprint_corridor: Iterable[Sequence[int]] = (),
) -> dict[str, Any]:
    """Create a bounded read-only local grid with explicit unknown cells."""
    center = (int(center_grid[0]), int(center_grid[1]))
    radius = max(1, int(radius_cells))
    free_set = {(int(cell[0]), int(cell[1])) for cell in free_cells}
    occupied_set = {(int(cell[0]), int(cell[1])) for cell in occupied_cells}
    channels = {"known_free": [], "occupied": [], "unknown": [], "semantic": [], "semantic_nodes": []}
    for row in range(center[0] - radius, center[0] + radius + 1):
        for col in range(center[1] - radius, center[1] + radius + 1):
            cell = [row, col]
            if (row, col) in occupied_set:
                channels["occupied"].append(cell)
            elif (row, col) in free_set:
                channels["known_free"].append(cell)
            else:
                channels["unknown"].append(cell)
    for cell in semantic_cells:
        row, col = int(cell[0]), int(cell[1])
        if abs(row - center[0]) <= radius and abs(col - center[1]) <= radius:
            channels["semantic"].append([row, col])
    for node in semantic_nodes:
        if not isinstance(node, Mapping):
            continue
        centroid = node.get("centroid")
        if isinstance(centroid, Sequence) and len(centroid) >= 2:
            # LSeg centroids are metric map XYZ; callers may provide grid in
            # the node when available, otherwise retain the metric point for
            # an offline renderer to transform.
            item = {"label": str(node.get("label") or "other"),
                    "confidence": node.get("mean_confidence"),
                    "evidence_tier": node.get("evidence_tier"),
                    "centroid": [float(v) for v in centroid[:3]]}
            if isinstance(node.get("grid"), Sequence):
                grid = [int(v) for v in node["grid"][:2]]
                if abs(grid[0] - center[0]) > radius or abs(grid[1] - center[1]) > radius:
                    continue
                item["grid"] = grid
            else:
                continue
            channels["semantic_nodes"].append(item)
    route = []
    for item in pose_trace:
        row, col = ((item.get("row"), item.get("col")) if isinstance(item, Mapping)
                    else (item[0], item[1]))
        if row is None or col is None:
            continue
        row, col = int(row), int(col)
        if abs(row - center[0]) <= radius and abs(col - center[1]) <= radius:
            route.append([row, col])
    pose = dict(current_pose or {})
    return {
        "schema_version": "stage38_recovery_bev_spatial_v1",
        "center_grid": list(center), "radius_cells": radius,
        "channels": channels, "executed_route": route,
        "current_pose": pose,
        "hfov_deg": None if hfov_deg is None else float(hfov_deg),
        "depth_endpoints": [dict(item) for item in depth_endpoints if isinstance(item, Mapping)],
        "candidate_path": [[int(v[0]), int(v[1])] for v in candidate_path if isinstance(v, Sequence) and len(v) >= 2],
        "footprint_corridor": [[int(v[0]), int(v[1])] for v in footprint_corridor if isinstance(v, Sequence) and len(v) >= 2],
        "unknown_is_free": False, "semantic_can_override_safety": False,
        "safety_authority": "SparseOcc_current_reaudit",
        "shadow_only": True, "action_applied": False, "gt_fields_used": [],
    }


def recovery_contract_ok(anchor: Mapping[str, Any], digest: Mapping[str, Any]) -> bool:
    current = dict(anchor.get("current_safety") or {})
    return bool(
        anchor.get("schema_version") == SCHEMA_VERSION
        and anchor.get("trigger_reaudit_required")
        and anchor.get("shadow_only")
        and not anchor.get("action_applied")
        and not anchor.get("gt_fields_used")
        and digest.get("unknown_is_free") is False
        and digest.get("semantic_can_override_safety") is False
        and digest.get("stair_height_is_inferred") is False
        and digest.get("safety_authority") == "current_sparseocc_reaudit"
        and digest.get("safety_status") == current.get("status", "not_revalidated")
    )

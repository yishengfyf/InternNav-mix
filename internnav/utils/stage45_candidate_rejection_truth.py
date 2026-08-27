"""Offline-only audit of SparseOcc candidate rejection against geometry GT."""

from collections import Counter
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


SCHEMA_VERSION = "stage45_candidate_rejection_truth_v1"


def _state_counts(values) -> Dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {
        name: int(counts.get(name, 0))
        for name in ("free", "occupied", "unknown")
    }


def audit_candidate_rejection_truth(
    candidate: Mapping[str, Any],
    *,
    sparse_2d_state: Callable[[int, int], str],
    sparse_floor_footprint_state: Callable[[int, int, float], str],
    navmesh_cell: Callable[[Tuple[int, int]], Mapping[str, Any]],
    navmesh_edge: Callable[
        [Tuple[int, int], Tuple[int, int]], Mapping[str, Any]
    ],
    footprint_radius_m: float,
) -> Dict[str, Any]:
    """Classify a frozen route-only candidate without changing its decision.

    Habitat callbacks are supplied by the evaluator so this module remains
    unit-testable and cannot accidentally expose GT to candidate generation.
    """
    cells = []
    for raw in candidate.get("path_cells") or ():
        try:
            cell = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if not cells or cell != cells[-1]:
            cells.append(cell)

    report: Dict[str, Any] = {
        "event_schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.get("candidate_id"),
        "source_step": candidate.get("source_step"),
        "valid": False,
        "reason": None,
        "shadow_only": True,
        "action_applied": False,
        "gt_used_for_navigation": False,
        "gt_reference": "Habitat pathfinder navmesh",
        "unknown_is_free": False,
        "footprint_radius_m": float(footprint_radius_m),
        "floor_z_m": float(candidate.get("floor_z_m", 0.0) or 0.0),
        "path_cell_count": int(len(cells)),
        "path_edge_count": int(max(0, len(cells) - 1)),
    }
    if not cells:
        report["reason"] = "missing_path_cells"
        return report

    floor_z_m = float(report["floor_z_m"])
    cell_rows = []
    sparse_2d_values = []
    sparse_floor_values = []
    navmesh_safe_count = 0
    navmesh_navigable_count = 0
    footprint_check_available_count = 0
    route_2d_false_block_count = 0
    floor_footprint_false_block_count = 0
    sparse_2d_false_free_count = 0
    floor_footprint_false_free_count = 0
    snap_distances = []
    clearances = []

    for cell in cells:
        sparse_2d = str(sparse_2d_state(cell[0], cell[1]))
        sparse_floor = str(
            sparse_floor_footprint_state(cell[0], cell[1], floor_z_m)
        )
        nav = dict(navmesh_cell(cell) or {})
        navigable = bool(nav.get("navigable", False))
        footprint_safe = nav.get("footprint_safe")
        clearance = nav.get("clearance_m")
        footprint_check_available = footprint_safe is not None
        gt_safe = bool(
            navigable
            and footprint_check_available
            and bool(footprint_safe)
        )
        sparse_2d_values.append(sparse_2d)
        sparse_floor_values.append(sparse_floor)
        navmesh_navigable_count += int(navigable)
        footprint_check_available_count += int(footprint_check_available)
        navmesh_safe_count += int(gt_safe)
        route_2d_false_block_count += int(
            sparse_2d in {"occupied", "unknown"} and gt_safe
        )
        floor_footprint_false_block_count += int(
            sparse_floor in {"occupied", "unknown"} and gt_safe
        )
        sparse_2d_false_free_count += int(sparse_2d == "free" and not gt_safe)
        floor_footprint_false_free_count += int(
            sparse_floor == "free" and not gt_safe
        )
        if nav.get("snap_distance_m") is not None:
            snap_distances.append(float(nav["snap_distance_m"]))
        if clearance is not None:
            clearances.append(float(clearance))
        cell_rows.append(
            {
                "grid": [int(cell[0]), int(cell[1])],
                "sparse_2d_state": sparse_2d,
                "sparse_floor_footprint_state": sparse_floor,
                "navmesh_navigable": navigable,
                "navmesh_footprint_safe": (
                    bool(footprint_safe) if footprint_check_available else None
                ),
                "navmesh_clearance_m": (
                    float(clearance) if clearance is not None else None
                ),
                "route_2d_false_block": bool(
                    sparse_2d in {"occupied", "unknown"} and gt_safe
                ),
                "floor_footprint_false_block": bool(
                    sparse_floor in {"occupied", "unknown"} and gt_safe
                ),
            }
        )

    edge_rows = []
    connected_edge_count = 0
    for first, second in zip(cells, cells[1:]):
        edge = dict(navmesh_edge(first, second) or {})
        connected = bool(edge.get("connected", False))
        connected_edge_count += int(connected)
        edge_rows.append(
            {
                "source_grid": [int(first[0]), int(first[1])],
                "target_grid": [int(second[0]), int(second[1])],
                "connected": connected,
                "geodesic_m": edge.get("geodesic_m"),
                "direct_m": edge.get("direct_m"),
                "geodesic_ratio": edge.get("geodesic_ratio"),
            }
        )

    all_cells_gt_safe = navmesh_safe_count == len(cells)
    all_edges_connected = connected_edge_count == max(0, len(cells) - 1)
    complete_gt_safe_corridor = bool(all_cells_gt_safe and all_edges_connected)
    report.update(
        {
            "valid": footprint_check_available_count == len(cells),
            "reason": (
                "ok"
                if footprint_check_available_count == len(cells)
                else "navmesh_footprint_check_unavailable"
            ),
            "sparse_2d_state_counts": _state_counts(sparse_2d_values),
            "sparse_floor_footprint_state_counts": _state_counts(
                sparse_floor_values
            ),
            "navmesh_navigable_cell_count": int(navmesh_navigable_count),
            "navmesh_footprint_safe_cell_count": int(navmesh_safe_count),
            "route_2d_false_block_cell_count": int(route_2d_false_block_count),
            "floor_footprint_false_block_cell_count": int(
                floor_footprint_false_block_count
            ),
            "sparse_2d_false_free_cell_count": int(
                sparse_2d_false_free_count
            ),
            "floor_footprint_false_free_cell_count": int(
                floor_footprint_false_free_count
            ),
            "navmesh_connected_edge_count": int(connected_edge_count),
            "all_navmesh_edges_connected": bool(all_edges_connected),
            "complete_gt_safe_corridor": bool(complete_gt_safe_corridor),
            "route_occ_false_block_candidate": bool(
                complete_gt_safe_corridor
                and any(value in {"occupied", "unknown"} for value in sparse_2d_values)
            ),
            "floor_footprint_false_block_candidate": bool(
                complete_gt_safe_corridor
                and any(
                    value in {"occupied", "unknown"}
                    for value in sparse_floor_values
                )
            ),
            "navmesh_snap_distance_m": {
                "max": max(snap_distances) if snap_distances else None,
            },
            "navmesh_clearance_m": {
                "min": min(clearances) if clearances else None,
            },
            "cells": cell_rows,
            "edges": edge_rows,
        }
    )
    return report


def summarize_event_audits(audits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in audits if item.get("valid")]
    return {
        "event_schema_version": SCHEMA_VERSION,
        "candidate_count": int(len(audits)),
        "valid_candidate_count": int(len(valid)),
        "complete_gt_safe_corridor_count": sum(
            bool(item.get("complete_gt_safe_corridor")) for item in valid
        ),
        "route_occ_false_block_candidate_count": sum(
            bool(item.get("route_occ_false_block_candidate")) for item in valid
        ),
        "floor_footprint_false_block_candidate_count": sum(
            bool(item.get("floor_footprint_false_block_candidate"))
            for item in valid
        ),
        "shadow_only": True,
        "action_applied": False,
        "gt_used_for_navigation": False,
        "unknown_is_free": False,
    }

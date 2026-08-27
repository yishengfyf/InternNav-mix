import importlib.util
import math
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage45_candidate_rejection_truth.py"
_spec = importlib.util.spec_from_file_location("stage45_candidate_rejection_truth", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _candidate():
    return {
        "candidate_id": "route_node:7",
        "source_step": 7,
        "path_cells": [[0, 0], [0, 1], [0, 2]],
        "floor_z_m": 0.0,
    }


def _edge(first, second):
    direct = math.hypot(second[0] - first[0], second[1] - first[1]) * 0.05
    return {"connected": True, "direct_m": direct, "geodesic_m": direct, "geodesic_ratio": 1.0}


def test_complete_corridor_identifies_sparse_false_block():
    report = _module.audit_candidate_rejection_truth(
        _candidate(),
        sparse_2d_state=lambda row, col: "occupied" if col == 1 else "free",
        sparse_floor_footprint_state=lambda row, col, floor: "occupied" if col == 1 else "free",
        navmesh_cell=lambda cell: {"navigable": True, "footprint_safe": True, "clearance_m": 0.4, "snap_distance_m": 0.0},
        navmesh_edge=_edge,
        footprint_radius_m=0.18,
    )
    assert report["valid"] is True
    assert report["complete_gt_safe_corridor"] is True
    assert report["route_occ_false_block_candidate"] is True
    assert report["floor_footprint_false_block_candidate"] is True
    assert report["route_2d_false_block_cell_count"] == 1
    assert report["gt_used_for_navigation"] is False
    assert report["unknown_is_free"] is False


def test_blocked_navmesh_never_promotes_candidate():
    report = _module.audit_candidate_rejection_truth(
        _candidate(),
        sparse_2d_state=lambda row, col: "occupied" if col == 1 else "free",
        sparse_floor_footprint_state=lambda row, col, floor: "occupied" if col == 1 else "free",
        navmesh_cell=lambda cell: {
            "navigable": cell[1] != 1,
            "footprint_safe": cell[1] != 1,
            "clearance_m": 0.05 if cell[1] == 1 else 0.4,
            "snap_distance_m": 0.2 if cell[1] == 1 else 0.0,
        },
        navmesh_edge=_edge,
        footprint_radius_m=0.18,
    )
    assert report["complete_gt_safe_corridor"] is False
    assert report["route_occ_false_block_candidate"] is False
    assert report["floor_footprint_false_block_candidate"] is False


def test_missing_clearance_fails_closed():
    report = _module.audit_candidate_rejection_truth(
        _candidate(),
        sparse_2d_state=lambda row, col: "unknown",
        sparse_floor_footprint_state=lambda row, col, floor: "unknown",
        navmesh_cell=lambda cell: {"navigable": True, "footprint_safe": None, "clearance_m": None},
        navmesh_edge=_edge,
        footprint_radius_m=0.18,
    )
    assert report["valid"] is False
    assert report["reason"] == "navmesh_footprint_check_unavailable"
    assert report["complete_gt_safe_corridor"] is False

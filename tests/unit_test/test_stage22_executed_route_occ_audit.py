import copy

import numpy as np

from internnav.utils.sparse_occ_memory import (
    SparseOccSemanticMemory,
    _cam_to_base_for_pitch,
    _default_cam_to_base_tf,
)


def _memory():
    return SparseOccSemanticMemory(
        {
            "enable": True,
            "frontier_enable": False,
            "grid_size": 64,
            "cell_size": 0.25,
        }
    )


def _node(step, row, col, x, y):
    return {
        "step_id": step,
        "row": row,
        "col": col,
        "x": x,
        "y": y,
        "z": 0.0,
        "yaw": 0.0,
    }


def _candidate(source_step=10):
    return {
        "grid": [32, 32],
        "semantic_resilience_source_step_id": source_step,
    }


def test_pitch_transform_preserves_horizontal_projection_and_rotates_lookdown():
    horizontal = _cam_to_base_for_pitch(1.5, 0.0)
    assert np.allclose(horizontal, _default_cam_to_base_tf(1.5))

    optical_forward = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    pitched_point = _cam_to_base_for_pitch(1.5, 30.0) @ optical_forward
    assert np.allclose(pitched_point[:2], [np.cos(np.deg2rad(30.0)), 0.0])
    assert np.isclose(pitched_point[2], 1.0)


def test_route_audit_collapses_rotation_poses_and_finds_known_free_path():
    memory = _memory()
    memory.pose_trace = [
        _node(10, 32, 32, 0.0, 0.0),
        _node(11, 32, 32, 0.0, 0.0),
        _node(12, 32, 31, 0.0, 0.25),
        _node(13, 32, 31, 0.0, 0.25),
        _node(14, 32, 30, 0.0, 0.50),
    ]
    for cell in ((32, 32), (32, 31), (32, 30)):
        memory.free2d_counts[cell] = 1

    result = memory.audit_executed_route_to_candidate(
        _candidate(), current_step=14
    )

    assert result["valid"] is True
    assert result["source_anchor_pose_match"] is True
    assert result["route_raw_pose_count"] == 5
    assert result["route_translation_node_count"] == 3
    assert result["rotation_or_duplicate_pose_count"] == 2
    assert result["route_movement_edge_count"] == 2
    assert result["route_length_m"] == 0.5
    assert result["route_chain_continuous"] is True
    assert result["route_cell_state_counts"] == {
        "free": 3,
        "unknown": 0,
        "occupied": 0,
    }
    assert result["known_free_connectivity"]["reachable"] is True
    assert result["continuous_but_ray_disconnected"] is False


def test_route_audit_separates_unknown_and_occupied_map_conflicts():
    memory = _memory()
    memory.pose_trace = [
        _node(10, 32, 32, 0.0, 0.0),
        _node(11, 32, 31, 0.0, 0.25),
        _node(12, 32, 30, 0.0, 0.50),
    ]
    memory.free2d_counts[(32, 32)] = 1
    memory.occ2d_counts[(32, 30)] = 1

    result = memory.audit_executed_route_to_candidate(
        _candidate(), current_step=12
    )

    assert result["route_chain_continuous"] is True
    assert result["route_cell_state_counts"] == {
        "free": 1,
        "unknown": 1,
        "occupied": 1,
    }
    assert result["route_conflict_edge_count"] == 1
    assert result["route_unknown_edge_count"] == 2
    assert result["first_unknown_gap"]["grid"] == [32, 31]
    assert result["first_occupied_conflict"]["grid"] == [32, 30]
    assert result["known_free_connectivity"]["reachable"] is False
    assert result["continuous_but_ray_disconnected"] is True


def test_route_audit_reports_missing_source_step_and_large_pose_jump():
    memory = _memory()
    memory.pose_trace = [
        _node(10, 32, 32, 0.0, 0.0),
        _node(11, 32, 24, 2.0, 0.0),
    ]

    missing = memory.audit_executed_route_to_candidate(
        _candidate(source_step=9), current_step=11
    )
    jumped = memory.audit_executed_route_to_candidate(
        _candidate(), current_step=11, max_edge_m=0.75
    )

    assert missing["valid"] is False
    assert missing["reason"] == "source_step_not_in_pose_trace"
    assert jumped["valid"] is True
    assert jumped["route_discontinuity_edge_count"] == 1
    assert jumped["route_chain_continuous"] is False


def test_route_audit_is_read_only():
    memory = _memory()
    memory.pose_trace = [
        _node(10, 32, 32, 0.0, 0.0),
        _node(11, 32, 31, 0.0, 0.25),
    ]
    memory.free2d_counts[(32, 32)] = 2
    memory.occ2d_counts[(31, 31)] = 3
    before = {
        "pose_trace": copy.deepcopy(memory.pose_trace),
        "free2d_counts": copy.deepcopy(memory.free2d_counts),
        "occ2d_counts": copy.deepcopy(memory.occ2d_counts),
        "visited2d_counts": copy.deepcopy(memory.visited2d_counts),
    }

    result = memory.audit_executed_route_to_candidate(
        _candidate(), current_step=11
    )

    assert result["action_applied"] is False
    assert result["output_rewritten"] is False
    assert result["gt_fields_used"] == []
    assert memory.pose_trace == before["pose_trace"]
    assert memory.free2d_counts == before["free2d_counts"]
    assert memory.occ2d_counts == before["occ2d_counts"]
    assert memory.visited2d_counts == before["visited2d_counts"]


def test_route_audit_reports_3d_height_conflicts_without_changing_2d_state():
    memory = _memory()
    memory.pose_trace = [
        _node(10, 32, 32, 0.0, 0.0),
        _node(11, 32, 31, 0.0, 0.25),
    ]
    # The flattened route cell is occupied, but its only evidence is below
    # the configured obstacle band. This must remain an occupied 2D audit
    # state while being diagnosed as ground-level evidence.
    memory.occ2d_counts[(32, 31)] = 1
    memory.occ_counts[(32, 31, 0)] = 3
    result = memory.audit_executed_route_to_candidate(_candidate(), current_step=11)
    diagnostics = result["route_occupied_height_diagnostics"]
    assert result["route_cell_state_counts"]["occupied"] == 1
    assert diagnostics["occupied_route_cell_count"] == 1
    assert diagnostics["low_or_ground_conflict_cell_count"] == 1
    assert diagnostics["obstacle_band_conflict_cell_count"] == 0
    assert diagnostics["cells"][0]["voxel_heights"][0]["z_m"] == 0.0


def test_route_cell_evidence_reports_occupied_free_and_visit_support_read_only():
    memory = _memory()
    memory.occ2d_counts[(32, 31)] = 3
    memory.free2d_counts[(32, 31)] = 1
    memory.visited2d_counts[(32, 31)] = 2
    before = copy.deepcopy(
        (memory.occ2d_counts, memory.free2d_counts, memory.visited2d_counts)
    )

    evidence = memory.audit_route_cell_evidence(
        {"route_cells": [[32, 32], [32, 31], [32, 31]]}
    )

    assert evidence["schema_version"] == "stage22d_route_cell_evidence_v1"
    assert evidence["cell_count"] == 2
    assert evidence["occupied_cell_count"] == 1
    cell = next(item for item in evidence["cells"] if item["grid"] == [32, 31])
    assert cell["occupied_hits"] == 3
    assert cell["free_hits"] == 1
    assert cell["visited_hits"] == 2
    assert cell["occupied_free_ratio"] == 0.75
    assert evidence["mutated_memory"] is False
    assert (memory.occ2d_counts, memory.free2d_counts, memory.visited2d_counts) == before


def test_route_cell_evidence_aligns_free_and_occupied_obstacle_heights_read_only():
    memory = _memory()
    memory.occ2d_counts[(32, 31)] = 3
    memory.free2d_counts[(32, 31)] = 109
    memory.visited2d_counts[(32, 31)] = 2
    memory.occ_counts[(32, 31, 2)] = 3
    memory.free_counts[(32, 31, 2)] = 10
    memory.free_counts[(32, 31, 0)] = 99
    before = copy.deepcopy(
        (
            memory.occ_counts,
            memory.free_counts,
            memory.occ2d_counts,
            memory.free2d_counts,
            memory.visited2d_counts,
        )
    )

    evidence = memory.audit_route_cell_evidence(
        {"route_cells": [[32, 31]]}, include_height_aligned=True
    )

    assert evidence["schema_version"] == "stage22e_height_aligned_route_cell_evidence_v1"
    assert evidence["height_aligned"] is True
    assert evidence["height_aligned_cell_count"] == 1
    assert evidence["band_free_dominant_cell_count"] == 1
    assert evidence["shared_band_height_cell_count"] == 1
    cell = evidence["cells"][0]
    assert cell["occupied_hits"] == 3
    assert cell["free_hits"] == 109
    assert cell["occupied_band_hits"] == 3
    assert cell["free_band_hits"] == 10
    assert cell["band_evidence_hits"] == 13
    assert cell["shared_band_height_indices"] == [2]
    assert cell["shared_band_occupied_hits"] == 3
    assert cell["shared_band_free_hits"] == 10
    assert cell["free_band_height_indices"] == [2]
    assert evidence["mutated_memory"] is False
    assert (
        memory.occ_counts,
        memory.free_counts,
        memory.occ2d_counts,
        memory.free2d_counts,
        memory.visited2d_counts,
    ) == before

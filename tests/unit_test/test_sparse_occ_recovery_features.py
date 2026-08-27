import math

import numpy as np

from internnav.utils.sparse_occ_memory import SparseOccSemanticMemory


def _memory():
    return SparseOccSemanticMemory(
        {
            "enable": True,
            "frontier_enable": False,
            "grid_size": 64,
            "cell_size": 0.25,
            "semantic_resilience_anchor_feature_radius_cells": 4,
            "semantic_resilience_cycle_radius_cells": 1,
            "semantic_resilience_cycle_window_steps": 16,
        }
    )


def test_open_multi_sector_anchor_has_high_branch_count():
    memory = _memory()
    center = (32, 32)
    for row, col in ((31, 32), (30, 32), (33, 32), (34, 32), (32, 31), (32, 30), (32, 33), (32, 34)):
        memory.free2d_counts[(row, col)] = 1

    info = memory._anchor_spatial_information(center)

    assert info["branch_count"] == 4
    assert info["direction_entropy"] > 0.95


def test_isolated_free_cells_do_not_become_executable_exits():
    memory = _memory()
    center = (32, 32)
    for cell in ((30, 32), (34, 32), (32, 30), (32, 34)):
        memory.free2d_counts[cell] = 1

    info = memory._anchor_spatial_information(center)

    assert info["branch_count"] == 0
    assert info["connected_component_count"] == 4


def test_a_b_a_trace_sets_short_cycle_risk():
    memory = _memory()
    memory.pose_trace = [
        {"row": 32, "col": 32, "step_id": 1},
        {"row": 32, "col": 36, "step_id": 2},
        {"row": 32, "col": 32, "step_id": 3},
    ]

    info = memory._anchor_trace_information((32, 32), latest_step=3)

    assert info["return_count"] == 2
    assert info["recent_cycle_count"] == 1
    assert info["short_cycle_risk"] > 0.0


def test_revisit_interval_uses_visit_segments_not_consecutive_near_steps():
    memory = _memory()
    memory.pose_trace = [
        {"row": 32, "col": 32, "step_id": 1},
        {"row": 32, "col": 32, "step_id": 2},
        {"row": 32, "col": 36, "step_id": 3},
        {"row": 32, "col": 36, "step_id": 4},
        {"row": 32, "col": 32, "step_id": 9},
    ]

    info = memory._anchor_trace_information((32, 32), latest_step=9)

    assert info["revisit_interval_steps"] == [8]


def test_path_bridge_derives_projection_geometry_without_mutating_candidate(monkeypatch):
    memory = _memory()
    path = [(32, col) for col in range(32, 37)]
    for cell in path:
        memory.free2d_counts[cell] = 1
    monkeypatch.setattr(
        memory,
        "_current_pose_state",
        lambda _obs: {"grid": [32, 32], "yaw": 0.0},
    )
    captured = {}

    def capture_projection(candidate, *_args, **_kwargs):
        captured.update(candidate)
        return {"valid": False, "reason": "test_capture", "sample_records": []}

    monkeypatch.setattr(memory, "plan_recovery_projection_bridge", capture_projection)
    candidate = {"candidate_id": "route_node:4", "grid": [32, 36]}
    original = dict(candidate)

    report = memory.plan_recovery_path_bridge(
        candidate,
        {},
        np.ones((8, 8), dtype=np.float32),
        reorient_lookahead_m=0.50,
    )

    assert report["path_reachable"] is True
    assert candidate == original
    assert captured["grid"] == [32, 36]
    assert math.isfinite(captured["direction_angle_deg"])
    assert captured["distance_m"] == report["path_m"]
    assert report["projection_candidate_geometry"] == {
        "source": "current_sparseocc_path_reaudit",
        "direction_bucket": captured["direction_bucket"],
        "direction_angle_deg": captured["direction_angle_deg"],
        "distance_m": captured["distance_m"],
        "candidate_mutated": False,
    }


def test_current_pose_state_can_prefer_temporary_observation_over_trace(monkeypatch):
    memory = _memory()
    memory.pose_trace = [{"row": 32, "col": 32, "x": 0.0, "y": 0.0, "yaw": 0.0}]
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    observed_tf = np.eye(4, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: observed_tf)
    monkeypatch.setattr(memory, "_relative_base_tf", lambda _tf: observed_tf)
    monkeypatch.setattr(memory, "_pose_to_grid", lambda _tf: (40, 41, 1.25))

    state = memory._current_pose_state({"_prefer_observation_pose": True})

    assert state == {"grid": [40, 41], "xy": [0.0, 0.0], "yaw": 1.25}
    assert memory._current_pose_state({})["grid"] == [32, 32]


def test_path_bridge_reports_counterfactual_observation_pose(monkeypatch):
    memory = _memory()
    path = [(32, col) for col in range(32, 37)]
    for cell in path:
        memory.free2d_counts[cell] = 1
    monkeypatch.setattr(
        memory,
        "_current_pose_state",
        lambda _obs: {"grid": [32, 32], "yaw": 1.25},
    )
    monkeypatch.setattr(
        memory,
        "plan_recovery_projection_bridge",
        lambda *_args, **_kwargs: {
            "valid": False,
            "reason": "test_capture",
            "sample_records": [],
        },
    )

    report = memory.plan_recovery_path_bridge(
        {"candidate_id": "route_node:4", "grid": [32, 36]},
        {"_prefer_observation_pose": True},
        np.ones((8, 8), dtype=np.float32),
    )

    assert report["path_pose_source"] == "current_observation"
    assert report["path_pose_yaw_rad"] == 1.25

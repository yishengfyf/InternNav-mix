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


def test_depth_short_lookahead_rejects_unknown_prefix_and_accepts_free_prefix(monkeypatch):
    memory = _memory()
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(memory, "_pixel_goal_to_grid", lambda *_args, **_kwargs: {
        "goal_grid": [32, 36], "start_grid": [32, 32], "start_yaw": 0.0,
        "depth_m": 2.0, "goal_world_z": 0.0,
    })
    monkeypatch.setattr(memory, "validation_floor_aligned_cell_evidence", lambda *_args, **_kwargs: {"state": "free"})
    monkeypatch.setattr(memory, "_grid_to_pixel_goal", lambda *_args, **_kwargs: [4, 4])
    for col in (32, 33, 34, 35, 36):
        memory.free2d_counts[(32, col)] = 1
    memory.free2d_counts.pop((32, 33))
    rejected = memory.plan_depth_short_lookahead_shadow({}, np.ones((8, 8), dtype=np.float32))
    assert rejected["eligible_count"] == 0
    assert rejected["reason"] == "no_current_depth_short_safe_path"
    memory.free2d_counts[(32, 33)] = 1
    accepted = memory.plan_depth_short_lookahead_shadow({}, np.ones((8, 8), dtype=np.float32))
    assert accepted["eligible_count"] > 0
    assert all(item["path_state"] == "free" for item in accepted["eligible_records"])
    assert accepted["unknown_is_free"] is False


def test_depth_short_lookahead_never_uses_surface_z_for_floor_gate(monkeypatch):
    memory = _memory()
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(memory, "_pixel_goal_to_grid", lambda *_args, **_kwargs: {
        "goal_grid": [32, 34], "start_grid": [32, 32], "start_yaw": 0.0,
        "depth_m": 2.0, "goal_world_z": 9.0,
    })
    floor_z_seen = []
    monkeypatch.setattr(memory, "validation_floor_aligned_cell_evidence", lambda _r, _c, z, **_kwargs: floor_z_seen.append(z) or {"state": "free"})
    monkeypatch.setattr(memory, "_grid_to_pixel_goal", lambda *_args, **_kwargs: [4, 4])
    for col in (32, 33, 34):
        memory.free2d_counts[(32, col)] = 1
    report = memory.plan_depth_short_lookahead_shadow({}, np.ones((8, 8), dtype=np.float32))
    assert report["eligible_count"] > 0
    assert floor_z_seen and all(abs(value) < 1e-6 for value in floor_z_seen)


def test_grid_to_pixel_goal_rejects_out_of_image_projection(monkeypatch):
    memory = _memory()
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.cam_to_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.array(
        [[1.0, 0.0, 1000.0], [0.0, 1.0, 1000.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    monkeypatch.setattr(memory, "_grid_to_xy", lambda _grid: (0.0, 0.0))
    assert memory._grid_to_pixel_goal(
        [32, 32], 1.0, np.eye(4, dtype=np.float32),
        {"image_width": 8, "image_height": 8}, np.ones((8, 8), dtype=np.float32),
    ) is None


def test_depth_short_local_search_can_route_around_blocked_direct_cell(monkeypatch):
    memory = _memory()
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(memory, "_pixel_goal_to_grid", lambda *_args, **_kwargs: {
        "goal_grid": [32, 36], "start_grid": [32, 32], "start_yaw": 0.0,
        "depth_m": 2.0, "goal_world_z": 0.0,
    })
    monkeypatch.setattr(memory, "validation_floor_aligned_cell_evidence", lambda *_args, **_kwargs: {
        "state": "free", "occupied_hits": 0, "occupied_voxel_count": 0,
        "height_index_min": 0, "height_index_max": 6,
    })
    monkeypatch.setattr(memory, "_grid_to_pixel_goal", lambda *_args, **_kwargs: [4, 4])
    for row, col in (
        (32, 32), (31, 32), (30, 32), (30, 33), (30, 34),
        (30, 35), (30, 36), (31, 36), (32, 36),
    ):
        memory.free2d_counts[(row, col)] = 1
    memory.occ2d_counts[(32, 33)] = 1
    report = memory.plan_depth_short_lookahead_shadow(
        {}, np.ones((8, 8), dtype=np.float32), local_search_enable=True,
        local_search_lateral_m=0.5, local_search_detour_m=1.0,
    )
    assert report["eligible_count"] > 0
    assert any(row["path_source"] == "local_8n_depth_supported" for row in report["eligible_records"])
    assert all(row["lookahead_pixel_in_bounds"] for row in report["eligible_records"])


def test_depth_short_origin_occupancy_does_not_block_future_free_cells(monkeypatch):
    memory = _memory()
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(memory, "_pixel_goal_to_grid", lambda *_args, **_kwargs: {
        "goal_grid": [32, 34], "start_grid": [32, 32], "start_yaw": 0.0,
        "depth_m": 2.0, "goal_world_z": 0.0,
    })
    monkeypatch.setattr(memory, "validation_floor_aligned_cell_evidence", lambda *_args, **_kwargs: {
        "state": "free", "occupied_hits": 0, "occupied_voxel_count": 0,
        "height_index_min": 0, "height_index_max": 6,
    })
    monkeypatch.setattr(memory, "_grid_to_pixel_goal", lambda *_args, **_kwargs: [4, 4])
    memory.occ2d_counts[(32, 32)] = 1
    memory.free2d_counts[(32, 33)] = 1
    memory.free2d_counts[(32, 34)] = 1
    report = memory.plan_depth_short_lookahead_shadow({}, np.ones((8, 8), dtype=np.float32))
    assert report["eligible_count"] > 0
    assert all(row["start_cell_state"] == "occupied" for row in report["eligible_records"])
    assert all(row["start_cell_exempted"] is True for row in report["eligible_records"])


def test_depth_short_footprint_uses_physical_radius_and_preserves_future_blocker(monkeypatch):
    memory = _memory()
    memory.cell_size = 0.05
    memory.cs = 0.05
    memory.init_base_tf = np.eye(4, dtype=np.float32)
    memory.camera_intrinsic = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(memory, "_pose_from_obs", lambda _obs: np.eye(4, dtype=np.float32))
    monkeypatch.setattr(memory, "_pixel_goal_to_grid", lambda *_args, **_kwargs: {
        "goal_grid": [31, 32], "start_grid": [32, 32], "start_yaw": 0.0,
        "depth_m": 2.0, "goal_world_z": 0.0,
    })
    evidence_calls = []
    def evidence(row, col, _z, **_kwargs):
        evidence_calls.append((row, col))
        # Outside the physical 0.18m radius of the current pose, but within
        # the old ceil-to-four-cell stencil for the first future pose.
        if (row, col) == (27, 33):
            return {"state": "blocked", "occupied_hits": 1, "occupied_voxel_count": 1}
        # A genuine future footprint conflict: inside the footprint of the
        # next pose, despite being outside the initial pose footprint.
        if (row, col) == (28, 33):
            return {"state": "blocked", "occupied_hits": 1, "occupied_voxel_count": 1}
        return {"state": "free", "occupied_hits": 0, "occupied_voxel_count": 0}
    monkeypatch.setattr(memory, "validation_floor_aligned_cell_evidence", evidence)
    monkeypatch.setattr(memory, "_grid_to_pixel_goal", lambda *_args, **_kwargs: [4, 4])
    for row in range(27, 33):
        memory.free2d_counts[(row, 32)] = 1
    report = memory.plan_depth_short_lookahead_shadow(
        {}, np.ones((8, 8), dtype=np.float32), lookahead_distances_m=(0.25,)
    )
    record = report["lookahead_records"][0]
    blocker = (record["floor_footprint_audit"] or {}).get("first_blocker")
    assert blocker is not None
    assert blocker["footprint_cell"] == [28, 33]
    assert record["floor_footprint_audit"]["initial_footprint_exempted_cell_count"] > 0
    # A cell exactly 0.20m from the next pose is outside the 0.18m gate.
    assert (27, 33) not in evidence_calls

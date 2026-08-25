from internnav.utils.stage29_hsgm_recovery_bridge import (
    angular_delta_deg,
    bridge_candidate,
    grid_to_xy,
)


def _event():
    return {"scene_id": "scene", "episode_id": 1, "step_id": 10}


def _candidate(path=None, **extra):
    row = {
        "candidate_id": "route_node:4",
        "source_type": "R-route-open",
        "source_step": 4,
        "route_support": "executed_transition_chain",
        "route_support_edge_count": 3,
        "path_cells": path or [[496, 500], [497, 500], [498, 500], [499, 500], [500, 500]],
        "path_length_m": 0.2,
        "route_occ_conflict": False,
        "unknown_fraction": 0.0,
        "occupied_fraction": 0.0,
        "floor_aligned_known_free": True,
        "floor_z_source": "gps_compass_2d",
        "gt_fields_used": [],
        "action_applied": False,
    }
    row.update(extra)
    return row


def _observation(compass=0.0):
    return {"pose": {"gps": [0.0, 0.0], "compass": [compass]}}


def test_grid_to_xy_and_angle_wrap():
    assert grid_to_xy([490, 510], origin=[500, 500], cell_size_m=0.05) == (0.5, -0.5)
    assert angular_delta_deg(-179.0, 179.0) == 2.0


def test_first_edge_visible_is_non_executing():
    result = bridge_candidate(_event(), _candidate(), _observation())
    assert result["bridge_status"] == "first_edge_horizontally_visible"
    assert result["shadow_only"] is True
    assert result["action_applied"] is False
    assert result["edge_reports"][0]["visibility_mode"] == "pose_bearing_only_no_depth_occlusion"


def test_offscreen_candidate_requires_reobserve_bridge():
    candidate = _candidate(path=[[504, 500], [503, 500], [502, 500], [501, 500], [500, 500]])
    result = bridge_candidate(_event(), candidate, _observation())
    assert result["bridge_status"] == "offscreen_requires_turn_reobserve"
    assert result["first_visible_subgoal"] is None


def test_unknown_and_route_conflict_are_rejected_before_visibility():
    result = bridge_candidate(
        _event(),
        _candidate(unknown_fraction=0.1, route_occ_conflict=True),
        _observation(),
    )
    assert result["bridge_status"] == "rejected_before_visibility"
    assert result["safety_reason"] == "route_occ_conflict"


def test_non_contiguous_route_path_is_not_bridgeable():
    result = bridge_candidate(
        _event(),
        _candidate(path=[[504, 500], [500, 500]]),
        _observation(),
    )
    assert result["bridge_status"] == "invalid_route_path"

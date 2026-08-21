import numpy as np

from scripts.eval.analyze_stage25_gt_detector import (
    compact_observation, cumulative_path_length, mine_events, route_revisit,
    semantic_cells, merge_geometry_intervals, resolve_episode_eval_seed,
)


def observation(index, x, action=1, collision=0, collision_delta=0):
    return {
        "record_index": index,
        "step_id": index,
        "observation_key": f"{index}:{index}",
        "pose": {"gps": [x, 0.0], "compass": [0.0]},
        "previous_action": action,
        "previous_action_applied": True,
        "occ_summary": {"occupied_added": 1, "free_added": 1},
        "audit_metrics": {
            "collision_count": collision,
            "collision_delta": collision_delta,
            "distance_to_goal": 5.0,
        },
    }


def test_forward_not_realized_is_geometry_candidate():
    rows = [observation(index, 0.0) for index in range(10)]
    result = mine_events(rows, [], [])
    assert any("commanded_forward_not_realized" in event["evidence"] for event in result["D1"])


def test_semantic_only_never_creates_event():
    rows = [observation(index, index * 0.25, action=2) for index in range(10)]
    semantic = [
        {"valid": True, "step_id": step, "class_surface_counts": {"door": 10}}
        for step in range(4)
    ]
    result = mine_events(rows, [], semantic)
    assert result["D2"] == []
    assert result["D3Q_confirmed"] == []


def test_future_motion_labels_self_recovery_without_changing_onset():
    rows = [observation(index, 0.0) for index in range(9)]
    rows += [observation(index, (index - 8) * 0.1) for index in range(9, 18)]
    result = mine_events(rows, [], [])
    assert result["D1"][0]["step_id"] <= 7
    assert result["D1"][0]["recoverability_proxy"] == "self_recovered_quick"


def test_route_revisit_requires_causal_low_progress_confirmation():
    positions = [
        0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50,
        1.25, 1.0, 0.75, 0.50, 0.25, 0.0,
    ]
    rows = [observation(index, x, action=2) for index, x in enumerate(positions)]
    rows += [observation(index, 0.0, action=2) for index in range(13, 26)]
    for row in rows:
        row["occ_summary"].update({
            "occupied_voxel_count": 100, "free_voxel_count": 200,
        })
    result = mine_events(
        rows, [], [], route_confirm_min_steps=8, route_confirm_max_steps=12
    )
    assert len(result["D2_raw_revisit"]) >= 1
    assert len(result["D2"]) == 1
    assert result["D2"][0]["signal_step"] == 12
    assert result["D2"][0]["confirmation_delay_steps"] == 8


def test_normal_route_revisit_is_not_confirmed_after_progress():
    positions = [
        0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50,
        1.25, 1.0, 0.75, 0.50, 0.25, 0.0,
    ]
    positions += [0.25 * (index - 12) for index in range(13, 28)]
    rows = [observation(index, x, action=1) for index, x in enumerate(positions)]
    for index, row in enumerate(rows):
        row["occ_summary"].update({
            "occupied_voxel_count": 100 + index * 100,
            "free_voxel_count": 200 + index * 100,
        })
    result = mine_events(rows, [], [])
    assert any(
        event["event_family"] == "G3_route_topology"
        for event in result["D2_raw_revisit"]
    )
    assert not any(
        event["event_family"] == "G3_route_topology"
        for event in result["D2"]
    )


def test_spatial_semantic_recurrence_confirms_but_never_triggers():
    rows = [observation(index, 0.0, action=1) for index in range(12)]
    cells = ["0:1:2:3", "8:2:2:3"]
    semantic = [
        {"valid": True, "step_id": step, "spatial_semantic_cells": cells}
        for step in (0, 2, 4, 6)
    ]
    result = mine_events(rows, [], semantic)
    assert result["D3Q_confirmed"]
    moving = [observation(index, index * 0.25, action=2) for index in range(12)]
    assert mine_events(moving, [], semantic)["D3Q_confirmed"] == []


def test_semantic_cells_require_support_and_confidence():
    points = np.asarray([[0.1, 0.1, 0.1]] * 8 + [[1.1, 0.1, 0.1]] * 7)
    classes = np.zeros(15, dtype=np.int16)
    confidence = np.asarray([0.5] * 8 + [0.9] * 7, dtype=np.float32)
    assert semantic_cells(points, classes, confidence) == ["0:0:0:0"]


def test_delayed_recovery_is_separate_from_persistent_episode():
    rows = [observation(index, 0.0) for index in range(45)]
    rows += [observation(index, 0.7) for index in range(45, 50)]
    result = mine_events(rows, [], [])
    assert result["D1"][0]["recoverability_proxy"] == "self_recovered_delayed"


def test_short_terminal_horizon_is_not_labeled_persistent():
    rows = [observation(index, 0.0, collision=index, collision_delta=1) for index in range(20)]
    result = mine_events(rows, [], [])
    assert result["D1"][0]["recoverability_proxy"] == "episode_ended"


def test_confirmed_revisit_is_merged_until_region_departure():
    positions = [
        0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50,
        1.25, 1.0, 0.75, 0.50, 0.25, 0.0,
    ] + [0.0] * 60
    rows = [observation(index, x, action=2) for index, x in enumerate(positions)]
    for row in rows:
        row["occ_summary"].update({
            "occupied_voxel_count": 100, "free_voxel_count": 200,
        })
    result = mine_events(rows, [], [])
    route_events = [
        event for event in result["D2"]
        if event["event_family"] == "G3_route_topology"
    ]
    assert len(route_events) == 1


def test_cumulative_route_length_preserves_revisit_result():
    positions = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.0, 0.5, 0.0]
    rows = [compact_observation(observation(index, x)) for index, x in enumerate(positions)]
    original = route_revisit(rows, len(rows) - 1, min_gap=4)
    optimized = route_revisit(
        rows, len(rows) - 1, min_gap=4,
        cumulative_path_m=cumulative_path_length(rows),
    )
    assert optimized == original


def test_continuous_geometry_evidence_becomes_one_interval():
    rows = [observation(index, 0.0, collision=index, collision_delta=1) for index in range(18)]
    result = mine_events(rows, [], [])
    geometry = [event for event in result["D1"] if event["event_family"] == "G1_geometry_execution"]
    assert len(geometry) == 1
    assert geometry[0]["step_id"] == 1
    assert geometry[0]["end_step"] == 17
    assert geometry[0]["support_count"] == 17


def test_geometry_intervals_split_after_gap_or_departure():
    base = {
        "event_family": "G1_geometry_execution", "evidence": ["collision"],
        "semantic_confirmation": {"supports_existing_suspicion": False},
        "recoverability_proxy": "persistent_episode",
    }
    events = [
        {**base, "step_id": 2, "position": [0.0, 0.0]},
        {**base, "step_id": 3, "position": [0.0, 0.0]},
        {**base, "step_id": 9, "position": [0.0, 0.0]},
        {**base, "step_id": 10, "position": [1.0, 0.0]},
    ]
    assert len(merge_geometry_intervals(events)) == 3


def test_episode_seed_prefers_meta_and_falls_back_to_progress():
    assert resolve_episode_eval_seed(
        {"episode_eval_seed": 41}, {"episode_eval_seed": 42}
    ) == (41, "episode_meta")
    assert resolve_episode_eval_seed(
        {"episode_eval_seed": None}, {"episode_eval_seed": 42}
    ) == (42, "progress_fallback")
    assert resolve_episode_eval_seed({}, {}) == (None, "missing")


def test_geometry_thresholds_are_explicit_and_default_compatible():
    rows = [
        observation(index, 0.0, collision=index, collision_delta=1)
        for index in range(8)
    ]
    default = mine_events(rows, [], [])
    strict = mine_events(
        rows, [], [], collision_burst_min=4, geometry_max_displacement_m=0.10
    )
    assert len(default["D1"]) == 1
    assert len(strict["D1"]) == 1
    assert default["D1"][0]["event_family"] == "G1_geometry_execution"


def test_executed_near_full_rotation_is_strict_separate_variant():
    rows = [observation(index, 0.0, action=3) for index in range(26)]
    for index, item in enumerate(rows):
        item["pose"]["compass"] = [np.deg2rad(index * 15.0)]
    result = mine_events(rows, [], [])
    assert result["D2"] == []
    rotation = [
        event for event in result["D2_executed_rotation"]
        if event["event_family"] == "G2_executed_rotation_loop"
    ]
    assert len(rotation) == 1
    assert rotation[0]["window"]["executed_rotation_degrees"] >= 345.0
    assert rotation[0]["window"]["executed_rotation_displacement_m"] == 0.0


def test_partial_scan_or_translating_turns_are_not_executed_rotation_loop():
    partial = [observation(index, 0.0, action=3) for index in range(14)]
    for index, item in enumerate(partial):
        item["pose"]["compass"] = [np.deg2rad(index * 15.0)]
    assert mine_events(partial, [], [])["D2_executed_rotation"] == []
    moving = [observation(index, index * 0.05, action=3) for index in range(26)]
    for index, item in enumerate(moving):
        item["pose"]["compass"] = [np.deg2rad(index * 15.0)]
    assert mine_events(moving, [], [])["D2_executed_rotation"] == []

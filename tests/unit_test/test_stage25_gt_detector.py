import numpy as np

from scripts.eval.analyze_stage25_gt_detector import mine_events, semantic_cells


def observation(index, x, action=1, collision=0, collision_delta=0):
    return {
        "record_index": index,
        "step_id": index,
        "observation_key": f"{index}:{index}",
        "pose": {"gps": [x, 0.0]},
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
    assert result["D1"][0]["recoverability_proxy"] == "self_recovered"


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

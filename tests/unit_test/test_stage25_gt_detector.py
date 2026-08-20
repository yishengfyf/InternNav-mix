from scripts.eval.analyze_stage25_gt_detector import mine_events


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

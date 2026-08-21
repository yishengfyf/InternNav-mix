from internnav.utils.stage25_event_gt_review import merge_windows, mine_review_windows


def row(step, x, goal):
    return {"step_id": step, "gps": [x, 0.0], "distance_to_goal": goal}


def test_stationary_window_is_review_candidate():
    rows = [row(step, 0.0, 5.0) for step in range(20)]
    result = mine_review_windows(rows, stall_window=8)
    assert len(result) == 1
    assert result[0]["review_family"] == "offline_local_stagnation"
    assert result[0]["uses_future_for_review_only"] is True


def test_moving_away_is_wrong_way_not_local_stagnation():
    rows = [row(step, step * 0.25, 5.0 + step * 0.10) for step in range(40)]
    result = mine_review_windows(rows, regression_window=16)
    assert {item["review_family"] for item in result} == {"offline_wrong_way_progress"}


def test_normal_progress_is_not_review_candidate():
    rows = [row(step, step * 0.25, 8.0 - step * 0.10) for step in range(40)]
    assert mine_review_windows(rows, regression_window=16) == []


def test_merged_window_uses_full_interval_and_latest_evidence_step():
    merged = merge_windows([
        {
            "review_family": "offline_local_stagnation",
            "onset_step": 1, "end_step": 8, "step_id": 8,
            "path_length_m": 0.1,
        },
        {
            "review_family": "offline_local_stagnation",
            "onset_step": 2, "end_step": 12, "step_id": 12,
            "path_length_m": 0.2,
        },
    ])
    assert len(merged) == 1
    assert merged[0]["end_step"] == 12
    assert merged[0]["step_id"] == 12
    assert merged[0]["duration_steps"] == 12
    assert merged[0]["path_length_m"] == 0.2

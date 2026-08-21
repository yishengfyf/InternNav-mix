from internnav.utils.stage25_event_gt_review import (
    action_interval_summary, merge_windows, mine_review_windows,
)


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


def test_interleaved_families_do_not_prevent_same_family_merge():
    merged = merge_windows([
        {
            "review_family": "offline_local_stagnation",
            "onset_step": 1, "end_step": 8, "step_id": 8,
        },
        {
            "review_family": "offline_wrong_way_progress",
            "onset_step": 2, "end_step": 9, "step_id": 9,
        },
        {
            "review_family": "offline_local_stagnation",
            "onset_step": 3, "end_step": 10, "step_id": 10,
        },
    ])
    assert len(merged) == 2
    local = next(
        row for row in merged
        if row["review_family"] == "offline_local_stagnation"
    )
    assert local["onset_step"] == 1
    assert local["end_step"] == 10
    assert local["support_count"] == 2


def test_action_interval_summary_uses_executed_interval_only():
    observations = [
        {
            **row(1, 0.0, 5.0), "compass": [0.0],
        },
        {
            **row(2, 0.0, 5.0), "compass": [0.2617993878],
        },
        {
            **row(3, 0.0, 5.2), "compass": [0.5235987756],
        },
    ]
    actions = [
        {
            "step_id": 0, "action": 1, "action_applied": True,
            "action_source": "outside", "audit_metrics": {},
        },
        {
            "step_id": 1, "action": 2, "action_applied": True,
            "action_source": "s2", "audit_metrics": {"collision_delta": 0},
        },
        {
            "step_id": 2, "action": 2, "action_applied": True,
            "action_source": "s2", "audit_metrics": {"collision_delta": 1},
        },
        {
            "step_id": 3, "action": 1, "action_applied": False,
            "action_source": "s2", "audit_metrics": {"collision_delta": 1},
        },
    ]
    summary = action_interval_summary(
        observations, actions, onset_step=1, end_step=3
    )
    assert summary["applied_action_count"] == 2
    assert summary["action_counts"]["left"] == 2
    assert summary["action_counts"]["forward"] == 0
    assert summary["collision_delta"] == 1.0
    assert abs(summary["total_abs_turn_deg"] - 30.0) < 1e-4
    assert abs(summary["goal_distance_delta_m"] - 0.2) < 1e-6

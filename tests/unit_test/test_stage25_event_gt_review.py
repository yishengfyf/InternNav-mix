from internnav.utils.stage25_event_gt_review import (
    action_interval_summary, annotate_review_candidates,
    evaluate_detector_against_gt_lite, intervals_overlap, merge_windows,
    mine_review_windows, objective_review_annotation, scene_split,
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


def review_candidate(family="offline_local_stagnation", scene="scene_a"):
    return {
        "review_family": family,
        "scene_id": scene,
        "episode_id": 1,
        "onset_step": 10,
        "end_step": 49,
        "step_id": 49,
        "duration_steps": 40,
        "displacement_m": 0.05,
        "path_length_m": 0.10 if family == "offline_local_stagnation" else 2.0,
        "goal_distance_increase_m": 1.5,
        "offline_action_audit": {
            "applied_action_count": 40,
            "action_counts": {
                "stop": 0, "forward": 0, "left": 24, "right": 16,
                "lookup": 0, "lookdown": 0,
            },
            "collision_delta": 0.0,
            "total_abs_turn_deg": 600.0,
            "turn_only_ratio": 1.0,
        },
        "outcome": {"success": 1.0, "steps": 100},
    }


def test_objective_rotation_stagnation_is_true_trap():
    annotation = objective_review_annotation(review_candidate())
    assert annotation["auto_status"] == "objective_confirmed"
    assert annotation["state"] == "true_trap"
    assert annotation["type"] == "G2_local_rotation_loop"
    assert annotation["recoverability"] == "self_recovered_delayed"


def test_wrong_way_is_confirmed_but_not_local_trap():
    annotation = objective_review_annotation(
        review_candidate("offline_wrong_way_progress")
    )
    assert annotation["auto_status"] == "objective_confirmed"
    assert annotation["state"] == "wrong_way_progress"
    assert annotation["recoverability"] == "not_a_local_trap"


def test_short_ambiguous_stagnation_abstains():
    candidate = review_candidate()
    candidate["duration_steps"] = 12
    candidate["offline_action_audit"]["applied_action_count"] = 12
    assert objective_review_annotation(candidate)["auto_status"] == "abstain"


def test_scene_split_is_deterministic_and_scene_disjoint():
    assert scene_split("scene_a") == scene_split("scene_a")
    rows = annotate_review_candidates([
        review_candidate(scene="scene_a"),
        {**review_candidate(scene="scene_a"), "episode_id": 2},
    ])
    assert len({row["split"] for row in rows}) == 1


def test_interval_overlap_requires_same_episode():
    first = review_candidate()
    detector = {"scene_id": "scene_a", "episode_id": 1, "step_id": 45}
    assert intervals_overlap(first, detector)
    assert not intervals_overlap(first, {**detector, "episode_id": 2})


def test_gt_lite_evaluation_reports_recall_not_precision():
    gt = annotate_review_candidates([
        review_candidate(),
        review_candidate("offline_wrong_way_progress", scene="scene_b"),
    ])
    detector = [{"scene_id": "scene_a", "episode_id": 1, "step_id": 45}]
    report = evaluate_detector_against_gt_lite(detector, gt)["all"]
    assert report["objective_true_trap_count"] == 1
    assert report["true_trap_recall"] == 1.0
    assert report["wrong_way_protection_rate"] == 1.0
    assert report["precision_status"].startswith("not_computed")

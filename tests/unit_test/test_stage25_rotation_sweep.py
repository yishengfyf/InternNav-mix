from internnav.utils.stage25_event_gt_review import annotate_review_candidates
from scripts.eval.sweep_stage25_executed_rotation import (
    evaluation_for_split, variant_rank,
)
from scripts.eval.finalize_stage25_event_gt_review import evaluate_final


def variant(recall, protection, events=10):
    return {
        "window_steps": 32,
        "minimum_degrees": 270.0,
        "minimum_turn_actions": 18,
        "maximum_displacement_m": 0.35,
        "median_confirmation_delay_steps": 31,
        "dev": {
            "true_trap_recall": recall,
            "wrong_way_protection_rate": protection,
            "detector_event_count": events,
        },
    }


def gt_candidate(scene, state):
    family = (
        "offline_local_stagnation"
        if state == "true_trap" else "offline_wrong_way_progress"
    )
    return {
        "review_family": family,
        "scene_id": scene,
        "episode_id": 1,
        "onset_step": 10,
        "end_step": 49,
        "step_id": 49,
        "duration_steps": 40,
        "displacement_m": 0.05,
        "path_length_m": 0.10 if state == "true_trap" else 2.0,
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


def test_rank_rejects_recall_gain_that_reduces_wrong_way_protection():
    baseline = {"wrong_way_protection_rate": 0.80}
    protected = variant(0.82, 0.80)
    risky = variant(0.95, 0.79)
    assert variant_rank(protected, baseline) < variant_rank(risky, baseline)


def test_evaluation_for_split_excludes_other_split_scenes():
    scenes = [f"scene_{index}" for index in range(100)]
    annotated = annotate_review_candidates([
        gt_candidate(scene, "true_trap") for scene in scenes
    ])
    dev_row = next(row for row in annotated if row["split"] == "dev")
    holdout_row = next(row for row in annotated if row["split"] == "holdout")
    events = [
        {
            "scene_id": dev_row["scene_id"], "episode_id": 1,
            "step_id": 40,
        },
        {
            "scene_id": holdout_row["scene_id"], "episode_id": 1,
            "step_id": 40,
        },
    ]
    report = evaluation_for_split(events, annotated, "dev")
    assert report["detector_event_count"] == 1
    assert report["detected_true_trap_count"] == 1


def test_final_audit_keeps_objective_and_visual_sources_separate():
    objective = annotate_review_candidates([
        gt_candidate("scene_a", "true_trap"),
        gt_candidate("scene_b", "wrong_way_progress"),
    ])
    events = [
        {"scene_id": "scene_a", "episode_id": 1, "step_id": 40,
         "event_family": "G2_policy_loop"},
        {"scene_id": "scene_c", "episode_id": 2, "step_id": 20,
         "event_family": "G1_geometry_execution"},
        {"scene_id": "scene_d", "episode_id": 3, "step_id": 30,
         "event_family": "G1_geometry_execution"},
    ]
    visual = [
        {**events[1], "state": "true_trap"},
        {**events[2], "state": "hesitation"},
    ]
    report = evaluate_final(events, objective, visual)["all"]
    assert report["event_adjudication_counts"] == {
        "objective_true_trap": 1,
        "visual_true_trap": 1,
        "visual_hesitation": 1,
    }
    assert report["event_precision_on_adjudicated"] == 2 / 3
    assert report["combined_confirmed_true_trap_count"] == 2
    assert report["combined_confirmed_recall"] == 1.0

import json

from internnav.utils.stage63_adaptive_reobserve import plan_adaptive_view_sweep
from scripts.eval.analyze_stage63_adaptive_reobserve import analyze


def test_view_sweep_has_budget_entry_center_and_overscan() -> None:
    probes = plan_adaptive_view_sweep(
        100.0,
        hfov_deg=79.0,
        turn_angle_deg=15.0,
        primitive_budgets=(1, 2, 4),
        max_turn_steps=12,
    )
    by_arm = {probe["arm"]: probe for probe in probes}
    assert by_arm["budget_1"]["planned_yaw_delta_deg"] == 15.0
    assert by_arm["budget_2"]["planned_yaw_delta_deg"] == 30.0
    assert by_arm["budget_4"]["planned_yaw_delta_deg"] == 60.0
    assert by_arm["fov_entry"]["turn_steps"] == 5
    assert by_arm["path_center"]["turn_steps"] == 7
    assert by_arm["path_center_overscan"]["turn_steps"] == 8
    assert all(probe["action_applied"] is False for probe in probes)


def test_view_sweep_deduplicates_equal_yaws() -> None:
    probes = plan_adaptive_view_sweep(
        -20.0,
        hfov_deg=79.0,
        turn_angle_deg=15.0,
        primitive_budgets=(1, 2, 4),
        max_turn_steps=4,
    )
    steps = [probe["turn_steps"] for probe in probes]
    assert len(steps) == len(set(steps))
    assert all(probe["planned_yaw_delta_deg"] <= 0.0 for probe in probes)
    aliases = {alias for probe in probes for alias in probe["arm_aliases"]}
    assert {"budget_1", "budget_2", "budget_4", "fov_entry", "path_center"} <= aliases


def test_view_sweep_rejects_invalid_sensor_contract() -> None:
    assert plan_adaptive_view_sweep(
        90.0,
        hfov_deg=0.0,
        turn_angle_deg=15.0,
    ) == []


def _write_adaptive_event(tmp_path, adaptive):
    event_dir = tmp_path / "vlmap_safety_debug" / "rank0_run_001"
    event_dir.mkdir(parents=True)
    event = {
        "scene_id": "scene",
        "episode_id": 1,
        "trigger_step": 2,
        "stage59_productive_onset": {
            "anchors": [{
                "anchor": "last_productive_pre_loop",
                "stage63_adaptive_reobserve": adaptive,
            }],
        },
    }
    (event_dir / "s2_loop_path_reobserve_active_events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )


def test_analyzer_accepts_missing_bearing_noop_from_existing_run(tmp_path) -> None:
    _write_adaptive_event(tmp_path, {
        "shadow_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "official_memory_mutated": False,
        "sim_pose_all_restored": False,
        "probes": [],
        "reason": "missing_retreat_bearing",
        "gt_fields_used": [],
    })
    report = analyze(tmp_path)
    assert report["integrity_passed"] is True
    assert report["no_probe_event_count"] == 1
    assert report["records"][0]["no_probe_reason"] == "missing_retreat_bearing"


def test_analyzer_rejects_unexplained_no_probe_event(tmp_path) -> None:
    _write_adaptive_event(tmp_path, {
        "shadow_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "official_memory_mutated": False,
        "sim_pose_all_restored": False,
        "probes": [],
        "reason": "observation_failed",
        "gt_fields_used": [],
    })
    assert analyze(tmp_path)["integrity_passed"] is False

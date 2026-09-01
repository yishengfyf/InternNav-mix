from internnav.utils.stage63_adaptive_reobserve import plan_adaptive_view_sweep


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

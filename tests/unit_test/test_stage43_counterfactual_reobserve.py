import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage43_counterfactual_reobserve.py"
_spec = importlib.util.spec_from_file_location("stage43_counterfactual_reobserve", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_plan_turns_back_target_into_real_hfov() -> None:
    report = _module.plan_bounded_reorientation(
        -168.8, hfov_deg=79.0, turn_angle_deg=15.0, center_margin_deg=10.0
    )
    assert report["valid"] is True
    assert report["turn_direction"] == "right"
    assert report["turn_steps"] == 10
    assert abs(report["residual_bearing_deg"]) <= report["half_visible_deg"]
    assert report["action_applied"] is False


def test_plan_does_not_turn_already_visible_target() -> None:
    report = _module.plan_bounded_reorientation(
        12.0, hfov_deg=79.0, turn_angle_deg=15.0, center_margin_deg=10.0
    )
    assert report["valid"] is True
    assert report["turn_steps"] == 0
    assert report["planned_yaw_delta_deg"] == 0.0


def test_plan_abstains_when_budget_cannot_reveal_target() -> None:
    report = _module.plan_bounded_reorientation(
        180.0,
        hfov_deg=79.0,
        turn_angle_deg=15.0,
        center_margin_deg=10.0,
        max_turn_steps=4,
    )
    assert report["valid"] is False
    assert report["reason"] == "turn_budget_insufficient"
    assert report["action_emitted"] is False


def test_counterfactual_contract_requires_restored_pose_and_isolation() -> None:
    record = {
        "schema_version": _module.SCHEMA_VERSION,
        "shadow_only": True,
        "action_emitted": False,
        "action_applied": False,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "sim_pose_restored": True,
        "official_memory_mutated": False,
        "safety_authority": "temporary_current_sparseocc_reaudit",
    }
    assert _module.counterfactual_contract_ok(record)
    record["sim_pose_restored"] = False
    assert not _module.counterfactual_contract_ok(record)

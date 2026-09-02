"""Regression tests for the mainline/legacy VLMaps boundary."""

import importlib
from pathlib import Path


MODULE = importlib.import_module(
    "internnav.habitat_extensions.vln.habitat_vln_evaluator"
)


def _evaluator(*, legacy=False):
    value = object.__new__(MODULE.HabitatVLNEvaluator)
    value._legacy_vlmaps_enabled = legacy
    value.vlmap_semantic = None
    return value


def test_legacy_disabled_does_not_validate_or_modify_normal_actions():
    evaluator = _evaluator()
    action, changed, decision = evaluator._postprocess_habitat_action_with_vlmap_safety(
        1, {"gps": [0.0, 0.0], "compass": [0.0]}, None
    )
    assert (action, changed, decision) == (1, False, {})


def test_legacy_disabled_recovery_trajectory_fails_closed():
    evaluator = _evaluator()
    rejected, decision = evaluator._validate_local_actions_with_vlmap(
        [1], {"gps": [0.0, 0.0], "compass": [0.0]}, None,
        recovery_active=True,
    )
    assert rejected is True
    assert decision["reason"] == "legacy_vlmaps_disabled_mainline_preflight_required"
    assert decision["reject_required"] is True


def test_legacy_disabled_normal_trajectory_is_unchanged():
    evaluator = _evaluator()
    rejected, decision = evaluator._validate_local_actions_with_vlmap(
        [1], {"gps": [0.0, 0.0], "compass": [0.0]}, None,
        recovery_active=False,
    )
    assert rejected is False
    assert decision == {}


def test_legacy_waypoint_stage_declares_explicit_opt_in():
    cfg_path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "configs" / "habitat_dual_system_vlmap_stage68_native_vlmap_shadow_cfg.py"
    text = cfg_path.read_text(encoding="utf-8")
    assert 'legacy_vlmaps_experiment"] = True' in text
    assert 'legacy_vlmaps_enable"] = True' in text


def test_action_safety_wrapper_itself_stays_uninitialized_without_opt_in():
    safety = MODULE.VLMapActionSafety(
        {"enable": True, "debug": False, "legacy_vlmaps_experiment": False},
        MODULE.np.eye(3, dtype=MODULE.np.float32),
    )
    # The evaluator normalizes inherited config before constructing this
    # wrapper; this fixture documents the required normalized contract.
    assert safety.enabled is False
    assert safety.builder is None

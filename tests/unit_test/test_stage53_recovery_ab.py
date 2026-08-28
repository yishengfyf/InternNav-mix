import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "analyze_stage53_recovery_ab.py"
)
SPEC = importlib.util.spec_from_file_location("stage53_recovery_ab", SCRIPT_PATH)
stage53 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage53)


def _arm(variant):
    return {
        "variant": variant,
        "status": "ok",
        "prompt_image_binding_valid": True,
        "forbidden_context_terms": [],
        "hinted_protocol_valid": True,
        "hinted_valid": True,
        "hinted_reobserve": False,
        "continues_repeated_error_direction": False,
        "change_type": "valid_to_valid",
    }


def _write_run(tmp_path, event):
    run_root = tmp_path / "run"
    event_dir = run_root / "vlmap_safety_debug" / "rank0_run_001"
    event_dir.mkdir(parents=True)
    (run_root / "progress.json").write_text(
        json.dumps({"scene_id": "scene", "episode_id": 7}) + "\n",
        encoding="utf-8",
    )
    (event_dir / "stage53_recovery_ab_events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    return run_root


def _event():
    return {
        "arms": [
            _arm("control"),
            _arm("lookdown_only"),
            _arm("context_only"),
            _arm("lookdown_context"),
        ],
        "official_memory_mutated": False,
        "action_applied": False,
        "gt_fields_used": [],
        "lookdown_geometry": {
            "observation_readable": True,
            "temporary_update_valid": True,
            "report": {"valid": True, "eligible_count": 0},
        },
    }


def test_stage53_integrity_requires_complete_bound_four_arm_event(tmp_path):
    report = stage53.analyze(_write_run(tmp_path, _event()), expected_episodes=1)
    assert report["integrity_passed"] is True
    assert report["complete_four_arm_event_count"] == 1


def test_stage53_integrity_rejects_binding_and_context_claims(tmp_path):
    event = _event()
    event["arms"][1]["prompt_image_binding_valid"] = False
    event["arms"][2]["forbidden_context_terms"] = ["geometry_safe"]
    report = stage53.analyze(_write_run(tmp_path, event), expected_episodes=1)
    assert report["integrity_passed"] is False
    assert report["prompt_image_binding_violation_count"] == 1
    assert report["recovery_context_claim_violation_count"] == 1


def test_stage53_integrity_rejects_failed_temporary_map_update(tmp_path):
    event = _event()
    event["lookdown_geometry"]["temporary_update_valid"] = False
    report = stage53.analyze(_write_run(tmp_path, event), expected_episodes=1)
    assert report["integrity_passed"] is False


def test_stage53_v2_requires_view_note_only_on_lookdown_arms(tmp_path):
    event = _event()
    event["event_schema_version"] = "stage53_recovery_ab_v2"
    for arm in event["arms"]:
        arm["lookdown_view_prompt_included"] = arm["variant"] in {
            "lookdown_only",
            "lookdown_context",
        }
    report = stage53.analyze(_write_run(tmp_path, event), expected_episodes=1)
    assert report["integrity_passed"] is True
    event["arms"][0]["lookdown_view_prompt_included"] = True
    report = stage53.analyze(_write_run(tmp_path, event), expected_episodes=1)
    assert report["integrity_passed"] is False
    assert report["lookdown_view_prompt_violation_count"] == 1

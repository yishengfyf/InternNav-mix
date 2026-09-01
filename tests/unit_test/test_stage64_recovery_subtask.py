import importlib.util
import json
from pathlib import Path

from internnav.utils.stage64_recovery_subtask import (
    IMAGE_TOKEN,
    build_programmatic_recovery_prompt,
    build_self_authored_execution_prompt,
    build_self_authoring_prompt,
    is_valid_self_authored_instruction,
    plan_recovery_state_reset,
    sanitize_self_authored_instruction,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "analyze_stage64_recovery_subtask.py"
)
SPEC = importlib.util.spec_from_file_location("stage64_recovery_subtask", SCRIPT_PATH)
stage64 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage64)


def test_recovery_prompts_replace_original_task_and_bind_images() -> None:
    kwargs = {
        "image_count": 3,
        "bearing_deg": -45.0,
        "distance_m": 0.25,
        "failure_type": "turn_loop",
        "repeated_direction": "left",
    }
    programmatic = build_programmatic_recovery_prompt(**kwargs)
    authoring = build_self_authoring_prompt(**kwargs)
    execution = build_self_authored_execution_prompt(
        instruction="Turn toward the prior hallway and reacquire a visible point.",
        image_count=3,
        bearing_deg=-45.0,
        distance_m=0.25,
    )
    original = "Walk through the kitchen and stop beside the sofa."
    assert all(prompt.count(IMAGE_TOKEN) == 3 for prompt in (programmatic, authoring, execution))
    assert all(original not in prompt for prompt in (programmatic, authoring, execution))
    assert "do not pursue" in programmatic
    assert "untrusted" in execution


def test_self_authored_instruction_is_sanitized() -> None:
    value = sanitize_self_authored_instruction("  Go  left <image> then observe.  ", max_chars=18)
    assert IMAGE_TOKEN not in value
    assert value == "Go left then obser"
    assert is_valid_self_authored_instruction("Turn right and reacquire a visible hallway point.")
    assert not is_valid_self_authored_instruction("200 100")
    assert not is_valid_self_authored_instruction("←←")


def test_recovery_reset_plan_clears_all_stale_execution_state() -> None:
    plan = plan_recovery_state_reset(
        {
            "action_seq": [2],
            "local_actions": [1, 1],
            "vlmap_recovery_actions": [3],
            "pixel_goal": [100, 200],
            "traj_latents": "present",
            "output_ids": "present",
        }
    )
    assert plan["applied"] is False
    assert set(plan["would_clear_fields"]) == set(plan["after_if_applied"])
    assert plan["after_if_applied"]["action_seq"] == []
    assert plan["after_if_applied"]["pixel_goal"] is None


def _arm(variant: str, *, joint: bool = False) -> dict:
    return {
        "variant": variant,
        "status": "ok",
        "prompt_image_binding_valid": True,
        "protocol_valid": True,
        "pixel_valid": joint,
        "path_consistent": joint,
        "support_safe": True,
        "joint_eligible": joint,
        "direct_turn_direction": None,
        "is_stop": False,
        "output": "200 100" if joint else "←",
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
    }


def _event() -> dict:
    return {
        "scene_id": "scene",
        "episode_id": 7,
        "trigger_step": 40,
        "reason": "ok",
        "shadow_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "official_memory_mutated": False,
        "original_instruction_restored": True,
        "original_instruction_leaked_to_recovery_prompt": False,
        "recovery_messages_prefix_empty": True,
        "queue_state_unchanged": True,
        "queue_reset_plan": {"shadow_only": True, "applied": False},
        "self_authored_instruction_valid": True,
        "authoring_prompt_image_binding_valid": True,
        "arms": [
            _arm("control"),
            _arm("programmatic_subtask", joint=True),
            _arm("self_authored_subtask"),
        ],
    }


def _write_run(tmp_path: Path, event: dict) -> Path:
    run_root = tmp_path / "run"
    event_dir = run_root / "vlmap_safety_debug" / "rank0_run_001"
    event_dir.mkdir(parents=True)
    (run_root / "progress.json").write_text(
        json.dumps({"scene_id": "scene", "episode_id": 7}) + "\n",
        encoding="utf-8",
    )
    (event_dir / "stage64_recovery_subtask_events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    return run_root


def test_stage64_analyzer_accepts_complete_shadow_event(tmp_path: Path) -> None:
    report = stage64.analyze(_write_run(tmp_path, _event()), expected_episodes=1)
    assert report["integrity_passed"] is True
    assert report["complete_three_arm_event_count"] == 1
    assert report["arm_summary"]["programmatic_subtask"]["joint_eligible_count"] == 1


def test_stage64_analyzer_rejects_instruction_or_queue_mutation(tmp_path: Path) -> None:
    event = _event()
    event["original_instruction_restored"] = False
    event["queue_state_unchanged"] = False
    report = stage64.analyze(_write_run(tmp_path, event), expected_episodes=1)
    assert report["integrity_passed"] is False
    assert report["integrity_violations"] == [["scene", 7, 40]]

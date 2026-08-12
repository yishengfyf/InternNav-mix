import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "analyze_stage21_s2_action_loops.py"
SPEC = importlib.util.spec_from_file_location("stage21_s2_action_loop_audit", SCRIPT_PATH)
loop_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop_audit)


def test_loop_audit_requires_shadow_and_contains_no_gt_fields(tmp_path):
    run_root = tmp_path / "run"
    debug_dir = run_root / "vlmap_safety_debug" / "run_001"
    debug_dir.mkdir(parents=True)
    (run_root / "progress.json").write_text(
        json.dumps({"scene_id": "scene", "episode_id": 7, "success": 0.0}) + "\n",
        encoding="utf-8",
    )
    event = {
        "event_type": "s2_action_loop_shadow",
        "event_schema_version": "stage21a_s2_loop_v1",
        "scene_id": "scene",
        "episode_id": 7,
        "step_id": 54,
        "transition": "start",
        "shadow_only": True,
        "applied": False,
        "candidate": {"geometry_safe": True},
        "triage_tier": "adapter_candidate",
        "failure_type": "s2_turn_loop_obstructed",
        "gt_fields_used": [],
        "rgb_file": "s2_action_loop_snapshots/scene_7_step54_loop1.jpg",
    }
    snapshot_path = debug_dir / event["rgb_file"]
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"jpeg")
    (debug_dir / "s2_action_loop_events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    summary = loop_audit.build_audit(run_root, 1)
    assert summary["passed"] is True
    assert summary["gt_leakage_scan"]["passed"] is True
    assert summary["shadow_safety"]["passed"] is True


def test_loop_audit_allows_unsaved_snapshot_beyond_episode_budget(tmp_path):
    run_root = tmp_path / "run"
    debug_dir = run_root / "vlmap_safety_debug" / "run_001"
    debug_dir.mkdir(parents=True)
    (run_root / "progress.json").write_text(
        json.dumps({"scene_id": "scene", "episode_id": 7, "success": 0.0}) + "\n",
        encoding="utf-8",
    )
    event = {
        "event_type": "s2_action_loop_shadow",
        "event_schema_version": "stage21a_s2_loop_v1",
        "scene_id": "scene",
        "episode_id": 7,
        "step_id": 391,
        "transition": "start",
        "loop_index": 3,
        "shadow_only": True,
        "applied": False,
        "candidate": {"geometry_safe": True},
        "triage_tier": "abstain",
        "failure_type": "s2_turn_loop_semantic",
        "gt_fields_used": [],
        "rgb_snapshot_expected": False,
    }
    (debug_dir / "s2_action_loop_events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )

    summary = loop_audit.build_audit(run_root, 1)

    assert summary["passed"] is True
    assert summary["missing_rgb_snapshots"] == []

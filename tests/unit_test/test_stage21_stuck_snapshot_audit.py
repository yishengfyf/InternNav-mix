import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "analyze_stage21_stuck_snapshots.py"
)
SPEC = importlib.util.spec_from_file_location("stage21_stuck_snapshot_audit", SCRIPT_PATH)
snapshot_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_audit)


def test_snapshot_audit_checks_seed_and_source_fields(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    snapshot_dir = run_root / "vlmap_safety_debug" / "run_001" / "stuck_snapshots"
    snapshot_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "summary.json"
    manifest.write_text(
        json.dumps([{"scene_id": "scene_a", "episode_id": 7}]),
        encoding="utf-8",
    )
    (run_root / "progress.json").write_text(
        json.dumps(
            {
                "scene_id": "scene_a",
                "episode_id": 7,
                "success": 0.0,
                "spl": 0.0,
                "ne": 1.0,
                "steps": 64,
                "collision_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = {
        "scene_id": "scene_a",
        "episode_id": 7,
        "step_id": 32,
        "rgb_file": "scene_a_7_step32.jpg",
        "trigger_reasons": ["repeated_action"],
        "current_action": 2,
        "pre_safety_action": 2,
        "action_source": "nextdit_local_queue",
        "dominant_action_source": "nextdit_local_queue",
        "environment_step_applied": True,
        "dominant_action": 2,
        "dominant_action_ratio": 1.0,
        "pixel_goal": [10, 20],
        "pixel_goal_age_steps": 32,
        "last_s2_query_step": 0,
        "s2_query_age_steps": 32,
        "local_action_queue_length": 4,
        "system2_action_queue_length": 0,
        "episode_eval_seed": 123,
    }
    (snapshot_dir / "scene_a_7_step32.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    (snapshot_dir / "scene_a_7_step32.jpg").write_bytes(b"jpeg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--run-root",
            str(run_root),
            "--episode-manifest",
            str(manifest),
            "--output",
            str(output),
            "--expected-seed",
            "scene_a/7=123",
            "--require-all",
        ],
    )

    snapshot_audit.main()

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["coverage_rate"] == 1.0
    assert summary["seed_mismatches"] == []
    assert summary["snapshots"][0]["diagnostic_hypotheses"] == [
        "nextdit_local_queue_loop",
        "stale_pixel_goal",
        "stale_s2_decision",
    ]

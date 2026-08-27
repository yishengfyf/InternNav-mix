import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "analyze_stage43_counterfactual_reobserve.py"
_spec = importlib.util.spec_from_file_location("analyze_stage43", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_analyzer_counts_zero_to_nonzero_and_clear(tmp_path: Path) -> None:
    row = {
        "scene_id": "scene",
        "episode_id": 1,
        "step_id": 2,
        "pre_candidate_count": 0,
        "post_candidate_count_max": 1,
        "probes": [{
            "reason": "ok",
            "observation_readable": True,
            "sim_pose_restored": True,
            "official_memory_mutated": False,
            "action_emitted": False,
            "post_contracts": [{"contract": {
                "first_edge_depth_checked": True,
                "first_edge_depth_clear": True,
                "executor_eligible": True,
            }}],
        }],
        "shadow_only": True,
        "action_applied": False,
        "unknown_is_free": False,
        "gt_fields_used": [],
    }
    (tmp_path / "stage43_counterfactual_reobserve_events.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([row]), encoding="utf-8")
    report = _module.analyze(tmp_path, manifest)
    assert report["integrity_passed"] is True
    assert report["zero_to_nonzero_event_count"] == 1
    assert report["post_executor_eligible_count"] == 1

import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "analyze_stage41_executor_contract.py"
_spec = importlib.util.spec_from_file_location("stage41_analyzer", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_stage41_analyzer_requires_exact_safe_depth_contract(tmp_path):
    row = {
        "scene_id": "s", "episode_id": 1, "step_id": 2,
        "shadow_only": True, "action_applied": False, "unknown_is_free": False, "gt_fields_used": [],
        "contracts": [{
            "contract": {"executor_eligible": False, "depth_readable": True, "sensor_hfov_deg": 79.0, "first_edge_depth_checked": True, "all_edges_sparseocc_reaudited": True},
            "edge_audits": [{"sparseocc_safe": True, "unknown": False, "occupied": False}],
        }],
    }
    log = tmp_path / "stage41_executor_contract_events.jsonl"
    log.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"scene_id": "s", "episode_id": 1, "step_id": 2}]))
    report = _module.analyze(tmp_path, manifest)
    assert report["integrity_passed"] is True
    assert report["abstain_count"] == 1

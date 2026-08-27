import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "analyze_stage45_candidate_rejection_truth.py"
_spec = importlib.util.spec_from_file_location("stage45_analyzer", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_analyzer_requires_isolated_gt_and_safe_corridor(tmp_path):
    event = {
        "shadow_only": True,
        "action_applied": False,
        "unknown_is_free": False,
        "candidate_feature_gt_fields_used": [],
        "gt_fields_used": [],
        "offline_rejection_truth_audit": {
            "gt_used_for_navigation": False,
            "unknown_is_free": False,
            "audits": [{
                "valid": True,
                "gt_used_for_navigation": False,
                "complete_gt_safe_corridor": True,
                "route_occ_false_block_candidate": True,
                "floor_footprint_false_block_candidate": True,
                "sparse_2d_false_free_cell_count": 0,
                "floor_footprint_false_free_cell_count": 0,
            }],
        },
    }
    path = tmp_path / "rank0" / "stage27_m3_candidate_events.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    report = _module.analyze(tmp_path)
    assert report["integrity_passed"] is True
    assert report["stage46_certificate_gate"] is True

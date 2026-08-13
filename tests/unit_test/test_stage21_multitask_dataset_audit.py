import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train" / "audit_stage21_multitask_dataset.py"
SPEC = importlib.util.spec_from_file_location("stage21_multitask_audit", SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _row(split, preference="positive", recovery=False):
    labels = {
        "preference_vs_s2": preference,
        "short_horizon_executability_proxy": 0.75,
        "geometry_safe_target": True,
        "proxy_is_causal_success_label": False,
        "recovery_proxy_route_w0": 0.65,
        "recovery_proxy_class": "promising",
    }
    return {
        "identity": {"scene_id": f"scene_{split}", "episode_id": 1, "step_id": 1, "split": split},
        "online_inputs": {"candidate": {"candidate_type": "resilience_backtrack" if recovery else "semantic_frontier", "distance_m": 1.0}, "current_policy_candidate": {}},
        "offline_labels": labels,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_accepts_leakage_safe_minimal_dataset(tmp_path):
    summary = {
        "task": "stage21_candidate_recoverability_dataset",
        "split_audit": {"scene_overlap_count": 0},
        "candidate_recoverability_rows": {"gt_leakage_scan": {"passed": True}},
        "active_safety_check": {"passed": True},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for stem, recovery in (("progress_rows", False), ("safety_rows", False), ("recovery_proxy_rows", True)):
        _write(tmp_path / f"{stem}_train.jsonl", [_row("train", recovery=recovery)])
        _write(tmp_path / f"{stem}_val.jsonl", [_row("val", recovery=recovery)])
    result = audit_module.audit(tmp_path)
    assert result["passed"] is True
    assert result["feature_dim"] > 0

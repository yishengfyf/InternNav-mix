import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train" / "stage21_training_common.py"
SPEC = importlib.util.spec_from_file_location("stage21_training_common", SCRIPT_PATH)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def _row(candidate=None):
    return {
        "identity": {"scene_id": "scene", "episode_id": 1, "step_id": 2},
        "online_inputs": {
            "candidate": {
                "candidate_type": "resilience_backtrack",
                "direction_bucket": "back",
                "distance_m": 2.0,
                "anchor_visible_free_ratio": 0.9,
                "geometry_safe": True,
                "active_gate_safe": False,
                **(candidate or {}),
            },
            "current_policy_candidate": {"distance_m": 1.0},
            "triage_context": {"tier": "adapter_candidate"},
        },
        "offline_labels": {
            "proxy_is_causal_success_label": False,
            "recovery_proxy_route_w0": 0.7,
            "recovery_proxy_class": "promising",
        },
    }


def test_encoder_is_stable_and_excludes_hard_gate_fields():
    names = common.feature_names()
    values = common.encode_row(_row())
    assert len(names) == len(values)
    assert not any("geometry_safe" in name for name in names)
    assert not any("active_gate_safe" in name for name in names)
    assert "route_progress" not in " ".join(names)
    assert not any(name.endswith("::score") for name in names)
    assert not any("goal_progress_score" in name for name in names)
    assert not any("semantic_resilience_score" in name for name in names)


def test_online_leakage_audit_rejects_outcome_key():
    row = _row({"success": 1.0})
    hits = common.audit_online_row(row)
    assert any("success" in hit for hit in hits)


def test_recovery_target_is_explicitly_non_causal():
    score, auxiliary = common.task_target(_row(), "recovery")
    assert score == 0.7
    assert auxiliary == 1.0

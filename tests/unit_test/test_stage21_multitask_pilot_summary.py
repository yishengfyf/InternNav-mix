import importlib.util
import json
import math
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train" / "summarize_stage21_multitask_pilot.py"
SPEC = importlib.util.spec_from_file_location("stage21_pilot_summary", SCRIPT_PATH)
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def test_training_artifact_audit_accepts_frozen_offline_run(tmp_path):
    for name in ("best.pt", "feature_schema.json", "normalizer.json"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    (tmp_path / "TRAINING_SCOPE.txt").write_text(
        "offline structured scorer only\n"
        "Frozen S2/NextDiT: true\n"
        "Episode-time parameter update: false\n"
        "Active navigation: false\n",
        encoding="utf-8",
    )
    (tmp_path / "training_config.json").write_text(
        json.dumps({"seed": 21, "frozen_navigation": True, "active_navigation": False}),
        encoding="utf-8",
    )
    record = {
        "epoch": 1,
        "global_step": 30,
        "selection_score": 0.7,
        "train_eval": {
            "progress": {"pairwise_accuracy": 0.8},
            "safety": {"mae": 0.1},
            "recovery": {"mae": 0.2},
        },
        "val": {
            "progress": {"pairwise_accuracy": 0.7, "event_top1_positive": 0.6},
            "safety": {"mae": 0.15, "rmse": 0.2, "aux_accuracy": 0.8},
            "recovery": {"mae": 0.25, "rmse": 0.3, "aux_accuracy": 0.7},
        },
        "val_heuristics": {
            "progress_candidate_score": {"pairwise_accuracy": 0.5, "event_top1_positive": 0.4},
            "progress_intent_alignment": {"pairwise_accuracy": 0.6, "event_top1_positive": 0.5},
            "safety_low_revisit_risk": {"mae": 0.3, "rmse": 0.4},
            "recovery_open_score": {"mae": 0.4, "rmse": 0.5},
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps([record]), encoding="utf-8")

    result = summary_module.audit_training_dir(tmp_path, expected_epochs=1, minimum_steps=30)
    assert result["best_epoch"] == 1
    assert math.isclose(result["comparison"]["progress_pairwise_minus_candidate_score"], 0.2)
    assert math.isclose(result["comparison"]["recovery_mae_improvement_over_open_score"], 0.15)

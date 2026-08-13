#!/usr/bin/env python3
"""Validate and summarize a Stage21b offline smoke-to-pilot run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_SCOPE_LINES = {
    "Frozen S2/NextDiT: true",
    "Episode-time parameter update: false",
    "Active navigation: false",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    return True


def audit_training_dir(path: Path, expected_epochs: Optional[int] = None,
                       minimum_steps: int = 1) -> Dict[str, Any]:
    required = [
        "best.pt", "metrics.json", "TRAINING_SCOPE.txt", "training_config.json",
        "feature_schema.json", "normalizer.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{path}: missing artifacts: {missing}")
    history = _read_json(path / "metrics.json")
    config = _read_json(path / "training_config.json")
    scope = set((path / "TRAINING_SCOPE.txt").read_text(encoding="utf-8").splitlines())
    if not history:
        raise ValueError(f"{path}: empty metrics history")
    if expected_epochs is not None and len(history) != expected_epochs:
        raise ValueError(f"{path}: expected {expected_epochs} epochs, got {len(history)}")
    if int(history[-1].get("global_step", 0)) < minimum_steps:
        raise ValueError(f"{path}: fewer than {minimum_steps} optimizer steps")
    if not REQUIRED_SCOPE_LINES.issubset(scope):
        raise ValueError(f"{path}: training scope guard is incomplete")
    if config.get("frozen_navigation") is not True or config.get("active_navigation") is not False:
        raise ValueError(f"{path}: frozen/active scope mismatch")
    if not _finite_numbers(history):
        raise ValueError(f"{path}: non-finite metric detected")
    best = max(history, key=lambda row: float(row["selection_score"]))
    val = best["val"]
    train = best.get("train_eval", {})
    heuristics = best["val_heuristics"]
    progress = val["progress"]
    safety = val["safety"]
    recovery = val["recovery"]
    candidate = heuristics["progress_candidate_score"]
    intent = heuristics["progress_intent_alignment"]
    safety_base = heuristics.get("safety_low_revisit_risk", {})
    recovery_base = heuristics["recovery_open_score"]
    return {
        "path": str(path.resolve()),
        "epochs_completed": len(history),
        "global_step": int(history[-1]["global_step"]),
        "best_epoch": int(best["epoch"]),
        "selection_score": float(best["selection_score"]),
        "val": val,
        "train_eval": train,
        "heuristics": heuristics,
        "comparison": {
            "progress_pairwise_minus_candidate_score": progress["pairwise_accuracy"] - candidate["pairwise_accuracy"],
            "progress_pairwise_minus_intent_alignment": progress["pairwise_accuracy"] - intent["pairwise_accuracy"],
            "progress_top1_minus_candidate_score": progress["event_top1_positive"] - candidate["event_top1_positive"],
            "progress_top1_minus_intent_alignment": progress["event_top1_positive"] - intent["event_top1_positive"],
            "safety_mae_improvement_over_low_revisit_risk": safety_base.get("mae", float("nan")) - safety["mae"],
            "recovery_mae_improvement_over_open_score": recovery_base["mae"] - recovery["mae"],
        },
        "train_val_gap": {
            "progress_pairwise": train.get("progress", {}).get("pairwise_accuracy", float("nan")) - progress["pairwise_accuracy"],
            "safety_mae_val_minus_train": safety["mae"] - train.get("safety", {}).get("mae", float("nan")),
            "recovery_mae_val_minus_train": recovery["mae"] - train.get("recovery", {}).get("mae", float("nan")),
        },
    }


def _aggregate(rows: Iterable[Dict[str, Any]], path: List[str]) -> Dict[str, float]:
    values: List[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"mean": mean(values), "median": median(values), "min": min(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 37, 53])
    parser.add_argument("--expected-smoke-steps", type=int, default=30)
    parser.add_argument("--expected-pilot-epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    smoke = audit_training_dir(args.smoke_dir, minimum_steps=args.expected_smoke_steps)
    seeds: Dict[str, Dict[str, Any]] = {}
    for seed in args.seeds:
        run = audit_training_dir(
            args.pilot_root / f"seed_{seed}",
            expected_epochs=args.expected_pilot_epochs,
            minimum_steps=args.expected_pilot_epochs,
        )
        config = _read_json(args.pilot_root / f"seed_{seed}" / "training_config.json")
        if int(config["seed"]) != seed:
            raise ValueError(f"seed directory/config mismatch for {seed}")
        seeds[str(seed)] = run
    seed_rows = list(seeds.values())
    paths = {
        "progress_pairwise_accuracy": ["val", "progress", "pairwise_accuracy"],
        "progress_event_top1_positive": ["val", "progress", "event_top1_positive"],
        "safety_mae": ["val", "safety", "mae"],
        "safety_rmse": ["val", "safety", "rmse"],
        "safety_aux_accuracy": ["val", "safety", "aux_accuracy"],
        "recovery_mae": ["val", "recovery", "mae"],
        "recovery_rmse": ["val", "recovery", "rmse"],
        "recovery_aux_accuracy": ["val", "recovery", "aux_accuracy"],
        "progress_pairwise_minus_candidate_score": ["comparison", "progress_pairwise_minus_candidate_score"],
        "progress_pairwise_minus_intent_alignment": ["comparison", "progress_pairwise_minus_intent_alignment"],
        "recovery_mae_improvement_over_open_score": ["comparison", "recovery_mae_improvement_over_open_score"],
    }
    result = {
        "task": "stage21b_multitask_scorer_offline_pilot_summary",
        "passed": True,
        "scope": {
            "offline_training_only": True,
            "frozen_s2_nextdit": True,
            "habitat_started": False,
            "episode_time_parameter_updates": False,
            "active_recovery": False,
        },
        "smoke": smoke,
        "seeds": seeds,
        "aggregate": {name: _aggregate(seed_rows, path) for name, path in paths.items()},
        "interpretation_guard": (
            "Artifact/audit pass means the offline pipeline completed. It does not establish navigation benefit; "
            "recovery validation has only 44 proxy rows and is not a causal success label."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

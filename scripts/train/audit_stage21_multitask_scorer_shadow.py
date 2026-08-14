#!/usr/bin/env python3
"""Offline audit for a frozen Stage21b multi-head scorer.

This evaluates a trained checkpoint on the scene-held-out JSONL validation
rows.  It never launches Habitat and never changes the policy.  The output is
intended to decide whether a learned scorer is safe to attach in shadow mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage21_training_common import (  # noqa: E402
    encode_row,
    feature_names,
    heuristic_score,
    iter_jsonl,
    task_target,
)
from train_stage21_multitask_scorer import (  # noqa: E402
    Stage21MultiHeadScorer,
)

import torch  # noqa: E402


TASKS = ("progress", "safety", "recovery")
TASK_FILES = {
    "progress": "progress_rows",
    "safety": "safety_rows",
    "recovery": "recovery_proxy_rows",
}


def _event_key(row: Mapping[str, Any]) -> str:
    identity = row.get("identity") or {}
    return "/".join(str(identity.get(key)) for key in ("scene_id", "episode_id", "step_id"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read_rows(data_dir: Path, task: str, split: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(data_dir / f"{TASK_FILES[task]}_{split}.jsonl"))


def _load_model(checkpoint_dir: Path, device: torch.device):
    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("training_config") or json.loads(
        (checkpoint_dir / "training_config.json").read_text(encoding="utf-8")
    )
    schema = checkpoint.get("feature_schema") or json.loads(
        (checkpoint_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    normalizer = checkpoint.get("normalizer") or json.loads(
        (checkpoint_dir / "normalizer.json").read_text(encoding="utf-8")
    )
    names = list(schema.get("feature_names") or feature_names())
    if names != feature_names():
        raise ValueError("Checkpoint feature schema does not match current encoder")
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    if len(mean) != len(names) or len(std) != len(names) or np.any(std <= 0.0):
        raise ValueError("Invalid checkpoint normalizer")
    model = Stage21MultiHeadScorer(
        len(names), int(config.get("hidden_dim", 128)), float(config.get("dropout", 0.10))
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, names, mean, std, config


def _matrix(rows: Sequence[Mapping[str, Any]], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray([encode_row(row) for row in rows], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite encoded feature")
    return (values - mean) / std


@torch.no_grad()
def _predict(model, matrix: np.ndarray, device: torch.device, batch_size: int = 512) -> Dict[str, np.ndarray]:
    outputs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for start in range(0, len(matrix), batch_size):
        batch = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
        raw = model(batch)
        for name, value in raw.items():
            # Progress is a ranking logit (sigmoid is monotonic), while the
            # scalar/auxiliary heads are audited in probability space to match
            # the training metrics and proxy targets in [0, 1].
            tensor = value if name == "progress" else torch.sigmoid(value)
            outputs[name].append(tensor.detach().cpu().numpy())
    return {name: np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)
            for name, values in outputs.items()}


def _progress_metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        target, _ = task_target(row, "progress")
        if target == 0.5:
            continue
        grouped[_event_key(row)].append((float(score), target))
    pair_count = pair_correct = event_count = top1 = 0
    for values in grouped.values():
        positive = [score for score, target in values if target > 0.5]
        negative = [score for score, target in values if target < 0.5]
        if not positive or not negative:
            continue
        pair_count += len(positive) * len(negative)
        pair_correct += sum(score > other for score in positive for other in negative)
        event_count += 1
        top1 += int(max(values, key=lambda item: item[0])[1] > 0.5)
    return {
        "pairwise_accuracy": pair_correct / max(1, pair_count),
        "event_top1_positive": top1 / max(1, event_count),
        "pair_count": float(pair_count),
        "event_count": float(event_count),
    }


def _scalar_metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], task: str,
                    heuristic_name: str) -> Dict[str, float]:
    targets = [task_target(row, task)[0] for row in rows]
    auxiliaries = [task_target(row, task)[1] for row in rows]
    errors = [float(score) - target for score, target in zip(scores, targets)]
    predictions = [int(float(score) >= 0.5) for score in scores]
    labels = [int(value >= 0.5) for value in auxiliaries]
    heuristic_errors = [heuristic_score(row, task, heuristic_name) - target
                        for row, target in zip(rows, targets)]
    return {
        "mae": sum(abs(value) for value in errors) / max(1, len(errors)),
        "rmse": math.sqrt(sum(value * value for value in errors) / max(1, len(errors))),
        "aux_accuracy": sum(pred == label for pred, label in zip(predictions, labels)) / max(1, len(labels)),
        "aux_positive_rate": sum(labels) / max(1, len(labels)),
        "heuristic_mae": sum(abs(value) for value in heuristic_errors) / max(1, len(errors)),
        "examples": float(len(rows)),
    }


def _group_key(row: Mapping[str, Any], group: str) -> str:
    identity = row.get("identity") or {}
    candidate = (row.get("online_inputs") or {}).get("candidate") or {}
    if group == "scene":
        return str(identity.get("scene_id") or "unknown")
    if group == "candidate_type":
        return str(candidate.get("candidate_type") or candidate.get("source") or "unknown")
    if group == "direction":
        return str(candidate.get("direction_bucket") or "unknown")
    if group == "tier":
        return str(((row.get("online_inputs") or {}).get("triage_context") or {}).get("tier") or "unknown")
    raise ValueError(group)


def _group_metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], task: str,
                   heuristic_name: str, group: str) -> Dict[str, Any]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_group_key(row, group)].append(index)
    result: Dict[str, Any] = {}
    for key, indices in sorted(grouped.items()):
        subset_rows = [rows[index] for index in indices]
        subset_scores = [scores[index] for index in indices]
        if task == "progress":
            metrics = _progress_metrics(subset_rows, subset_scores)
        else:
            metrics = _scalar_metrics(subset_rows, subset_scores, task, heuristic_name)
        metrics["rows"] = len(indices)
        result[key] = metrics
    return result


def _ablation_mask(names: Sequence[str], name: str) -> np.ndarray:
    def has_any(feature: str, tokens: Sequence[str]) -> bool:
        return any(token in feature for token in tokens)

    mask = np.zeros(len(names), dtype=bool)
    for index, feature in enumerate(names):
        if name == "no_delta" and feature.startswith("delta::"):
            mask[index] = True
        elif name == "no_current_context" and (feature.startswith("current_s2::") or feature.startswith("current_s2_present::") or feature.startswith("delta::")):
            mask[index] = True
        elif name == "no_candidate_context" and (feature.startswith("candidate::") or feature.startswith("candidate_present::") or feature.startswith("candidate_bool::") or feature.startswith("delta::")):
            mask[index] = True
        elif name == "no_anchor_semantics" and has_any(feature, ("anchor_", "semantic_", "landmark", "instruction", "target_frontier", "completed_landmark")):
            mask[index] = True
        elif name == "no_type_direction" and (feature.startswith("candidate_type=") or feature.startswith("direction_")):
            mask[index] = True
    return mask


def _shadow_disagreement(rows: Sequence[Mapping[str, Any]], learned: Sequence[float]) -> Dict[str, Any]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_event_key(row)].append(index)
    changed_candidate = changed_intent = 0
    examples = []
    for event_key, indices in sorted(grouped.items()):
        if len(indices) < 2:
            continue
        candidate_best = max(indices, key=lambda index: heuristic_score(rows[index], "progress", "candidate_score"))
        intent_best = max(indices, key=lambda index: heuristic_score(rows[index], "progress", "intent_alignment"))
        learned_best = max(indices, key=lambda index: learned[index])
        if learned_best != candidate_best:
            changed_candidate += 1
        if learned_best != intent_best:
            changed_intent += 1
        if learned_best != candidate_best or learned_best != intent_best:
            examples.append({
                "event_key": event_key,
                "learned_index": learned_best,
                "candidate_score_index": candidate_best,
                "intent_alignment_index": intent_best,
                "learned_candidate_type": _group_key(rows[learned_best], "candidate_type"),
                "learned_direction": _group_key(rows[learned_best], "direction"),
            })
    return {
        "event_count": len(grouped),
        "learned_vs_candidate_score_changed_events": changed_candidate,
        "learned_vs_intent_alignment_changed_events": changed_intent,
        "learned_vs_candidate_score_change_rate": changed_candidate / max(1, len(grouped)),
        "learned_vs_intent_alignment_change_rate": changed_intent / max(1, len(grouped)),
        "examples": examples[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, names, mean, std, config = _load_model(args.checkpoint_dir, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = {task: _read_rows(args.data_dir, task, "val") for task in TASKS}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    matrices: Dict[str, np.ndarray] = {}
    for task, rows in validation.items():
        matrices[task] = _matrix(rows, mean, std)
        predictions[task] = _predict(model, matrices[task], device)

    metrics: Dict[str, Any] = {}
    heuristic_names = {"progress": "candidate_score", "safety": "low_revisit_risk", "recovery": "open_score"}
    for task, rows in validation.items():
        learned = predictions[task]
        if task == "progress":
            metrics[task] = {"overall": _progress_metrics(rows, learned["progress"]),
                             "by_scene": _group_metrics(rows, learned["progress"], task, heuristic_names[task], "scene"),
                             "by_candidate_type": _group_metrics(rows, learned["progress"], task, heuristic_names[task], "candidate_type"),
                             "by_direction": _group_metrics(rows, learned["progress"], task, heuristic_names[task], "direction"),
                             "by_tier": _group_metrics(rows, learned["progress"], task, heuristic_names[task], "tier"),
                             "disagreement": _shadow_disagreement(rows, learned["progress"])}
        else:
            metrics[task] = {"overall": _scalar_metrics(rows, learned[task], task, heuristic_names[task]),
                             "by_scene": _group_metrics(rows, learned[task], task, heuristic_names[task], "scene"),
                             "by_candidate_type": _group_metrics(rows, learned[task], task, heuristic_names[task], "candidate_type"),
                             "by_direction": _group_metrics(rows, learned[task], task, heuristic_names[task], "direction"),
                             "by_tier": _group_metrics(rows, learned[task], task, heuristic_names[task], "tier")}

    ablation_names = ("no_delta", "no_current_context", "no_candidate_context", "no_anchor_semantics", "no_type_direction")
    ablations: Dict[str, Any] = {}
    for ablation in ablation_names:
        mask = _ablation_mask(names, ablation)
        ablation_metrics: Dict[str, Any] = {"zeroed_feature_count": int(mask.sum())}
        for task, rows in validation.items():
            ablated_matrix = matrices[task].copy()
            ablated_matrix[:, mask] = 0.0
            output = _predict(model, ablated_matrix, device)
            if task == "progress":
                ablation_metrics[task] = _progress_metrics(rows, output["progress"])
            else:
                ablation_metrics[task] = _scalar_metrics(rows, output[task], task, heuristic_names[task])
        ablations[ablation] = ablation_metrics

    prediction_path = args.output_dir / "val_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for task, rows in validation.items():
            for index, row in enumerate(rows):
                candidate = (row.get("online_inputs") or {}).get("candidate") or {}
                item = {
                    "task": task,
                    "identity": row.get("identity"),
                    "candidate_type": candidate.get("candidate_type") or candidate.get("source") or "unknown",
                    "direction": candidate.get("direction_bucket") or "unknown",
                    "tier": ((row.get("online_inputs") or {}).get("triage_context") or {}).get("tier") or "unknown",
                    "target": task_target(row, task),
                    "learned": {name: float(values[index]) for name, values in predictions[task].items()},
                    "heuristics": {name: float(heuristic_score(row, task, name)) for name in (
                        ("candidate_score", "intent_alignment") if task == "progress" else
                        (("low_revisit_risk",) if task == "safety" else ("open_score", "anchor_free_ratio"))
                    )},
                }
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

    result = {
        "task": "stage21b_frozen_multitask_scorer_offline_shadow_audit",
        "passed": True,
        "scope": {"offline_only": True, "habitat_started": False, "frozen_s2_nextdit": True,
                   "episode_time_parameter_update": False, "active_recovery": False},
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "device": str(device),
        "feature_dim": len(names),
        "validation_rows": {task: len(rows) for task, rows in validation.items()},
        "metrics": metrics,
        "feature_family_ablation": ablations,
        "interpretation_guard": "Proxy-head performance is not causal navigation recovery; progress validation is internal train-split scene-held-out data, not official R2R val-unseen.",
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

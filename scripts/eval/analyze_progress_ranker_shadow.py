"""Offline Stage17c shadow audit for progress and resilience-aware reranking.

This tool does not run Habitat and never changes navigation actions.  It loads
an already-trained ``ProgressCandidateRanker`` and compares its hypothetical
selection with the existing candidate-score and target-frontier heuristics.

The ``ranker_resilience`` and ``ranker_abstain`` policies are deliberately
diagnostic policies.  Their safety/recovery terms are interpretable proxies
derived from online candidate features, not ground-truth future rollouts:

* future observability: frontier escape, novelty, and topology signals;
* recoverability: geometry/active-gate safety and reduced revisit/completion risk;
* uncertainty handling: low-margin ranker choices can fall back to the
  target-frontier heuristic.

This keeps the Stage17c step faithful to the FutureNav/DreamNav/P2DNav ideas
without prematurely making the learned ranker active in the real evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_DIR = SCRIPT_DIR.parents[0] / "train"
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from progress_ranker_model import ProgressCandidateRanker  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{lineno}")
            yield value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _load_rows(data_dir: Path, split: str) -> List[Dict[str, Any]]:
    if split == "all":
        return _load_rows(data_dir, "train") + _load_rows(data_dir, "val")
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    rows = list(_read_jsonl(path))
    if not rows:
        raise ValueError(f"Dataset split is empty: {path}")
    return rows


def _feature_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {name: index for index, name in enumerate(feature_names)}


def _value(feature: Sequence[float], index: Mapping[str, int], name: str) -> float:
    position = index.get(name)
    return _safe_float(feature[position]) if position is not None else 0.0


def _bool_value(feature: Sequence[float], index: Mapping[str, int], name: str) -> bool:
    return _value(feature, index, name) > 0.5


def _heuristic_scores(
    features: Sequence[Sequence[float]],
    index: Mapping[str, int],
    name: str,
) -> List[float]:
    if name == "candidate_score":
        return [_value(feature, index, "score") for feature in features]
    if name == "target_frontier_score":
        return [_value(feature, index, "target_frontier_score") for feature in features]
    if name == "front_bucket":
        return [_value(feature, index, "direction_bucket=front") for feature in features]
    raise ValueError(f"Unknown heuristic: {name}")


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper - lower <= 1e-8:
        return [0.5 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def _resilience_prior(
    feature: Sequence[float],
    index: Mapping[str, int],
) -> Tuple[float, Dict[str, float]]:
    """Return an interpretable future-observability/recoverability proxy."""

    geometry_safe = float(_bool_value(feature, index, "geometry_safe"))
    active_gate_safe = float(_bool_value(feature, index, "active_gate_safe"))
    escape = float(_bool_value(feature, index, "target_frontier_escape_candidate"))
    target_frontier = float(_bool_value(feature, index, "target_frontier_candidate"))
    not_revisited = 1.0 - float(_bool_value(feature, index, "points_to_revisited_region"))
    not_completed = 1.0 - _value(feature, index, "is_completed_landmark")

    novelty = (
        _clamp01(_value(feature, index, "topology_novelty_score"))
        + _clamp01(_value(feature, index, "semantic_novelty_score"))
        + _clamp01(_value(feature, index, "unknown_target_frontier_bonus"))
    ) / 3.0
    low_revisit_risk = 1.0 - _clamp01(_value(feature, index, "revisit_risk"))
    future_observability = 0.60 * escape + 0.25 * target_frontier + 0.15 * novelty
    recoverability = (
        0.25 * geometry_safe
        + 0.20 * active_gate_safe
        + 0.20 * not_revisited
        + 0.15 * not_completed
        + 0.20 * low_revisit_risk
    )
    prior = 0.55 * future_observability + 0.45 * recoverability
    return prior, {
        "future_observability_proxy": future_observability,
        "recoverability_proxy": recoverability,
        "geometry_safe": geometry_safe,
        "active_gate_safe": active_gate_safe,
        "target_frontier_escape": escape,
        "target_frontier": target_frontier,
        "points_to_revisited_region": 1.0 - not_revisited,
        "completed_landmark": 1.0 - not_completed,
    }


def _load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = ("model_state", "feature_dim", "hidden_dim", "dropout")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint missing fields {missing}: {checkpoint_path}")
    model = ProgressCandidateRanker(
        int(checkpoint["feature_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _ranker_scores(
    model: torch.nn.Module,
    features: Sequence[Sequence[float]],
    device: torch.device,
) -> List[float]:
    tensor = torch.tensor(features, dtype=torch.float32, device=device)
    return [float(value) for value in model(tensor).detach().cpu().tolist()]


def _ndcg(labels: Sequence[float], order: Sequence[int]) -> float:
    gains = sum(
        max(0.0, float(labels[index])) / math.log2(rank + 2.0)
        for rank, index in enumerate(order)
    )
    ideal_order = sorted(range(len(labels)), key=lambda index: float(labels[index]), reverse=True)
    ideal = sum(
        max(0.0, float(labels[index])) / math.log2(rank + 2.0)
        for rank, index in enumerate(ideal_order)
    )
    return gains / ideal if ideal > 0.0 else 0.0


def _mrr(labels: Sequence[float], order: Sequence[int]) -> float:
    for rank, index in enumerate(order, start=1):
        if float(labels[index]) > 0.0:
            return 1.0 / float(rank)
    return 0.0


def _list_get(values: Any, index: int, default: Any = None) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return default
    return values[index]


def _select_order(
    row: Mapping[str, Any],
    policy: str,
    ranker_scores: Sequence[float],
    index: Mapping[str, int],
    resilience_weight: float,
    abstain_margin: float,
) -> Tuple[List[int], bool]:
    features = row["features"]
    target_scores = _heuristic_scores(features, index, "target_frontier_score")
    if policy == "candidate_score":
        scores = _heuristic_scores(features, index, "candidate_score")
        return sorted(range(len(scores)), key=lambda item: scores[item], reverse=True), False
    if policy == "target_frontier":
        return sorted(range(len(target_scores)), key=lambda item: target_scores[item], reverse=True), False
    if policy == "ranker":
        return sorted(range(len(ranker_scores)), key=lambda item: ranker_scores[item], reverse=True), False

    normalized_ranker = _minmax(ranker_scores)
    if policy == "ranker_resilience":
        combined = []
        for candidate, base_score in zip(features, normalized_ranker):
            prior, _ = _resilience_prior(candidate, index)
            combined.append(base_score + resilience_weight * prior)
        return sorted(range(len(combined)), key=lambda item: combined[item], reverse=True), False

    if policy == "ranker_abstain":
        order = sorted(range(len(ranker_scores)), key=lambda item: ranker_scores[item], reverse=True)
        margin = ranker_scores[order[0]] - ranker_scores[order[1]] if len(order) > 1 else 0.0
        if margin < abstain_margin:
            return (
                sorted(range(len(target_scores)), key=lambda item: target_scores[item], reverse=True),
                True,
            )
        return order, False

    raise ValueError(f"Unknown policy: {policy}")


def _selection_record(
    row: Mapping[str, Any],
    order: Sequence[int],
    index: Mapping[str, int],
    abstained: bool,
) -> Dict[str, Any]:
    selected = int(order[0])
    feature = row["features"][selected]
    prior, proxy = _resilience_prior(feature, index)
    labels = row["labels"]
    raw_values = row.get("raw_values", [])
    return {
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "step_id": row.get("step_id"),
        "hard_row": bool(row.get("hard_row")),
        "selected_index": selected,
        "selected_candidate_id": _list_get(row.get("candidate_ids"), selected),
        "selected_label": _safe_float(labels[selected]),
        "selected_raw_value": _safe_float(raw_values[selected]) if selected < len(raw_values) else None,
        "selected_candidate_status": _list_get(row.get("candidate_statuses"), selected, ""),
        "selected_resilience_prior": prior,
        "selected_future_observability_proxy": proxy["future_observability_proxy"],
        "selected_recoverability_proxy": proxy["recoverability_proxy"],
        "selected_geometry_safe": proxy["geometry_safe"],
        "selected_active_gate_safe": proxy["active_gate_safe"],
        "selected_target_frontier_escape": proxy["target_frontier_escape"],
        "selected_target_frontier": proxy["target_frontier"],
        "selected_revisited": proxy["points_to_revisited_region"],
        "selected_completed": proxy["completed_landmark"],
        "abstained_to_target_frontier": abstained,
    }


def _metrics(records: Sequence[Dict[str, Any]], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [float(record["selected_label"]) for record in records]
    raw_values = [
        float(record["selected_raw_value"])
        for record in records
        if record["selected_raw_value"] is not None
    ]
    total = float(max(1, len(records)))
    top1 = sum(value > 0.0 for value in labels) / total
    mrr = sum(float(record["_mrr"]) for record in records) / total
    policy_ndcg = sum(float(record["_ndcg"]) for record in records) / total
    return {
        "rows": float(len(records)),
        "top1_positive": top1,
        "mrr": mrr,
        "mean_selected_label": sum(labels) / total,
        "mean_selected_raw_value": sum(raw_values) / max(1, len(raw_values)),
        "ndcg": policy_ndcg,
        "completed_selected_rate": sum(record["selected_completed"] > 0.5 for record in records) / total,
        "repeated_selected_rate": sum(record["selected_revisited"] > 0.5 for record in records) / total,
        "unsafe_selected_rate": sum(
            record["selected_geometry_safe"] < 0.5 or record["selected_active_gate_safe"] < 0.5
            for record in records
        ) / total,
        "target_frontier_escape_selected_rate": sum(
            record["selected_target_frontier_escape"] > 0.5 for record in records
        ) / total,
        "future_observability_proxy_mean": sum(
            record["selected_future_observability_proxy"] for record in records
        ) / total,
        "recoverability_proxy_mean": sum(record["selected_recoverability_proxy"] for record in records) / total,
        "abstain_rate": sum(record["abstained_to_target_frontier"] for record in records) / total,
        "hard_rows": sum(bool(row.get("hard_row")) for row in rows),
    }


def _paired(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Paired policies have different row counts")
    left_wins = right_wins = ties = 0
    label_delta = raw_delta = 0.0
    for left_record, right_record in zip(left, right):
        left_label = float(left_record["selected_label"])
        right_label = float(right_record["selected_label"])
        left_raw = _safe_float(left_record["selected_raw_value"])
        right_raw = _safe_float(right_record["selected_raw_value"])
        label_delta += left_label - right_label
        raw_delta += left_raw - right_raw
        if left_label > right_label:
            left_wins += 1
        elif left_label < right_label:
            right_wins += 1
        else:
            ties += 1
    denom = float(max(1, len(left)))
    return {
        "left_wins": left_wins / denom,
        "right_wins": right_wins / denom,
        "ties": ties / denom,
        "mean_label_delta": label_delta / denom,
        "mean_raw_value_delta": raw_delta / denom,
    }


def _audit_split(
    rows: Sequence[Dict[str, Any]],
    model: torch.nn.Module,
    index: Mapping[str, int],
    device: torch.device,
    resilience_weight: float,
    abstain_margin: float,
) -> Dict[str, Any]:
    policies = ("candidate_score", "target_frontier", "ranker", "ranker_resilience", "ranker_abstain")
    records_by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ranker_scores = _ranker_scores(model, row["features"], device)
        for policy in policies:
            order, abstained = _select_order(
                row,
                policy,
                ranker_scores,
                index,
                resilience_weight,
                abstain_margin,
            )
            record = _selection_record(row, order, index, abstained)
            record["_mrr"] = _mrr(row["labels"], order)
            record["_ndcg"] = _ndcg(row["labels"], order)
            records_by_policy[policy].append(record)

    metrics = {
        policy: _metrics(records, rows)
        for policy, records in records_by_policy.items()
    }
    hard_metrics = {
        policy: _metrics(
            [record for record, row in zip(records, rows) if bool(row.get("hard_row"))],
            [row for row in rows if bool(row.get("hard_row"))],
        )
        for policy, records in records_by_policy.items()
    }
    comparisons = {}
    for policy in ("ranker", "ranker_resilience", "ranker_abstain"):
        comparisons[f"{policy}_vs_target_frontier"] = _paired(
            records_by_policy[policy],
            records_by_policy["target_frontier"],
        )
        comparisons[f"{policy}_vs_candidate_score"] = _paired(
            records_by_policy[policy],
            records_by_policy["candidate_score"],
        )
    return {
        "metrics": metrics,
        "hard_metrics": hard_metrics,
        "comparisons": comparisons,
        "selections": {
            policy: [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]
            for policy, records in records_by_policy.items()
        },
    }


def audit(
    data_dir: Path,
    checkpoint: Path,
    split: str,
    output: Optional[Path],
    device_name: str,
    resilience_weight: float,
    abstain_margin: float,
) -> Dict[str, Any]:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    feature_names = summary.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"Dataset summary has no feature_names: {data_dir}")
    rows = _load_rows(data_dir, split)
    feature_dim = len(feature_names)
    if any(len(feature) != feature_dim for row in rows for feature in row["features"]):
        raise ValueError(f"Feature dimension mismatch in {data_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint_meta = _load_model(checkpoint, device)
    if int(checkpoint_meta["feature_dim"]) != feature_dim:
        raise ValueError(
            f"Checkpoint feature_dim={checkpoint_meta['feature_dim']} does not match dataset={feature_dim}"
        )
    result = {
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "split": split,
        "device": str(device),
        "feature_dim": feature_dim,
        "rows": len(rows),
        "resilience_weight": resilience_weight,
        "abstain_margin": abstain_margin,
        "policy_definitions": {
            "candidate_score": "existing online candidate score",
            "target_frontier": "existing target-frontier heuristic",
            "ranker": "learned route-progress ranker only",
            "ranker_resilience": "ranker plus future-observability/recoverability proxy",
            "ranker_abstain": "ranker, falling back to target-frontier on low score margin",
        },
        "checkpoint_metrics": checkpoint_meta.get("metrics"),
        "audit": _audit_split(
            rows,
            model,
            _feature_index(feature_names),
            device,
            resilience_weight,
            abstain_margin,
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--resilience-weight", type=float, default=0.20)
    parser.add_argument("--abstain-margin", type=float, default=0.05)
    args = parser.parse_args()
    if args.resilience_weight < 0.0:
        raise ValueError("--resilience-weight must be non-negative")
    if args.abstain_margin < 0.0:
        raise ValueError("--abstain-margin must be non-negative")
    result = audit(
        args.data_dir,
        args.checkpoint,
        args.split,
        args.output,
        args.device,
        args.resilience_weight,
        args.abstain_margin,
    )
    print(json.dumps(result["audit"]["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(result["audit"]["comparisons"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

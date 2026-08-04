"""Train the Stage18c S2-aware candidate-advantage scorer.

This trainer predicts whether an online-safe OccMem candidate has a directional
advantage over the frozen S2/current waypoint.  It is deliberately not an
active intervention policy.  Active gating remains a later, stricter step.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from progress_ranker_model import ProgressCandidateRanker


class AdvantageCandidateDataset(Dataset):
    """Candidate-level JSONL dataset for Stage18c advantage scoring."""

    def __init__(self, path: Path):
        self.items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
                if "features" not in item or "label" not in item or "event_key" not in item:
                    raise ValueError(f"Missing Stage18c fields at {path}:{lineno}")
                self.items.append(item)
        if not self.items:
            raise ValueError(f"No usable Stage18c rows in {path}")
        self.feature_dim = len(self.items[0]["features"])
        for item in self.items:
            if len(item["features"]) != self.feature_dim:
                raise ValueError(f"Inconsistent feature dimension in {path}")
            label = int(item["label"])
            if label not in {0, 1}:
                raise ValueError(f"Invalid binary label at {path}")

    def label_counts(self) -> Counter:
        return Counter(int(item["label"]) for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def collate_rows(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "features": torch.tensor([item["features"] for item in items], dtype=torch.float32),
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32),
        "event_keys": [str(item["event_key"]) for item in items],
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pos_weight(
    dataset: AdvantageCandidateDataset,
    device: torch.device,
    *,
    cap: float,
    disabled: bool,
) -> Optional[torch.Tensor]:
    if disabled:
        return None
    if cap <= 0.0:
        raise ValueError("--pos-weight-cap must be positive")
    counts = dataset.label_counts()
    positives = float(counts.get(1, 0))
    negatives = float(counts.get(0, 0))
    if positives <= 0.0:
        raise ValueError("Training split lacks positive advantage candidates.")
    return torch.tensor([min(float(cap), negatives / positives)], dtype=torch.float32, device=device)


def _binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
        "predicted_positive_count": float(tp + fp),
        "positive_count": float(tp + fn),
        "prediction_rate": (tp + fp) / max(1, tp + fp + fn + tn),
    }


def _threshold_sweep(
    labels: Sequence[int],
    scores: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    result = []
    for threshold in thresholds:
        predictions = [int(score >= threshold) for score in scores]
        metrics = _binary_metrics(labels, predictions)
        metrics["threshold"] = float(threshold)
        result.append(metrics)
    return result


def _best_threshold(
    threshold_metrics: Sequence[Dict[str, float]],
    *,
    key: str,
) -> Dict[str, float]:
    if not threshold_metrics:
        return {}
    return dict(
        max(
            threshold_metrics,
            key=lambda item: (
                float(item.get(key, 0.0)),
                float(item.get("precision", 0.0)),
                float(item.get("recall", 0.0)),
                -float(item.get("predicted_positive_count", 0.0)),
            ),
        )
    )


def _group_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    event_keys: Sequence[str],
) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for label, score, event_key in zip(labels, scores, event_keys):
        grouped[str(event_key)].append((float(score), int(label)))
    positive_groups = [
        items for items in grouped.values() if any(label == 1 for _, label in items)
    ]
    multi_candidate_positive_groups = [
        items for items in positive_groups if len(items) > 1
    ]
    top1 = mrr = 0.0
    for items in positive_groups:
        ordered = sorted(items, key=lambda pair: pair[0], reverse=True)
        top1 += float(ordered[0][1] == 1)
        for index, (_, label) in enumerate(ordered, start=1):
            if label == 1:
                mrr += 1.0 / float(index)
                break
    multi_top1 = multi_mrr = 0.0
    for items in multi_candidate_positive_groups:
        ordered = sorted(items, key=lambda pair: pair[0], reverse=True)
        multi_top1 += float(ordered[0][1] == 1)
        for index, (_, label) in enumerate(ordered, start=1):
            if label == 1:
                multi_mrr += 1.0 / float(index)
                break
    return {
        "event_count": float(len(grouped)),
        "positive_event_count": float(len(positive_groups)),
        "multi_candidate_positive_event_count": float(len(multi_candidate_positive_groups)),
        "event_top1_positive": top1 / max(1, len(positive_groups)),
        "event_mrr": mrr / max(1, len(positive_groups)),
        "multi_candidate_event_top1_positive": multi_top1
        / max(1, len(multi_candidate_positive_groups)),
        "multi_candidate_event_mrr": multi_mrr / max(1, len(multi_candidate_positive_groups)),
    }


def _selection_score(metrics: Dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "f1":
        return float(metrics["f1"])
    if selection_metric == "precision_first":
        return (
            0.60 * float(metrics["precision"])
            + 0.25 * float(metrics["f1"])
            + 0.15 * float(metrics["event_top1_positive"])
        )
    if selection_metric == "sweep_f1":
        return float(metrics.get("best_threshold_by_f1", {}).get("f1", 0.0))
    if selection_metric == "top1":
        return float(metrics["event_top1_positive"])
    raise ValueError(f"Unknown selection metric: {selection_metric}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: Optional[torch.Tensor],
    eval_threshold: float,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    model.eval()
    total_loss = total_examples = 0.0
    labels: List[int] = []
    scores: List[float] = []
    event_keys: List[str] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels, pos_weight=pos_weight)
        batch_size = float(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        labels.extend(int(value) for value in batch_labels.detach().cpu().tolist())
        scores.extend(float(value) for value in torch.sigmoid(logits).detach().cpu().tolist())
        event_keys.extend(batch["event_keys"])

    predictions = [int(score >= float(eval_threshold)) for score in scores]
    threshold_metrics = _threshold_sweep(labels, scores, thresholds)
    metrics: Dict[str, Any] = {
        "loss": total_loss / max(1.0, total_examples),
        "examples": total_examples,
        "eval_threshold": float(eval_threshold),
        "label_counts": {
            "negative": int(sum(label == 0 for label in labels)),
            "positive": int(sum(label == 1 for label in labels)),
        },
        "threshold_sweep": threshold_metrics,
        "best_threshold_by_f1": _best_threshold(threshold_metrics, key="f1"),
        "best_threshold_by_precision": _best_threshold(threshold_metrics, key="precision"),
    }
    metrics.update(_binary_metrics(labels, predictions))
    metrics.update(_group_metrics(labels, scores, event_keys))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--eval-threshold", type=float, default=0.50)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "precision_first", "sweep_f1", "top1"),
        default="precision_first",
    )
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage18c advantage scorer training requires one CUDA GPU.")
    if not 0.0 <= args.eval_threshold <= 1.0:
        raise ValueError("--eval-threshold must be in [0, 1]")
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("--thresholds must be in [0, 1]")

    _set_seed(args.seed)
    device = torch.device("cuda")
    summary_path = args.data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("task") != "candidate_advantage_not_active_gate":
        raise ValueError("Stage18c trainer expects a candidate-advantage dataset.")

    train_dataset = AdvantageCandidateDataset(args.data_dir / "train.jsonl")
    val_dataset = AdvantageCandidateDataset(args.data_dir / "val.jsonl")
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("Train/val feature dimensions differ")
    if train_dataset.label_counts().get(1, 0) <= 0 or val_dataset.label_counts().get(1, 0) <= 0:
        raise ValueError("Stage18c train and val splits must both contain positives.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_rows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_rows,
    )
    model = ProgressCandidateRanker(train_dataset.feature_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = _pos_weight(
        train_dataset,
        device,
        cap=args.pos_weight_cap,
        disabled=args.disable_pos_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight_cap": args.pos_weight_cap,
        "disable_pos_weight": args.disable_pos_weight,
        "pos_weight": None if pos_weight is None else pos_weight.detach().cpu().tolist(),
        "eval_threshold": args.eval_threshold,
        "thresholds": args.thresholds,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "train_label_counts": {
            "negative": int(train_dataset.label_counts().get(0, 0)),
            "positive": int(train_dataset.label_counts().get(1, 0)),
        },
        "val_label_counts": {
            "negative": int(val_dataset.label_counts().get(0, 0)),
            "positive": int(val_dataset.label_counts().get(1, 0)),
        },
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n",
        encoding="utf-8",
    )

    history: List[Dict[str, Any]] = []
    best_score = float("-inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break

        metrics = evaluate(
            model,
            val_loader,
            device,
            pos_weight=pos_weight,
            eval_threshold=args.eval_threshold,
            thresholds=args.thresholds,
        )
        metrics.update({"epoch": epoch, "global_step": global_step})
        metrics["selection_score"] = _selection_score(metrics, args.selection_metric)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if metrics["selection_score"] > best_score:
            best_score = float(metrics["selection_score"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": train_dataset.feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "dataset_summary": dataset_summary,
                    "training_config": training_config,
                    "metrics": metrics,
                },
                args.output_dir / "best.pt",
            )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.smoke_steps and global_step >= args.smoke_steps:
            break


if __name__ == "__main__":
    main()

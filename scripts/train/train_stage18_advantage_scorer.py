"""Train the Stage18c S2-aware candidate-advantage scorer.

Stage18c is not an active intervention policy.  It learns whether, within one
online-safe OccMem candidate set, some candidate appears more advantageous than
the frozen S2/current waypoint.  Active gating remains a later, stricter step.

The trainer supports both the original candidate-level BCE smoke objective and
event-level ranking objectives.  The ranking path is the preferred Stage18c
mode because the downstream question is set-wise: which candidate should be
preferred inside the current event, not whether an isolated row crosses a
global active threshold.
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


class AdvantageEventDataset(Dataset):
    """Event-level dataset with a variable-size safe candidate set."""

    def __init__(self, path: Path):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.feature_dim: Optional[int] = None
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
                feature_dim = len(item["features"])
                if self.feature_dim is None:
                    self.feature_dim = feature_dim
                elif feature_dim != self.feature_dim:
                    raise ValueError(f"Inconsistent feature dimension in {path}")
                label = int(item["label"])
                if label not in {0, 1}:
                    raise ValueError(f"Invalid binary label at {path}")
                grouped[str(item["event_key"])].append(item)

        if not grouped:
            raise ValueError(f"No usable Stage18c rows in {path}")
        if self.feature_dim is None:
            raise ValueError(f"No feature rows in {path}")

        self.items: List[Dict[str, Any]] = []
        for event_key, rows in sorted(grouped.items()):
            self.items.append(
                {
                    "event_key": event_key,
                    "features": [row["features"] for row in rows],
                    "labels": [int(row["label"]) for row in rows],
                    "candidate_ids": [row.get("candidate_id") for row in rows],
                }
            )

    def label_counts(self) -> Counter:
        counter: Counter = Counter()
        for item in self.items:
            counter.update(int(label) for label in item["labels"])
        return counter

    def event_counts(self) -> Dict[str, int]:
        positive = sum(any(int(label) == 1 for label in item["labels"]) for item in self.items)
        multi_positive = sum(
            any(int(label) == 1 for label in item["labels"]) and len(item["labels"]) > 1
            for item in self.items
        )
        return {
            "events": len(self.items),
            "positive_events": int(positive),
            "multi_candidate_positive_events": int(multi_positive),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def collate_events(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    feature_dim = len(items[0]["features"][0])
    max_candidates = max(len(item["features"]) for item in items)
    features = torch.zeros((len(items), max_candidates, feature_dim), dtype=torch.float32)
    labels = torch.zeros((len(items), max_candidates), dtype=torch.float32)
    mask = torch.zeros((len(items), max_candidates), dtype=torch.bool)
    event_keys: List[str] = []
    for row, item in enumerate(items):
        count = len(item["features"])
        features[row, :count] = torch.tensor(item["features"], dtype=torch.float32)
        labels[row, :count] = torch.tensor(item["labels"], dtype=torch.float32)
        mask[row, :count] = True
        event_keys.append(str(item["event_key"]))
    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "event_keys": event_keys,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pos_weight(
    dataset: AdvantageEventDataset,
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


def _masked_bce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        scores[mask],
        labels[mask],
        pos_weight=pos_weight,
    )


def _listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ListNet-style loss over events that contain at least one positive."""

    positive_mask = (labels > 0.0) & mask
    valid_rows = positive_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return scores.sum() * 0.0
    row_scores = scores[valid_rows]
    row_labels = labels[valid_rows]
    row_mask = mask[valid_rows]
    masked_scores = row_scores.masked_fill(~row_mask, float("-inf"))
    target = row_labels.masked_fill(~row_mask, 0.0).clamp_min(0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(masked_scores, dim=1).masked_fill(~row_mask, 0.0)
    return -(target * log_probs).sum(dim=1).mean()


def _pairwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pairwise positive-vs-negative ranking loss inside each event."""

    losses: List[torch.Tensor] = []
    for row in range(scores.shape[0]):
        row_mask = mask[row]
        positives = row_mask & (labels[row] > 0.0)
        negatives = row_mask & (labels[row] <= 0.0)
        if not bool(positives.any()) or not bool(negatives.any()):
            continue
        diff = scores[row][positives].unsqueeze(1) - scores[row][negatives].unsqueeze(0)
        losses.append(-F.logsigmoid(diff).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _combined_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_mode: str,
    pos_weight: Optional[torch.Tensor],
    bce_loss_weight: float,
    listwise_loss_weight: float,
    pairwise_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = _masked_bce_loss(scores, labels, mask, pos_weight=pos_weight)
    listwise = _listwise_loss(scores, labels, mask)
    pairwise = _pairwise_loss(scores, labels, mask)
    if loss_mode == "bce":
        loss = bce
    elif loss_mode == "listwise":
        loss = float(listwise_loss_weight) * listwise + float(bce_loss_weight) * bce
    elif loss_mode == "pairwise":
        loss = float(pairwise_loss_weight) * pairwise + float(bce_loss_weight) * bce
    elif loss_mode == "hybrid":
        loss = (
            float(listwise_loss_weight) * listwise
            + float(pairwise_loss_weight) * pairwise
            + float(bce_loss_weight) * bce
        )
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    return loss, {
        "bce_loss": float(bce.detach().item()),
        "listwise_loss": float(listwise.detach().item()),
        "pairwise_loss": float(pairwise.detach().item()),
    }


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
    if selection_metric == "ranking":
        return (
            0.45 * float(metrics["multi_candidate_event_top1_positive"])
            + 0.30 * float(metrics["multi_candidate_event_mrr"])
            + 0.15 * float(metrics["event_top1_positive"])
            + 0.10 * float(metrics["event_mrr"])
        )
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def _flatten_event_batch(
    labels: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    event_keys: Sequence[str],
) -> Tuple[List[int], List[float], List[str]]:
    flat_labels: List[int] = []
    flat_scores: List[float] = []
    flat_event_keys: List[str] = []
    cpu_labels = labels.detach().cpu()
    cpu_scores = scores.detach().cpu()
    cpu_mask = mask.detach().cpu()
    for row, event_key in enumerate(event_keys):
        valid_count = int(cpu_mask[row].sum().item())
        flat_labels.extend(int(value) for value in cpu_labels[row, :valid_count].tolist())
        flat_scores.extend(float(value) for value in cpu_scores[row, :valid_count].tolist())
        flat_event_keys.extend([str(event_key)] * valid_count)
    return flat_labels, flat_scores, flat_event_keys


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: Optional[torch.Tensor],
    eval_threshold: float,
    thresholds: Sequence[float],
    loss_mode: str,
    bce_loss_weight: float,
    listwise_loss_weight: float,
    pairwise_loss_weight: float,
) -> Dict[str, Any]:
    model.eval()
    total_loss = total_bce = total_listwise = total_pairwise = total_examples = 0.0
    labels: List[int] = []
    scores: List[float] = []
    event_keys: List[str] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(features)
        loss, parts = _combined_loss(
            logits,
            batch_labels,
            mask,
            loss_mode=loss_mode,
            pos_weight=pos_weight,
            bce_loss_weight=bce_loss_weight,
            listwise_loss_weight=listwise_loss_weight,
            pairwise_loss_weight=pairwise_loss_weight,
        )
        batch_size = float(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_bce += float(parts["bce_loss"]) * batch_size
        total_listwise += float(parts["listwise_loss"]) * batch_size
        total_pairwise += float(parts["pairwise_loss"]) * batch_size
        total_examples += batch_size
        flat_labels, flat_scores, flat_event_keys = _flatten_event_batch(
            batch_labels,
            torch.sigmoid(logits),
            mask,
            batch["event_keys"],
        )
        labels.extend(flat_labels)
        scores.extend(flat_scores)
        event_keys.extend(flat_event_keys)

    predictions = [int(score >= float(eval_threshold)) for score in scores]
    threshold_metrics = _threshold_sweep(labels, scores, thresholds)
    metrics: Dict[str, Any] = {
        "loss": total_loss / max(1.0, total_examples),
        "bce_loss": total_bce / max(1.0, total_examples),
        "listwise_loss": total_listwise / max(1.0, total_examples),
        "pairwise_loss": total_pairwise / max(1.0, total_examples),
        "examples": total_examples,
        "candidate_examples": float(len(labels)),
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
    parser.add_argument(
        "--loss-mode",
        choices=("bce", "listwise", "pairwise", "hybrid"),
        default="bce",
    )
    parser.add_argument("--bce-loss-weight", type=float, default=0.25)
    parser.add_argument("--listwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=1.0)
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
        choices=("f1", "precision_first", "sweep_f1", "top1", "ranking"),
        default="precision_first",
    )
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage18c advantage scorer training requires one CUDA GPU.")
    if not 0.0 <= args.eval_threshold <= 1.0:
        raise ValueError("--eval-threshold must be in [0, 1]")
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("--thresholds must be in [0, 1]")
    if args.bce_loss_weight < 0.0 or args.listwise_loss_weight < 0.0 or args.pairwise_loss_weight < 0.0:
        raise ValueError("Loss weights must be non-negative")

    _set_seed(args.seed)
    device = torch.device("cuda")
    summary_path = args.data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("task") != "candidate_advantage_not_active_gate":
        raise ValueError("Stage18c trainer expects a candidate-advantage dataset.")

    train_dataset = AdvantageEventDataset(args.data_dir / "train.jsonl")
    val_dataset = AdvantageEventDataset(args.data_dir / "val.jsonl")
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
        collate_fn=collate_events,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_events,
    )
    model = ProgressCandidateRanker(int(train_dataset.feature_dim), args.hidden_dim, args.dropout).to(device)
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
        "loss_mode": args.loss_mode,
        "bce_loss_weight": args.bce_loss_weight,
        "listwise_loss_weight": args.listwise_loss_weight,
        "pairwise_loss_weight": args.pairwise_loss_weight,
        "pos_weight_cap": args.pos_weight_cap,
        "disable_pos_weight": args.disable_pos_weight,
        "pos_weight": None if pos_weight is None else pos_weight.detach().cpu().tolist(),
        "eval_threshold": args.eval_threshold,
        "thresholds": args.thresholds,
        "selection_metric": args.selection_metric,
        "eval_before_train": args.eval_before_train,
        "seed": args.seed,
        "train_label_counts": {
            "negative": int(train_dataset.label_counts().get(0, 0)),
            "positive": int(train_dataset.label_counts().get(1, 0)),
        },
        "val_label_counts": {
            "negative": int(val_dataset.label_counts().get(0, 0)),
            "positive": int(val_dataset.label_counts().get(1, 0)),
        },
        "train_event_counts": train_dataset.event_counts(),
        "val_event_counts": val_dataset.event_counts(),
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n",
        encoding="utf-8",
    )

    history: List[Dict[str, Any]] = []
    best_score = float("-inf")
    global_step = 0

    def evaluate_and_save(epoch: int) -> None:
        nonlocal best_score
        metrics = evaluate(
            model,
            val_loader,
            device,
            pos_weight=pos_weight,
            eval_threshold=args.eval_threshold,
            thresholds=args.thresholds,
            loss_mode=args.loss_mode,
            bce_loss_weight=args.bce_loss_weight,
            listwise_loss_weight=args.listwise_loss_weight,
            pairwise_loss_weight=args.pairwise_loss_weight,
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
                    "feature_dim": int(train_dataset.feature_dim),
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

    if args.eval_before_train:
        evaluate_and_save(epoch=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            scores = model(features)
            loss, _ = _combined_loss(
                scores,
                labels,
                mask,
                loss_mode=args.loss_mode,
                pos_weight=pos_weight,
                bce_loss_weight=args.bce_loss_weight,
                listwise_loss_weight=args.listwise_loss_weight,
                pairwise_loss_weight=args.pairwise_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break

        evaluate_and_save(epoch=epoch)
        if args.smoke_steps and global_step >= args.smoke_steps:
            break


if __name__ == "__main__":
    main()

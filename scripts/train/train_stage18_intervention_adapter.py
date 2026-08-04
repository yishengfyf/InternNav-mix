"""Train the lightweight offline Stage18b S2-aware intervention adapter.

This trainer never loads or updates S2/NextDiT.  It learns from privileged
offline labels whether to keep the frozen policy, select one safe OccMem
recovery candidate, or abstain and fall back to S2.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stage18_intervention_model import S2AwareInterventionAdapter


DECISION_NAMES = ("keep", "intervene", "abstain")


class InterventionDataset(Dataset):
    """Event-level dataset with a variable-size safe candidate set."""

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
                if "decision_label" not in item or "event_context_features" not in item:
                    raise ValueError(f"Missing Stage18b fields at {path}:{lineno}")
                self.items.append(item)
        if not self.items:
            raise ValueError(f"No usable Stage18b rows in {path}")

        self.context_dim = len(self.items[0]["event_context_features"])
        pair_examples = [
            feature
            for item in self.items
            for feature in item.get("candidate_pair_features") or []
        ]
        if not pair_examples:
            raise ValueError(f"No safe candidate pair features in {path}")
        self.pair_dim = len(pair_examples[0])
        for item in self.items:
            if len(item["event_context_features"]) != self.context_dim:
                raise ValueError(f"Inconsistent event context dimension in {path}")
            for feature in item.get("candidate_pair_features") or []:
                if len(feature) != self.pair_dim:
                    raise ValueError(f"Inconsistent pair feature dimension in {path}")
            label = int(item["decision_label"])
            if label < 0 or label >= len(DECISION_NAMES):
                raise ValueError(f"Invalid decision label at {path}")
            target = int(item.get("target_candidate_index", -1))
            if label == 1 and not (0 <= target < len(item.get("candidate_pair_features") or [])):
                raise ValueError(f"Intervene row has invalid candidate target at {path}")

    def decision_counts(self) -> Counter:
        return Counter(int(item["decision_label"]) for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def collate_intervention_rows(
    items: Sequence[Dict[str, Any]],
    *,
    pair_dim: int,
) -> Dict[str, torch.Tensor]:
    context_dim = len(items[0]["event_context_features"])
    max_candidates = max(1, max(len(item.get("candidate_pair_features") or []) for item in items))
    pair_features = torch.zeros((len(items), max_candidates, pair_dim), dtype=torch.float32)
    candidate_mask = torch.zeros((len(items), max_candidates), dtype=torch.bool)
    context_features = torch.zeros((len(items), context_dim), dtype=torch.float32)
    decision_labels = torch.zeros((len(items),), dtype=torch.long)
    target_indices = torch.full((len(items),), -1, dtype=torch.long)
    for row, item in enumerate(items):
        candidates = item.get("candidate_pair_features") or []
        if candidates:
            count = len(candidates)
            pair_features[row, :count] = torch.tensor(candidates, dtype=torch.float32)
            candidate_mask[row, :count] = True
        context_features[row] = torch.tensor(item["event_context_features"], dtype=torch.float32)
        decision_labels[row] = int(item["decision_label"])
        target_indices[row] = int(item.get("target_candidate_index", -1))
    return {
        "candidate_pair_features": pair_features,
        "candidate_mask": candidate_mask,
        "event_context_features": context_features,
        "decision_labels": decision_labels,
        "target_indices": target_indices,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_weights(dataset: InterventionDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.decision_counts()
    total = float(len(dataset))
    weights = []
    for label in range(len(DECISION_NAMES)):
        count = float(counts.get(label, 0))
        if count <= 0.0:
            raise ValueError(f"Training split lacks decision class {DECISION_NAMES[label]}")
        weights.append(min(8.0, total / (float(len(DECISION_NAMES)) * count)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _balanced_sampler(dataset: InterventionDataset) -> WeightedRandomSampler:
    counts = dataset.decision_counts()
    weights = [1.0 / float(counts[int(item["decision_label"])]) for item in dataset.items]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def _decision_metrics(labels: List[int], predictions: List[int]) -> Dict[str, float]:
    result: Dict[str, float] = {"decision_accuracy": 0.0}
    if not labels:
        return result
    result["decision_accuracy"] = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
    f1_values = []
    for index, name in enumerate(DECISION_NAMES):
        true_positive = sum(a == index and b == index for a, b in zip(labels, predictions))
        false_positive = sum(a != index and b == index for a, b in zip(labels, predictions))
        false_negative = sum(a == index and b != index for a, b in zip(labels, predictions))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = f1
        f1_values.append(f1)
    result["macro_f1"] = sum(f1_values) / len(f1_values)
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    candidate_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = total_decision_loss = total_candidate_loss = 0.0
    labels: List[int] = []
    predictions: List[int] = []
    candidate_total = candidate_correct = 0
    for batch in loader:
        pair_features = batch["candidate_pair_features"].to(device, non_blocking=True)
        candidate_mask = batch["candidate_mask"].to(device, non_blocking=True)
        context_features = batch["event_context_features"].to(device, non_blocking=True)
        decision_labels = batch["decision_labels"].to(device, non_blocking=True)
        target_indices = batch["target_indices"].to(device, non_blocking=True)
        candidate_scores, decision_logits = model(pair_features, candidate_mask, context_features)
        decision_loss = F.cross_entropy(decision_logits, decision_labels, weight=class_weights)
        intervene_mask = (decision_labels == 1) & (target_indices >= 0)
        candidate_loss = torch.zeros((), device=device)
        if intervene_mask.any():
            masked_scores = candidate_scores.masked_fill(~candidate_mask, float("-inf"))
            candidate_loss = F.cross_entropy(masked_scores[intervene_mask], target_indices[intervene_mask])
            predicted_targets = masked_scores[intervene_mask].argmax(dim=1)
            candidate_correct += int(
                (predicted_targets == target_indices[intervene_mask]).sum().item()
            )
            candidate_total += int(intervene_mask.sum().item())
        loss = decision_loss + float(candidate_loss_weight) * candidate_loss
        batch_size = int(decision_labels.shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_decision_loss += float(decision_loss.item()) * batch_size
        total_candidate_loss += float(candidate_loss.item()) * batch_size
        labels.extend(int(value) for value in decision_labels.detach().cpu().tolist())
        predictions.extend(int(value) for value in decision_logits.argmax(dim=1).detach().cpu().tolist())

    metrics = {
        "loss": total_loss / max(1, total_examples),
        "decision_loss": total_decision_loss / max(1, total_examples),
        "candidate_loss": total_candidate_loss / max(1, total_examples),
        "candidate_top1_accuracy": candidate_correct / max(1, candidate_total),
        "candidate_examples": float(candidate_total),
        "examples": float(total_examples),
    }
    metrics.update(_decision_metrics(labels, predictions))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Stage18b S2-aware intervention adapter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--disable-balanced-sampler", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage18b adapter training requires one CUDA GPU.")
    _set_seed(args.seed)
    device = torch.device("cuda")
    summary_path = args.data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("split_key") != "scene":
        raise ValueError("Stage18b requires a scene-disjoint train/val split.")

    train_dataset = InterventionDataset(args.data_dir / "train.jsonl")
    val_dataset = InterventionDataset(args.data_dir / "val.jsonl")
    if train_dataset.pair_dim != val_dataset.pair_dim:
        raise ValueError("Train/val pair dimensions differ")
    if train_dataset.context_dim != val_dataset.context_dim:
        raise ValueError("Train/val context dimensions differ")
    if train_dataset.decision_counts().get(1, 0) <= 0 or val_dataset.decision_counts().get(1, 0) <= 0:
        raise ValueError("Stage18b train and val splits must both contain intervene labels.")

    train_sampler = None if args.disable_balanced_sampler else _balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_intervention_rows, pair_dim=train_dataset.pair_dim),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_intervention_rows, pair_dim=val_dataset.pair_dim),
    )
    model = S2AwareInterventionAdapter(
        pair_feature_dim=train_dataset.pair_dim,
        event_context_dim=train_dataset.context_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _class_weights(train_dataset, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_score = float("-inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            pair_features = batch["candidate_pair_features"].to(device, non_blocking=True)
            candidate_mask = batch["candidate_mask"].to(device, non_blocking=True)
            context_features = batch["event_context_features"].to(device, non_blocking=True)
            decision_labels = batch["decision_labels"].to(device, non_blocking=True)
            target_indices = batch["target_indices"].to(device, non_blocking=True)
            candidate_scores, decision_logits = model(pair_features, candidate_mask, context_features)
            decision_loss = F.cross_entropy(decision_logits, decision_labels, weight=class_weights)
            intervene_mask = (decision_labels == 1) & (target_indices >= 0)
            candidate_loss = torch.zeros((), device=device)
            if intervene_mask.any():
                masked_scores = candidate_scores.masked_fill(~candidate_mask, float("-inf"))
                candidate_loss = F.cross_entropy(
                    masked_scores[intervene_mask],
                    target_indices[intervene_mask],
                )
            loss = decision_loss + float(args.candidate_loss_weight) * candidate_loss
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
            class_weights,
            args.candidate_loss_weight,
        )
        metrics.update({"epoch": epoch, "global_step": global_step})
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        selection_score = 0.5 * metrics["intervene_f1"] + 0.5 * metrics["candidate_top1_accuracy"]
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "pair_feature_dim": train_dataset.pair_dim,
                    "event_context_dim": train_dataset.context_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "decision_names": list(DECISION_NAMES),
                    "dataset_summary": dataset_summary,
                    "class_weights": class_weights.detach().cpu().tolist(),
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

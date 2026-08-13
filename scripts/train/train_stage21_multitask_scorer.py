"""Train the first offline Stage21 structured multi-head scorer.

The model is intentionally small and consumes only frozen-policy online OCC
features.  It never loads Habitat, S2, NextDiT, route references, or episode
outcomes as inputs.  The three heads are:

* progress: event-level positive-vs-negative candidate ranking relative to S2;
* safety: short-horizon executability proxy regression plus geometry-safe audit;
* recovery: safe-decision-state-restoration proxy regression (explicitly
  non-causal; final success is never its label).

This is an offline training smoke/full scorer, not an active policy or an
episode-time OCC memory update.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from stage21_training_common import (
    audit_online_row,
    encode_row,
    feature_names,
    heuristic_score,
    iter_jsonl,
    task_target,
)


TASKS = ("progress", "safety", "recovery")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _event_key(row: Mapping[str, Any]) -> str:
    identity = row.get("identity") or {}
    return "/".join(str(identity.get(key)) for key in ("scene_id", "episode_id", "step_id"))


def _read_task_rows(data_dir: Path, task: str, split: str) -> List[Dict[str, Any]]:
    filename = {
        "progress": "progress_rows",
        "safety": "safety_rows",
        "recovery": "recovery_proxy_rows",
    }[task]
    path = data_dir / f"{filename}_{split}.jsonl"
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No {task}/{split} rows: {path}")
    return rows


def audit_dataset(data_dir: Path) -> Dict[str, Any]:
    summary_path = data_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = ("event_schema_version", "candidate_recoverability_rows", "active_safety_check", "split_audit")
    for key in required:
        if key not in summary:
            raise ValueError(f"Stage21 summary missing {key}")
    if summary["event_schema_version"] != "stage21a_r3_v3":
        raise ValueError("Unexpected Stage21 event schema")
    if not summary["active_safety_check"].get("passed"):
        raise ValueError("Active safety audit failed; refusing offline training")
    if summary["split_audit"].get("scene_overlap_count") != 0:
        raise ValueError("Scene overlap detected; refusing offline training")
    candidate_audit = summary["candidate_recoverability_rows"]
    if not candidate_audit.get("gt_leakage_scan", {}).get("passed"):
        raise ValueError("Dataset GT leakage audit failed")
    if candidate_audit.get("active_gate_safe_used_as_recovery_target"):
        raise ValueError("active_gate_safe is not an allowed recovery target")
    rows_audit: Dict[str, Any] = {"summary": summary, "splits": {}}
    for split in ("train", "val"):
        split_result: Dict[str, Any] = {}
        for task in TASKS:
            rows = _read_task_rows(data_dir, task, split)
            hits = []
            scene_ids = set()
            for row in rows:
                hits.extend(audit_online_row(row))
                scene_ids.add(str((row.get("identity") or {}).get("scene_id")))
                # Materialize now to catch malformed values before torch starts.
                encode_row(row)
                task_target(row, task)
            split_result[task] = {
                "rows": len(rows),
                "scene_count": len(scene_ids),
                "leakage_hit_count": len(hits),
                "sample_leakage_hits": hits[:10],
            }
            if hits:
                raise ValueError(f"GT leakage detected in {task}/{split}: {hits[:3]}")
        rows_audit["splits"][split] = split_result
    return rows_audit


class FeatureNormalizer:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != self.std.shape or np.any(self.std <= 0.0):
            raise ValueError("Invalid Stage21 feature normalizer")

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]]) -> "FeatureNormalizer":
        matrix = np.asarray([encode_row(row) for row in rows], dtype=np.float32)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, row: Mapping[str, Any]) -> List[float]:
        values = np.asarray(encode_row(row), dtype=np.float32)
        return ((values - self.mean) / self.std).tolist()

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class ProgressEventDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]], normalizer: FeatureNormalizer):
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_event_key(row)].append(row)
        self.items: List[Dict[str, Any]] = []
        for event_key, event_rows in sorted(grouped.items()):
            # Ambiguous/tie rows have no ordinal supervision.  Excluding them
            # from this pilot prevents a no-loss candidate from deciding the
            # event top-1 metric while preserving them in the source dataset
            # for later uncertainty work.
            event_rows = [
                row for row in event_rows if task_target(row, "progress")[0] != 0.5
            ]
            labels = [task_target(row, "progress")[0] for row in event_rows]
            if 1.0 not in labels or 0.0 not in labels:
                continue
            self.items.append({
                "event_key": event_key,
                "features": [normalizer.transform(row) for row in event_rows],
                "labels": labels,
                "rows": list(event_rows),
            })
        if not self.items:
            raise ValueError("Progress split contains no positive/negative events")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


class ScalarDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]], task: str, normalizer: FeatureNormalizer):
        self.task = task
        self.rows = list(rows)
        self.features = [normalizer.transform(row) for row in self.rows]
        self.targets = [task_target(row, task) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        score, auxiliary = self.targets[index]
        return {
            "features": self.features[index],
            "score_target": score,
            "aux_target": auxiliary,
        }


def collate_progress(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    max_count = max(len(item["features"]) for item in items)
    dim = len(items[0]["features"][0])
    features = torch.zeros((len(items), max_count, dim), dtype=torch.float32)
    labels = torch.zeros((len(items), max_count), dtype=torch.float32)
    mask = torch.zeros((len(items), max_count), dtype=torch.bool)
    keys = []
    for row, item in enumerate(items):
        count = len(item["features"])
        features[row, :count] = torch.tensor(item["features"], dtype=torch.float32)
        labels[row, :count] = torch.tensor(item["labels"], dtype=torch.float32)
        mask[row, :count] = True
        keys.append(item["event_key"])
    return {"features": features, "labels": labels, "mask": mask, "event_keys": keys}


def collate_scalar(items: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.tensor([item["features"] for item in items], dtype=torch.float32),
        "score_target": torch.tensor([item["score_target"] for item in items], dtype=torch.float32),
        "aux_target": torch.tensor([item["aux_target"] for item in items], dtype=torch.float32),
    }


class Stage21MultiHeadScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.safety_head = nn.Linear(hidden_dim, 1)
        self.safety_geometry_head = nn.Linear(hidden_dim, 1)
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.recovery_promising_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "progress": self.progress_head(hidden).squeeze(-1),
            "safety": self.safety_head(hidden).squeeze(-1),
            "safety_geometry": self.safety_geometry_head(hidden).squeeze(-1),
            "recovery": self.recovery_head(hidden).squeeze(-1),
            "recovery_promising": self.recovery_promising_head(hidden).squeeze(-1),
        }


def _pairwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for row in range(scores.shape[0]):
        positive = mask[row] & (labels[row] > 0.5)
        negative = mask[row] & (labels[row] < 0.5)
        if bool(positive.any()) and bool(negative.any()):
            delta = scores[row][positive].unsqueeze(1) - scores[row][negative].unsqueeze(0)
            losses.append(-F.logsigmoid(delta).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _progress_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    pair_total = pair_correct = event_total = event_top1 = 0
    all_rows: List[Mapping[str, Any]] = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            scores = model(batch["features"])["progress"]
            for row in range(scores.shape[0]):
                valid = batch["mask"][row]
                row_scores = scores[row][valid]
                row_labels = batch["labels"][row][valid]
                pos = row_scores[row_labels > 0.5]
                neg = row_scores[row_labels < 0.5]
                if pos.numel() and neg.numel():
                    pair_total += int(pos.numel() * neg.numel())
                    pair_correct += int((pos.unsqueeze(1) > neg.unsqueeze(0)).sum().item())
                if row_scores.numel():
                    event_total += 1
                    event_top1 += int(row_labels[int(torch.argmax(row_scores))] > 0.5)
    return {
        "pairwise_accuracy": pair_correct / max(1, pair_total),
        "event_top1_positive": event_top1 / max(1, event_total),
        "pair_count": float(pair_total),
        "event_count": float(event_total),
    }


@torch.no_grad()
def _scalar_metrics(model: nn.Module, loader: DataLoader, device: torch.device, task: str) -> Dict[str, float]:
    model.eval()
    predictions: List[float] = []
    targets: List[float] = []
    auxiliary_targets: List[float] = []
    auxiliary_predictions: List[float] = []
    for raw in loader:
        batch = _move_batch(raw, device)
        outputs = model(batch["features"])
        output = torch.sigmoid(outputs[task]).detach().cpu().tolist()
        auxiliary_name = "safety_geometry" if task == "safety" else "recovery_promising"
        auxiliary_output = torch.sigmoid(outputs[auxiliary_name]).detach().cpu().tolist()
        predictions.extend(float(value) for value in output)
        targets.extend(float(value) for value in batch["score_target"].detach().cpu().tolist())
        auxiliary_targets.extend(float(value) for value in batch["aux_target"].detach().cpu().tolist())
        auxiliary_predictions.extend(float(value) for value in auxiliary_output)
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    aux_predictions = [int(value >= 0.5) for value in auxiliary_predictions]
    aux_labels = [int(value >= 0.5) for value in auxiliary_targets]
    return {
        "mae": sum(abs(value) for value in errors) / max(1, len(errors)),
        "rmse": math.sqrt(sum(value * value for value in errors) / max(1, len(errors))),
        "aux_accuracy": sum(a == b for a, b in zip(aux_predictions, aux_labels)) / max(1, len(aux_labels)),
        "examples": float(len(targets)),
    }


def _heuristic_progress_metrics(rows: Sequence[Mapping[str, Any]], name: str) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        labels = task_target(row, "progress")[0]
        grouped[_event_key(row)].append((heuristic_score(row, "progress", name), labels))
    pairs = correct = events = top1 = 0
    for values in grouped.values():
        positive = [score for score, label in values if label > 0.5]
        negative = [score for score, label in values if label < 0.5]
        if not positive or not negative:
            continue
        pairs += len(positive) * len(negative)
        correct += sum(score > neg for score in positive for neg in negative)
        events += 1
        top1 += int(max(values, key=lambda value: value[0])[1] > 0.5)
    return {"pairwise_accuracy": correct / max(1, pairs), "event_top1_positive": top1 / max(1, events)}


def _heuristic_scalar_metrics(
    rows: Sequence[Mapping[str, Any]], task: str, name: str
) -> Dict[str, float]:
    errors = []
    for row in rows:
        target, _ = task_target(row, task)
        errors.append(heuristic_score(row, task, name) - target)
    return {
        "mae": sum(abs(value) for value in errors) / max(1, len(errors)),
        "rmse": math.sqrt(sum(value * value for value in errors) / max(1, len(errors))),
        "examples": float(len(errors)),
    }


def _loss_safety(outputs: Mapping[str, torch.Tensor], score_target: torch.Tensor, aux_target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(outputs["safety"])
    return F.mse_loss(probability, score_target) + 0.25 * F.binary_cross_entropy_with_logits(outputs["safety_geometry"], aux_target)


def _loss_recovery(outputs: Mapping[str, torch.Tensor], score_target: torch.Tensor, aux_target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(outputs["recovery"])
    # Auxiliary class agreement is diagnostic only; the proxy score remains the
    # primary target and no episode outcome is used.
    return F.mse_loss(probability, score_target) + 0.10 * F.binary_cross_entropy_with_logits(outputs["recovery_promising"], aux_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    _set_seed(args.seed)
    audit = audit_dataset(args.data_dir)
    train_rows = {task: _read_task_rows(args.data_dir, task, "train") for task in TASKS}
    val_rows = {task: _read_task_rows(args.data_dir, task, "val") for task in TASKS}
    normalizer = FeatureNormalizer.fit([row for task in TASKS for row in train_rows[task]])
    progress_train = ProgressEventDataset(train_rows["progress"], normalizer)
    progress_val = ProgressEventDataset(val_rows["progress"], normalizer)
    safety_train = ScalarDataset(train_rows["safety"], "safety", normalizer)
    safety_val = ScalarDataset(val_rows["safety"], "safety", normalizer)
    recovery_train = ScalarDataset(train_rows["recovery"], "recovery", normalizer)
    recovery_val = ScalarDataset(val_rows["recovery"], "recovery", normalizer)
    loaders = {
        "progress_train": DataLoader(progress_train, args.progress_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_progress),
        "progress_val": DataLoader(progress_val, args.progress_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_progress),
        "safety_train": DataLoader(safety_train, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_scalar),
        "safety_val": DataLoader(safety_val, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_scalar),
        "recovery_train": DataLoader(recovery_train, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_scalar),
        "recovery_val": DataLoader(recovery_val, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_scalar),
    }
    model = Stage21MultiHeadScorer(len(feature_names()), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": "stage21_structured_online_v1",
        "feature_names": feature_names(),
        "input_dim": len(feature_names()),
        "excluded_from_inputs": [
            "geometry_safe", "active_gate_safe", "success", "spl", "ne",
            "route/reference fields", "episode identifiers", "candidate_id",
            "handcrafted aggregate scores (score, route/goal/semantic progress, target frontier, resilience score)",
        ],
        "tasks": {
            "progress": "S2-relative positive/negative candidate preference",
            "safety": "short_horizon_executability_proxy + geometry-safe auxiliary audit",
            "recovery": "safe_decision_state_restoration_v2 proxy; non-causal auxiliary target",
        },
    }
    (args.output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8")
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "progress_batch_size": args.progress_batch_size, "hidden_dim": args.hidden_dim,
        "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
        "device": str(device), "frozen_navigation": True, "active_navigation": False,
        "dataset_audit": audit["splits"], "train_event_count": len(progress_train),
        "val_event_count": len(progress_val), "train_recovery_rows": len(recovery_train),
        "val_recovery_rows": len(recovery_val),
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    history: List[Dict[str, Any]] = []
    best_score = float("-inf")
    global_step = 0

    # Use equal task turns rather than letting the 17k-row safety/progress
    # streams drown the 345-row proxy recovery stream.  Each task still sees
    # its full loader; shorter loaders are cycled for the epoch.
    task_loaders = {task: loaders[f"{task}_train"] for task in TASKS}
    task_steps = max(len(loader) for loader in task_loaders.values())
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = Counter()
        task_seen = Counter()
        task_iterators = {task: iter(loader) for task, loader in task_loaders.items()}
        for step in range(task_steps):
            for task in TASKS:
                try:
                    raw = next(task_iterators[task])
                except StopIteration:
                    task_iterators[task] = iter(task_loaders[task])
                    raw = next(task_iterators[task])
                batch = _move_batch(raw, device)
                if task == "progress":
                    logits = model(batch["features"])["progress"]
                    loss = _pairwise_loss(logits, batch["labels"], batch["mask"])
                else:
                    outputs = model(batch["features"])
                    if task == "safety":
                        loss = _loss_safety(outputs, batch["score_target"], batch["aux_target"])
                    else:
                        loss = _loss_recovery(outputs, batch["score_target"], batch["aux_target"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                task_seen[task] += 1
                train_losses[task] += float(loss.detach().item())
                if args.smoke_steps and global_step >= args.smoke_steps:
                    break
            if args.smoke_steps and global_step >= args.smoke_steps:
                break
        progress_metrics = _progress_metrics(model, loaders["progress_val"], device)
        safety_metrics = _scalar_metrics(model, loaders["safety_val"], device, "safety")
        recovery_metrics = _scalar_metrics(model, loaders["recovery_val"], device, "recovery")
        record = {
            "epoch": epoch, "global_step": global_step,
            "train_loss": {task: train_losses[task] / max(1, task_seen[task]) for task in TASKS},
            "train_steps": dict(task_seen),
            "train_eval": {
                "progress": _progress_metrics(model, loaders["progress_train"], device),
                "safety": _scalar_metrics(model, loaders["safety_train"], device, "safety"),
                "recovery": _scalar_metrics(model, loaders["recovery_train"], device, "recovery"),
            },
            "val": {"progress": progress_metrics, "safety": safety_metrics, "recovery": recovery_metrics},
            "val_heuristics": {
                "progress_candidate_score": _heuristic_progress_metrics(val_rows["progress"], "candidate_score"),
                "progress_intent_alignment": _heuristic_progress_metrics(val_rows["progress"], "intent_alignment"),
                "safety_low_revisit_risk": _heuristic_scalar_metrics(val_rows["safety"], "safety", "low_revisit_risk"),
                "recovery_open_score": _heuristic_scalar_metrics(val_rows["recovery"], "recovery", "open_score") if val_rows["recovery"] else {},
            },
        }
        # Pairwise progress is the primary selection criterion; scalar heads are
        # tracked but cannot silently dominate the navigation-facing ranker.
        record["selection_score"] = float(progress_metrics["pairwise_accuracy"])
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if record["selection_score"] > best_score:
            best_score = record["selection_score"]
            torch.save({"model_state": model.state_dict(), "feature_schema": schema,
                        "normalizer": normalizer.to_dict(), "training_config": config,
                        "metrics": record}, args.output_dir / "best.pt")
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if args.smoke_steps and global_step >= args.smoke_steps:
            break
    (args.output_dir / "TRAINING_SCOPE.txt").write_text(
        "offline structured scorer only\nFrozen S2/NextDiT: true\nEpisode-time parameter update: false\nActive navigation: false\nRecovery target is non-causal safe_decision_state_restoration_v2 proxy\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "best_selection_score": best_score, "global_step": global_step}, indent=2))


if __name__ == "__main__":
    main()

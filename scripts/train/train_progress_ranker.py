"""Train the lightweight Stage17 OccMem progress candidate ranker.

The trainer is intentionally standalone: it does not load InternVLA, images, or
Habitat. That keeps the first training signal cheap to validate before touching
S2/NextDiT. It supports torchrun DDP, while a one-GPU smoke run is sufficient
for the initial angle-proxy sanity check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from progress_ranker_model import ProgressCandidateRanker


class CandidateListDataset(Dataset):
    """JSONL dataset whose item contains a variable-size candidate list."""

    def __init__(self, path: Path):
        self.items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
                if not item.get("features") or not item.get("labels"):
                    raise ValueError(f"Missing features or labels at {path}:{lineno}")
                if len(item["features"]) != len(item["labels"]):
                    raise ValueError(f"Feature/label length mismatch at {path}:{lineno}")
                self.items.append(item)
        if not self.items:
            raise ValueError(f"No usable rows in {path}")
        self.feature_dim = len(self.items[0]["features"][0])
        for item in self.items:
            if any(len(feature) != self.feature_dim for feature in item["features"]):
                raise ValueError(f"Inconsistent feature dimension in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def collate_candidate_lists(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    feature_dim = len(items[0]["features"][0])
    max_candidates = max(len(item["features"]) for item in items)
    features = torch.zeros((len(items), max_candidates, feature_dim), dtype=torch.float32)
    labels = torch.zeros((len(items), max_candidates), dtype=torch.float32)
    mask = torch.zeros((len(items), max_candidates), dtype=torch.bool)
    for row, item in enumerate(items):
        count = len(item["features"])
        features[row, :count] = torch.tensor(item["features"], dtype=torch.float32)
        labels[row, :count] = torch.tensor(item["labels"], dtype=torch.float32)
        mask[row, :count] = True
    return {"features": features, "labels": labels, "mask": mask}


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Uniform-positive ListNet target; each row is guaranteed to have a positive."""

    masked_scores = scores.masked_fill(~mask, float("-inf"))
    target = labels.masked_fill(~mask, 0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(masked_scores, dim=1).masked_fill(~mask, 0.0)
    return -(target * log_probs).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = total_rows = top1_correct = mrr_sum = ndcg_sum = 0.0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        scores = model(features)
        batch_size = float(features.shape[0])
        total_loss += float(listwise_loss(scores, labels, mask).item()) * batch_size
        total_rows += batch_size
        ranked = scores.masked_fill(~mask, float("-inf")).argsort(dim=1, descending=True)
        for row in range(features.shape[0]):
            valid_count = int(mask[row].sum().item())
            ordered_labels = labels[row, ranked[row][:valid_count]]
            top1_correct += float(ordered_labels[0] > 0.0)
            positive_positions = torch.nonzero(ordered_labels > 0.0, as_tuple=False)
            if len(positive_positions):
                mrr_sum += 1.0 / float(positive_positions[0].item() + 1)
            gains = ordered_labels.clamp_min(0.0)
            discounts = torch.log2(
                torch.arange(valid_count, device=device, dtype=torch.float32) + 2.0
            )
            dcg = torch.sum(gains / discounts)
            ideal_gains = torch.sort(labels[row, mask[row]].clamp_min(0.0), descending=True).values
            idcg = torch.sum(ideal_gains / discounts)
            if float(idcg.item()) > 0.0:
                ndcg_sum += float((dcg / idcg).item())

    return {
        "loss": total_loss / max(1.0, total_rows),
        "top1_accuracy": top1_correct / max(1.0, total_rows),
        "mrr": mrr_sum / max(1.0, total_rows),
        "ndcg": ndcg_sum / max(1.0, total_rows),
        "examples": total_rows,
    }


def _setup_distributed() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("The progress ranker trainer requires CUDA.")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _set_seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_main_process(rank: int) -> bool:
    return rank == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Stage17 OccMem progress candidate ranker.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-steps", type=int, default=0, help="Stop after this many optimizer steps.")
    parser.add_argument(
        "--allow-angle-proxy-training",
        action="store_true",
        help="Required only for the temporary Stage17 angle-label smoke experiment.",
    )
    args = parser.parse_args()

    rank, local_rank, world_size, device = _setup_distributed()
    _set_seed(args.seed, rank)
    summary_path = args.data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("label_source") == "gt_direction_angle_proxy" and not args.allow_angle_proxy_training:
        raise ValueError(
            "This dataset uses temporary angle-only proxy labels. Pass "
            "--allow-angle-proxy-training only for a pipeline smoke run."
        )
    train_dataset = CandidateListDataset(args.data_dir / "train.jsonl")
    val_dataset = CandidateListDataset(args.data_dir / "val.jsonl")
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("Train/val feature dimensions differ")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True,
                     "collate_fn": collate_candidate_lists}
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    # Evaluate once on rank zero. DistributedSampler pads uneven shards, which
    # would duplicate examples and bias listwise validation metrics.
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if _is_main_process(rank) else None

    model: nn.Module = ProgressCandidateRanker(train_dataset.feature_dim, args.hidden_dim, args.dropout).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = float("-inf")
    global_step = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = listwise_loss(model(features), labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break

        if _is_main_process(rank):
            eval_model = model.module if isinstance(model, DistributedDataParallel) else model
            metrics = evaluate(eval_model, val_loader, device)
            metrics.update({"epoch": epoch, "global_step": global_step})
            history.append(metrics)
            print(json.dumps(metrics, sort_keys=True))
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save(
                    {
                        "model_state": (model.module if isinstance(model, DistributedDataParallel) else model).state_dict(),
                        "feature_dim": train_dataset.feature_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "metrics": metrics,
                    },
                    args.output_dir / "best.pt",
                )
            (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if world_size > 1:
            dist.barrier()
        if args.smoke_steps and global_step >= args.smoke_steps:
            break

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

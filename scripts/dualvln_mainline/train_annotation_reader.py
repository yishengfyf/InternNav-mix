#!/usr/bin/env python3
"""Train a reader-only relevance matrix from frozen InternVLA features."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from internnav.model.basemodel.internvla_n1.evidence_conditioning import ObservableTaskStateEstimator
from internnav.model.basemodel.internvla_n1.evidence_memory import (
    TaskConditionedEvidenceMemory,
    multi_positive_relevance_loss,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cards(features_path, supervision_path):
    loaded = np.load(features_path)
    data = {key: loaded[key] for key in loaded.files}
    supervision = {row["annotation_id"]: row for row in read_jsonl(supervision_path)}
    cards = []
    for annotation_id in sorted(set(data["annotation_ids"])):
        indices = np.flatnonzero(data["annotation_ids"] == annotation_id)
        order = np.argsort(data["candidate_labels"][indices])
        indices = indices[order]
        row = supervision.get(str(annotation_id))
        if row is None:
            raise ValueError(f"supervision 缺少 {annotation_id}")
        labels = [str(label) for label in data["candidate_labels"][indices]]
        if labels != row["candidate_labels"]:
            raise ValueError(f"{annotation_id} 的候选顺序不一致")
        metadata = data["metadata"][indices]
        cards.append(
            {
                "annotation_id": str(annotation_id),
                "episode_id": str(data["episode_ids"][indices[0]]),
                "history": data["visual"][indices, 1],
                "current": data["visual"][indices[0], 0],
                "text": data["text"][indices[0]],
                "poses": metadata[:, [1, 2, 4, 5]],
                "ages": metadata[:, [0]],
                "qualities": np.ones((len(indices), 2), dtype=np.float32),
                "targets": np.asarray(row["targets"], dtype=np.float32),
            }
        )
    if len(cards) != len(supervision):
        raise ValueError("feature cards 与 supervision 记录数不一致")
    return cards


def tensor_batch(cards, indices, device):
    def stack(key):
        return torch.tensor(np.stack([cards[index][key] for index in indices]), device=device)

    return {
        "history": stack("history"),
        "current": stack("current"),
        "text": stack("text"),
        "poses": stack("poses"),
        "ages": stack("ages"),
        "qualities": stack("qualities"),
        "targets": stack("targets"),
        "valid": torch.ones((len(indices), 3), dtype=torch.bool, device=device),
    }


def forward_reader(memory, estimator, batch, variant):
    if variant == "content":
        poses, ages, qualities = torch.zeros_like(batch["poses"]), torch.zeros_like(batch["ages"]), torch.zeros_like(batch["qualities"])
    else:
        poses, ages, qualities = batch["poses"], batch["ages"], batch["qualities"]
    if variant == "task_spatial":
        task_state = estimator(
            batch["current"], batch["text"].unsqueeze(1),
            torch.ones((len(batch["current"]), 1), dtype=torch.bool, device=batch["current"].device),
        )
    else:
        task_state = torch.zeros(
            (len(batch["current"]), memory.task_dim), device=batch["current"].device
        )
    return memory(batch["history"], poses, ages, qualities, batch["valid"], task_state)


def evaluate(memory, estimator, batch, variant):
    memory.eval(); estimator.eval()
    with torch.no_grad():
        output = forward_reader(memory, estimator, batch, variant)
        loss = multi_positive_relevance_loss(
            output.read_weights, output.null_weights, batch["targets"], batch["valid"]
        )
        reads = output.read_weights.mean(dim=1)
        null = output.null_weights.mean(dim=1)
    yes_hits, pairwise, no_hits, masses = [], [], [], []
    for sample in range(len(batch["targets"])):
        targets = batch["targets"][sample]
        if targets.lt(0).all():
            continue
        positive = targets.gt(0.5)
        if positive.any():
            yes_hits.append(int(bool(positive[reads[sample].argmax()])))
            pairwise.extend(
                float(pos > neg)
                for pos in reads[sample][positive]
                for neg in reads[sample][~positive]
            )
            masses.append(float(reads[sample][positive].sum()))
        else:
            no_hits.append(int(float(null[sample]) > float(reads[sample].max())))
            masses.append(float(null[sample]))
    return {
        "loss": float(loss),
        "yes_top1": float(np.mean(yes_hits)) if yes_hits else None,
        "pairwise_accuracy": float(np.mean(pairwise)) if pairwise else None,
        "no_null_accuracy": float(np.mean(no_hits)) if no_hits else None,
        "positive_mass": float(np.mean(masses)) if masses else None,
        "yes_cards": len(yes_hits),
        "no_cards": len(no_hits),
    }


def run_one(cards, train_indices, val_indices, variant, seed, steps, learning_rate, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    feature_dim = cards[0]["history"].shape[-1]
    memory = TaskConditionedEvidenceMemory(
        feature_dim=feature_dim, task_dim=128, hidden_dim=128, output_dim=128,
        num_evidence_tokens=4, num_heads=4, num_stages=4,
    ).to(device)
    estimator = ObservableTaskStateEstimator(feature_dim, 128).to(device)
    parameters = list(memory.parameters())
    if variant == "task_spatial":
        parameters += list(estimator.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    train_batch = tensor_batch(cards, train_indices, device)
    val_batch = tensor_batch(cards, val_indices, device)
    initial_train = evaluate(memory, estimator, train_batch, variant)
    initial_val = evaluate(memory, estimator, val_batch, variant)
    curve = []
    for step in range(steps):
        memory.train(); estimator.train(); optimizer.zero_grad(set_to_none=True)
        output = forward_reader(memory, estimator, train_batch, variant)
        loss = multi_positive_relevance_loss(
            output.read_weights, output.null_weights, train_batch["targets"], train_batch["valid"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite relevance loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            curve.append({"step": step + 1, "train_loss": float(loss.detach())})
    return {
        "variant": variant,
        "seed": seed,
        "initial_train": initial_train,
        "initial_val": initial_val,
        "final_train": evaluate(memory, estimator, train_batch, variant),
        "final_val": evaluate(memory, estimator, val_batch, variant),
        "curve": curve,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }


def write_svg(summary, path):
    rows = []
    for index, item in enumerate(summary["variants"]):
        value = item["mean_val_positive_mass"]
        y = 64 + index * 44
        rows.append(
            f'<text x="20" y="{y + 17}">{item["variant"]}</text>'
            f'<rect x="150" y="{y}" width="380" height="22" fill="#edf2f7"/>'
            f'<rect x="150" y="{y}" width="{380 * value:.1f}" height="22" fill="#2f855a"/>'
            f'<text x="540" y="{y + 17}">{value:.1%}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="630" height="230">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="31" font-size="20" font-weight="700">Reader-only relevance validation mass</text>'
        + "".join(rows) + "</svg>\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="训练人工 evidence relevance reader-only 矩阵")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 47, 71])
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cards = load_cards(args.features, args.supervision)
    runs = []
    for fold in range(args.folds):
        val_indices = [index for index, card in enumerate(cards) if int(card["episode_id"]) % args.folds == fold]
        train_indices = [index for index in range(len(cards)) if index not in val_indices]
        for seed in args.seeds:
            for variant in ("content", "spatial", "task_spatial"):
                run = run_one(
                    cards, train_indices, val_indices, variant, seed + fold * 1000,
                    args.steps, args.learning_rate, device,
                )
                run["fold"] = fold
                runs.append(run)
                print(json.dumps({"fold": fold, "seed": seed, "variant": variant, "final_val": run["final_val"]}, ensure_ascii=False), flush=True)
    variants = []
    for variant in ("content", "spatial", "task_spatial"):
        selected = [run for run in runs if run["variant"] == variant]
        variants.append(
            {
                "variant": variant,
                "mean_val_loss": float(np.mean([run["final_val"]["loss"] for run in selected])),
                "mean_val_positive_mass": float(np.mean([run["final_val"]["positive_mass"] for run in selected])),
                "mean_val_yes_top1": float(np.mean([run["final_val"]["yes_top1"] for run in selected if run["final_val"]["yes_top1"] is not None])),
                "mean_val_pairwise": float(np.mean([run["final_val"]["pairwise_accuracy"] for run in selected if run["final_val"]["pairwise_accuracy"] is not None])),
                "mean_val_no_null_accuracy": float(np.mean([run["final_val"]["no_null_accuracy"] for run in selected if run["final_val"]["no_null_accuracy"] is not None])),
            }
        )
    best = min(variants, key=lambda item: item["mean_val_loss"])
    summary = {
        "schema_version": 1, "status": "completed", "device": str(device),
        "cards": len(cards), "folds": args.folds, "seeds": args.seeds, "steps": args.steps,
        "learning_rate": args.learning_rate, "variants": variants, "best_by_val_loss": best["variant"], "runs": runs,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_svg(summary, args.output_dir / "metrics.svg")
    report = "# Reader-only 多正例训练结果\n\n" + "\n".join(
        "- `%s`：val loss %.4f，正例质量 %.1f%%，yes Top-1 %.1f%%，pairwise %.1f%%，no/null %.1f%%" % (
            item["variant"], item["mean_val_loss"], item["mean_val_positive_mass"] * 100,
            item["mean_val_yes_top1"] * 100, item["mean_val_pairwise"] * 100,
            item["mean_val_no_null_accuracy"] * 100,
        ) for item in variants
    )
    report += f"\n\n按验证 loss 最佳：`{best['variant']}`。本阶段只训练独立 reader，不更新 S2 checkpoint；单场景小样本结果不代表导航收益。\n"
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

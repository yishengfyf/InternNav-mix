#!/usr/bin/env python3
"""Evaluate a saved evidence adapter on episode-disjoint relevance cards."""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")
sys.path.insert(0, str(REPO_ROOT))
if ISOLATED_DEPS.is_dir():
    sys.path.append(str(ISOLATED_DEPS))
os.environ["INTERNNAV_TRAJ_DATA_ROOT"] = str(DATA_ROOT)
os.environ["INTERNNAV_CHECKPOINT_ROOT"] = str(CHECKPOINT.parent)


def mean(values):
    return statistics.fmean(values) if values else None


def summarize(rows):
    yes = [row for row in rows if row["target_type"] == "history"]
    no = [row for row in rows if row["target_type"] == "null"]
    pairwise = [value for row in yes for value in row["pairwise_results"]]
    return {
        "samples": len(rows),
        "yes_samples": len(yes),
        "no_samples": len(no),
        "mean_target_mass": mean([row["target_mass"] for row in rows]),
        "uniform_mean_target_mass": mean([row["uniform_target_mass"] for row in rows]),
        "yes_top1_accuracy": mean([row["top1_correct"] for row in yes]),
        "uniform_yes_top1_accuracy": mean([row["uniform_top1_probability"] for row in yes]),
        "yes_history_top1_accuracy": mean([row["history_top1_correct"] for row in yes]),
        "uniform_yes_history_top1_accuracy": mean(
            [row["uniform_history_top1_probability"] for row in yes]
        ),
        "yes_conditional_target_mass": mean([row["conditional_target_mass"] for row in yes]),
        "pairwise_accuracy": mean(pairwise),
        "no_null_accuracy": mean([row["top1_correct"] for row in no]),
        "uniform_no_null_accuracy": mean([row["uniform_top1_probability"] for row in no]),
        "need_history_binary_accuracy": mean([row["binary_correct"] for row in rows]),
        "mean_normalized_entropy": mean([row["normalized_entropy"] for row in rows]),
        "mean_null_mass": mean([row["null_mass"] for row in rows]),
    }


def write_svg(report, path):
    train = report["splits"]["train"]
    heldout = report["splits"]["heldout"]
    fields = (
        ("目标质量", "mean_target_mass", "uniform_mean_target_mass"),
        ("Yes Top-1", "yes_top1_accuracy", "uniform_yes_top1_accuracy"),
        ("Pairwise", "pairwise_accuracy", None),
        ("No/null", "no_null_accuracy", "uniform_no_null_accuracy"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390">',
        '<rect width="900" height="390" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">Evidence adapter relevance 诊断</text>',
    ]
    for index, (label, key, baseline_key) in enumerate(fields):
        y = 62 + index * 72
        parts.append(f'<text x="24" y="{y + 18}" font-size="14">{label}</text>')
        for offset, (name, values, color) in enumerate(
            (("train", train, "#2b6cb0"), ("held-out", heldout, "#2f855a"))
        ):
            value = values[key] or 0.0
            baseline = values[baseline_key] if baseline_key else None
            by = y + offset * 25
            parts.append(f'<rect x="160" y="{by}" width="{500 * value:.1f}" height="18" fill="{color}"/>')
            parts.append(f'<text x="{170 + 500 * value:.1f}" y="{by + 14}" font-size="12">{name} {value:.3f}</text>')
            if baseline is not None:
                x = 160 + 500 * baseline
                parts.append(f'<line x1="{x:.1f}" y1="{by - 2}" x2="{x:.1f}" y2="{by + 20}" stroke="#c53030" stroke-width="2"/>')
    parts.append('<text x="24" y="370" font-size="12" fill="#555">红线：按每张卡有效候选数计算的均匀随机期望；held-out 按 episode 隔离。</text>')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--relevance-supervision", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from internnav.dataset.internvla_n1_lerobot_dataset import (
        DataCollatorForSupervisedDataset,
        NavPixelGoalDataset,
        evidence_sample_identity,
    )
    from internnav.model.basemodel.internvla_n1.evidence_inference import (
        configure_task_spatial_inference,
        load_evidence_adapter,
    )
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM,
        InternVLAN1ModelConfig,
    )
    from scripts.dualvln_mainline.real_overfit import select_relevance_samples

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=False)
    tokenizer.model_max_length = 1024
    processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    data_args = SimpleNamespace(
        vln_dataset_use="r2r_125cm_0_30",
        video_max_total_pixels=1664 * 28 * 28,
        video_min_total_pixels=256 * 28 * 28,
        model_type="internvla-n1",
        sample_step=1,
        predict_step_num=32,
        pixel_goal_only=True,
        num_future_steps=4,
        num_history=3,
        image_processor=processor.image_processor,
        transform_train=None,
        use_evidence_memory=True,
        task_state_supervision=False,
        task_alignment_path=None,
        evidence_relevance_path=str(args.relevance_supervision),
        evidence_relevance_only=True,
        max_pixels=224 * 224,
        min_pixels=224 * 224,
    )
    dataset = NavPixelGoalDataset(tokenizer, data_args)
    train_indices = select_relevance_samples(dataset, args.samples, args.seed)
    train_episodes = {evidence_sample_identity(dataset.list_data_dict[index])[:2] for index in train_indices}
    heldout_indices = select_relevance_samples(
        dataset,
        args.validation_samples,
        args.seed + 100,
        excluded_episodes=train_episodes,
    )

    config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
    configure_task_spatial_inference(config)
    config.evidence_normalize_task_state = True
    config.evidence_latent_query_bypass = False
    config.evidence_record_diagnostics = True
    config.use_cache = False
    max_memory = {index: "20GiB" for index in range(torch.cuda.device_count())}
    max_memory["cpu"] = "64GiB"
    model = InternVLAN1ForCausalLM.from_pretrained(
        CHECKPOINT,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
    )
    adapter_manifest = load_evidence_adapter(model, args.adapter)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)

    def evaluate(indices, split):
        rows = []
        for offset, index in enumerate(indices):
            random.seed(args.seed * 1000 + offset)
            torch.manual_seed(args.seed * 1000 + offset)
            torch.cuda.manual_seed_all(args.seed * 1000 + offset)
            cpu_batch = collator([dataset[index]])
            batch = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in cpu_batch.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch)
            diagnostics = model.latest_evidence_diagnostics
            candidate_mass = torch.tensor(diagnostics["read_weights"])[0].mean(dim=0)
            null_mass = float(torch.tensor(diagnostics["null_weights"])[0].mean())
            targets = cpu_batch["evidence_relevance_targets"][0]
            valid = cpu_batch["evidence_valid_mask"][0].bool()
            positive = targets.gt(0.5) & valid
            supervised_negative = targets.eq(0) & valid
            valid_scores = candidate_mass[valid]
            distribution = torch.cat((valid_scores, torch.tensor([null_mass])))
            distribution = distribution / distribution.sum().clamp_min(1e-8)
            entropy = float(-(distribution * distribution.clamp_min(1e-8).log()).sum())
            normalized_entropy = entropy / math.log(max(distribution.numel(), 2))
            if positive.any():
                target_type = "history"
                target_mass = float(candidate_mass[positive].sum())
                best = int(torch.cat((candidate_mass, torch.tensor([null_mass]))).argmax())
                top1_correct = float(best < positive.numel() and positive[best])
                history_best = int(candidate_mass.masked_fill(~valid, float("-inf")).argmax())
                history_top1_correct = float(positive[history_best])
                negatives = torch.cat((candidate_mass[supervised_negative], torch.tensor([null_mass])))
                pairwise_results = [
                    float(pos > neg) for pos in candidate_mass[positive] for neg in negatives
                ]
                uniform_target_mass = float(positive.sum()) / distribution.numel()
                uniform_top1_probability = uniform_target_mass
                uniform_history_top1_probability = float(positive.sum()) / int(valid.sum())
                conditional_target_mass = float(
                    candidate_mass[positive].sum() / candidate_mass[valid].sum().clamp_min(1e-8)
                )
                binary_correct = float(float(candidate_mass[valid].sum()) > null_mass)
            else:
                target_type = "null"
                target_mass = null_mass
                top1_correct = float(null_mass > float(candidate_mass[valid].max()))
                pairwise_results = []
                uniform_target_mass = 1.0 / distribution.numel()
                uniform_top1_probability = uniform_target_mass
                history_top1_correct = None
                uniform_history_top1_probability = None
                conditional_target_mass = None
                binary_correct = float(null_mass > float(candidate_mass[valid].sum()))
            entry = dataset.list_data_dict[index]
            scene_id, _, _ = evidence_sample_identity(entry)
            rows.append(
                {
                    "split": split,
                    "dataset_index": index,
                    "episode_id": int(entry[0]),
                    "scene_id": scene_id,
                    "current_frame_id": int(entry[7][0]),
                    "target_type": target_type,
                    "target_mass": target_mass,
                    "uniform_target_mass": uniform_target_mass,
                    "top1_correct": top1_correct,
                    "uniform_top1_probability": uniform_top1_probability,
                    "history_top1_correct": history_top1_correct,
                    "uniform_history_top1_probability": uniform_history_top1_probability,
                    "conditional_target_mass": conditional_target_mass,
                    "binary_correct": binary_correct,
                    "pairwise_results": pairwise_results,
                    "null_mass": null_mass,
                    "candidate_mass": candidate_mass.tolist(),
                    "targets": targets.tolist(),
                    "normalized_entropy": normalized_entropy,
                }
            )
        return rows

    train_rows = evaluate(train_indices, "train")
    heldout_rows = evaluate(heldout_indices, "heldout")
    all_rows = train_rows + heldout_rows
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "single_scene_episode_disjoint_relevance_diagnostic",
        "git_commit": args.commit,
        "adapter": adapter_manifest,
        "splits": {"train": summarize(train_rows), "heldout": summarize(heldout_rows)},
        "duration_s": time.monotonic() - start,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train = report["splits"]["train"]
    heldout = report["splits"]["heldout"]
    summary = f"""# Evidence adapter relevance 诊断简报

- 状态：`通过`
- 范围：R2R train 单场景，训练/验证按 episode 隔离
- 样本：训练 `{train['samples']}`，held-out `{heldout['samples']}`
- 原轨迹模块：冻结；latent-query bypass：关闭

|划分|目标质量 / 均匀基线|Yes 全局 Top-1 / 均匀基线|Yes 历史内 Top-1 / 均匀基线|Pairwise|No/null Top-1|历史需求二分类|归一化熵|
|---|---:|---:|---:|---:|---:|---:|---:|
|train|{train['mean_target_mass']:.3f} / {train['uniform_mean_target_mass']:.3f}|{train['yes_top1_accuracy']:.3f} / {train['uniform_yes_top1_accuracy']:.3f}|{train['yes_history_top1_accuracy']:.3f} / {train['uniform_yes_history_top1_accuracy']:.3f}|{train['pairwise_accuracy']:.3f}|{train['no_null_accuracy']:.3f}|{train['need_history_binary_accuracy']:.3f}|{train['mean_normalized_entropy']:.3f}|
|held-out|{heldout['mean_target_mass']:.3f} / {heldout['uniform_mean_target_mass']:.3f}|{heldout['yes_top1_accuracy']:.3f} / {heldout['uniform_yes_top1_accuracy']:.3f}|{heldout['yes_history_top1_accuracy']:.3f} / {heldout['uniform_yes_history_top1_accuracy']:.3f}|{heldout['pairwise_accuracy']:.3f}|{heldout['no_null_accuracy']:.3f}|{heldout['need_history_binary_accuracy']:.3f}|{heldout['mean_normalized_entropy']:.3f}|

## 解释边界

目标质量和 Top-1 均把 null 作为竞争项；红线按每张卡的有效候选数与正例数计算，不是假定固定四分类。该结果只诊断当前人工单场景标签上的 reader，不代表 R2R val-unseen 导航泛化。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(report, args.output_dir / "metrics.svg")
    print(json.dumps(report["splits"], ensure_ascii=False))


if __name__ == "__main__":
    main()

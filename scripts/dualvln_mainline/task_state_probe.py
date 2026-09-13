#!/usr/bin/env python3
import argparse
import copy
import itertools
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ISOLATED_DEPS))
os.environ["INTERNNAV_TRAJ_DATA_ROOT"] = str(DATA_ROOT)
os.environ["INTERNNAV_CHECKPOINT_ROOT"] = str(CHECKPOINT.parent)


def write_svg(metrics, path):
    values = (
        ("task cosine", metrics.get("mean_instruction_task_cosine", 0.0), 1.0, "#4a5568"),
        ("read L1 delta", metrics.get("mean_instruction_read_l1", 0.0), 1.0, "#2b6cb0"),
        ("stage flip rate", metrics.get("instruction_stage_flip_rate", 0.0), 1.0, "#d69e2e"),
        ("progress probe MAE", metrics.get("progress_probe_mae", 0.0), 1.0, "#2f855a"),
        ("constant MAE", metrics.get("progress_constant_mae", 0.0), 1.0, "#c53030"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="390">',
        '<rect width="850" height="390" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">Task-state 语义与指令敏感性探针</text>',
    ]
    for index, (label, value, maximum, color) in enumerate(values):
        y = 62 + index * 58
        width = 520 * min(max(value / maximum, 0.0), 1.0)
        parts.extend(
            [
                f'<text x="24" y="{y + 17}" font-size="14">{label}</text>',
                f'<rect x="190" y="{y}" width="{width:.1f}" height="22" fill="{color}"/>',
                f'<text x="{200 + width:.1f}" y="{y + 16}" font-size="13">{value:.4f}</text>',
            ]
        )
    parts.append('<text x="24" y="365" font-size="13" fill="#8b1a1a">粗路线进度只用于诊断，不等同于语言子任务阶段。</text></svg>\n')
    path.write_text("".join(parts), encoding="utf-8")


def write_report(args, report):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = report.get("metrics", {})
    summary = f"""# Task-state 语义与指令敏感性探针简报

- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`
- 真实 train 因果样本：`{metrics.get('samples', 0)}`
- 独立 episode / 指令：`{metrics.get('unique_episodes', 0)} / {metrics.get('unique_instructions', 0)}`
- 四个粗进度区间样本数：`{metrics.get('samples_per_progress_bin', [])}`
- 权重：`M2-lr01 seed 23`
- 诊断标签：路线帧进度四分位，不是自然语言子任务真值

|指标|结果|
|---|---:|
|原/打乱指令 task-state 平均余弦|{metrics.get('mean_instruction_task_cosine', 0):.6f}|
|task-state 相对 L2 变化|{metrics.get('mean_instruction_task_relative_l2', 0):.6f}|
|读取权重平均 L1 变化|{metrics.get('mean_instruction_read_l1', 0):.6f}|
|stage argmax 翻转率|{metrics.get('instruction_stage_flip_rate', 0):.3f}|
|stage 最佳置换后四分位准确率|{metrics.get('stage_best_permutation_accuracy', 0):.3f}|
|四折 PCA-ridge 进度 MAE / 常数 MAE|{metrics.get('progress_probe_mae', 0):.4f} / {metrics.get('progress_constant_mae', 0):.4f}|
|耗时（秒）|{metrics.get('duration_s', 0):.2f}|

## 简述与分析

{report['analysis']}

逐样本结果见 `probe.jsonl`，样本与粗进度标签见 `sample_manifest.json`。本探针不更新参数，也不把 route progress 当作已经成立的任务阶段监督。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(metrics, args.output_dir / "metrics.svg")


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def select_stratified(dataset, samples, seed):
    bins = [[] for _ in range(4)]
    for index, entry in enumerate(dataset.list_data_dict):
        frame_id = int(entry[7][0])
        episode_length = len(entry[10])
        if frame_id < 2 or entry[9] is None or episode_length < 2:
            continue
        progress = frame_id / max(episode_length - 1, 1)
        bins[min(int(progress * 4), 3)].append((index, progress, entry[0]))
    per_bin = samples // 4
    if samples % 4 or any(len(values) < per_bin for values in bins):
        raise RuntimeError(f"cannot select {samples} balanced samples from progress bins: {[len(v) for v in bins]}")
    rng = random.Random(seed)
    for values in bins:
        rng.shuffle(values)
    selected = []
    used_episodes = set()
    for stage, values in enumerate(bins):
        stage_values = []
        for index, progress, episode_id in values:
            if episode_id in used_episodes:
                continue
            stage_values.append((index, progress, episode_id))
            used_episodes.add(episode_id)
            if len(stage_values) == per_bin:
                break
        if len(stage_values) != per_bin:
            raise RuntimeError(
                f"cannot select {per_bin} distinct episodes for progress bin {stage}; "
                f"selected={len(stage_values)} total_unique={len(used_episodes)}"
            )
        for offset, (index, progress, episode_id) in enumerate(stage_values):
            selected.append(
                {
                    "index": index,
                    "episode_id": episode_id,
                    "progress": progress,
                    "stage": stage,
                    "fold": offset % 4,
                }
            )
    return selected


def mismatch_instructions(instructions, seed):
    """Return a deterministic permutation with no unchanged instruction text."""
    if len(set(instructions)) < 2:
        raise RuntimeError("instruction mismatch control needs at least two distinct instructions")
    order = list(range(len(instructions)))
    random.Random(seed).shuffle(order)
    shuffled = [instructions[index] for index in order]
    for offset in range(1, len(shuffled)):
        rotated = shuffled[offset:] + shuffled[:offset]
        if all(original != replacement for original, replacement in zip(instructions, rotated)):
            return rotated
    raise RuntimeError("could not construct an instruction permutation without fixed text")


def best_stage_accuracy(predictions, targets):
    best = 0.0
    for permutation in itertools.permutations(range(4)):
        correct = sum(permutation[prediction] == target for prediction, target in zip(predictions, targets))
        best = max(best, correct / len(targets))
    return best


def ridge_progress_probe(features, targets, folds, torch, alpha=1.0, max_components=8):
    # Fit PCA inside each fold. The raw task-state is 512-D, so an unconstrained
    # linear probe on only 24 training points would be an unreliable decoder.
    features = features.to(dtype=torch.float64, device="cpu")
    targets = targets.to(dtype=torch.float64, device="cpu")
    predictions = torch.empty_like(targets)
    baselines = torch.empty_like(targets)
    for fold in sorted(set(folds)):
        train_mask = torch.tensor([value != fold for value in folds], dtype=torch.bool)
        test_mask = ~train_mask
        train_x = features[train_mask]
        test_x = features[test_mask]
        mean = train_x.mean(dim=0, keepdim=True)
        train_x = train_x - mean
        test_x = test_x - mean
        _, _, components = torch.linalg.svd(train_x, full_matrices=False)
        component_count = min(max_components, len(train_x) - 1, components.shape[0])
        components = components[:component_count]
        train_x = train_x @ components.T
        test_x = test_x @ components.T
        scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-8)
        train_x = train_x / scale
        test_x = test_x / scale
        train_y = targets[train_mask]
        y_mean = train_y.mean()
        centered_y = train_y - y_mean
        gram = train_x.T @ train_x
        ridge = gram + alpha * torch.eye(component_count, dtype=train_x.dtype)
        weights = torch.linalg.solve(ridge, train_x.T @ centered_y)
        predictions[test_mask] = y_mean + test_x @ weights
        baselines[test_mask] = y_mean
    return float((predictions - targets).abs().mean()), float((baselines - targets).abs().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {
        "schema_version": 1,
        "git_commit": args.commit,
        "status": "failed",
        "metrics": {},
        "analysis": "Task-state 探针未完成。",
    }
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoProcessor, AutoTokenizer

        from internnav.dataset.internvla_n1_lerobot_dataset import DataCollatorForSupervisedDataset, NavPixelGoalDataset
        from internnav.model.basemodel.internvla_n1.evidence_inference import (
            configure_task_spatial_inference,
            load_evidence_adapter,
        )
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig

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
            sample_step=4,
            predict_step_num=32,
            pixel_goal_only=True,
            num_future_steps=4,
            num_history=2,
            image_processor=processor.image_processor,
            transform_train=None,
            use_evidence_memory=True,
            max_pixels=224 * 224,
            min_pixels=224 * 224,
        )
        dataset = NavPixelGoalDataset(tokenizer, data_args)
        selected = select_stratified(dataset, args.samples, args.seed)
        instructions = [dataset.list_data_dict[item["index"]][6] for item in selected]
        shifted_instructions = mismatch_instructions(instructions, args.seed + 1)
        manifest = []
        for item, shuffled_instruction in zip(selected, shifted_instructions):
            entry = dataset.list_data_dict[item["index"]]
            manifest.append(
                {
                    **item,
                    "current_frame_id": entry[7][0],
                    "episode_length": len(entry[10]),
                    "instruction": entry[6],
                    "shuffled_instruction": shuffled_instruction,
                }
            )
        (args.output_dir / "sample_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        configure_task_spatial_inference(config)
        config.use_cache = False
        model = InternVLAN1ForCausalLM.from_pretrained(
            CHECKPOINT,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB", 3: "20GiB", "cpu": "64GiB"},
        )
        adapter_manifest = load_evidence_adapter(model, args.adapter)
        model.eval()
        input_device = model.get_input_embeddings().weight.device
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)
        captured = []

        def capture_task_state(module, inputs, output):
            del module, inputs
            captured.append(output.detach().float().cpu())

        handle = model.get_model().task_state_estimator.register_forward_hook(capture_task_state)

        def evaluate_sample(index, instruction, sample_seed):
            original_entry = dataset.list_data_dict[index]
            replaced = list(original_entry)
            replaced[6] = instruction
            dataset.list_data_dict[index] = tuple(replaced)
            try:
                random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                batch = collator([dataset[index]])
                captured.clear()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**move_batch(batch, input_device))
                if len(captured) != 1:
                    raise RuntimeError(f"expected one task-state call, got {len(captured)}")
                return captured[0][0].clone(), copy.deepcopy(model.latest_evidence_diagnostics)
            finally:
                dataset.list_data_dict[index] = original_entry

        rows = []
        original_features = []
        for item, original_instruction, shuffled_instruction in zip(selected, instructions, shifted_instructions):
            sample_seed = args.seed * 1000 + item["index"]
            original_state, original_diag = evaluate_sample(item["index"], original_instruction, sample_seed)
            shuffled_state, shuffled_diag = evaluate_sample(item["index"], shuffled_instruction, sample_seed)
            cosine = float(F.cosine_similarity(original_state[None], shuffled_state[None]))
            relative_l2 = float((original_state - shuffled_state).norm() / original_state.norm().clamp_min(1e-6))
            original_read = torch.tensor(original_diag["read_weights"])
            shuffled_read = torch.tensor(shuffled_diag["read_weights"])
            read_l1 = float((original_read - shuffled_read).abs().mean())
            original_stage = int(torch.tensor(original_diag["stage_logits"])[0].argmax())
            shuffled_stage = int(torch.tensor(shuffled_diag["stage_logits"])[0].argmax())
            rows.append(
                {
                    **item,
                    "task_cosine": cosine,
                    "task_relative_l2": relative_l2,
                    "read_l1": read_l1,
                    "stage_original": original_stage,
                    "stage_shuffled": shuffled_stage,
                    "stage_probabilities": original_diag["stage_probabilities"][0],
                    "null_weights": original_diag["null_weights"][0],
                }
            )
            original_features.append(original_state)
        handle.remove()
        with (args.output_dir / "probe.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        features = torch.stack(original_features)
        targets = torch.tensor([item["progress"] for item in selected], dtype=torch.float32)
        folds = [item["fold"] for item in selected]
        probe_mae, constant_mae = ridge_progress_probe(features, targets, folds, torch)
        stage_accuracy = best_stage_accuracy(
            [row["stage_original"] for row in rows], [item["stage"] for item in selected]
        )
        cosine = statistics.fmean(row["task_cosine"] for row in rows)
        read_l1 = statistics.fmean(row["read_l1"] for row in rows)
        stage_flip = statistics.fmean(row["stage_original"] != row["stage_shuffled"] for row in rows)
        metrics = {
            "samples": len(rows),
            "unique_episodes": len({item["episode_id"] for item in selected}),
            "unique_instructions": len(set(instructions)),
            "samples_per_progress_bin": [sum(item["stage"] == stage for item in selected) for stage in range(4)],
            "mean_instruction_task_cosine": cosine,
            "mean_instruction_task_relative_l2": statistics.fmean(row["task_relative_l2"] for row in rows),
            "mean_instruction_read_l1": read_l1,
            "instruction_stage_flip_rate": stage_flip,
            "stage_best_permutation_accuracy": stage_accuracy,
            "progress_probe_mae": probe_mae,
            "progress_constant_mae": constant_mae,
            "adapter_manifest": adapter_manifest,
            "duration_s": time.monotonic() - start,
        }
        weak_instruction_signal = cosine > 0.995 and read_l1 < 0.01 and stage_flip < 0.1
        weak_progress_signal = probe_mae >= 0.95 * constant_mae and stage_accuracy <= 0.40
        if weak_instruction_signal and weak_progress_signal:
            analysis = "Task-state 对打乱指令近乎不敏感，且粗路线进度不比常数基线更可解码；当前不能支持任务阶段语义，应先增加因果阶段/进度监督并重训。"
        elif weak_instruction_signal:
            analysis = "粗路线进度存在部分可解码信号，但 task-state 对指令置换近乎不敏感；更可能编码视觉/位置进度，而非指令条件化任务阶段。"
        elif weak_progress_signal:
            analysis = "Task-state 会随指令变化，但尚未显示稳定的粗路线进度结构；需要更贴近指令子句的阶段标签或对比监督。"
        else:
            analysis = "探针观察到指令敏感性和部分粗进度结构；仍需跨 episode/scene 复验并与视觉-only probe 比较，不能直接宣称任务阶段理解。"
        report.update(status="passed", metrics=metrics, analysis=analysis)
    except Exception as error:
        report["metrics"]["duration_s"] = time.monotonic() - start
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["analysis"] = "Task-state 探针执行失败；保留异常与已有闭环结论，不进入阶段监督训练。"
    write_report(args, report)
    print(f"task_state_probe={report['status']} metrics={json.dumps(report.get('metrics', {}), ensure_ascii=False)}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

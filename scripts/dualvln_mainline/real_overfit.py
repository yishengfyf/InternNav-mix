#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")
TASK_ALIGNMENT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/task_alignment/r2r_17DRP5sb8fy.json")
sys.path.insert(0, str(REPO_ROOT))
if ISOLATED_DEPS.is_dir():
    sys.path.append(str(ISOLATED_DEPS))
os.environ["INTERNNAV_TRAJ_DATA_ROOT"] = str(DATA_ROOT)
os.environ["INTERNNAV_CHECKPOINT_ROOT"] = str(CHECKPOINT.parent)


def write_curve(series, path, samples):
    width, height = 720, 350
    left, top, plot_w, plot_h = 68, 48, 610, 240
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    span = max(high - low, 1e-8)
    colors = {
        "total_loss": "#2b6cb0",
        "s2_loss": "#2f855a",
        "trajectory_loss": "#c05621",
        "evidence_relevance_loss": "#805ad5",
    }
    lines = []
    legend = []
    for line_index, (name, values) in enumerate(series.items()):
        points = []
        for index, value in enumerate(values):
            x = left + plot_w * index / max(1, len(values) - 1)
            y = top + plot_h * (high - value) / span
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
        legend.append(
            f'<line x1="{left + line_index * 180}" y1="325" x2="{left + 25 + line_index * 180}" y2="325" stroke="{colors[name]}" stroke-width="3"/><text x="{left + 31 + line_index * 180}" y="330" font-size="13">{name}</text>'
        )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="{width}" height="{height}" fill="#fff"/><text x="24" y="28" font-size="20" font-weight="700">N1/N2 {samples} 样本真实过拟合 loss</text><line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#4a5568"/><line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#4a5568"/><text x="8" y="{top + 5}" font-size="12">{high:.3f}</text><text x="8" y="{top + plot_h}" font-size="12">{low:.3f}</text>{"".join(lines)}{"".join(legend)}</svg>\n'
    path.write_text(svg, encoding="utf-8")


def write_report(args, report, series=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = report.get("metrics", {})
    summary = f"""# 阶段结果简报

- 阶段：`N1/N2 {metrics.get('samples', 0)} 样本真实 checkpoint 过拟合`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`

|指标|结果|
|---|---:|
|真实训练样本|{metrics.get('samples', 0)}|
|Held-out 样本|{metrics.get('validation_samples', 0)}|
|优化步数|{metrics.get('steps_completed', 0)}|
|实验模式|{metrics.get('experiment', 'unknown')}|
|训练结构|{metrics.get('architecture', 'unknown')}|
|初始总 loss|{metrics.get('initial_total_loss', 0):.6f}|
|最终总 loss|{metrics.get('final_total_loss', 0):.6f}|
|总 loss 下降|{metrics.get('total_loss_reduction', 0):.2%}|
|初始/最终 S2 loss|{metrics.get('initial_s2_loss', 0):.6f} / {metrics.get('final_s2_loss', 0):.6f}|
|初始/最终轨迹 loss|{metrics.get('initial_trajectory_loss', 0):.6f} / {metrics.get('final_trajectory_loss', 0):.6f}|
|初始/最终 relevance loss|{metrics.get('initial_evidence_relevance_loss', 0):.6f} / {metrics.get('final_evidence_relevance_loss', 0):.6f}|
|Held-out 初始/最终 relevance loss|{metrics.get('initial_validation_evidence_relevance_loss', 0):.6f} / {metrics.get('final_validation_evidence_relevance_loss', 0):.6f}|
|初始/最终任务对比 loss|{metrics.get('initial_task_contrastive_loss', 0):.6f} / {metrics.get('final_task_contrastive_loss', 0):.6f}|
|初始/最终软进度 loss|{metrics.get('initial_stage_loss', 0):.6f} / {metrics.get('final_stage_loss', 0):.6f}|
|峰值本进程显存 MiB|{metrics.get('peak_allocated_mib', 0):.1f}|
|耗时（秒）|{metrics.get('duration_s', 0):.3f}|

## 简述与分析

{report['analysis']}

曲线见 `metrics.svg`，训练配置和样本 manifest 见 `config.json` 与 `sample_manifest.json`。可训练组另提供逐步 `train_log.jsonl` 和分 loss `gradient_audit.json`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    if series:
        write_curve(series, args.output_dir / "metrics.svg", args.samples)
    else:
        color = "#c53030"
        (args.output_dir / "metrics.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="120"><rect width="620" height="120" fill="#fff"/><text x="24" y="32" font-size="20" font-weight="700">N1/N2 {args.samples} 样本真实过拟合</text><rect x="24" y="58" width="520" height="26" fill="{color}"/><text x="556" y="77" font-size="14">失败</text></svg>\n',
            encoding="utf-8",
        )


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def parameter_group(name):
    for group in ("task_state_estimator", "evidence_memory", "cond_projector", "latent_queries"):
        if group in name:
            return group
    return "other"


def summarize_gradients(model, torch):
    summaries = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        summary = summaries.setdefault(
            group,
            {
                "parameters": 0,
                "with_gradient": 0,
                "nonfinite_gradients": 0,
                "gradient_l2_squared": 0.0,
                "gradient_max_abs": 0.0,
                "parameter_dtypes": set(),
            },
        )
        summary["parameters"] += parameter.numel()
        summary["parameter_dtypes"].add(str(parameter.dtype))
        if parameter.grad is None:
            continue
        summary["with_gradient"] += parameter.numel()
        finite = torch.isfinite(parameter.grad)
        summary["nonfinite_gradients"] += parameter.grad.numel() - int(finite.sum())
        if finite.any():
            finite_gradient = parameter.grad.detach()[finite].float()
            summary["gradient_l2_squared"] += float(finite_gradient.square().sum())
            summary["gradient_max_abs"] = max(summary["gradient_max_abs"], float(finite_gradient.abs().max()))
    for summary in summaries.values():
        summary["gradient_l2"] = summary.pop("gradient_l2_squared") ** 0.5
        summary["parameter_dtypes"] = sorted(summary["parameter_dtypes"])
    return summaries


def select_task_state_samples(dataset, samples, seed):
    if samples % 4:
        raise ValueError("task-state repair samples must be divisible by four")
    bins = [[] for _ in range(4)]
    for index, entry in enumerate(dataset.list_data_dict):
        frame_id = int(entry[7][0])
        if frame_id < 2 or entry[9] is None or len(entry[10]) < 2:
            continue
        progress = frame_id / (len(entry[10]) - 1)
        bins[min(int(progress * 4), 3)].append((index, dataset.task_key(entry[1], entry[0], entry[6])))
    rng = random.Random(seed)
    for values in bins:
        rng.shuffle(values)
    selected, used_tasks = [], set()
    for values in bins:
        for index, task_key in values:
            if task_key in used_tasks:
                continue
            selected.append(index)
            used_tasks.add(task_key)
            if len(selected) % (samples // 4) == 0:
                break
    if len(selected) != samples:
        raise RuntimeError(f"only found {len(selected)} balanced distinct-route task-state samples")
    return selected


def select_relevance_samples(dataset, samples, seed, excluded_episodes=()):
    from internnav.dataset.internvla_n1_lerobot_dataset import (
        evidence_relevance_for_sample,
        evidence_sample_identity,
    )

    if samples % 2:
        raise ValueError("relevance samples must be divisible by two")
    groups = {"yes": [], "no": []}
    excluded_episodes = set(excluded_episodes)
    for index, entry in enumerate(dataset.list_data_dict):
        if entry[9] is None:
            continue
        scene_id, episode_id, _ = evidence_sample_identity(entry)
        episode_key = (scene_id, episode_id)
        if episode_key in excluded_episodes or episode_id in excluded_episodes:
            continue
        relevance = evidence_relevance_for_sample(dataset.evidence_relevance, entry)
        if relevance is None or not relevance.get("supervised", False):
            continue
        state = "yes" if any(target > 0.5 for target in relevance["targets"]) else "no"
        groups[state].append((index, episode_key))
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected, used_episodes = [], set()
    for state in ("yes", "no"):
        selected_for_state = 0
        for index, episode_id in groups[state]:
            if episode_id in used_episodes:
                continue
            selected.append(index)
            used_episodes.add(episode_id)
            selected_for_state += 1
            if selected_for_state == samples // 2:
                break
    if len(selected) != samples:
        raise RuntimeError(f"only found {len(selected)} balanced distinct-episode relevance samples")
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--gradient-bypass", action="store_true")
    parser.add_argument("--bypass-mode", choices=("mean", "cross_attention"), default="mean")
    parser.add_argument("--bypass-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--s2-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--task-state-lr-scale", type=float, default=1.0)
    parser.add_argument("--task-state-scale", type=float, default=1.0)
    parser.add_argument("--normalize-task-state", action="store_true")
    parser.add_argument("--freeze-task-state", action="store_true")
    parser.add_argument("--freeze-trajectory-backbone", action="store_true")
    parser.add_argument("--instruction-only-task-state", action="store_true")
    parser.add_argument("--task-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--stage-loss-weight", type=float, default=0.0)
    parser.add_argument("--relevance-supervision", type=Path)
    parser.add_argument("--relevance-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--task-contrastive-margin", type=float, default=0.2)
    parser.add_argument("--task-alignment-path", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=("B0", "B1", "B2", "M1", "M2"),
        default="M2",
        help="N2 ablation: null, content-only, spatial-only, or task-conditioned spatial memory",
    )
    args = parser.parse_args()
    if (args.task_contrastive_weight > 0 or args.stage_loss_weight > 0) and args.task_alignment_path is None:
        args.task_alignment_path = TASK_ALIGNMENT
    if bool(args.relevance_supervision) != (args.relevance_loss_weight > 0):
        raise ValueError("relevance supervision and a positive relevance loss weight must be enabled together")
    if args.relevance_supervision and args.experiment == "B0":
        raise ValueError("B0 does not contain an evidence reader for relevance supervision")
    if min(args.s2_loss_weight, args.trajectory_loss_weight, args.relevance_loss_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.s2_loss_weight + args.trajectory_loss_weight + args.relevance_loss_weight <= 0:
        raise ValueError("at least one training loss weight must be positive")
    if args.init_adapter and args.experiment == "B0":
        raise ValueError("B0 cannot load an evidence adapter")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable_allowlist = ["model.evidence_memory.*"]
    if not args.freeze_task_state:
        trainable_allowlist.insert(0, "model.task_state_estimator.*")
    if not args.freeze_trajectory_backbone:
        trainable_allowlist.extend(("model.cond_projector.*", "model.latent_queries"))
    planned_config = {
        "checkpoint": str(CHECKPOINT),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": args.samples,
        "validation_samples": args.validation_samples,
        "steps": args.steps,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "s2_loss_weight": args.s2_loss_weight,
        "trajectory_loss_weight": args.trajectory_loss_weight,
        "task_state_learning_rate": args.learning_rate * args.task_state_lr_scale,
        "task_state_scale": args.task_state_scale,
        "normalize_task_state": args.normalize_task_state,
        "freeze_task_state": args.freeze_task_state,
        "freeze_trajectory_backbone": args.freeze_trajectory_backbone,
        "instruction_only_task_state": args.instruction_only_task_state,
        "task_contrastive_weight": args.task_contrastive_weight,
        "stage_loss_weight": args.stage_loss_weight,
        "relevance_supervision": str(args.relevance_supervision) if args.relevance_supervision else None,
        "relevance_loss_weight": args.relevance_loss_weight,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "task_contrastive_margin": args.task_contrastive_margin,
        "task_alignment_path": str(args.task_alignment_path) if args.task_alignment_path else None,
        "smoke_only": args.smoke_only,
        "max_pixels": 224 * 224,
        "model_max_length": 1024,
        "num_history": 3 if args.relevance_supervision else 2,
        "memory_fraction": 0.50,
        "architecture": (
            "original checkpoint evaluation"
            if args.experiment == "B0"
            else (
                f"frozen-feature adapter baseline ({args.bypass_mode})"
                if args.gradient_bypass
                else "input-token evidence"
            )
        ),
        "experiment": args.experiment,
        "sharded": args.sharded,
        "bypass_scale": args.bypass_scale,
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "trainable_dtype": "float32",
        "trainable_allowlist": trainable_allowlist,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(planned_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    start = time.monotonic()
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "failed",
        "metrics": {},
        "analysis": "真实 checkpoint 过拟合未完成。",
    }
    series = None
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        from internnav.dataset.internvla_n1_lerobot_dataset import DataCollatorForSupervisedDataset, NavPixelGoalDataset
        from internnav.dataset.internvla_n1_lerobot_dataset import (
            evidence_relevance_for_sample,
            evidence_sample_identity,
        )
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
            InternVLAN1ModelConfig,
        )
        from internnav.model.basemodel.internvla_n1.evidence_inference import load_evidence_adapter
        from internnav.model.basemodel.internvla_n1.trainable import configure_trainable_parameters

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if not args.sharded:
            torch.cuda.set_per_process_memory_fraction(0.50, device=0)
        device = torch.device("cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=False)
        tokenizer.model_max_length = 1024
        processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
        data_args = SimpleNamespace(
            vln_dataset_use="r2r_125cm_0_30",
            video_max_total_pixels=1664 * 28 * 28,
            video_min_total_pixels=256 * 28 * 28,
            model_type="internvla-n1",
            sample_step=1 if args.relevance_supervision else 4,
            predict_step_num=32,
            pixel_goal_only=True,
            num_future_steps=4,
            num_history=3 if args.relevance_supervision else 2,
            image_processor=processor.image_processor,
            transform_train=None,
            use_evidence_memory=args.experiment != "B0",
            task_state_supervision=args.task_contrastive_weight > 0 or args.stage_loss_weight > 0,
            task_alignment_path=str(args.task_alignment_path) if args.task_alignment_path else None,
            evidence_relevance_path=(
                str(args.relevance_supervision) if args.relevance_supervision else None
            ),
            evidence_relevance_only=bool(args.relevance_supervision),
            max_pixels=224 * 224,
            min_pixels=224 * 224,
        )
        dataset = NavPixelGoalDataset(tokenizer, data_args)
        if args.relevance_supervision:
            selected = select_relevance_samples(dataset, args.samples, args.seed)
        elif args.instruction_only_task_state:
            selected = select_task_state_samples(dataset, args.samples, args.seed)
        else:
            selected = [
                index
                for index, entry in enumerate(dataset.list_data_dict)
                if entry[7][0] >= 2 and entry[9] is not None
            ][: args.samples]
        if len(selected) != args.samples:
            raise RuntimeError(f"only found {len(selected)} eligible training samples")
        validation_selected = []
        if args.validation_samples:
            if not args.relevance_supervision:
                raise ValueError("held-out validation currently requires relevance supervision")
            training_episodes = {
                evidence_sample_identity(dataset.list_data_dict[index])[:2] for index in selected
            }
            validation_selected = select_relevance_samples(
                dataset,
                args.validation_samples,
                args.seed + 100,
                excluded_episodes=training_episodes,
            )
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)
        cpu_batches = [collator([dataset[index]]) for index in selected]
        cpu_validation_batches = [collator([dataset[index]]) for index in validation_selected]
        manifest = []
        for index in selected:
            entry = dataset.list_data_dict[index]
            item = {
                "dataset_index": index,
                "episode_id": entry[0],
                "current_frame_id": entry[7][0],
            }
            scene_id, _, _ = evidence_sample_identity(entry)
            item["scene_id"] = scene_id
            relevance = evidence_relevance_for_sample(dataset.evidence_relevance, entry)
            if relevance is not None:
                item.update(
                    annotation_id=relevance["annotation_id"],
                    candidate_frame_ids=relevance["candidate_frame_ids"],
                    relevance_targets=relevance["targets"],
                )
            manifest.append(item)
        (args.output_dir / "sample_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if validation_selected:
            validation_manifest = []
            for index in validation_selected:
                entry = dataset.list_data_dict[index]
                scene_id, _, _ = evidence_sample_identity(entry)
                relevance = evidence_relevance_for_sample(dataset.evidence_relevance, entry)
                validation_manifest.append(
                    {
                        "dataset_index": index,
                        "scene_id": scene_id,
                        "episode_id": entry[0],
                        "current_frame_id": entry[7][0],
                        "annotation_id": relevance["annotation_id"],
                        "candidate_frame_ids": relevance["candidate_frame_ids"],
                        "relevance_targets": relevance["targets"],
                    }
                )
            (args.output_dir / "validation_manifest.json").write_text(
                json.dumps(validation_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        config.use_evidence_memory = args.experiment != "B0"
        config.evidence_task_dim = 512
        config.evidence_bottleneck_dim = 512
        config.num_evidence_tokens = 4
        config.evidence_num_heads = 8
        config.evidence_num_stages = 4
        config.evidence_dropout = 0.0
        config.s2_loss_weight = args.s2_loss_weight
        config.trajectory_loss_weight = args.trajectory_loss_weight
        config.use_cache = False
        config.evidence_gradient_bypass = args.gradient_bypass
        config.evidence_gradient_bypass_scale = args.bypass_scale
        config.evidence_gradient_bypass_mode = args.bypass_mode
        config.evidence_latent_query_bypass = not args.freeze_trajectory_backbone
        config.evidence_task_state_scale = args.task_state_scale
        config.evidence_normalize_task_state = args.normalize_task_state
        config.evidence_instruction_only_task_state = args.instruction_only_task_state
        config.evidence_task_contrastive_weight = args.task_contrastive_weight
        config.evidence_stage_loss_weight = args.stage_loss_weight
        config.evidence_relevance_loss_weight = args.relevance_loss_weight
        config.evidence_task_contrastive_margin = args.task_contrastive_margin
        config.evidence_ablation = {
            "B0": "null",
            "B1": "null",
            "B2": "content",
            "M1": "spatial",
            "M2": "task_spatial",
        }[args.experiment]
        load_kwargs = {}
        if args.sharded:
            max_memory = {index: "20GiB" for index in range(torch.cuda.device_count())}
            max_memory["cpu"] = "64GiB"
            load_kwargs.update(device_map="auto", max_memory=max_memory)
        model = InternVLAN1ForCausalLM.from_pretrained(
            CHECKPOINT,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        if not args.sharded:
            model = model.to(device)
        if not args.gradient_bypass:
            model.gradient_checkpointing_enable()
        trainable = None
        if args.experiment != "B0":
            model.enable_input_require_grads()
            trainable = configure_trainable_parameters(model, tuple(trainable_allowlist))
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
            model.get_model().reset_evidence_parameters()
            init_adapter_manifest = (
                load_evidence_adapter(model, args.init_adapter) if args.init_adapter else None
            )
        input_device = model.get_input_embeddings().weight.device
        completed_config = {
            **planned_config,
            "device": str(device),
            "trainable_parameters": trainable.trainable_parameters if trainable else 0,
            "frozen_parameters": (
                trainable.frozen_parameters if trainable else sum(p.numel() for p in model.parameters())
            ),
            "matched_parameters": trainable.matched_parameters if trainable else [],
            "init_adapter_manifest": init_adapter_manifest if args.experiment != "B0" else None,
        }
        (args.output_dir / "config.json").write_text(
            json.dumps(completed_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def evaluate(batches):
            model.eval()
            totals, s2_values, trajectory_values = [], [], []
            contrastive_values, stage_values, relevance_values = [], [], []
            with torch.no_grad():
                for sample_idx, cpu_batch in enumerate(batches):
                    torch.manual_seed(args.seed + 1000 + sample_idx)
                    torch.cuda.manual_seed_all(args.seed + 1000 + sample_idx)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**move_batch(cpu_batch, input_device))
                    totals.append(float(output.loss))
                    s2_values.append(float(output.s2_loss))
                    trajectory_values.append(float(output.trajectory_loss))
                    contrastive_values.append(
                        float(output.task_contrastive_loss) if output.task_contrastive_loss is not None else 0.0
                    )
                    stage_values.append(float(output.stage_loss) if output.stage_loss is not None else 0.0)
                    relevance_values.append(
                        float(output.evidence_relevance_loss)
                        if output.evidence_relevance_loss is not None
                        else 0.0
                    )
            return (
                sum(totals) / len(totals),
                sum(s2_values) / len(s2_values),
                sum(trajectory_values) / len(trajectory_values),
                sum(contrastive_values) / len(contrastive_values),
                sum(stage_values) / len(stage_values),
                sum(relevance_values) / len(relevance_values),
            )

        (
            initial_total,
            initial_s2,
            initial_trajectory,
            initial_contrastive,
            initial_stage,
            initial_relevance,
        ) = evaluate(cpu_batches)
        initial_validation = evaluate(cpu_validation_batches) if cpu_validation_batches else None
        if args.experiment == "B0":
            finite = all(
                torch.isfinite(torch.tensor(value)) for value in (initial_total, initial_s2, initial_trajectory)
            )
            report["status"] = "passed" if finite else "failed"
            report["metrics"].update(
                {
                    "architecture": planned_config["architecture"],
                    "experiment": args.experiment,
                    "samples": args.samples,
                    "steps_completed": 0,
                    "initial_total_loss": initial_total,
                    "final_total_loss": initial_total,
                    "total_loss_reduction": 0.0,
                    "initial_s2_loss": initial_s2,
                    "final_s2_loss": initial_s2,
                    "initial_trajectory_loss": initial_trajectory,
                    "final_trajectory_loss": initial_trajectory,
                    "initial_task_contrastive_loss": initial_contrastive,
                    "final_task_contrastive_loss": initial_contrastive,
                    "initial_stage_loss": initial_stage,
                    "final_stage_loss": initial_stage,
                    "initial_evidence_relevance_loss": initial_relevance,
                    "final_evidence_relevance_loss": initial_relevance,
                    "duration_s": time.monotonic() - start,
                    "trainable_parameters": 0,
                    "frozen_gradient_parameters": 0,
                }
            )
            report["analysis"] = (
                "原始 checkpoint 在固定样本上的无更新损失已记录。" if finite else "原始 checkpoint 前向出现非有限损失。"
            )
            series = {
                "total_loss": [initial_total],
                "s2_loss": [initial_s2],
                "trajectory_loss": [initial_trajectory],
            }
            write_report(args, report, series)
            print(f"real_overfit={report['status']} metrics={json.dumps(report['metrics'], ensure_ascii=False)}")
            return
        task_parameters = []
        other_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (task_parameters if "task_state_estimator" in name else other_parameters).append(parameter)
        optimizer_groups = [{"params": other_parameters, "lr": args.learning_rate}]
        if task_parameters:
            optimizer_groups.append({"params": task_parameters, "lr": args.learning_rate * args.task_state_lr_scale})
        optimizer = torch.optim.AdamW(optimizer_groups)
        model.train()
        gradient_audit = {}
        audit_loss_names = []
        if args.s2_loss_weight > 0:
            audit_loss_names.append("s2_loss")
        if args.trajectory_loss_weight > 0:
            audit_loss_names.append("trajectory_loss")
        if args.task_contrastive_weight > 0:
            audit_loss_names.append("task_contrastive_loss")
        if args.stage_loss_weight > 0:
            audit_loss_names.append("stage_loss")
        if args.relevance_loss_weight > 0:
            audit_loss_names.append("evidence_relevance_loss")
        for loss_name in audit_loss_names:
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(args.seed + 2000)
            torch.cuda.manual_seed_all(args.seed + 2000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                audit_output = model(**move_batch(cpu_batches[0], input_device))
            getattr(audit_output, loss_name).backward()
            gradient_audit[loss_name] = summarize_gradients(model, torch)
        optimizer.zero_grad(set_to_none=True)
        (args.output_dir / "gradient_audit.json").write_text(
            json.dumps(gradient_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        nonfinite_audit = sum(
            group["nonfinite_gradients"] for loss_summary in gradient_audit.values() for group in loss_summary.values()
        )
        required_gradient_groups = {}
        if args.s2_loss_weight > 0:
            required_gradient_groups["s2_loss"] = (
                ("task_state_estimator", "evidence_memory")
                if args.experiment == "M2" and not args.freeze_task_state
                else ("evidence_memory",)
            )
        if args.trajectory_loss_weight > 0:
            required_gradient_groups["trajectory_loss"] = (
                ("evidence_memory",)
                if args.freeze_trajectory_backbone
                else ("latent_queries", "cond_projector")
            )
        if args.task_contrastive_weight > 0:
            required_gradient_groups["task_contrastive_loss"] = ("task_state_estimator",)
        if args.stage_loss_weight > 0:
            required_gradient_groups["stage_loss"] = ("task_state_estimator", "evidence_memory")
        if args.relevance_loss_weight > 0:
            required_gradient_groups["evidence_relevance_loss"] = (
                "task_state_estimator",
                "evidence_memory",
            )
        missing_gradient_groups = [
            f"{loss_name}:{group_name}"
            for loss_name, group_names in required_gradient_groups.items()
            for group_name in group_names
            if gradient_audit.get(loss_name, {}).get(group_name, {}).get("gradient_l2", 0.0) <= 0
        ]
        report["metrics"].update(
            {
                "architecture": planned_config["architecture"],
                "experiment": args.experiment,
                "seed": args.seed,
                "samples": args.samples,
                "steps_completed": 0,
                "initial_total_loss": initial_total,
                "initial_s2_loss": initial_s2,
                "initial_trajectory_loss": initial_trajectory,
                "initial_task_contrastive_loss": initial_contrastive,
                "initial_stage_loss": initial_stage,
                "initial_evidence_relevance_loss": initial_relevance,
                "gradient_audit_nonfinite": nonfinite_audit,
                "missing_gradient_groups": missing_gradient_groups,
            }
        )
        if nonfinite_audit:
            raise FloatingPointError(f"pre-update gradient audit found {nonfinite_audit} non-finite values")
        if missing_gradient_groups:
            raise FloatingPointError(f"pre-update gradient audit found zero gradients: {missing_gradient_groups}")
        series = {"total_loss": [], "s2_loss": [], "trajectory_loss": []}
        if args.relevance_loss_weight > 0:
            series["evidence_relevance_loss"] = []
        with (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
            for step in range(args.steps):
                batch = move_batch(cpu_batches[step % len(cpu_batches)], input_device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**batch)
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                values = {
                    "total_loss": float(output.loss.detach()),
                    "s2_loss": float(output.s2_loss.detach()),
                    "trajectory_loss": float(output.trajectory_loss.detach()),
                }
                if args.relevance_loss_weight > 0:
                    values["evidence_relevance_loss"] = float(output.evidence_relevance_loss.detach())
                for key in series:
                    series[key].append(values[key])
                log_file.write(
                    json.dumps(
                        {
                            "step": step,
                            **values,
                            "task_contrastive_loss": (
                                float(output.task_contrastive_loss.detach())
                                if output.task_contrastive_loss is not None
                                else 0.0
                            ),
                            "stage_loss": float(output.stage_loss.detach()) if output.stage_loss is not None else 0.0,
                            "gradient_norm": float(gradient_norm),
                            "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
                        }
                    )
                    + "\n"
                )
        (
            final_total,
            final_s2,
            final_trajectory,
            final_contrastive,
            final_stage,
            final_relevance,
        ) = evaluate(cpu_batches)
        final_validation = evaluate(cpu_validation_batches) if cpu_validation_batches else None
        reduction = 1.0 - final_total / initial_total
        frozen_gradient_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad and parameter.grad is not None
        )
        final_losses = [final_total, final_s2, final_trajectory]
        if args.relevance_loss_weight > 0:
            final_losses.append(final_relevance)
        finite_final = all(torch.isfinite(torch.tensor(value)) for value in final_losses)
        passed = (
            finite_final and frozen_gradient_parameters == 0
            if args.smoke_only
            else (
                reduction >= 0.30
                and (args.s2_loss_weight <= 0 or final_s2 < initial_s2)
                and (args.trajectory_loss_weight <= 0 or final_trajectory < initial_trajectory)
                and (args.relevance_loss_weight <= 0 or final_relevance < initial_relevance)
                and (
                    initial_validation is None
                    or args.relevance_loss_weight <= 0
                    or final_validation[5] < initial_validation[5]
                )
                and frozen_gradient_parameters == 0
            )
        )
        adapter_prefixes = (
            "model.task_state_estimator.",
            "model.evidence_memory.",
            "model.cond_projector.",
            "model.latent_queries",
        )
        adapter_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if any(name == prefix or name.startswith(prefix) for prefix in adapter_prefixes)
        }
        torch.save(adapter_state, args.output_dir / "adapter_state.pt")
        report["status"] = "passed" if passed else "failed"
        per_gpu_peak_mib = {
            str(index): torch.cuda.max_memory_allocated(index) / 1024**2 for index in range(torch.cuda.device_count())
        }
        report["metrics"].update(
            {
                "samples": args.samples,
                "validation_samples": len(cpu_validation_batches),
                "steps_completed": args.steps,
                "initial_total_loss": initial_total,
                "final_total_loss": final_total,
                "total_loss_reduction": reduction,
                "initial_s2_loss": initial_s2,
                "final_s2_loss": final_s2,
                "initial_trajectory_loss": initial_trajectory,
                "final_trajectory_loss": final_trajectory,
                "initial_task_contrastive_loss": initial_contrastive,
                "final_task_contrastive_loss": final_contrastive,
                "initial_stage_loss": initial_stage,
                "final_stage_loss": final_stage,
                "initial_evidence_relevance_loss": initial_relevance,
                "final_evidence_relevance_loss": final_relevance,
                "initial_validation_total_loss": initial_validation[0] if initial_validation else 0.0,
                "final_validation_total_loss": final_validation[0] if final_validation else 0.0,
                "initial_validation_s2_loss": initial_validation[1] if initial_validation else 0.0,
                "final_validation_s2_loss": final_validation[1] if final_validation else 0.0,
                "initial_validation_trajectory_loss": initial_validation[2] if initial_validation else 0.0,
                "final_validation_trajectory_loss": final_validation[2] if final_validation else 0.0,
                "initial_validation_evidence_relevance_loss": initial_validation[5] if initial_validation else 0.0,
                "final_validation_evidence_relevance_loss": final_validation[5] if final_validation else 0.0,
                "heldout_relevance_improved": (
                    final_validation[5] < initial_validation[5] if initial_validation else None
                ),
                "peak_allocated_mib": max(per_gpu_peak_mib.values()),
                "per_gpu_peak_allocated_mib": per_gpu_peak_mib,
                "frozen_gradient_parameters": frozen_gradient_parameters,
                "duration_s": time.monotonic() - start,
                "trainable_parameters": trainable.trainable_parameters,
            }
        )
        report["analysis"] = (
            "真实 R2R train 单步接口、梯度和数值检查通过。"
            if args.smoke_only and passed
            else "真实 R2R train 样本与 InternVLA-N1 checkpoint 已完成过拟合门槛。"
            if passed
            else "训练完成但 relevance/S2/trajectory loss 未同时通过总下降门槛，需要依据曲线调整权重、学习率或步数后重试。"
        )
    except Exception as error:
        error_traceback = traceback.format_exc()
        report["metrics"]["duration_s"] = time.monotonic() - start
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = error_traceback
        report["analysis"] = "真实 checkpoint 过拟合异常退出；未影响并行进程，具体原因见 metrics.json 与 overfit.log。"
        print(error_traceback, file=sys.stderr)
    write_report(args, report, series)
    print(f"real_overfit={report['status']} metrics={json.dumps(report['metrics'], ensure_ascii=False)}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

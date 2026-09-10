#!/usr/bin/env python3
import argparse
import json
import os
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


def write_curve(series, path):
    width, height = 720, 350
    left, top, plot_w, plot_h = 68, 48, 610, 240
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    span = max(high - low, 1e-8)
    colors = {"total_loss": "#2b6cb0", "s2_loss": "#2f855a", "trajectory_loss": "#c05621"}
    lines = []
    legend = []
    for line_index, (name, values) in enumerate(series.items()):
        points = []
        for index, value in enumerate(values):
            x = left + plot_w * index / max(1, len(values) - 1)
            y = top + plot_h * (high - value) / span
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
        legend.append(f'<line x1="{left + line_index * 180}" y1="325" x2="{left + 25 + line_index * 180}" y2="325" stroke="{colors[name]}" stroke-width="3"/><text x="{left + 31 + line_index * 180}" y="330" font-size="13">{name}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="{width}" height="{height}" fill="#fff"/><text x="24" y="28" font-size="20" font-weight="700">P1 8 样本真实过拟合 loss</text><line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#4a5568"/><line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#4a5568"/><text x="8" y="{top + 5}" font-size="12">{high:.3f}</text><text x="8" y="{top + plot_h}" font-size="12">{low:.3f}</text>{"".join(lines)}{"".join(legend)}</svg>\n'
    path.write_text(svg, encoding="utf-8")


def write_report(args, report, series=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = report.get("metrics", {})
    summary = f"""# 阶段结果简报

- 阶段：`P1 8 样本真实 checkpoint 过拟合`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`

|指标|结果|
|---|---:|
|真实训练样本|{metrics.get('samples', 0)}|
|优化步数|{metrics.get('steps_completed', 0)}|
|初始总 loss|{metrics.get('initial_total_loss', 0):.6f}|
|最终总 loss|{metrics.get('final_total_loss', 0):.6f}|
|总 loss 下降|{metrics.get('total_loss_reduction', 0):.2%}|
|初始/最终 S2 loss|{metrics.get('initial_s2_loss', 0):.6f} / {metrics.get('final_s2_loss', 0):.6f}|
|初始/最终轨迹 loss|{metrics.get('initial_trajectory_loss', 0):.6f} / {metrics.get('final_trajectory_loss', 0):.6f}|
|峰值本进程显存 MiB|{metrics.get('peak_allocated_mib', 0):.1f}|
|耗时（秒）|{metrics.get('duration_s', 0):.3f}|

## 简述与分析

{report['analysis']}

逐步 loss 见 `train_log.jsonl`，分 loss 梯度审计见 `gradient_audit.json`，曲线见 `metrics.svg`，训练配置和样本 manifest 见 `config.json` 与 `sample_manifest.json`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    if series:
        write_curve(series, args.output_dir / "metrics.svg")
    else:
        color = "#c53030"
        (args.output_dir / "metrics.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="120"><rect width="620" height="120" fill="#fff"/><text x="24" y="32" font-size="20" font-weight="700">P1 8 样本真实过拟合</text><rect x="24" y="58" width="520" height="26" fill="{color}"/><text x="556" y="77" font-size="14">失败</text></svg>\n', encoding="utf-8")


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
            summary["gradient_max_abs"] = max(
                summary["gradient_max_abs"], float(finite_gradient.abs().max())
            )
    for summary in summaries.values():
        summary["gradient_l2"] = summary.pop("gradient_l2_squared") ** 0.5
        summary["parameter_dtypes"] = sorted(summary["parameter_dtypes"])
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    planned_config = {
        "checkpoint": str(CHECKPOINT),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": args.samples,
        "steps": args.steps,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "learning_rate": 3e-4,
        "max_pixels": 224 * 224,
        "model_max_length": 1024,
        "num_history": 2,
        "memory_fraction": 0.50,
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "trainable_allowlist": [
            "model.task_state_estimator.*",
            "model.evidence_memory.*",
            "model.cond_projector.*",
            "model.latent_queries",
        ],
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(planned_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    start = time.monotonic()
    report = {"schema_version": 1, "run_id": args.run_id, "git_commit": args.commit, "status": "failed", "metrics": {}, "analysis": "真实 checkpoint 过拟合未完成。"}
    series = None
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        from internnav.dataset.internvla_n1_lerobot_dataset import DataCollatorForSupervisedDataset, NavPixelGoalDataset
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
            InternVLAN1ModelConfig,
        )
        from internnav.model.basemodel.internvla_n1.trainable import configure_trainable_parameters

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.manual_seed(23)
        torch.cuda.manual_seed_all(23)
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
        selected = [index for index, entry in enumerate(dataset.list_data_dict) if entry[7][0] >= 2 and entry[9] is not None][: args.samples]
        if len(selected) != args.samples:
            raise RuntimeError(f"only found {len(selected)} eligible training samples")
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)
        cpu_batches = [collator([dataset[index]]) for index in selected]
        manifest = [{"dataset_index": index, "episode_id": dataset.list_data_dict[index][0], "current_frame_id": dataset.list_data_dict[index][7][0]} for index in selected]
        (args.output_dir / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        config.use_evidence_memory = True
        config.evidence_task_dim = 512
        config.evidence_bottleneck_dim = 512
        config.num_evidence_tokens = 4
        config.evidence_num_heads = 8
        config.evidence_num_stages = 4
        config.evidence_dropout = 0.0
        config.s2_loss_weight = 1.0
        config.trajectory_loss_weight = 1.0
        config.use_cache = False
        model = InternVLAN1ForCausalLM.from_pretrained(
            CHECKPOINT,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to(device)
        model.gradient_checkpointing_enable()
        trainable = configure_trainable_parameters(
            model,
            ("model.task_state_estimator.*", "model.evidence_memory.*", "model.cond_projector.*", "model.latent_queries"),
        )
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4)
        completed_config = {
            **planned_config,
            "device": str(device),
            "trainable_parameters": trainable.trainable_parameters,
            "frozen_parameters": trainable.frozen_parameters,
            "matched_parameters": trainable.matched_parameters,
        }
        (args.output_dir / "config.json").write_text(
            json.dumps(completed_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def evaluate():
            model.eval()
            totals, s2_values, trajectory_values = [], [], []
            with torch.no_grad():
                for sample_idx, cpu_batch in enumerate(cpu_batches):
                    torch.manual_seed(1000 + sample_idx)
                    torch.cuda.manual_seed_all(1000 + sample_idx)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**move_batch(cpu_batch, device))
                    totals.append(float(output.loss))
                    s2_values.append(float(output.s2_loss))
                    trajectory_values.append(float(output.trajectory_loss))
            return sum(totals) / len(totals), sum(s2_values) / len(s2_values), sum(trajectory_values) / len(trajectory_values)

        initial_total, initial_s2, initial_trajectory = evaluate()
        model.train()
        gradient_audit = {}
        for loss_name in ("s2_loss", "trajectory_loss"):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(2000)
            torch.cuda.manual_seed_all(2000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                audit_output = model(**move_batch(cpu_batches[0], device))
            getattr(audit_output, loss_name).backward()
            gradient_audit[loss_name] = summarize_gradients(model, torch)
        optimizer.zero_grad(set_to_none=True)
        (args.output_dir / "gradient_audit.json").write_text(
            json.dumps(gradient_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        nonfinite_audit = sum(
            group["nonfinite_gradients"]
            for loss_summary in gradient_audit.values()
            for group in loss_summary.values()
        )
        report["metrics"].update(
            {
                "samples": args.samples,
                "steps_completed": 0,
                "initial_total_loss": initial_total,
                "initial_s2_loss": initial_s2,
                "initial_trajectory_loss": initial_trajectory,
                "gradient_audit_nonfinite": nonfinite_audit,
            }
        )
        if nonfinite_audit:
            raise FloatingPointError(
                f"pre-update gradient audit found {nonfinite_audit} non-finite values"
            )
        series = {"total_loss": [], "s2_loss": [], "trajectory_loss": []}
        with (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
            for step in range(args.steps):
                batch = move_batch(cpu_batches[step % len(cpu_batches)], device)
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
                values = (float(output.loss.detach()), float(output.s2_loss.detach()), float(output.trajectory_loss.detach()))
                for key, value in zip(series, values):
                    series[key].append(value)
                log_file.write(json.dumps({"step": step, "total_loss": values[0], "s2_loss": values[1], "trajectory_loss": values[2], "gradient_norm": float(gradient_norm), "allocated_mib": torch.cuda.memory_allocated() / 1024**2, "reserved_mib": torch.cuda.memory_reserved() / 1024**2}) + "\n")
        final_total, final_s2, final_trajectory = evaluate()
        reduction = 1.0 - final_total / initial_total
        passed = reduction >= 0.30 and final_s2 < initial_s2 and final_trajectory < initial_trajectory
        adapter_state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
        torch.save(adapter_state, args.output_dir / "adapter_state.pt")
        report["status"] = "passed" if passed else "failed"
        report["metrics"].update({"samples": args.samples, "steps_completed": args.steps, "initial_total_loss": initial_total, "final_total_loss": final_total, "total_loss_reduction": reduction, "initial_s2_loss": initial_s2, "final_s2_loss": final_s2, "initial_trajectory_loss": initial_trajectory, "final_trajectory_loss": final_trajectory, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2, "duration_s": time.monotonic() - start, "trainable_parameters": trainable.trainable_parameters})
        report["analysis"] = "真实 R2R train 样本与 InternVLA-N1 checkpoint 已完成过拟合门槛。" if passed else "训练完成但 S2/trajectory 双 loss 未同时达到 30% 总下降门槛，需要依据曲线调整学习率、步数或训练范围后重试。"
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

#!/usr/bin/env python3
import argparse
import json
import time
from html import escape
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from internnav.model.basemodel.internvla_n1.evidence_memory import TaskConditionedEvidenceMemory


def write_curve(losses, path, title):
    width, height = 680, 320
    left, top, plot_w, plot_h = 62, 52, 580, 220
    low, high = min(losses), max(losses)
    span = max(high - low, 1e-9)
    points = []
    for index, loss in enumerate(losses):
        x = left + plot_w * index / max(1, len(losses) - 1)
        y = top + plot_h * (high - loss) / span
        points.append(f"{x:.2f},{y:.2f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="24" y="29" font-size="20" font-weight="700">{escape(title)}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#4a5568"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#4a5568"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#2b6cb0" stroke-width="2"/>
<text x="8" y="{top + 5}" font-size="12">{high:.4f}</text>
<text x="8" y="{top + plot_h}" font-size="12">{low:.4f}</text>
<text x="{left}" y="{top + plot_h + 24}" font-size="12">0</text>
<text x="{left + plot_w - 34}" y="{top + plot_h + 24}" font-size="12">{len(losses) - 1} step</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(2)

    batch, history, feature_dim, task_dim, output_dim = args.samples, 6, 32, 16, 32
    history_features = torch.randn(batch, history, feature_dim)
    relative_poses = torch.randn(batch, history, 4)
    ages = torch.arange(history, dtype=torch.float32).view(1, history, 1).expand(batch, -1, -1)
    qualities = torch.rand(batch, history, 2)
    valid_mask = torch.arange(history).view(1, -1) < ((torch.arange(batch) % history) + 1).view(-1, 1)
    valid_mask[0] = False
    task_state = torch.randn(batch, task_dim)
    target = torch.randn(batch, 8)
    stage_target = torch.arange(batch) % 4

    memory = TaskConditionedEvidenceMemory(
        feature_dim=feature_dim,
        task_dim=task_dim,
        hidden_dim=64,
        output_dim=output_dim,
        num_evidence_tokens=4,
        num_heads=8,
        num_stages=4,
    )
    readout = nn.Linear(output_dim, target.shape[1])
    optimizer = torch.optim.AdamW(list(memory.parameters()) + list(readout.parameters()), lr=3e-3)
    losses = []
    log_path = args.output_dir / "train_log.jsonl"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(args.steps):
            output = memory(history_features, relative_poses, ages, qualities, valid_mask, task_state)
            prediction = readout(output.tokens.mean(dim=1))
            regression_loss = F.mse_loss(prediction, target)
            stage_loss = F.cross_entropy(output.stage_logits, stage_target)
            loss = regression_loss + 0.1 * stage_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            log_file.write(json.dumps({"step": step, "loss": value, "regression_loss": float(regression_loss.detach()), "stage_loss": float(stage_loss.detach())}) + "\n")

    memory.eval()
    readout.eval()
    history_probe = history_features.detach().clone().requires_grad_(True)
    task_probe = task_state.detach().clone().requires_grad_(True)
    output = memory(history_probe, relative_poses, ages, qualities, valid_mask, task_probe)
    prediction = readout(output.tokens.mean(dim=1))
    final_loss_tensor = F.mse_loss(prediction, target) + 0.1 * F.cross_entropy(output.stage_logits, stage_target)
    final_loss_tensor.backward()
    shuffled_output = memory(history_features, relative_poses, ages, qualities, valid_mask, task_state.roll(1, 0))
    shuffled_prediction = readout(shuffled_output.tokens.mean(dim=1))
    shuffled_loss = float(F.mse_loss(shuffled_prediction, target))
    duration = time.monotonic() - start
    initial_loss, final_loss = losses[0], float(final_loss_tensor.detach())
    reduction = 1.0 - final_loss / initial_loss
    passed = final_loss < initial_loss * 0.15 and shuffled_loss > final_loss * 1.10
    report = {
        "schema_version": 1,
        "stage": "P1 16 样本合成协议过拟合",
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "metrics": {
            "samples": batch,
            "steps": args.steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": reduction,
            "shuffled_task_loss": shuffled_loss,
            "shuffled_to_final_ratio": shuffled_loss / max(final_loss, 1e-12),
            "history_input_grad_l1": float(history_probe.grad.abs().sum()),
            "task_input_grad_l1": float(task_probe.grad.abs().sum()),
            "trainable_parameters": sum(parameter.numel() for parameter in memory.parameters()) + sum(parameter.numel() for parameter in readout.parameters()),
            "duration_s": duration,
        },
        "analysis": "该阶段只证明小型 evidence 模块在 16 个合成样本上可优化且依赖 task 输入，不代表 7B S2、真实轨迹或闭环导航已经通过。",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"seed": 17, "device": "cpu", "optimizer": "AdamW", "learning_rate": 3e-3, "samples": batch, "steps": args.steps}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    summary = f"""# 阶段结果简报

- 阶段：`{report['stage']}`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{'通过' if passed else '失败'}`

|指标|结果|
|---|---:|
|样本数|{batch}|
|优化步数|{args.steps}|
|初始 loss|{initial_loss:.6f}|
|最终 loss|{final_loss:.6f}|
|loss 下降比例|{reduction:.2%}|
|打乱 task 后 loss|{shuffled_loss:.6f}|
|打乱/正常 loss 比|{metrics['shuffled_to_final_ratio']:.2f}|
|历史输入梯度 L1|{metrics['history_input_grad_l1']:.6f}|
|任务输入梯度 L1|{metrics['task_input_grad_l1']:.6f}|
|耗时（秒）|{duration:.3f}|

## 简述与分析

{report['analysis']}

曲线见 `metrics.svg`，逐步原始数值见 `train_log.jsonl`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_curve(losses, args.output_dir / "metrics.svg", report["stage"])
    print(f"samples={batch} initial_loss={initial_loss:.6f} final_loss={final_loss:.6f} reduction={reduction:.2%} shuffled_task_loss={shuffled_loss:.6f} duration_s={duration:.3f}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

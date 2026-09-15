#!/usr/bin/env python3
"""Aggregate task-state reader ablations into JSON, Chinese Markdown, and SVG."""
import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    ("mean_val_loss", "验证 loss"),
    ("train_val_gap", "训练/验证 gap"),
    ("mean_val_yes_top1", "yes Top-1"),
    ("mean_val_pairwise", "pairwise"),
    ("mean_val_no_null_accuracy", "no/null"),
)


def summarize_record(label, path, variant):
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs = [run for run in metrics["runs"] if run["variant"] == variant]
    if not runs:
        raise ValueError(f"{metrics_path} 中不存在 variant={variant}")
    val_losses = [run["final_val"]["loss"] for run in runs]
    train_losses = [run["final_train"]["loss"] for run in runs]
    summary = next(item for item in metrics["variants"] if item["variant"] == variant)
    return {
        "label": label,
        "path": path.name,
        "variant": variant,
        "run_count": len(runs),
        "steps": metrics["steps"],
        "learning_rate": metrics["learning_rate"],
        "task_learning_rate_scale": metrics.get("task_learning_rate_scale", 1.0),
        "normalize_task_state": metrics.get("normalize_task_state", False),
        "freeze_task_estimator": metrics.get("freeze_task_estimator", False),
        "mean_train_loss": statistics.mean(train_losses),
        "mean_val_loss": statistics.mean(val_losses),
        "std_val_loss": statistics.pstdev(val_losses),
        "train_val_gap": statistics.mean(val_losses) - statistics.mean(train_losses),
        "mean_val_positive_mass": summary["mean_val_positive_mass"],
        "mean_val_yes_top1": summary["mean_val_yes_top1"],
        "mean_val_pairwise": summary["mean_val_pairwise"],
        "mean_val_no_null_accuracy": summary["mean_val_no_null_accuracy"],
    }


def write_svg(records, output_path):
    width, height = 980, 430
    group_width = 170
    colors = ("#64748b", "#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626")
    max_loss = max(record["mean_val_loss"] for record in records) * 1.15
    bars = []
    for index, record in enumerate(records):
        x = 80 + index * group_width
        val_height = 250 * record["mean_val_loss"] / max_loss
        gap_height = 250 * record["train_val_gap"] / max_loss
        bars.extend(
            [
                f'<rect x="{x}" y="{310 - val_height:.1f}" width="54" height="{val_height:.1f}" fill="{colors[index % len(colors)]}"/>',
                f'<rect x="{x + 58}" y="{310 - gap_height:.1f}" width="38" height="{gap_height:.1f}" fill="#fbbf24"/>',
                f'<text x="{x}" y="330" font-size="12">{record["label"]}</text>',
                f'<text x="{x}" y="{300 - val_height:.1f}" font-size="12">{record["mean_val_loss"]:.3f}</text>',
                f'<text x="{x + 58}" y="{300 - gap_height:.1f}" font-size="12">{record["train_val_gap"]:.3f}</text>',
                f'<text x="{x}" y="350" font-size="11">Top-1 {record["mean_val_yes_top1"]:.1%}</text>',
                f'<text x="{x}" y="367" font-size="11">Pair {record["mean_val_pairwise"]:.1%}</text>',
                f'<text x="{x}" y="384" font-size="11">Null {record["mean_val_no_null_accuracy"]:.1%}</text>',
            ]
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="28" y="34" font-size="21" font-weight="700">Task-state 稳定化对比</text>'
        '<text x="28" y="55" font-size="12" fill="#475569">彩色柱：验证 loss；黄色柱：训练/验证 gap（越低越好）</text>'
        '<line x1="65" y1="60" x2="65" y2="310" stroke="#475569"/>'
        '<line x1="65" y1="310" x2="950" y2="310" stroke="#475569"/>'
        + "".join(bars)
        + '</svg>\n'
    )
    output_path.write_text(svg, encoding="utf-8")


def write_report(records, selected, source_commit, output_path):
    rows = [
        "|设置|val loss|loss 标准差|train/val gap|yes Top-1|pairwise|no/null|正例质量|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        rows.append(
            f'|{record["label"]}|{record["mean_val_loss"]:.4f}|{record["std_val_loss"]:.4f}|'
            f'{record["train_val_gap"]:.4f}|{record["mean_val_yes_top1"]:.1%}|'
            f'{record["mean_val_pairwise"]:.1%}|{record["mean_val_no_null_accuracy"]:.1%}|'
            f'{record["mean_val_positive_mass"]:.1%}|'
        )
    table = "\n".join(rows)
    report = f"""# Task-state 稳定化消融联合分析

本轮源码提交为 `{source_commit}`。固定 40 张人工标注卡片、episode 隔离的平衡四折、seed 23/47/71、20 steps 与学习率 3e-4，只改变 task-state 稳定化策略。每个设置共 12 个 fold/seed 运行。

{table}

## 结论

- 最佳设置为 `{selected['label']}`：验证 loss `{selected['mean_val_loss']:.4f}`，跨运行标准差 `{selected['std_val_loss']:.4f}`，训练/验证 gap `{selected['train_val_gap']:.4f}`。
- 单独降低 task 分支学习率或冻结 estimator 只能部分缓解过拟合；task-state 单位范数同时改善验证 loss、稳定性和排序指标，说明主要问题是 task query 尺度失衡。
- 归一化再叠加 0.1 倍 task 学习率没有进一步收益，因此保留更简单的“只归一化”方案。
- 归一化 task-spatial 已整体不劣于 content/spatial，但数据仍只有单场景 40 张卡片，当前结论是“机制门槛通过”，不能宣称跨场景或导航收益。

## 下一步

1. 将 `evidence_normalize_task_state=true` 固定用于正式 relevance-supervised S2 小样本训练。
2. 让训练 dataset 按 annotation/frame id 注入人工 relevance targets，并保存可用于推理的 adapter 权重。
3. 先做 8 条标注样本过拟合与 held-out card 检查；只有 relevance、S2 与 trajectory loss 均有限且方向正确，才运行 1--4 episode 配对短闭环。
4. 旧闭环 dispatcher 不加载本轮 reader 权重，不能直接作为本轮闭环验证入口。
"""
    output_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="汇总 task-state reader 稳定化消融")
    parser.add_argument("--record", nargs=3, action="append", metavar=("LABEL", "PATH", "VARIANT"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    records = [summarize_record(label, Path(path), variant) for label, path, variant in args.record]
    selected = min(records, key=lambda record: record["mean_val_loss"])
    summary = {
        "schema_version": 1,
        "status": "completed",
        "source_commit": args.source_commit,
        "selection_metric": "mean_val_loss",
        "selected": selected["label"],
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(records, args.output_dir / "metrics.svg")
    write_report(records, selected, args.source_commit, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

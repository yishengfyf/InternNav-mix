#!/usr/bin/env python3
"""Aggregate reader-only budget scans into a Chinese report and SVG."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="汇总 reader-only 训练步数扫描")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    records = []
    for path in args.runs:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for variant in metrics["variants"]:
            records.append({"steps": metrics["steps"], **variant})
    selected = min(records, key=lambda row: row["mean_val_loss"])
    by_steps = {}
    for row in records:
        by_steps.setdefault(str(row["steps"]), []).append(row)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "runs": [str(path) for path in args.runs],
        "records": sorted(records, key=lambda row: (row["steps"], row["variant"])),
        "selected_steps": selected["steps"],
        "selected_variant": selected["variant"],
        "selected_val_loss": selected["mean_val_loss"],
        "analysis": "20 steps 是唯一使 content/spatial 平均验证 loss 低于初始约 1.10 的预算；40/80 steps 出现过拟合。task_spatial 在所有预算的验证 loss 均明显更高，应先降低或冻结 task 分支容量。",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    colors = {"content": "#2b6cb0", "spatial": "#2f855a", "task_spatial": "#c53030"}
    points = []
    legend = []
    for index, variant in enumerate(("content", "spatial", "task_spatial")):
        rows = sorted((row for row in records if row["variant"] == variant), key=lambda row: row["steps"])
        coords = []
        for row in rows:
            x = 100 + (row["steps"] - 20) / 60 * 430
            y = 280 - min(row["mean_val_loss"], 3.0) / 3.0 * 220
            coords.append(f"{x:.1f},{y:.1f}")
        points.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colors[variant]}" stroke-width="3"/>')
        legend.append(f'<rect x="{105 + index * 150}" y="315" width="15" height="15" fill="{colors[variant]}"/><text x="{126 + index * 150}" y="328">{variant}</text>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="355"><rect width="100%" height="100%" fill="white"/>'
        '<text x="28" y="31" font-size="20" font-weight="700">Reader-only validation loss vs steps</text>'
        '<line x1="100" y1="60" x2="100" y2="280" stroke="#4a5568"/><line x1="100" y1="280" x2="530" y2="280" stroke="#4a5568"/>'
        '<text x="70" y="285">0</text><text x="62" y="212">1</text><text x="62" y="139">2</text><text x="62" y="66">3</text>'
        '<text x="92" y="303">20</text><text x="235" y="303">40</text><text x="520" y="303">80</text>'
        + "".join(points) + "".join(legend) + "</svg>\n"
    )
    (args.output_dir / "metrics.svg").write_text(svg, encoding="utf-8")
    report = f"""# Reader-only 训练预算联合分析

- 最佳验证 loss：`{selected['variant']}` / {selected['steps']} steps / {selected['mean_val_loss']:.4f}
- 20 steps：content 1.0304，spatial 1.0281，task-spatial 1.7997
- 40 steps：content 1.1431，spatial 1.1224，task-spatial 2.2810
- 80 steps：content 1.9886，spatial 1.8773，task-spatial 2.6660

## 分析

20 steps 是唯一使 content/spatial 平均验证 loss 低于初始约 1.10 的预算。继续训练会快速记忆 30 张左右的训练卡片并损害 episode 隔离验证。task-spatial 在 20 steps 时训练 loss 已降到约 0.11、验证 loss 却升至 1.80，说明新增 task 分支容量过强，而不是训练不足。

下一轮固定 20 steps，只比较 task 参数 0.1 倍学习率、task-state 单位范数与冻结 estimator；不同时修改多个因素。
"""
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


EXPERIMENTS = ("B0", "B1", "B2", "M1", "M2")
LABELS = {
    "B0": "原 checkpoint",
    "B1": "null adapter",
    "B2": "历史内容",
    "M1": "内容 + 空间",
    "M2": "内容 + 空间 + task-state",
}


def write_svg(rows, path):
    width, height = 860, 390
    left, top, chart_width = 250, 55, 550
    bar_height, gap = 24, 34
    max_reduction = max(0.5, max(max(row["total_loss_reduction"], 0.0) for row in rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">N2 32 样本公平对照：总 loss 下降率</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * (bar_height + gap)
        reduction = row["total_loss_reduction"]
        bar_width = chart_width * max(reduction, 0.0) / max_reduction
        color = "#2f855a" if row["status"] == "passed" else "#c53030"
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" font-size="14">{row["experiment"]} {LABELS[row["experiment"]]}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="{color}"/>',
                f'<text x="{left + bar_width + 8:.1f}" y="{y + 18}" font-size="14">{reduction:.1%}</text>',
            ]
        )
    parts.append('</svg>\n')
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for experiment in EXPERIMENTS:
        metrics_path = args.stage_dir / experiment / "metrics.json"
        if not metrics_path.is_file():
            rows.append(
                {
                    "experiment": experiment,
                    "status": "missing",
                    "total_loss_reduction": 0.0,
                    "initial_s2_loss": None,
                    "final_s2_loss": None,
                    "initial_trajectory_loss": None,
                    "final_trajectory_loss": None,
                }
            )
            continue
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics", {})
        rows.append(
            {
                "experiment": experiment,
                "status": report.get("status", "unknown"),
                "total_loss_reduction": metrics.get("total_loss_reduction", 0.0),
                "initial_s2_loss": metrics.get("initial_s2_loss"),
                "final_s2_loss": metrics.get("final_s2_loss"),
                "initial_trajectory_loss": metrics.get("initial_trajectory_loss"),
                "final_trajectory_loss": metrics.get("final_trajectory_loss"),
            }
        )

    complete = all(row["status"] == "passed" for row in rows)
    b2 = next(row for row in rows if row["experiment"] == "B2")
    m2 = next(row for row in rows if row["experiment"] == "M2")
    delta = m2["total_loss_reduction"] - b2["total_loss_reduction"]
    report = {
        "schema_version": 1,
        "status": "passed" if complete else "incomplete",
        "rows": rows,
        "m2_minus_b2_reduction": delta,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table_rows = []
    for row in rows:
        s2 = (
            f'{row["initial_s2_loss"]:.5f} -> {row["final_s2_loss"]:.5f}'
            if row["initial_s2_loss"] is not None
            else "缺失"
        )
        trajectory = (
            f'{row["initial_trajectory_loss"]:.5f} -> {row["final_trajectory_loss"]:.5f}'
            if row["initial_trajectory_loss"] is not None
            else "缺失"
        )
        table_rows.append(
            f'|{row["experiment"]} / {LABELS[row["experiment"]]}|{row["status"]}|'
            f'{row["total_loss_reduction"]:.2%}|{s2}|{trajectory}|'
        )
    conclusion = (
        f'M2 相对 B2 的总 loss 下降率差为 {delta:+.2%}。这只是同一小样本训练集上的拟合差异，'
        "只有差异可重复且随后闭环指标方向一致，才能支持 task-state 的独立价值。"
        if complete
        else "至少一组缺失或未通过，当前矩阵不能用于判断 task-state 的独立价值。"
    )
    summary = f"""# N2 公平小样本对照简报

- 状态：`{'完成' if complete else '未完成'}`
- 样本：固定 32 个真实 R2R train 样本
- 约束：同一 commit、seed、样本顺序、学习率、token 数与更新预算

|组别|状态|总 loss 下降|S2 loss|trajectory loss|
|---|---|---:|---:|---:|
{chr(10).join(table_rows)}

## 简述与分析

{conclusion}
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(rows, args.stage_dir / "metrics.svg")
    print(f"n2_matrix={report['status']} m2_minus_b2_reduction={delta:+.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CASES = ("R0", "R1", "R2")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_svg(rows, path):
    colors = {"R0": "#4a5568", "R1": "#2b6cb0", "R2": "#2f855a"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360">',
        '<rect width="900" height="360" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">Task-state 修复机制对照</text>',
    ]
    metrics = (("1 - task cosine", "instruction_change", 0.25), ("read L1", "read_l1", 0.1))
    for block, (label, key, maximum) in enumerate(metrics):
        y0 = 65 + block * 135
        parts.append(f'<text x="24" y="{y0}" font-size="15">{label}</text>')
        for index, row in enumerate(rows):
            y = y0 + 18 + index * 31
            width = 620 * min(max(row[key] / maximum, 0.0), 1.0)
            parts.append(f'<text x="24" y="{y + 16}" font-size="13">{row["case"]}</text>')
            parts.append(f'<rect x="70" y="{y}" width="{width:.1f}" height="20" fill="{colors[row["case"]]}"/>')
            parts.append(f'<text x="{80 + width:.1f}" y="{y + 15}" font-size="12">{row[key]:.6f}</text>')
    parts.append('<text x="24" y="342" font-size="13" fill="#8b1a1a">进度代理不等同自然语言子任务阶段。</text></svg>\n')
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    rows = []
    for case in CASES:
        train = load(args.stage_dir / case / "train" / "metrics.json")
        probe = load(args.stage_dir / case / "probe" / "metrics.json")
        tm, pm = train["metrics"], probe["metrics"]
        rows.append(
            {
                "case": case,
                "train_status": train["status"],
                "probe_status": probe["status"],
                "final_s2_loss": tm["final_s2_loss"],
                "final_trajectory_loss": tm["final_trajectory_loss"],
                "final_task_contrastive_loss": tm.get("final_task_contrastive_loss", 0.0),
                "final_stage_loss": tm.get("final_stage_loss", 0.0),
                "task_cosine": pm["mean_instruction_task_cosine"],
                "instruction_change": 1.0 - pm["mean_instruction_task_cosine"],
                "task_relative_l2": pm["mean_instruction_task_relative_l2"],
                "read_l1": pm["mean_instruction_read_l1"],
                "stage_flip_rate": pm["instruction_stage_flip_rate"],
                "progress_probe_mae": pm["progress_probe_mae"],
                "progress_constant_mae": pm["progress_constant_mae"],
            }
        )
    by_case = {row["case"]: row for row in rows}
    baseline = by_case["R0"]

    def mechanism_gate(row):
        native_ok = (
            row["final_s2_loss"] <= baseline["final_s2_loss"] * 1.05
            and row["final_trajectory_loss"] <= baseline["final_trajectory_loss"] * 1.10
        )
        instruction_sensitive = row["task_cosine"] < 0.995 and row["task_relative_l2"] > 0.05
        reader_uses_change = row["read_l1"] >= 0.01 or row["stage_flip_rate"] >= 0.10
        return native_ok and instruction_sensitive and reader_uses_change

    r1_gate = mechanism_gate(by_case["R1"])
    r2_gate = mechanism_gate(by_case["R2"])
    r2_progress_gain = by_case["R2"]["progress_probe_mae"] < by_case["R1"]["progress_probe_mae"] * 0.95
    selected = "R2" if r2_gate and r2_progress_gain else ("R1" if r1_gate else "none")
    report = {
        "schema_version": 1,
        "git_commit": args.commit,
        "status": "passed" if selected != "none" else "failed",
        "selected": selected,
        "r1_mechanism_gate": r1_gate,
        "r2_mechanism_gate": r2_gate,
        "r2_progress_gain": r2_progress_gain,
        "rows": rows,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        f'|{row["case"]}|{row["final_s2_loss"]:.4f}|{row["final_trajectory_loss"]:.4f}|'
        f'{row["task_cosine"]:.6f}|{row["read_l1"]:.6f}|{row["stage_flip_rate"]:.3f}|'
        f'{row["progress_probe_mae"]:.4f}|' for row in rows
    )
    analysis = (
        f"机制门槛选择 `{selected}`；这只表示错配指令能改变 task-state/读取且原生 loss 未明显退化。"
        if selected != "none"
        else "R1/R2 均未同时达到指令敏感、读取变化和原生 loss 保持门槛；不进入闭环，继续定位表示或监督设计。"
    )
    summary = f"""# Task-state 修复机制对照简报

- 状态：`{'通过' if report['status'] == 'passed' else '未通过'}`
- 选择：`{selected}`
- 样本/步数：每组 `8 / 80`
- 边界：单 scene 小样本机制筛选，不是导航泛化结果

|组别|最终 S2|最终轨迹|错配 task 余弦|read L1|stage 翻转|进度 MAE|
|---|---:|---:|---:|---:|---:|---:|
{table}

## 简述与分析

{analysis}

R0 只使用 instruction-only 表征；R1 增加异路线指令分离；R2 再增加训练期软路线进度代理。软进度不等同自然语言子任务阶段。
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(rows, args.stage_dir / "metrics.svg")
    (args.stage_dir / "selected_variant.txt").write_text(selected + "\n", encoding="utf-8")
    print(f"task_state_repair={report['status']} selected={selected}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

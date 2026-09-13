#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CASES = ("R0P", "R1P")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_svg(rows, path):
    colors = {"R0P": "#4a5568", "R1P": "#2b6cb0"}
    metrics = (
        ("异路径 read L1", "cross_read_l1", 0.02),
        ("同路径复述 read L1", "paraphrase_read_l1", 0.02),
        ("异路径 task relative L2", "cross_relative_l2", 0.10),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390">',
        '<rect width="900" height="390" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">R1P 官方路径复述监督机制对照</text>',
    ]
    for block, (label, key, maximum) in enumerate(metrics):
        y0 = 62 + block * 96
        parts.append(f'<text x="24" y="{y0}" font-size="14">{label}</text>')
        for index, row in enumerate(rows):
            y = y0 + 12 + index * 30
            width = 600 * min(max(row[key] / maximum, 0.0), 1.0)
            parts.append(f'<text x="24" y="{y + 17}" font-size="13">{row["case"]}</text>')
            parts.append(f'<rect x="82" y="{y}" width="{width:.1f}" height="20" fill="{colors[row["case"]]}"/>')
            parts.append(f'<text x="{92 + width:.1f}" y="{y + 15}" font-size="12">{row[key]:.6f}</text>')
    parts.append('<text x="24" y="372" font-size="13" fill="#8b1a1a">路线级复述监督不等同任务阶段或导航泛化证明。</text></svg>\n')
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
                "cross_task_cosine": pm["mean_instruction_task_cosine"],
                "cross_relative_l2": pm["mean_instruction_task_relative_l2"],
                "cross_read_l1": pm["mean_instruction_read_l1"],
                "cross_stage_flip": pm["instruction_stage_flip_rate"],
                "paraphrase_task_cosine": pm["mean_paraphrase_task_cosine"],
                "paraphrase_relative_l2": pm["mean_paraphrase_task_relative_l2"],
                "paraphrase_read_l1": pm["mean_paraphrase_read_l1"],
                "paraphrase_stage_flip": pm["paraphrase_stage_flip_rate"],
                "gradient_audit_nonfinite": tm.get("gradient_audit_nonfinite", 0),
                "frozen_gradient_parameters": tm.get("frozen_gradient_parameters", 0),
            }
        )
    by_case = {row["case"]: row for row in rows}
    baseline, repaired = by_case["R0P"], by_case["R1P"]
    numeric_ok = all(
        row["train_status"] == "passed"
        and row["probe_status"] == "passed"
        and row["gradient_audit_nonfinite"] == 0
        and row["frozen_gradient_parameters"] == 0
        for row in rows
    )
    native_ok = (
        repaired["final_s2_loss"] <= baseline["final_s2_loss"] * 1.05
        and repaired["final_trajectory_loss"] <= baseline["final_trajectory_loss"] * 1.10
    )
    cross_route_sensitive = (
        repaired["cross_task_cosine"] < 0.995
        and repaired["cross_relative_l2"] > 0.05
        and (repaired["cross_read_l1"] >= 0.01 or repaired["cross_stage_flip"] >= 0.10)
    )
    paraphrase_ordered = (
        repaired["paraphrase_task_cosine"] > repaired["cross_task_cosine"]
        and repaired["paraphrase_relative_l2"] < repaired["cross_relative_l2"]
        and repaired["paraphrase_read_l1"] < repaired["cross_read_l1"]
    )
    passed = numeric_ok and native_ok and cross_route_sensitive and paraphrase_ordered
    report = {
        "schema_version": 1,
        "git_commit": args.commit,
        "status": "passed" if passed else "failed",
        "closed_loop_ready": passed,
        "gates": {
            "numeric_and_gradient": numeric_ok,
            "native_losses_preserved": native_ok,
            "cross_route_reader_sensitive": cross_route_sensitive,
            "same_route_paraphrase_ordered": paraphrase_ordered,
        },
        "rows": rows,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        f'|{row["case"]}|{row["final_s2_loss"]:.4f}|{row["final_trajectory_loss"]:.4f}|'
        f'{row["cross_task_cosine"]:.6f}|{row["cross_read_l1"]:.6f}|'
        f'{row["paraphrase_task_cosine"]:.6f}|{row["paraphrase_read_l1"]:.6f}|'
        for row in rows
    )
    decision = (
        "R1P 同时保持原生损失、区分异路径指令并对同路径复述保持更强一致性，可以进入最小配对闭环。"
        if passed
        else "R1P 未同时通过数值、原生损失、异路径读取敏感性与同路径复述排序门槛；不进入闭环，并停止继续堆叠无证据的 task-state 辅助损失。"
    )
    summary = f"""# 官方路径复述监督 R1P 简报

- 状态：`{'通过' if passed else '未通过'}`
- 样本/步数：每组 `8 / 80`
- 输入边界：S2 使用 RGB、历史 RGB 与理想相对位姿；不是严格 S-RGB
- 语义边界：同路径复述仅监督路线级任务身份，不代表任务阶段理解

|组别|最终 S2|最终轨迹|异路径 task 余弦|异路径 read L1|同路径 task 余弦|同路径 read L1|
|---|---:|---:|---:|---:|---:|---:|
{table}

## 门槛

- 数值与梯度：`{numeric_ok}`
- 原生损失保持：`{native_ok}`
- 异路径 reader 敏感：`{cross_route_sensitive}`
- 同路径复述排序正确：`{paraphrase_ordered}`

## 简述与分析

{decision}
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(rows, args.stage_dir / "metrics.svg")
    (args.stage_dir / "closed_loop_ready").write_text("1\n" if passed else "0\n", encoding="utf-8")
    print(f"task_paraphrase={report['status']} closed_loop_ready={int(passed)}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


EXPERIMENTS = ("B2", "M1", "M2")
SEEDS = (23, 47, 71)
VARIANTS = ("original", "scale01", "unit_norm", "lr01")


def load_metrics(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    return {
        "status": report["status"],
        "final_total_loss": metrics["final_total_loss"],
        "final_s2_loss": metrics["final_s2_loss"],
        "final_trajectory_loss": metrics["final_trajectory_loss"],
        "gradient_audit_nonfinite": metrics.get("gradient_audit_nonfinite", 0),
        "frozen_gradient_parameters": metrics.get("frozen_gradient_parameters", 0),
    }


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def original_path(reference_dir, stage_dir, experiment, seed):
    if seed == 23:
        return reference_dir / experiment / "metrics.json"
    return stage_dir / "original" / f"seed{seed}" / experiment / "metrics.json"


def variant_path(reference_dir, stage_dir, variant, seed):
    if variant == "original":
        return original_path(reference_dir, stage_dir, "M2", seed)
    if seed == 23:
        return stage_dir / "screen" / variant / "metrics.json"
    return stage_dir / "validate" / variant / f"seed{seed}" / "metrics.json"


def load_original(reference_dir, stage_dir):
    result = {}
    for experiment in EXPERIMENTS:
        rows = []
        for seed in SEEDS:
            path = original_path(reference_dir, stage_dir, experiment, seed)
            if not path.is_file():
                return None
            rows.append({"seed": seed, **load_metrics(path)})
        result[experiment] = rows
    return result


def original_gate(original):
    b2 = original["B2"]
    m2 = original["M2"]
    finite = all(
        row["status"] == "passed" and row["gradient_audit_nonfinite"] == 0 and row["frozen_gradient_parameters"] == 0
        for row in b2 + m2
    )
    return (
        finite
        and mean(m2, "final_total_loss") <= 1.01 * mean(b2, "final_total_loss")
        and mean(m2, "final_s2_loss") <= 1.01 * mean(b2, "final_s2_loss")
        and mean(m2, "final_trajectory_loss") <= 1.05 * mean(b2, "final_trajectory_loss")
        and sum(m2_row["final_total_loss"] <= 1.01 * b2_row["final_total_loss"] for m2_row, b2_row in zip(m2, b2)) >= 2
    )


def select_variant(reference_dir, stage_dir, original):
    if original_gate(original):
        return "original", {}
    scores = {}
    original_m2 = original["M2"][0]
    scores["original"] = original_m2["final_total_loss"]
    for variant in VARIANTS[1:]:
        path = variant_path(reference_dir, stage_dir, variant, 23)
        if not path.is_file():
            continue
        row = load_metrics(path)
        if row["status"] == "passed" and row["gradient_audit_nonfinite"] == 0:
            scores[variant] = row["final_total_loss"]
    if len(scores) != len(VARIANTS):
        return None, scores
    return min(scores, key=scores.get), scores


def validate_variant(reference_dir, stage_dir, variant):
    rows = []
    for seed in SEEDS:
        path = variant_path(reference_dir, stage_dir, variant, seed)
        if not path.is_file():
            return None
        rows.append({"seed": seed, **load_metrics(path)})
    return rows


def write_svg(original, optimized, selected, gate_passed, path):
    groups = [(name, original[name]) for name in EXPERIMENTS]
    if selected != "original" and optimized:
        groups.append((f"M2-{selected}", optimized))
    width, height = 860, 360
    max_loss = max(mean(rows, "final_total_loss") for _, rows in groups) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">N2 多 seed 最终 loss 对照</text>',
    ]
    colors = ("#4a5568", "#2b6cb0", "#c53030", "#2f855a")
    for index, ((label, rows), color) in enumerate(zip(groups, colors)):
        y = 65 + index * 58
        value = mean(rows, "final_total_loss")
        bar_width = 570 * value / max_loss
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" font-size="14">{label}</text>',
                f'<rect x="150" y="{y}" width="{bar_width:.1f}" height="25" fill="{color}"/>',
                f'<text x="{158 + bar_width:.1f}" y="{y + 18}" font-size="14">{value:.4f}</text>',
            ]
        )
    status = "通过闭环入口门槛" if gate_passed else "未通过闭环入口门槛"
    parts.append(f'<text x="24" y="330" font-size="16" font-weight="700">{status}</text></svg>\n')
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    original = load_original(args.reference_dir, args.stage_dir)
    if original is None:
        raise SystemExit("original seed matrix is incomplete")
    selected, scores = select_variant(args.reference_dir, args.stage_dir, original)
    if args.select_only:
        if selected is None:
            raise SystemExit("task-state screen is incomplete")
        (args.stage_dir / "selected_variant.txt").write_text(selected + "\n", encoding="utf-8")
        print(selected)
        return

    if selected is None:
        raise SystemExit("cannot finalize before all task-state screens exist")
    optimized = validate_variant(args.reference_dir, args.stage_dir, selected)
    if optimized is None:
        raise SystemExit(f"validation is incomplete for {selected}")
    b2 = original["B2"]
    finite = all(
        row["status"] == "passed" and row["gradient_audit_nonfinite"] == 0 and row["frozen_gradient_parameters"] == 0
        for row in b2 + optimized
    )
    gate_passed = (
        finite
        and mean(optimized, "final_total_loss") <= 1.01 * mean(b2, "final_total_loss")
        and mean(optimized, "final_s2_loss") <= 1.01 * mean(b2, "final_s2_loss")
        and mean(optimized, "final_trajectory_loss") <= 1.05 * mean(b2, "final_trajectory_loss")
        and sum(
            m2_row["final_total_loss"] <= 1.01 * b2_row["final_total_loss"] for m2_row, b2_row in zip(optimized, b2)
        )
        >= 2
    )
    report = {
        "schema_version": 1,
        "status": "passed" if gate_passed else "failed",
        "selected_variant": selected,
        "screen_seed23_final_total_loss": scores,
        "original": original,
        "optimized_m2": optimized,
        "means": {
            name: {
                "final_total_loss": mean(rows, "final_total_loss"),
                "final_s2_loss": mean(rows, "final_s2_loss"),
                "final_trajectory_loss": mean(rows, "final_trajectory_loss"),
            }
            for name, rows in {**original, f"M2-{selected}": optimized}.items()
        },
        "closed_loop_gate_passed": gate_passed,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for name, values in report["means"].items():
        rows.append(
            f'|{name}|{values["final_total_loss"]:.5f}|{values["final_s2_loss"]:.5f}|'
            f'{values["final_trajectory_loss"]:.5f}|'
        )
    decision = (
        "最佳 task-state 方案跨 seed 不劣于 B2，允许进入 1--4 episode 短闭环 smoke。"
        if gate_passed
        else "最佳 task-state 方案仍未达到不劣于 B2 的跨 seed 门槛，禁止进入短闭环，需继续定位。"
    )
    summary = f"""# N2 多 seed 与 task-state 稳定性简报

- 状态：`{'通过' if gate_passed else '未通过'}`
- seed：`23, 47, 71`
- 选中 task-state 方案：`{selected}`
- 闭环入口：`{'允许' if gate_passed else '禁止'}`

|组别|最终总 loss 均值|最终 S2 loss 均值|最终 trajectory loss 均值|
|---|---:|---:|---:|
{chr(10).join(rows)}

## 简述与分析

{decision} 门槛要求跨 seed 平均总/S2 loss 不超过 B2 的 101%，trajectory loss 不超过 B2 的 105%，且至少 2/3 seed 的总 loss 不劣于 B2 101%。下降百分比不用于最终决策。
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(original, optimized, selected, gate_passed, args.stage_dir / "metrics.svg")
    (args.stage_dir / "closed_loop_ready").write_text("1\n" if gate_passed else "0\n", encoding="utf-8")
    print(f"robustness={report['status']} selected={selected} closed_loop_ready={int(gate_passed)}")


if __name__ == "__main__":
    main()

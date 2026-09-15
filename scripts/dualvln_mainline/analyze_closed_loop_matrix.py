#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


VARIANTS = ("B0", "B1", "B2", "M1", "M2Z", "M2")
LABELS = {
    "B0": "原 checkpoint",
    "B1": "adapter 开启、历史输入置空",
    "B2": "adapter 开启、仅历史视觉",
    "M1": "adapter 开启、历史视觉+位姿",
    "M2Z": "同 M2 权重、task-state 置零",
    "M2": "完整历史视觉+位姿+task-state",
}


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_navigation(rows):
    return {
        "episodes": len(rows),
        "sr": mean(rows, "success"),
        "spl": mean(rows, "spl"),
        "os": mean(rows, "os"),
        "ne": mean(rows, "ne"),
        "steps": mean(rows, "steps"),
        "mean_s2_latency_s": mean(rows, "mean_s2_latency_s"),
    }


def flatten_events(rows):
    return [event for row in rows for event in row.get("events", [])]


def summarize_evidence(rows):
    events = flatten_events(rows)
    history_events = [event for event in events if max(event["valid_history_count"], default=0) > 0]
    null_weights = [weight for event in events for sample in event["null_weights"] for weight in sample]
    task_norms = [value for event in events for value in event["task_state_norm"]]
    history_mass = [
        sum(query)
        for event in history_events
        for sample in event["read_weights"]
        for query in sample
    ]
    normalized_entropy = []
    for event in history_events:
        for sample in event["read_weights"]:
            valid_count = max(event["valid_history_count"])
            for query in sample:
                values = query[:valid_count]
                total = sum(values)
                if total > 0 and valid_count > 1:
                    probs = [value / total for value in values if value > 0]
                    normalized_entropy.append(-sum(p * math.log(p) for p in probs) / math.log(valid_count))
    return {
        "events": len(events),
        "history_conditioned_events": len(history_events),
        "mean_task_state_norm": statistics.fmean(task_norms) if task_norms else None,
        "mean_null_weight": statistics.fmean(null_weights) if null_weights else None,
        "mean_history_mass": statistics.fmean(history_mass) if history_mass else None,
        "mean_normalized_read_entropy": statistics.fmean(normalized_entropy) if normalized_entropy else None,
    }


def write_svg(report, path):
    width, height = 980, 500
    max_ne = max(values["navigation"]["ne"] for values in report["variants"].values()) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="24" y="32" font-size="20" font-weight="700">DualVLN 4-episode 闭环归因矩阵</text>',
        '<text x="24" y="55" font-size="13" fill="#555">NE 越低越好；极小样本只用于方向与机制诊断</text>',
    ]
    colors = ("#4a5568", "#805ad5", "#2b6cb0", "#319795", "#d69e2e", "#2f855a")
    for index, (variant, color) in enumerate(zip(VARIANTS, colors)):
        values = report["variants"][variant]
        navigation = values["navigation"]
        evidence = values["evidence"]
        y = 82 + index * 62
        bar_width = 520 * navigation["ne"] / max_ne
        parts.extend(
            [
                f'<text x="24" y="{y + 17}" font-size="14">{variant}</text>',
                f'<rect x="90" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}"/>',
                f'<text x="{100 + bar_width:.1f}" y="{y + 16}" font-size="13">NE {navigation["ne"]:.3f} m</text>',
                f'<text x="700" y="{y + 16}" font-size="12">SR {navigation["sr"]:.2f}  null {evidence["mean_null_weight"] if evidence["mean_null_weight"] is not None else 0:.3f}</text>',
            ]
        )
    parts.append('<text x="24" y="475" font-size="13" fill="#8b1a1a">不把 4 episodes 解释为 val-unseen 泛化结果；stage logits 尚无标签监督。</text>')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    progress = {variant: read_jsonl(args.stage_dir / variant / "progress.json") for variant in VARIANTS}
    episode_keys = {
        variant: [(row["scene_id"], row["episode_id"], row["episode_instruction"]) for row in rows]
        for variant, rows in progress.items()
    }
    paired = bool(episode_keys["B0"]) and all(keys == episode_keys["B0"] for keys in episode_keys.values())
    variants = {}
    manifests = {}
    for variant in VARIANTS:
        manifests[variant] = json.loads(
            (args.stage_dir / variant / "inference_manifest.json").read_text(encoding="utf-8")
        )
        variants[variant] = {
            "label": LABELS[variant],
            "navigation": summarize_navigation(progress[variant]),
            "evidence": summarize_evidence(read_jsonl(args.stage_dir / variant / "evidence_trace.jsonl")),
        }
    expected_ablation = {"B1": "null", "B2": "content", "M1": "spatial", "M2Z": "spatial", "M2": "task_spatial"}
    protocol_ok = paired and not manifests["B0"]["evidence_enabled"]
    protocol_ok &= all(manifests[name]["evidence_ablation"] == mode for name, mode in expected_ablation.items())
    protocol_ok &= all(variants[name]["evidence"]["events"] > 0 for name in VARIANTS[1:])
    latent_query_bypass = {
        manifests[name].get("evidence_latent_query_bypass", True) for name in VARIANTS[1:]
    }
    protocol_ok &= len(latent_query_bypass) == 1
    non_base_adapters = {
        manifests[name].get("evidence_adapter", {}).get("adapter_path") for name in VARIANTS[1:]
    }
    shared_adapter = len(non_base_adapters) == 1 and None not in non_base_adapters
    report = {
        "schema_version": 1,
        "status": "passed" if protocol_ok else "failed",
        "scope": "paired_closed_loop_attribution_smoke_not_generalization",
        "paired_episodes": paired,
        "shared_adapter_across_ablations": shared_adapter,
        "latent_query_bypass": latent_query_bypass.pop(),
        "episodes": episode_keys["B0"],
        "variants": variants,
        "delta_vs_b0": {
            name: {
                key: variants[name]["navigation"][key] - variants["B0"]["navigation"][key]
                for key in ("sr", "spl", "os", "ne", "steps", "mean_s2_latency_s")
            }
            for name in VARIANTS[1:]
        },
        "delta_m2_vs_m2z": {
            key: variants["M2"]["navigation"][key] - variants["M2Z"]["navigation"][key]
            for key in ("sr", "spl", "os", "ne", "steps", "mean_s2_latency_s")
        },
        "manifests": manifests,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = []
    for variant in VARIANTS:
        nav = variants[variant]["navigation"]
        evidence = variants[variant]["evidence"]
        null = "-" if evidence["mean_null_weight"] is None else f'{evidence["mean_null_weight"]:.3f}'
        mass = "-" if evidence["mean_history_mass"] is None else f'{evidence["mean_history_mass"]:.3f}'
        table.append(
            f'|{variant}|{LABELS[variant]}|{nav["sr"]:.3f}|{nav["spl"]:.3f}|{nav["ne"]:.3f}|'
            f'{nav["mean_s2_latency_s"]:.3f}|{null}|{mass}|'
        )
    m2_delta = report["delta_vs_b0"]["M2"]
    task_delta = report["delta_m2_vs_m2z"]
    summary = f"""# DualVLN 4-episode 闭环归因矩阵简报

- 协议状态：`{'通过' if protocol_ok else '失败'}`
- 配对 episode 数：`{len(episode_keys['B0'])}`
- 数据：R2R `val_unseen` 排序后的相同 episodes
- 在线输入协议：S2 为 RGB+Habitat GPS/compass 理想相对位姿；S1/执行使用 RGB-D+pose
- 归因协议：`{'同一 adapter 权重，仅改变输入消融' if shared_adapter else '各组 adapter 权重不同，结果同时包含训练差异'}`

|组别|含义|SR|SPL|NE (m)|S2 延迟 (s)|null 权重|历史读取质量|
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## 简述与分析

M2 相对 B0 的 `SR/SPL/NE` 差值为 `{m2_delta['sr']:+.3f}/{m2_delta['spl']:+.3f}/{m2_delta['ne']:+.3f}`；完整 M2 相对同权重 task-state 置零的 M2Z 差值为 `{task_delta['sr']:+.3f}/{task_delta['spl']:+.3f}/{task_delta['ne']:+.3f}`。B1 用于识别仅由 adapter/null evidence 带来的变化，B2/M1 用于区分历史视觉与位姿元数据。read/null 权重只用于检查模型是否读取历史；stage logits 尚无可靠阶段标签，不能解释为任务阶段准确率。本阶段是接口、行为方向和归因 smoke，不是 val-unseen 泛化结论。
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(report, args.stage_dir / "metrics.svg")
    print(f"closed_loop_matrix={report['status']} episodes={len(episode_keys['B0'])} paired={int(paired)}")
    raise SystemExit(0 if protocol_ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(rows):
    return {
        "episodes": len(rows),
        "sr": statistics.fmean(row["success"] for row in rows),
        "spl": statistics.fmean(row["spl"] for row in rows),
        "os": statistics.fmean(row["os"] for row in rows),
        "ne": statistics.fmean(row["ne"] for row in rows),
        "steps": statistics.fmean(row["steps"] for row in rows),
        "duration_s": statistics.fmean(row["duration_s"] for row in rows),
        "mean_s2_latency_s": statistics.fmean(row["mean_s2_latency_s"] for row in rows),
        "s2_calls": sum(row["s2_calls"] for row in rows),
    }


def write_svg(report, path):
    b0 = report["variants"]["B0"]
    m2 = report["variants"]["M2"]
    width, height = 900, 420
    panels = (
        ("SR", "sr", 1.0, True),
        ("SPL", "spl", 1.0, True),
        ("OS", "os", 1.0, True),
        ("NE (m, 越低越好)", "ne", max(b0["ne"], m2["ne"], 1.0) * 1.15, False),
        ("S2 latency (s)", "mean_s2_latency_s", max(b0["mean_s2_latency_s"], m2["mean_s2_latency_s"], 1.0) * 1.15, False),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700">DualVLN 1--4 episode closed-loop smoke</text>',
        '<text x="24" y="54" font-size="13" fill="#555">Paired val-unseen episodes: B0 vs M2-lr01</text>',
    ]
    for index, (label, key, maximum, _) in enumerate(panels):
        y = 82 + index * 58
        parts.append(f'<text x="24" y="{y + 18}" font-size="14">{label}</text>')
        for offset, (name, values, color) in enumerate((("B0", b0, "#4a5568"), ("M2", m2, "#2f855a"))):
            value = values[key]
            bar_width = 500 * value / maximum
            by = y + offset * 21
            parts.append(f'<rect x="210" y="{by}" width="{bar_width:.1f}" height="16" fill="{color}"/>')
            parts.append(f'<text x="{218 + bar_width:.1f}" y="{by + 13}" font-size="12">{name} {value:.4f}</text>')
    parts.append(
        f'<text x="24" y="395" font-size="13" fill="#8b1a1a">Only {b0["episodes"]} paired episodes; interface/direction smoke, not a generalization claim.</text>'
    )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    manifests = {}
    for variant in ("B0", "M2"):
        case_dir = args.stage_dir / variant
        rows = read_jsonl(case_dir / "progress.json")
        if not rows:
            raise SystemExit(f"{variant} has no completed episode")
        cases[variant] = rows
        manifests[variant] = json.loads((case_dir / "inference_manifest.json").read_text(encoding="utf-8"))

    episode_keys = {
        variant: [(row["scene_id"], row["episode_id"], row["episode_instruction"]) for row in rows]
        for variant, rows in cases.items()
    }
    paired = episode_keys["B0"] == episode_keys["M2"]
    b0 = summarize(cases["B0"])
    m2 = summarize(cases["M2"])
    adapter_loaded = bool(manifests["M2"].get("evidence_adapter")) and manifests["M2"].get("evidence_enabled")
    base_clean = not manifests["B0"].get("evidence_enabled") and not manifests["B0"].get("evidence_adapter")
    finite = all(
        isinstance(value, (int, float)) and value == value
        for values in (b0, m2)
        for key, value in values.items()
        if key != "episodes"
    )
    protocol_passed = paired and adapter_loaded and base_clean and finite
    report = {
        "schema_version": 1,
        "status": "passed" if protocol_passed else "failed",
        "scope": "short_closed_loop_smoke_not_generalization",
        "paired_episodes": paired,
        "adapter_loaded": adapter_loaded,
        "base_evidence_disabled": base_clean,
        "finite_metrics": finite,
        "variants": {"B0": b0, "M2": m2},
        "delta_m2_minus_b0": {key: m2[key] - b0[key] for key in ("sr", "spl", "os", "ne", "steps", "mean_s2_latency_s")},
        "episodes": episode_keys["B0"],
        "manifests": manifests,
    }
    (args.stage_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    delta = report["delta_m2_minus_b0"]
    direction = (
        "M2 在本次配对 smoke 上至少一个导航指标改善且没有 SR 退化。"
        if m2["sr"] >= b0["sr"] and (m2["ne"] < b0["ne"] or m2["spl"] > b0["spl"])
        else "本次极小样本尚未显示 M2 的闭环方向改善；先定位输出、轨迹和 episode 级差异，不扩大结论。"
    )
    summary = f"""# DualVLN 短闭环 smoke 简报

- 协议状态：`{'通过' if protocol_passed else '失败'}`
- 配对 episode 数：`{b0['episodes']}`
- 数据：R2R `val_unseen` 排序后的相同 episode
- 位姿输入：Habitat GPS/compass 理想里程计，仅使用当步及历史
- 深度滤波：`{manifests['M2']['depth_filter_backend']}`

|方案|SR|SPL|OS|NE (m)|平均步数|平均 S2 延迟 (s)|
|---|---:|---:|---:|---:|---:|---:|
|B0|{b0['sr']:.4f}|{b0['spl']:.4f}|{b0['os']:.4f}|{b0['ne']:.4f}|{b0['steps']:.1f}|{b0['mean_s2_latency_s']:.3f}|
|M2-lr01|{m2['sr']:.4f}|{m2['spl']:.4f}|{m2['os']:.4f}|{m2['ne']:.4f}|{m2['steps']:.1f}|{m2['mean_s2_latency_s']:.3f}|

## 简述与分析

{direction} M2 相对 B0 的 `SR/SPL/OS/NE` 差值为 `{delta['sr']:+.4f}/{delta['spl']:+.4f}/{delta['os']:+.4f}/{delta['ne']:+.4f}`。本阶段只证明闭环接口、因果历史、adapter 加载和小样本行为可测，不把 1--4 episode 当作 val-unseen 泛化结果。
"""
    (args.stage_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(report, args.stage_dir / "metrics.svg")
    print(f"closed_loop_smoke={report['status']} episodes={b0['episodes']} paired={int(paired)}")
    raise SystemExit(0 if protocol_passed else 1)


if __name__ == "__main__":
    main()

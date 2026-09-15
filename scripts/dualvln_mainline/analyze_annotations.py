#!/usr/bin/env python3
"""Run a small, transparent, episode-grouped relevance probe over human labels."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidates(row):
    ann = row.get("annotation", {})
    positives = set(x for x in ann.get("preferred_evidence", []) if x in {"A", "B", "C"})
    return positives, {c["label"]: c for c in row.get("candidates", [])}


def rank_hit(row, key, reverse=False):
    positives, cs = candidates(row)
    if not positives or not cs:
        return None
    ordered = sorted(cs.values(), key=key, reverse=reverse)
    return int(ordered[0]["label"] in positives)


def pairwise(row, key, reverse=False):
    positives, cs = candidates(row)
    if not positives or len(positives) == len(cs):
        return None
    pos = [c for label, c in cs.items() if label in positives]
    neg = [c for label, c in cs.items() if label not in positives]
    good = 0; total = 0
    for p in pos:
        for n in neg:
            pv, nv = key(p), key(n)
            good += int(pv > nv if reverse else pv < nv)
            total += 1
    return good / total if total else None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def svg(scores, path):
    maximum = max(1.0, *(v for v in scores.values()))
    parts = []
    for i, (label, value) in enumerate(scores.items()):
        y = 58 + i * 42
        width = 410 * value / maximum
        parts.append(f'<text x="24" y="{y+17}" font-size="14">{label}</text><rect x="150" y="{y}" width="410" height="22" fill="#edf2f7"/><rect x="150" y="{y}" width="{width:.1f}" height="22" fill="#3878c7"/><text x="568" y="{y+17}" font-size="14">{value:.1%}</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="260"><rect width="100%" height="100%" fill="white"/><text x="24" y="30" font-size="20" font-weight="700">DualVLN 人工 evidence relevance probe</text>' + ''.join(parts) + '</svg>\n', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="按 episode 分组统计人工 evidence 偏好")
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    data = rows(args.annotations)
    yes = [r for r in data if (r.get("annotation", {}).get("need_history") == "yes") and candidates(r)[0]]
    grouped = defaultdict(list)
    for r in yes:
        grouped[str(r.get("episode_id"))].append(r)
    recent = lambda c: c.get("age", 10**9)
    nearest = lambda c: c.get("relative_pose", {}).get("distance", 10**9)
    oldest = lambda c: c.get("age", -1)
    metrics = {
        "schema_version": 1, "rows": len(data), "yes_rows_with_positive": len(yes), "episodes": len(grouped),
        "preferred_count": sum(len(candidates(r)[0]) for r in yes),
        "preferred_card_rate": mean([len(candidates(r)[0]) / max(1, len(r.get("candidates", []))) for r in yes]),
        "top1_hit": {"recent": mean([rank_hit(r, recent) for r in yes]), "nearest_pose": mean([rank_hit(r, nearest) for r in yes]), "oldest": mean([rank_hit(r, oldest, reverse=True) for r in yes])},
        "pairwise_accuracy": {"recent": mean([pairwise(r, recent) for r in yes]), "nearest_pose": mean([pairwise(r, nearest) for r in yes]), "oldest": mean([pairwise(r, oldest, reverse=True) for r in yes])},
        "position_selected": Counter(x for r in yes for x in candidates(r)[0]),
        "role": Counter(r.get("annotation", {}).get("evidence_role") for r in data),
        "confidence": Counter(r.get("annotation", {}).get("confidence") for r in data),
        "episode_rows": {ep: len(rs) for ep, rs in grouped.items()},
    }
    # Random top-1 is estimated within each card, not by assuming a fixed candidate count.
    rng = random.Random(args.seed)
    random_hits = []
    for r in yes:
        positives, cs = candidates(r)
        random_hits.append(sum(int(rng.choice(list(cs)) in positives) for _ in range(200)) / 200)
    metrics["top1_hit"]["random_card_baseline"] = mean(random_hits)
    metrics["analysis_scope"] = "single_scene_17DRP5sb8fy; grouped by episode; descriptive probe, not generalization"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    scores = {"最近帧": metrics["top1_hit"]["recent"], "最近位姿": metrics["top1_hit"]["nearest_pose"], "较旧帧": metrics["top1_hit"]["oldest"], "随机基线": metrics["top1_hit"]["random_card_baseline"]}
    svg(scores, args.output_dir / "metrics.svg")
    gain = metrics["top1_hit"]["recent"] - metrics["top1_hit"]["random_card_baseline"]
    summary = f"""# DualVLN 人工 evidence relevance probe

- 有效 `need_history=yes` 样本：{len(yes)} 条，覆盖 {len(grouped)} 个 episode
- 人工选择总数：{metrics['preferred_count']}（允许多正例）
- Top-1 命中：最近帧 {metrics['top1_hit']['recent']:.1%}；最近位姿 {metrics['top1_hit']['nearest_pose']:.1%}；较旧帧 {metrics['top1_hit']['oldest']:.1%}；随机卡片基线 {metrics['top1_hit']['random_card_baseline']:.1%}
- 最近帧相对随机基线差值：{gain:+.1%}

## 结果分析

该 probe 只检验人工偏好是否能被简单年龄/位姿启发式区分，不能替代 evidence reader 训练，也不能代表跨场景泛化。人工标签中允许两张正例，因此同时保留 pairwise accuracy 和 Top-1 结果；若简单基线没有稳定超过随机，下一步应先复核标注规则或扩展跨场景 pilot，而不是直接训练模型。

完整计数、角色分布和按 episode 划分信息见 `metrics.json`；柱状图见 `metrics.svg`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"样本: {len(yes)} | episode: {len(grouped)} | 最近帧 Top-1: {metrics['top1_hit']['recent']:.1%} | 随机: {metrics['top1_hit']['random_card_baseline']:.1%}")
    print(f"报告目录: {args.output_dir}")


if __name__ == "__main__":
    main()

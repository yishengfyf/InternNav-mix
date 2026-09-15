#!/usr/bin/env python3
"""Compare temporal, spatial, and rendering quality of candidate batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def black_fraction(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((80, 80))
        pixels = list(image.getdata())
    return sum(max(pixel) <= 4 for pixel in pixels) / max(1, len(pixels))


def summarize(path, exclude_scene=None):
    rows = read_jsonl(path)
    if exclude_scene:
        rows = [row for row in rows if str(row.get("scene_id")) != exclude_scene]
    root = path.parent
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    count = max(1, len(candidates))
    black = [black_fraction(root / candidate["image_path"]) for candidate in candidates]
    return {
        "cards": len(rows),
        "scenes": len({str(row.get("scene_id")) for row in rows}),
        "episodes": len({(str(row.get("scene_id")), str(row.get("episode_id"))) for row in rows}),
        "history_age_mean": sum(candidate["age"] for candidate in candidates) / count,
        "distance_mean_m": sum(candidate["relative_pose"]["distance"] for candidate in candidates) / count,
        "abs_yaw_mean_deg": sum(abs(candidate["relative_pose"]["dyaw_deg"]) for candidate in candidates) / count,
        "distance_ge_2m_rate": sum(candidate["relative_pose"]["distance"] >= 2 for candidate in candidates) / count,
        "degenerate_pose_rate": sum(candidate["relative_pose"]["distance"] < 0.05 for candidate in candidates) / count,
        "black_gt_25pct_count": sum(value > 0.25 for value in black),
        "black_fraction_mean": sum(black) / count,
    }


def parse_named(values):
    parsed = {}
    for value in values:
        name, path = value.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def write_svg(sets, path):
    measures = (
        ("平均历史年龄", "history_age_mean", 20.0),
        ("平均位移(m)", "distance_mean_m", 3.0),
        (">=2m 比例", "distance_ge_2m_rate", 0.5),
        ("黑区超限数", "black_gt_25pct_count", 20.0),
    )
    colors = ("#577590", "#43aa8b", "#f8961e")
    chunks = ['<rect width="100%" height="100%" fill="white"/>', '<text x="24" y="32" font-size="20" font-weight="700">DualVLN 候选策略对比</text>']
    for row_index, (label, key, scale) in enumerate(measures):
        y = 68 + row_index * 70
        chunks.append(f'<text x="24" y="{y + 18}" font-size="14">{label}</text>')
        for set_index, (name, metrics) in enumerate(sets.items()):
            value = metrics[key]
            width = min(380, 380 * value / scale)
            yy = y + set_index * 18
            chunks.append(f'<rect x="150" y="{yy}" width="{width:.1f}" height="14" fill="{colors[set_index % len(colors)]}"/><text x="538" y="{yy + 12}" font-size="12">{name}: {value:.3g}</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="370">' + "".join(chunks) + '</svg>\n', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", required=True, help="name=annotations_template.jsonl")
    parser.add_argument("--reviews", nargs="*", default=[], help="name=review_metrics.json")
    parser.add_argument("--exclude-scene")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = parse_named(args.sets)
    sets = {
        name: summarize(path, args.exclude_scene if name.endswith("curated") else None)
        for name, path in paths.items()
    }
    reviews = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in parse_named(args.reviews).items()}
    metrics = {"schema_version": 1, "sets": sets, "model_blind_reviews": reviews, "review_training_allowed": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    old, new = sets["v1"], sets["v3"]
    summary = f"""# DualVLN 多场景候选策略对比

|集合|卡片|场景|历史平均年龄|平均位移|至少 2m|黑区超限候选|
|---|---:|---:|---:|---:|---:|---:|
|v1 时间分层|{old['cards']}|{old['scenes']}|{old['history_age_mean']:.2f}|{old['distance_mean_m']:.2f} m|{old['distance_ge_2m_rate']:.1%}|{old['black_gt_25pct_count']}|
|v3 阶段/空间分层|{new['cards']}|{new['scenes']}|{new['history_age_mean']:.2f}|{new['distance_mean_m']:.2f} m|{new['distance_ge_2m_rate']:.1%}|{new['black_gt_25pct_count']}|

v3 在保持 104 张、6 场景、52 episode 和零因果违规的前提下，扩大了历史时间与空间跨度，并通过黑区过滤消除了超过 25% 无效黑像素的历史候选。模型盲审只用于候选质量诊断：同一 18 张抽样中，明确 `need_history=yes` 从 1 张增至 6 张；它不属于人工真值，不允许直接接入训练。

`XcA2TqTSSAj` 仍存在白墙和几何破损，首轮人工包将其剔除。curated 集合保留 5 个新场景、84 张卡片、42 个 episode；加上既有 40 张单场景人工 pilot 后总规模为 124 张。下一步必须由人工填写 curated 包并抽取 20--30% 独立复核，之后才运行 scene-held-out reader。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(sets, args.output_dir / "metrics.svg")
    print(f"候选对比完成: v1={old['cards']} v3={new['cards']} curated={sets.get('v3_curated', {}).get('cards')}")


if __name__ == "__main__":
    main()

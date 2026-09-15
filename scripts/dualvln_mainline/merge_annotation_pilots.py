#!/usr/bin/env python3
"""Merge scene-specific candidate packages into one auditable annotation batch."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_image(source_root, relative_path, output_root):
    source = source_root / relative_path
    target = output_root / relative_path
    if not source.is_file():
        raise ValueError(f"候选图片不存在: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"同名候选图片内容冲突: {relative_path}")
        return
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description="合并多场景人工标注候选包")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    templates, manifests = [], []
    seen_ids = set()
    scene_counts = Counter()
    episode_keys = set()
    for source_root in args.inputs:
        source_templates = read_jsonl(source_root / "annotations_template.jsonl")
        source_manifests = read_jsonl(source_root / "candidate_manifest.jsonl")
        if [row["annotation_id"] for row in source_templates] != [
            row["annotation_id"] for row in source_manifests
        ]:
            raise ValueError(f"模板与 manifest 顺序不一致: {source_root}")
        for template, manifest in zip(source_templates, source_manifests):
            annotation_id = template["annotation_id"]
            if annotation_id in seen_ids:
                raise ValueError(f"重复 annotation_id: {annotation_id}")
            seen_ids.add(annotation_id)
            scene = str(template["scene_id"])
            scene_counts[scene] += 1
            episode_keys.add((scene, str(template["episode_id"])))
            image_paths = [template["current_image_path"]] + [
                candidate["image_path"] for candidate in template["candidates"]
            ]
            for relative_path in image_paths:
                copy_image(source_root, relative_path, args.output)
            templates.append(template)
            manifests.append(manifest)
    templates.sort(key=lambda row: row["annotation_id"])
    manifests.sort(key=lambda row: row["annotation_id"])
    (args.output / "annotations_template.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in templates) + "\n",
        encoding="utf-8",
    )
    (args.output / "candidate_manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifests) + "\n",
        encoding="utf-8",
    )
    for filename in ("index.html", "schema.json"):
        shutil.copy2(args.inputs[0] / filename, args.output / filename)
    metrics = {
        "schema_version": 1,
        "status": "passed",
        "candidate_count": len(templates),
        "scene_count": len(scene_counts),
        "episode_count": len(episode_keys),
        "scene_candidates": dict(sorted(scene_counts.items())),
        "unique_annotation_ids": len(seen_ids) == len(templates),
        "causal_violations": sum(
            not row["causal_check"]["all_history_before_current"]
            or row["causal_check"]["future_input_exposed"]
            for row in templates
        ),
        "template_sha256": sha256(args.output / "annotations_template.jsonl"),
        "manifest_sha256": sha256(args.output / "candidate_manifest.jsonl"),
    }
    if metrics["causal_violations"]:
        raise ValueError("合并包包含非因果候选")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    scene_rows = "\n".join(f"|{scene}|{count}|" for scene, count in scene_counts.most_common())
    summary = f"""# DualVLN 第二批多场景标注候选

- 状态：`通过`
- 候选卡片：`{len(templates)}`
- 场景：`{len(scene_counts)}`
- episode：`{len(episode_keys)}`
- 因果违规：`0`
- 自动标签：`无`

|场景|候选数|
|---|---:|
{scene_rows}

本目录只合并程序挖掘的因果候选与图片，不包含人工或模型生成的 relevance 标签。使用 `index.html` 标注后另行导出 JSONL；不要覆盖第一批人工原文件。
"""
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    (args.output / "README.md").write_text(
        "# 第二批多场景标注\n\n在本目录启动静态 HTTP 服务并打开 `index.html`。页面读取 `annotations_template.jsonl`，支持自动暂存和导出。\n",
        encoding="utf-8",
    )
    bars = []
    for index, (scene, count) in enumerate(scene_counts.most_common()):
        y = 62 + index * 34
        bars.append(
            f'<text x="20" y="{y + 16}" font-size="13">{scene}</text>'
            f'<rect x="155" y="{y}" width="{count * 20}" height="20" fill="#2f855a"/>'
            f'<text x="{165 + count * 20}" y="{y + 15}" font-size="12">{count}</text>'
        )
    height = 95 + len(scene_counts) * 34
    (args.output / "metrics.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="700" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="30" font-size="20" font-weight="700">第二批因果候选场景分布</text>'
        + "".join(bars)
        + "</svg>\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert reviewed annotations into a minimal, hash-bound relevance manifest."""
import argparse
import hashlib
import json
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


def build_rows(annotations, manifest):
    candidates = {row["annotation_id"]: row for row in manifest}
    output = []
    for row in annotations:
        annotation_id = row["annotation_id"]
        source = candidates.get(annotation_id)
        if source is None:
            raise ValueError(f"candidate manifest 缺少 {annotation_id}")
        labels = [candidate["label"] for candidate in source["candidates"]]
        annotation = row["annotation"]
        need_history = annotation["need_history"]
        preferred = set(annotation["preferred_evidence"])
        if need_history == "yes":
            targets, null_target, supervised = [int(label in preferred) for label in labels], 0, True
        elif need_history == "no":
            targets, null_target, supervised = [0] * len(labels), 1, True
        else:
            targets, null_target, supervised = [-1] * len(labels), -1, False
        output.append(
            {
                "schema_version": 1,
                "annotation_id": annotation_id,
                "episode_id": str(source["episode_id"]),
                "candidate_labels": labels,
                "candidate_frame_ids": [candidate["frame_id"] for candidate in source["candidates"]],
                "targets": targets,
                "null_target": null_target,
                "supervised": supervised,
            }
        )
    if len(output) != len(candidates):
        raise ValueError("人工记录数与 candidate manifest 不一致")
    return output


def main():
    parser = argparse.ArgumentParser(description="生成最小多正例 relevance 监督 manifest")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    annotations, manifest = read_jsonl(args.annotations), read_jsonl(args.candidate_manifest)
    output = build_rows(annotations, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.output_dir / "relevance_supervision.jsonl"
    target_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n", encoding="utf-8"
    )
    states = Counter(
        "uncertain" if not row["supervised"] else "no" if row["null_target"] == 1 else "yes"
        for row in output
    )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "rows": len(output),
        "states": dict(states),
        "positive_candidate_targets": sum(sum(max(0, target) for target in row["targets"]) for row in output),
        "source_annotations_sha256": sha256(args.annotations),
        "source_candidate_manifest_sha256": sha256(args.candidate_manifest),
        "output_sha256": sha256(target_path),
        "privacy_fields_excluded": ["instruction", "image_path", "short_reason", "annotator_id", "created_at"],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# 最小 relevance 监督 manifest

- 记录：{len(output)}；状态：{dict(states)}
- 候选正例 target：{metrics['positive_candidate_targets']}
- 输出 SHA256：`{metrics['output_sha256']}`
- 已排除：指令、图像路径、人工理由、标注者和时间信息

该文件仍属于未公开研究监督数据。只有获得对这一具体最小文件的明确批准后，才能复制到实验服务器。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

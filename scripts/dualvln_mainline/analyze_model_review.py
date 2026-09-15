#!/usr/bin/env python3
"""Materialize and summarize annotation-blind model review decisions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "need_history",
    "preferred_evidence",
    "evidence_role",
    "misleading_evidence",
    "confidence",
    "short_reason",
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_svg(metrics, path: Path):
    labels = [("需要历史", "yes", "#277da1"), ("不需要", "no", "#43aa8b"), ("不确定", "uncertain", "#f9c74f")]
    maximum = max(1, *(metrics["need_history"].get(key, 0) for _, key, _ in labels))
    bars = []
    for index, (label, key, color) in enumerate(labels):
        value = metrics["need_history"].get(key, 0)
        y = 68 + index * 44
        width = 390 * value / maximum
        bars.append(
            f'<text x="24" y="{y + 18}" font-size="15">{label}</text>'
            f'<rect x="112" y="{y}" width="390" height="24" fill="#edf2f7"/>'
            f'<rect x="112" y="{y}" width="{width:.1f}" height="24" fill="{color}"/>'
            f'<text x="514" y="{y + 18}" font-size="15">{value}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="580" height="230">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="24" y="32" font-size="20" font-weight="700">DualVLN 模型独立复核</text>'
        + "".join(bars)
        + '</svg>\n',
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.template)
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    row_ids = {row["annotation_id"] for row in rows}
    if row_ids != set(decisions):
        missing = sorted(row_ids - set(decisions))
        extra = sorted(set(decisions) - row_ids)
        raise ValueError(f"复核决策与模板不一致: missing={missing}, extra={extra}")

    created_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        values = decisions[row["annotation_id"]]
        if len(values) != len(FIELDS):
            raise ValueError(f"{row['annotation_id']} 的决策字段数错误")
        row["annotation"] = dict(zip(FIELDS, values))
        row["annotator_id"] = "model_reviewer"
        row["created_at"] = created_at

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "model_review.jsonl"
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    need = Counter(row["annotation"]["need_history"] for row in rows)
    role = Counter(row["annotation"]["evidence_role"] for row in rows)
    confidence = Counter(row["annotation"]["confidence"] for row in rows)
    scene = Counter(row["scene_id"] for row in rows)
    metrics = {
        "schema_version": 1,
        "review_protocol": "annotation_blind_v1",
        "reviewer_id": "model_reviewer",
        "intended_use": "candidate_triage_and_disagreement_analysis_only",
        "training_supervision_allowed": False,
        "rows": len(rows),
        "scenes": dict(scene),
        "need_history": dict(need),
        "evidence_role": dict(role),
        "confidence": dict(confidence),
        "usable_yes_rows": need.get("yes", 0),
        "uncertain_rate": need.get("uncertain", 0) / len(rows),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# DualVLN 模型独立复核结果

- 共复核 `{len(rows)}` 张，覆盖 `{len(scene)}` 个场景。
- `need_history`：yes `{need.get('yes', 0)}`，no `{need.get('no', 0)}`，uncertain `{need.get('uncertain', 0)}`。
- 不确定率：`{metrics['uncertain_rate']:.1%}`；高/中/低置信度：`{confidence.get('high', 0)}/{confidence.get('medium', 0)}/{confidence.get('low', 0)}`。

## 分析

本轮是与第一批人工真值隔离的模型盲审，只用于候选筛查与后续人工分歧分析，**不得作为第二位人工标注，也不得直接转换为训练监督**。

仅 `{need.get('yes', 0)}` 张能较明确地从候选历史获得当前帧缺失的路线进度信息；`{need.get('uncertain', 0)}` 张因目标地标不可见、候选同质、渲染缺损或位姿退化而无法可靠判断。这说明当前均匀时间间隔候选主要覆盖相邻视角，任务相关历史密度偏低。下一轮人工标注应优先复核 uncertain 和模型判定 yes 的卡片；候选挖掘应增加阶段转折、地标出现/消失和有效位移筛选。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(metrics, args.output_dir / "metrics.svg")
    print(f"模型复核: {len(rows)} | yes={need.get('yes', 0)} no={need.get('no', 0)} uncertain={need.get('uncertain', 0)}")
    print(f"输出: {args.output_dir}")


if __name__ == "__main__":
    main()

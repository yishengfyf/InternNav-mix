#!/usr/bin/env python3
"""Audit a Stage21 multi-head training dataset without importing torch."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from stage21_training_common import audit_online_row, encode_row, feature_names, iter_jsonl, task_target


TASK_FILES = {
    "progress": "progress_rows",
    "safety": "safety_rows",
    "recovery": "recovery_proxy_rows",
}


def audit(data_dir: Path) -> Dict[str, Any]:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("task") != "stage21_candidate_recoverability_dataset":
        raise ValueError("Expected a Stage21 candidate-recoverability dataset")
    result: Dict[str, Any] = {
        "task": "stage21_multitask_training_dataset_audit",
        "data_dir": str(data_dir.resolve()),
        "feature_schema_version": "stage21_structured_online_v1",
        "feature_dim": len(feature_names()),
        "scene_overlap_count": summary.get("split_audit", {}).get("scene_overlap_count"),
        "dataset_gt_leakage_passed": summary.get("candidate_recoverability_rows", {}).get("gt_leakage_scan", {}).get("passed"),
        "active_safety_passed": summary.get("active_safety_check", {}).get("passed"),
        "tasks": {},
    }
    for task, stem in TASK_FILES.items():
        task_audit: Dict[str, Any] = {}
        for split in ("train", "val"):
            rows = list(iter_jsonl(data_dir / f"{stem}_{split}.jsonl"))
            leakage = []
            targets = []
            auxiliary = []
            preference = Counter()
            recovery_class = Counter()
            scene_ids = set()
            for row in rows:
                # identity.split records the source navigation split (train),
                # while this file split is the deterministic held-out-scene
                # model split.  Do not conflate them.
                scene_ids.add(str((row.get("identity") or {}).get("scene_id")))
                leakage.extend(audit_online_row(row))
                encode_row(row)
                target, aux = task_target(row, task)
                targets.append(target)
                auxiliary.append(aux)
                preference[str((row.get("offline_labels") or {}).get("preference_vs_s2") or "none")] += 1
                recovery_class[str((row.get("offline_labels") or {}).get("recovery_proxy_class") or "none")] += 1
            task_audit[split] = {
                "row_count": len(rows),
                "scene_count": len(scene_ids),
                "leakage_hit_count": len(leakage),
                "sample_leakage_hits": leakage[:10],
                "target_mean": sum(targets) / max(1, len(targets)),
                "target_min": min(targets) if targets else None,
                "target_max": max(targets) if targets else None,
                "auxiliary_positive_count": int(sum(value >= 0.5 for value in auxiliary)),
                "preference_counts": dict(preference),
                "recovery_class_counts": dict(recovery_class),
            }
        result["tasks"][task] = task_audit
        train_scenes = {
            str((row.get("identity") or {}).get("scene_id"))
            for row in iter_jsonl(data_dir / f"{stem}_train.jsonl")
        }
        val_scenes = {
            str((row.get("identity") or {}).get("scene_id"))
            for row in iter_jsonl(data_dir / f"{stem}_val.jsonl")
        }
        task_audit["scene_overlap_count"] = len(train_scenes & val_scenes)
    result["passed"] = bool(
        result["scene_overlap_count"] == 0
        and result["dataset_gt_leakage_passed"] is True
        and result["active_safety_passed"] is True
        and all(
            split["leakage_hit_count"] == 0
            for task in result["tasks"].values() for split in task.values()
            if isinstance(split, dict) and "leakage_hit_count" in split
        )
        and all(task["scene_overlap_count"] == 0 for task in result["tasks"].values())
        and result["tasks"]["progress"]["train"]["preference_counts"].get("positive", 0) > 0
        and result["tasks"]["progress"]["val"]["preference_counts"].get("positive", 0) > 0
        and result["tasks"]["recovery"]["train"]["row_count"] > 0
        and result["tasks"]["recovery"]["val"]["row_count"] > 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

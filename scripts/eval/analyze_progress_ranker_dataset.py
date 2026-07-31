"""Analyze Stage17 ranker datasets against simple non-learned heuristics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield item


def _percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _ranking_metrics(rows: List[Dict[str, Any]], score_fn: Callable[[List[float], int, Dict[str, Any]], float]) -> Dict[str, float]:
    total = top1 = 0
    mrr = ndcg = 0.0
    for row in rows:
        labels = [float(value) for value in row["labels"]]
        scores = [float(score_fn(feature, idx, row)) for idx, feature in enumerate(row["features"])]
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        total += 1
        top1 += int(labels[order[0]] > 0.0)
        for rank, idx in enumerate(order, start=1):
            if labels[idx] > 0.0:
                mrr += 1.0 / rank
                break
        dcg = sum(labels[idx] / math.log2(rank + 2) for rank, idx in enumerate(order))
        ideal = sorted(labels, reverse=True)
        idcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(ideal))
        ndcg += dcg / idcg if idcg > 0.0 else 0.0
    denom = float(max(1, total))
    return {
        "top1_accuracy": top1 / denom,
        "mrr": mrr / denom,
        "ndcg": ndcg / denom,
        "examples": float(total),
    }


def _label_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [float(value) for row in rows for value in row["labels"]]
    positives = [value for value in labels if value > 0.0]
    raw_values = [float(value) for row in rows for value in row.get("raw_values", [])]

    def summarize(values: List[float]) -> Dict[str, float | None]:
        return {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "median": _percentile(values, 0.5),
            "p90": _percentile(values, 0.9),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "rows": len(rows),
        "candidates": len(labels),
        "positive_candidates": len(positives),
        "positive_per_row": len(positives) / max(1, len(rows)),
        "positive_count_per_row": dict(Counter(sum(1 for value in row["labels"] if value > 0.0) for row in rows)),
        "labels_all": summarize(labels),
        "labels_positive": summarize(positives),
        "raw_values": summarize(raw_values),
    }


def _feature_distribution(rows: List[Dict[str, Any]], feature_names: List[str]) -> Dict[str, Any]:
    idx = {name: index for index, name in enumerate(feature_names)}
    counters = {
        "all_completed": 0,
        "positive_completed": 0,
        "all_type": Counter(),
        "positive_type": Counter(),
        "all_direction": Counter(),
        "positive_direction": Counter(),
    }
    type_names = ("frontier", "semantic_frontier", "semantic_keyframe", "open_floor")
    direction_names = ("front", "left", "right", "back")
    for row in rows:
        for feature, label in zip(row["features"], row["labels"]):
            is_positive = float(label) > 0.0
            if feature[idx["is_completed_landmark"]] > 0.5:
                counters["all_completed"] += 1
                if is_positive:
                    counters["positive_completed"] += 1
            candidate_type = "unknown"
            for name in type_names:
                if feature[idx[f"candidate_type={name}"]] > 0.5:
                    candidate_type = name
                    break
            counters["all_type"][candidate_type] += 1
            if is_positive:
                counters["positive_type"][candidate_type] += 1
            direction = "unknown"
            for name in direction_names:
                if feature[idx[f"direction_bucket={name}"]] > 0.5:
                    direction = name
                    break
            counters["all_direction"][direction] += 1
            if is_positive:
                counters["positive_direction"][direction] += 1

    total_candidates = sum(counters["all_type"].values())
    positive_candidates = sum(counters["positive_type"].values())
    return {
        "all_completed_rate": counters["all_completed"] / max(1, total_candidates),
        "positive_completed_rate": counters["positive_completed"] / max(1, positive_candidates),
        "all_type": dict(counters["all_type"]),
        "positive_type": dict(counters["positive_type"]),
        "all_direction": dict(counters["all_direction"]),
        "positive_direction": dict(counters["positive_direction"]),
    }


def analyze(data_dir: Path) -> Dict[str, Any]:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    feature_names = summary["feature_names"]
    idx = {name: index for index, name in enumerate(feature_names)}
    rows_by_split = {
        split: list(_read_jsonl(data_dir / f"{split}.jsonl"))
        for split in ("train", "val")
    }
    rows_by_split["all"] = rows_by_split["train"] + rows_by_split["val"]

    heuristics = {
        "candidate_score": lambda feature, _idx, _row: feature[idx["score"]],
        "intent_alignment": lambda feature, _idx, _row: feature[idx["intent_alignment_score"]],
        "neg_angle_to_waypoint": lambda feature, _idx, _row: -feature[idx["angle_to_current_waypoint_deg"]],
        "neg_distance_to_waypoint": lambda feature, _idx, _row: -feature[idx["distance_to_current_waypoint_m"]],
        "target_frontier_score": lambda feature, _idx, _row: feature[idx["target_frontier_score"]],
        "semantic_progress_score": lambda feature, _idx, _row: feature[idx["semantic_progress_score"]],
        "not_completed": lambda feature, _idx, _row: -feature[idx["is_completed_landmark"]],
        "front_bucket": lambda feature, _idx, _row: feature[idx["direction_bucket=front"]],
        "label_oracle": lambda _feature, cand_idx, row: row["labels"][cand_idx],
    }
    result = {
        "dataset_summary": summary,
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        result["splits"][split] = {
            "label_stats": _label_stats(rows),
            "feature_distribution": _feature_distribution(rows, feature_names),
            "heuristic_metrics": {
                name: _ranking_metrics(rows, fn)
                for name, fn in heuristics.items()
            },
        }
    episode_splits = defaultdict(set)
    scene_splits = defaultdict(set)
    for split, rows in rows_by_split.items():
        if split == "all":
            continue
        for row in rows:
            episode_splits[(row.get("scene_id"), row.get("episode_id"))].add(split)
            scene_splits[row.get("scene_id")].add(split)
    result["split_overlap"] = {
        "episodes_total": len(episode_splits),
        "episodes_in_both_train_val": sum(len(value) > 1 for value in episode_splits.values()),
        "scenes_total": len(scene_splits),
        "scenes_in_both_train_val": sum(len(value) > 1 for value in scene_splits.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(args.data_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

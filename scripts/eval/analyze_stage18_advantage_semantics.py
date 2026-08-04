"""Analyze Stage18 advantage labels through semantic/progress features.

This is an offline diagnostic tool.  It does not run Habitat and it does not
train a model.  It reads a Stage18c ``candidate_advantage_not_active_gate``
dataset and answers:

* Are positive candidates actually enriched with semantic/progress evidence?
* Which online features separate positive-vs-negative candidates?
* Do simple heuristics beat learned rankers because semantic features are weak?
* Are failures concentrated in events whose candidates have no semantic signal?

The script intentionally analyzes only online features stored in the dataset.
GT fields in rows are used for grouping/diagnostics only when already present
in the offline dataset rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "semantic": (
        "candidate::semantic_relevance_score",
        "candidate::semantic_novelty_score",
        "candidate::semantic_confidence_score",
        "candidate::semantic_bind_score",
        "candidate::next_landmark_relevance",
        "candidate::completed_landmark_penalty",
        "candidate::repeated_semantic_penalty",
        "candidate::semantic_progress_score",
        "candidate::semanticized_candidate",
        "candidate::instruction_relevant",
        "candidate::candidate_type=semantic_frontier",
        "candidate::candidate_type=semantic_keyframe",
    ),
    "target_frontier": (
        "candidate::target_frontier_score",
        "candidate::target_frontier_doorway_like_score",
        "candidate::target_frontier_corridor_continuation_score",
        "candidate::target_frontier_intent_deviation_penalty",
        "candidate::target_frontier_candidate",
        "candidate::target_frontier_escape_candidate",
        "candidate::target_frontier_intent_safe",
        "candidate::unknown_target_frontier_bonus",
    ),
    "geometry": (
        "candidate::distance_m",
        "candidate::frontier_distance_m",
        "candidate::frontier_progress_score",
        "candidate::topology_novelty_score",
        "candidate::nearby_visit_count",
        "candidate::revisit_risk",
        "candidate::geometry_safe",
        "candidate::active_gate_safe",
        "candidate::points_to_revisited_region",
    ),
    "s2_relation": (
        "candidate::angle_to_current_waypoint_deg",
        "candidate::intent_alignment_score",
        "candidate::distance_to_current_waypoint_m",
        "candidate::aligned_with_current_waypoint",
        "candidate_minus_current_angle_norm",
        "candidate_minus_current_distance_norm",
    ),
    "context": (
        "context::current_semantic_dead_zone",
        "context::current_semantic_dead_zone_score",
        "context::current_stagnation_active",
        "context::current_revisited",
        "context::candidate_count_norm",
        "context::safe_candidate_count_norm",
    ),
}

DEFAULT_HEURISTIC_FEATURES = (
    "candidate::score",
    "candidate::target_frontier_score",
    "candidate::goal_progress_score",
    "candidate::frontier_progress_score",
    "candidate::semantic_progress_score",
    "candidate::next_landmark_relevance",
    "candidate::semantic_relevance_score",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield row


def _load_split(path: Path, split: str) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        item = dict(row)
        item["_split"] = split
        rows.append(item)
    return rows


def _feature_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {name: index for index, name in enumerate(feature_names)}


def _feature_value(row: Mapping[str, Any], index: Mapping[str, int], name: str) -> float:
    feature_index = index.get(name)
    features = row.get("features") or []
    if feature_index is None or feature_index >= len(features):
        return 0.0
    return _safe_float(features[feature_index])


def _available_names(feature_names: Sequence[str], names: Sequence[str]) -> List[str]:
    available = set(feature_names)
    return [name for name in names if name in available]


def _group_by_event(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("event_key"))].append(row)
    return grouped


def _label(row: Mapping[str, Any]) -> int:
    return int(row.get("label", 0))


def _success_key(row: Mapping[str, Any]) -> str:
    value = row.get("final_success")
    if value is None:
        return "success_unknown"
    return f"success_{bool(value)}"


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _rate(values: Sequence[float], *, threshold: float = 0.0) -> Optional[float]:
    if not values:
        return None
    return float(sum(value > threshold for value in values) / len(values))


def _summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "nonzero_rate": None,
            "positive_rate": None,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "nonzero_rate": float(sum(abs(value) > 1e-9 for value in values) / len(values)),
        "positive_rate": float(sum(value > 0.0 for value in values) / len(values)),
    }


def _feature_label_stats(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    selected_names: Sequence[str],
) -> Dict[str, Any]:
    index = _feature_index(feature_names)
    result: Dict[str, Any] = {}
    for name in _available_names(feature_names, selected_names):
        positive = [_feature_value(row, index, name) for row in rows if _label(row) == 1]
        negative = [_feature_value(row, index, name) for row in rows if _label(row) == 0]
        positive_mean = _mean(positive)
        negative_mean = _mean(negative)
        result[name] = {
            "positive": _summarize_values(positive),
            "negative": _summarize_values(negative),
            "mean_delta_pos_minus_neg": (
                None
                if positive_mean is None or negative_mean is None
                else float(positive_mean - negative_mean)
            ),
            "positive_nonzero_rate_delta": (
                None
                if _rate(positive) is None or _rate(negative) is None
                else float(_rate(positive) - _rate(negative))
            ),
        }
    return result


def _semantic_signal(row: Mapping[str, Any], feature_names: Sequence[str]) -> bool:
    index = _feature_index(feature_names)
    signal_names = (
        "candidate::semantic_relevance_score",
        "candidate::next_landmark_relevance",
        "candidate::semantic_progress_score",
        "candidate::instruction_relevant",
        "candidate::semanticized_candidate",
        "candidate::candidate_type=semantic_frontier",
        "candidate::candidate_type=semantic_keyframe",
    )
    return any(_feature_value(row, index, name) > 0.0 for name in signal_names)


def _coverage_by_group(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {
        "all": list(rows),
        "positive": [row for row in rows if _label(row) == 1],
        "negative": [row for row in rows if _label(row) == 0],
    }
    for split in sorted({str(row.get("_split")) for row in rows}):
        groups[f"split={split}"] = [row for row in rows if str(row.get("_split")) == split]
    for key in sorted({_success_key(row) for row in rows}):
        groups[key] = [row for row in rows if _success_key(row) == key]

    result = {}
    for name, items in groups.items():
        if not items:
            continue
        positives = [row for row in items if _label(row) == 1]
        negatives = [row for row in items if _label(row) == 0]
        result[name] = {
            "rows": len(items),
            "positive_rows": len(positives),
            "negative_rows": len(negatives),
            "semantic_signal_rate_all": float(
                sum(_semantic_signal(row, feature_names) for row in items) / max(1, len(items))
            ),
            "semantic_signal_rate_positive": float(
                sum(_semantic_signal(row, feature_names) for row in positives) / max(1, len(positives))
            ),
            "semantic_signal_rate_negative": float(
                sum(_semantic_signal(row, feature_names) for row in negatives) / max(1, len(negatives))
            ),
        }
    return result


def _ordered_by_score(
    items: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    score_name: str,
) -> List[Mapping[str, Any]]:
    index = _feature_index(feature_names)
    return sorted(
        items,
        key=lambda row: _feature_value(row, index, score_name),
        reverse=True,
    )


def _ranking_metrics_for_score(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    score_name: str,
) -> Dict[str, Any]:
    grouped = _group_by_event(rows)
    positive_events = []
    multi_candidate_positive_events = []
    top1 = mrr = multi_top1 = multi_mrr = 0.0
    selected_positive_semantic = selected_negative_semantic = 0
    miss_cases = []

    for event_key, items in grouped.items():
        if not any(_label(row) == 1 for row in items):
            continue
        ordered = _ordered_by_score(items, feature_names, score_name)
        positive_events.append(items)
        selected = ordered[0]
        if _label(selected) == 1:
            top1 += 1.0
            if _semantic_signal(selected, feature_names):
                selected_positive_semantic += 1
        else:
            if _semantic_signal(selected, feature_names):
                selected_negative_semantic += 1
        for index, row in enumerate(ordered, start=1):
            if _label(row) == 1:
                mrr += 1.0 / float(index)
                break
        if len(items) > 1:
            multi_candidate_positive_events.append(items)
            if _label(selected) == 1:
                multi_top1 += 1.0
            for index, row in enumerate(ordered, start=1):
                if _label(row) == 1:
                    multi_mrr += 1.0 / float(index)
                    break
            if _label(selected) == 0:
                positive_rows = [row for row in items if _label(row) == 1]
                miss_cases.append(
                    {
                        "event_key": event_key,
                        "split": selected.get("_split"),
                        "final_success": selected.get("final_success"),
                        "selected_candidate_id": selected.get("candidate_id"),
                        "selected_has_semantic_signal": _semantic_signal(selected, feature_names),
                        "positive_candidate_count": len(positive_rows),
                        "positive_semantic_signal_count": sum(
                            _semantic_signal(row, feature_names) for row in positive_rows
                        ),
                    }
                )

    return {
        "score_name": score_name,
        "positive_event_count": len(positive_events),
        "multi_candidate_positive_event_count": len(multi_candidate_positive_events),
        "event_top1_positive": top1 / max(1, len(positive_events)),
        "event_mrr": mrr / max(1, len(positive_events)),
        "multi_candidate_event_top1_positive": multi_top1
        / max(1, len(multi_candidate_positive_events)),
        "multi_candidate_event_mrr": multi_mrr / max(1, len(multi_candidate_positive_events)),
        "selected_positive_semantic_rate": selected_positive_semantic / max(1, int(top1)),
        "selected_negative_semantic_rate": selected_negative_semantic
        / max(1, len(positive_events) - int(top1)),
        "miss_case_count": len(miss_cases),
        "miss_case_examples": miss_cases[:20],
    }


def _pairwise_feature_separability(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    selected_names: Sequence[str],
) -> Dict[str, Any]:
    index = _feature_index(feature_names)
    grouped = _group_by_event(rows)
    result: Dict[str, Any] = {}
    for name in _available_names(feature_names, selected_names):
        comparisons = greater = equal = lower = 0
        deltas = []
        for items in grouped.values():
            positives = [row for row in items if _label(row) == 1]
            negatives = [row for row in items if _label(row) == 0]
            for positive in positives:
                pos_value = _feature_value(positive, index, name)
                for negative in negatives:
                    neg_value = _feature_value(negative, index, name)
                    delta = pos_value - neg_value
                    deltas.append(delta)
                    comparisons += 1
                    if delta > 1e-9:
                        greater += 1
                    elif delta < -1e-9:
                        lower += 1
                    else:
                        equal += 1
        result[name] = {
            "pair_count": comparisons,
            "positive_greater_rate": greater / max(1, comparisons),
            "equal_rate": equal / max(1, comparisons),
            "positive_lower_rate": lower / max(1, comparisons),
            "mean_delta": _mean(deltas),
        }
    return result


def _event_semantic_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    grouped = _group_by_event(rows)
    counters = Counter()
    split_counters: Dict[str, Counter] = defaultdict(Counter)
    examples = []
    for event_key, items in grouped.items():
        positives = [row for row in items if _label(row) == 1]
        if not positives:
            continue
        split = str(items[0].get("_split"))
        is_multi = len(items) > 1
        counters["positive_events"] += 1
        counters["multi_candidate_positive_events"] += int(is_multi)
        split_counters[split]["positive_events"] += 1
        split_counters[split]["multi_candidate_positive_events"] += int(is_multi)
        positive_semantic_count = sum(_semantic_signal(row, feature_names) for row in positives)
        if positive_semantic_count <= 0:
            counters["positive_events_without_positive_semantic_signal"] += 1
            split_counters[split]["positive_events_without_positive_semantic_signal"] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "event_key": event_key,
                        "split": split,
                        "final_success": items[0].get("final_success"),
                        "candidate_count": len(items),
                        "positive_candidate_count": len(positives),
                    }
                )
        else:
            counters["positive_events_with_positive_semantic_signal"] += 1
            split_counters[split]["positive_events_with_positive_semantic_signal"] += 1

    return {
        "counts": dict(counters),
        "by_split": {key: dict(value) for key, value in sorted(split_counters.items())},
        "positive_event_without_semantic_rate": (
            counters["positive_events_without_positive_semantic_signal"]
            / max(1, counters["positive_events"])
        ),
        "examples_without_positive_semantic_signal": examples,
    }


def _overview(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = _group_by_event(rows)
    positive_events = [
        items for items in grouped.values() if any(_label(row) == 1 for row in items)
    ]
    multi_positive_events = [items for items in positive_events if len(items) > 1]
    labels = Counter(_label(row) for row in rows)
    return {
        "rows": len(rows),
        "negative_rows": int(labels.get(0, 0)),
        "positive_rows": int(labels.get(1, 0)),
        "events": len(grouped),
        "positive_events": len(positive_events),
        "multi_candidate_positive_events": len(multi_positive_events),
        "mean_candidates_per_event": (
            sum(len(items) for items in grouped.values()) / max(1, len(grouped))
        ),
    }


def analyze(
    *,
    data_dir: Path,
    extra_features: Sequence[str],
    heuristic_features: Sequence[str],
    top_feature_count: int,
) -> Dict[str, Any]:
    summary_path = data_dir / "summary.json"
    if not summary_path.exists() and (data_dir / "dataset_summary.json").exists():
        summary_path = data_dir / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("task") != "candidate_advantage_not_active_gate":
        raise ValueError("Expected a Stage18c candidate_advantage_not_active_gate dataset.")
    feature_names = list(dataset_summary.get("feature_names") or [])
    if not feature_names:
        raise ValueError("Dataset summary does not contain feature_names.")
    missing_splits = [
        str(data_dir / f"{split}.jsonl")
        for split in ("train", "val")
        if not (data_dir / f"{split}.jsonl").exists()
    ]
    if missing_splits:
        raise FileNotFoundError(
            "Stage18 semantic diagnostics require train.jsonl and val.jsonl. "
            f"Missing: {missing_splits}"
        )

    rows_by_split = {
        "train": _load_split(data_dir / "train.jsonl", "train"),
        "val": _load_split(data_dir / "val.jsonl", "val"),
    }
    all_rows = rows_by_split["train"] + rows_by_split["val"]

    selected_features = []
    for names in DEFAULT_FEATURE_GROUPS.values():
        selected_features.extend(names)
    selected_features.extend(extra_features)
    selected_features = _available_names(feature_names, list(dict.fromkeys(selected_features)))
    heuristic_names = _available_names(
        feature_names,
        list(dict.fromkeys([*DEFAULT_HEURISTIC_FEATURES, *heuristic_features])),
    )

    split_analysis = {}
    for split, rows in rows_by_split.items():
        split_analysis[split] = {
            "overview": _overview(rows),
            "semantic_coverage": _coverage_by_group(rows, feature_names),
            "feature_label_stats": _feature_label_stats(rows, feature_names, selected_features),
            "heuristic_rankings": {
                name: _ranking_metrics_for_score(rows, feature_names, name)
                for name in heuristic_names
            },
            "pairwise_feature_separability": _pairwise_feature_separability(
                rows,
                feature_names,
                selected_features,
            ),
            "event_semantic_diagnostics": _event_semantic_diagnostics(rows, feature_names),
        }

    all_feature_stats = _feature_label_stats(all_rows, feature_names, selected_features)
    sortable = [
        (
            name,
            abs(_safe_float(stats.get("mean_delta_pos_minus_neg"))),
            stats,
        )
        for name, stats in all_feature_stats.items()
        if stats.get("mean_delta_pos_minus_neg") is not None
    ]
    top_features = [
        {"feature": name, **stats}
        for name, _score, stats in sorted(sortable, key=lambda item: item[1], reverse=True)[
            : int(top_feature_count)
        ]
    ]

    return {
        "data_dir": str(data_dir),
        "dataset": {
            "task": dataset_summary.get("task"),
            "feature_dim": dataset_summary.get("feature_dim"),
            "split_key": dataset_summary.get("split_key"),
            "class_counts": dataset_summary.get("class_counts"),
            "event_diagnostics": dataset_summary.get("event_diagnostics"),
            "success_split_counts": dataset_summary.get("success_split_counts"),
        },
        "feature_groups": {
            group: _available_names(feature_names, names)
            for group, names in DEFAULT_FEATURE_GROUPS.items()
        },
        "heuristic_features": heuristic_names,
        "overview": {
            "train": split_analysis["train"]["overview"],
            "val": split_analysis["val"]["overview"],
            "all": _overview(all_rows),
        },
        "all_semantic_coverage": _coverage_by_group(all_rows, feature_names),
        "all_feature_label_stats": all_feature_stats,
        "top_abs_mean_delta_features": top_features,
        "splits": split_analysis,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help="Additional feature name to include in label/separability stats. Repeat as needed.",
    )
    parser.add_argument(
        "--heuristic-feature",
        action="append",
        default=[],
        help="Additional feature name to use as a one-feature ranking heuristic.",
    )
    parser.add_argument("--top-feature-count", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_feature_count <= 0:
        raise ValueError("--top-feature-count must be positive")
    result = analyze(
        data_dir=args.data_dir,
        extra_features=args.extra_feature,
        heuristic_features=args.heuristic_feature,
        top_feature_count=args.top_feature_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

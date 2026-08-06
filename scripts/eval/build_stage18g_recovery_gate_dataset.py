"""Build a Stage18g recovery-gate dataset from Stage18f rows.

The goal is to turn the offline recovery advantage audit into a small, stable
gate dataset for a lightweight classifier or rule audit.

This script does not touch Habitat. It reads Stage18f ``rows.jsonl`` and emits:

* ``train.jsonl`` / ``val.jsonl`` split by episode id
* ``summary.json`` with label balance and simple rule-audit metrics

The default label is ``strong_recovery_proxy`` because it is the clearest
positive signal we have so far: geometry-safe, utility-positive, and tied to a
real recovery context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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
            if isinstance(row, dict):
                yield row


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _hash_fraction(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _split_rows(rows: Sequence[Dict[str, Any]], val_fraction: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(_episode_key(row), []).append(row)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for episode_key in sorted(by_episode):
        bucket = val_fraction if val_fraction > 0.0 else 0.0
        if _hash_fraction(episode_key) < bucket:
            val.extend(by_episode[episode_key])
        else:
            train.extend(by_episode[episode_key])
    return train, val


def _bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / float(len(rows))


def _row_to_record(
    row: Mapping[str, Any],
    *,
    label_key: str,
    utility_threshold: float,
    margin_threshold: float,
    open_threshold: float,
    min_backtrack_m: float,
    max_backtrack_m: float,
    max_step_gap: int,
) -> Dict[str, Any]:
    current = row.get("current_policy_candidate") or {}
    backtrack = row.get("best_backtrack_candidate") or {}
    trigger_reasons = list(row.get("trigger_reasons") or [])
    context_tags = list(row.get("recovery_context_tags") or backtrack.get("semantic_resilience_recovery_context_tags") or [])
    label = bool(row.get(label_key))
    geometry_safe = bool(backtrack.get("geometry_safe"))
    utility = _safe_float(row.get("backtrack_utility_proxy"), 0.0)
    open_score = _safe_float(backtrack.get("semantic_resilience_open_score"), 0.0)
    backtrack_distance = _safe_float(
        backtrack.get("semantic_resilience_backtrack_distance_m"),
        0.0,
    )
    step_gap = _safe_int(backtrack.get("semantic_resilience_step_gap"), 999999)
    margin = row.get("advantage_margin_proxy")
    if margin is None:
        margin = utility - _safe_float(row.get("current_risk_proxy"), 0.0)
    margin = _safe_float(margin, 0.0)

    rule_recovery = bool(
        row.get("trigger")
        and row.get("backtrack_present")
        and geometry_safe
        and utility >= utility_threshold
        and margin >= margin_threshold
        and open_score >= open_threshold
        and min_backtrack_m <= backtrack_distance <= max_backtrack_m
        and step_gap <= max_step_gap
        and (
            bool(row.get("current_problem"))
            or "semantic_dead_zone" in trigger_reasons
            or "semantic_stagnation" in trigger_reasons
            or "current_waypoint_occupied" in trigger_reasons
            or "current_points_to_revisited_region" in trigger_reasons
        )
    )
    rule_keep = bool(
        not row.get("trigger")
        and bool(current.get("geometry_safe", True))
        and not bool(row.get("current_problem"))
    )
    if rule_recovery:
        rule = "recovery"
    elif rule_keep:
        rule = "keep"
    else:
        rule = "abstain"

    features = {
        "trigger": bool(row.get("trigger")),
        "current_problem": bool(row.get("current_problem")),
        "current_risk_proxy": _safe_float(row.get("current_risk_proxy"), 0.0),
        "backtrack_utility_proxy": utility,
        "advantage_margin_proxy": margin,
        "query_count": _safe_int(row.get("query_count")),
        "candidate_count": _safe_int(row.get("candidate_count")),
        "geometry_safe_candidate_count": _safe_int(row.get("geometry_safe_candidate_count")),
        "active_gate_safe_candidate_count": _safe_int(row.get("active_gate_safe_candidate_count")),
        "recovery_candidate_count": _safe_int(row.get("recovery_candidate_count")),
        "recovery_recommended_count": _safe_int(row.get("recovery_recommended_count")),
        "current_goal_state": current.get("goal_state"),
        "current_direction_bucket": current.get("direction_bucket"),
        "current_grid": current.get("grid"),
        "current_xy": current.get("xy"),
        "current_frontier_distance_m": current.get("frontier_distance_m"),
        "backtrack_direction_bucket": backtrack.get("direction_bucket"),
        "backtrack_source": backtrack.get("semantic_resilience_source"),
        "backtrack_step_gap": backtrack.get("semantic_resilience_step_gap"),
        "backtrack_distance_m": backtrack.get("semantic_resilience_backtrack_distance_m"),
        "backtrack_open_score": backtrack.get("semantic_resilience_open_score"),
        "backtrack_semantic_score": backtrack.get("semantic_resilience_score"),
        "backtrack_passage_term_count": backtrack.get("semantic_resilience_passage_term_count"),
        "backtrack_obstacle_term_count": backtrack.get("semantic_resilience_obstacle_term_count"),
        "backtrack_recovery_context_tags": context_tags,
        "trigger_reasons": trigger_reasons,
        "instruction_terms": list(row.get("instruction_terms") or []),
        "recovery_ready": bool(row.get("recovery_ready")),
        "strong_recovery_proxy": label,
        "rule_decision": rule,
    }
    return {
        "episode_key": _episode_key(row),
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "step_id": row.get("step_id"),
        "event_key": row.get("event_key"),
        "success": bool(row.get("success")) if row.get("success") is not None else None,
        "spl": row.get("spl"),
        "ne": row.get("ne"),
        "label": label,
        "features": features,
        "current_policy_candidate": current,
        "best_backtrack_candidate": backtrack,
    }


def build_dataset(
    rows_path: Path,
    *,
    label_key: str,
    val_fraction: float,
    utility_threshold: float,
    margin_threshold: float,
    open_threshold: float,
    min_backtrack_m: float,
    max_backtrack_m: float,
    max_step_gap: int,
) -> Dict[str, Any]:
    rows = list(_read_jsonl(rows_path))
    records = [
        _row_to_record(
            row,
            label_key=label_key,
            utility_threshold=utility_threshold,
            margin_threshold=margin_threshold,
            open_threshold=open_threshold,
            min_backtrack_m=min_backtrack_m,
            max_backtrack_m=max_backtrack_m,
            max_step_gap=max_step_gap,
        )
        for row in rows
    ]
    train_records, val_records = _split_rows(records, val_fraction)

    def _rule_precision_recall(subset: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        tp = sum(
            1
            for row in subset
            if row.get("features", {}).get("rule_decision") == "recovery"
            and bool(row.get("label"))
        )
        fp = sum(
            1
            for row in subset
            if row.get("features", {}).get("rule_decision") == "recovery"
            and not bool(row.get("label"))
        )
        fn = sum(
            1
            for row in subset
            if row.get("features", {}).get("rule_decision") != "recovery"
            and bool(row.get("label"))
        )
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    def _summary(name: str, subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        positive = [row for row in subset if bool(row.get("label"))]
        rule_recovery = [row for row in subset if row.get("features", {}).get("rule_decision") == "recovery"]
        return {
            "name": name,
            "records": len(subset),
            "episodes": len({row.get("episode_key") for row in subset}),
            "positives": len(positive),
            "positive_rate": len(positive) / float(len(subset)) if subset else 0.0,
            "rule_recovery": len(rule_recovery),
            "rule_recovery_rate": len(rule_recovery) / float(len(subset)) if subset else 0.0,
            "success_rate": _bool_rate(subset, "success"),
            "mean_margin": mean(
                [float(row.get("features", {}).get("advantage_margin_proxy", 0.0)) for row in subset]
            )
            if subset
            else 0.0,
            "mean_utility": mean(
                [float(row.get("features", {}).get("backtrack_utility_proxy", 0.0)) for row in subset]
            )
            if subset
            else 0.0,
            "precision_recall_rule_recovery": _rule_precision_recall(subset),
        }

    def _flatten_rule_keys(record: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(record)
        features = dict(out.pop("features", {}))
        out.update(features)
        return out

    result = {
        "source": str(rows_path),
        "label_key": label_key,
        "val_fraction": val_fraction,
        "utility_threshold": utility_threshold,
        "margin_threshold": margin_threshold,
        "open_threshold": open_threshold,
        "min_backtrack_m": min_backtrack_m,
        "max_backtrack_m": max_backtrack_m,
        "max_step_gap": max_step_gap,
        "records": len(records),
        "episodes": len({row.get("episode_key") for row in records}),
        "positive_rate": len([row for row in records if bool(row.get("label"))]) / float(len(records)) if records else 0.0,
        "train": _summary("train", train_records),
        "val": _summary("val", val_records),
        "trigger_rate": (
            sum(1 for row in records if bool(row.get("features", {}).get("trigger"))) / float(len(records))
            if records
            else 0.0
        ),
        "current_problem_rate": (
            sum(1 for row in records if bool(row.get("features", {}).get("current_problem"))) / float(len(records))
            if records
            else 0.0
        ),
    }
    return {
        "summary": result,
        "train_records": [_flatten_rule_keys(row) for row in train_records],
        "val_records": [_flatten_rule_keys(row) for row in val_records],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-key", default="strong_recovery_proxy")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--utility-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--open-threshold", type=float, default=0.65)
    parser.add_argument("--min-backtrack-m", type=float, default=1.0)
    parser.add_argument("--max-backtrack-m", type=float, default=3.5)
    parser.add_argument("--max-step-gap", type=int, default=45)
    args = parser.parse_args()

    built = build_dataset(
        args.rows,
        label_key=args.label_key,
        val_fraction=float(args.val_fraction),
        utility_threshold=float(args.utility_threshold),
        margin_threshold=float(args.margin_threshold),
        open_threshold=float(args.open_threshold),
        min_backtrack_m=float(args.min_backtrack_m),
        max_backtrack_m=float(args.max_backtrack_m),
        max_step_gap=int(args.max_step_gap),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train.jsonl", built["train_records"])
    _write_jsonl(args.output_dir / "val.jsonl", built["val_records"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(built["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(built["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {args.output_dir / 'val.jsonl'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

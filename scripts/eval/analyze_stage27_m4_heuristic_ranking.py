#!/usr/bin/env python3
"""Compare explainable M4 ranking rules on a frozen Stage27 candidate set.

This is an offline ordering audit only.  It never executes a candidate, uses
success/future fields as labels, or adds candidates that failed the M3 safety
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


STAGE = "route_occ_clearance_frontier"
RULES = ("near_first", "open_first", "route_support_first", "composite")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _key(row: Mapping[str, Any]) -> Tuple[str, int, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1)), int(row.get("step_id", -1))


def _load_events(root: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    events: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for path in sorted(root.glob("**/stage27_m3_candidate_events.jsonl")):
        for row in _jsonl(path):
            events[_key(row)] = row
    return events


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else list(payload.get("selected_detector_events", []))


def _num(candidate: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(candidate.get(name, default) or default)
    except (TypeError, ValueError):
        return default


def _rank_key(rule: str, candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    path = _num(candidate, "path_length_m", 10**6)
    openness = _num(candidate, "local_free_fraction", 0.0)
    free = _num(candidate, "free_fraction", 0.0)
    support = -_num(candidate, "route_support_edge_count", 0.0)
    source_step = -_num(candidate, "source_step", -10**6)
    if rule == "near_first":
        return path, support, -openness, source_step
    if rule == "open_first":
        return -openness, -free, support, path, source_step
    if rule == "route_support_first":
        return support, source_step, path, -openness
    return support, -openness, path, source_step


def _direction_bucket(event: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    trigger = event.get("trigger_grid") or [0, 0]
    grid = candidate.get("grid") or [0, 0]
    return int(round(math.degrees(math.atan2(
        float(grid[0]) - float(trigger[0]),
        float(grid[1]) - float(trigger[1]),
    )) / 45.0))


def _safe(candidate: Mapping[str, Any]) -> bool:
    return (
        bool(candidate.get("floor_aligned_known_free"))
        and _num(candidate, "unknown_fraction") == 0.0
        and _num(candidate, "occupied_fraction") == 0.0
        and bool(candidate.get("shadow_only"))
        and not bool(candidate.get("action_applied"))
        and not candidate.get("gt_fields_used")
    )


def _rule_report(rows: Sequence[Tuple[Mapping[str, Any], List[Mapping[str, Any]]]], rule: str) -> Dict[str, Any]:
    selected = [sorted(pool, key=lambda item: _rank_key(rule, item))[0] for _, pool in rows]
    return {
        "event_count": len(rows),
        "top1_candidate_family_counts": dict(sorted(Counter(
            str(item.get("source_type", "unknown")) for item in selected
        ).items())),
        "top1_path_length_mean_m": sum(_num(item, "path_length_m") for item in selected) / max(1, len(selected)),
        "top1_local_free_fraction_mean": sum(_num(item, "local_free_fraction") for item in selected) / max(1, len(selected)),
        "top1_route_support_edge_mean": sum(_num(item, "route_support_edge_count") for item in selected) / max(1, len(selected)),
        "top1_direction_counts": dict(sorted(Counter(
            _direction_bucket(event, item) for event, pool in rows
            for item in [sorted(pool, key=lambda candidate: _rank_key(rule, candidate))[0]]
        ).items())),
        "all_top1_safe": all(_safe(item) for item in selected),
        "selected_candidate_ids": [str(item.get("candidate_id")) for item in selected],
    }


def analyze(run_root: Path, manifest_path: Path) -> Dict[str, Any]:
    events = _load_events(run_root)
    manifest = _load_manifest(manifest_path)
    expected = [_key(row) for row in manifest]
    fixed_rows: List[Tuple[Mapping[str, Any], List[Mapping[str, Any]]]] = []
    missing = []
    unsafe = 0
    for key in expected:
        event = events.get(key)
        if event is None:
            missing.append(key)
            continue
        pool = list(event.get("ablation", {}).get(STAGE, {}).get("candidates") or [])
        if len(pool) < 2:
            continue
        unsafe += sum(not _safe(item) for item in pool)
        fixed_rows.append((event, pool))

    def grouped(field: str) -> Dict[str, Any]:
        values = sorted({str(row.get(field) or "unspecified") for row in manifest})
        result: Dict[str, Any] = {}
        for value in values:
            rows = [item for item in fixed_rows if str(item[0].get("audit_selection", {}).get(field) or "unspecified") == value]
            result[value] = {rule: _rule_report(rows, rule) for rule in RULES}
        return result

    top1 = {
        rule: _rule_report(fixed_rows, rule)
        for rule in RULES
    }
    top1_ids = {
        rule: top1[rule]["selected_candidate_ids"] for rule in RULES
    }
    disagreement = {
        f"{left}__vs__{right}": sum(a != b for a, b in zip(top1_ids[left], top1_ids[right]))
        for index, left in enumerate(RULES)
        for right in RULES[index + 1:]
    }
    canonical = "\n".join(
        json.dumps(events[key], sort_keys=True, ensure_ascii=False)
        for key in sorted(events)
    ).encode("utf-8")
    snapshot_sha = hashlib.sha256(canonical).hexdigest()
    return {
        "task": "stage27_m4_heuristic_ranking_fixed_candidate_audit",
        "candidate_stage": STAGE,
        "candidate_snapshot_sha256": snapshot_sha,
        "event_manifest_count": len(manifest),
        "observed_event_count": len(events),
        "missing_manifest_event_count": len(missing),
        "multi_candidate_event_count": len(fixed_rows),
        "single_or_zero_event_count": len(manifest) - len(fixed_rows),
        "unsafe_candidate_record_count": unsafe,
        "rules": list(RULES),
        "top1": top1,
        "top1_disagreement_event_counts": disagreement,
        "by_gt_split": grouped("gt_split"),
        "oracle_labels": "not_used",
        "success_used_as_label": False,
        "future_fields_used": False,
        "ranker_trained": False,
        "executor_called": False,
        "active_recovery_enabled": False,
        "semantic_tie_break": "not_available_as_candidate_level_feature_in_frozen_E1_snapshot",
        "integrity_passed": not missing and unsafe == 0 and all(
            top1[rule]["all_top1_safe"] for rule in RULES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Stage27 M3 candidate-generation shadow logs.

The report measures coverage and safety evidence only.  It does not score a
ranker, execute a candidate, or interpret episode success as event ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1)), int(row.get("step_id", -1))


def _load_events(root: Path) -> List[Dict[str, Any]]:
    events = []
    for path in sorted(root.glob("**/stage27_m3_candidate_events.jsonl")):
        events.extend(_jsonl(path))
    unique = {_event_key(row): row for row in events}
    return [unique[key] for key in sorted(unique)]


def _load_gt(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else list(payload.get("selected_detector_events", []))


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any], tolerance: int = 8) -> bool:
    if str(a.get("scene_id")) != str(b.get("scene_id")) or int(a.get("episode_id", -1)) != int(b.get("episode_id", -1)):
        return False
    return abs(int(a.get("step_id", 0)) - int(b.get("step_id", 0))) <= int(tolerance)


def _stage_report(events: Iterable[Mapping[str, Any]], stage: str) -> Dict[str, Any]:
    rows = [row.get("ablation", {}).get(stage, {}) for row in events]
    pools = [list(row.get("candidates") or []) for row in rows]
    candidates = [item for pool in pools for item in pool]
    family_counts = Counter(item.get("source_type", "unknown") for item in candidates)
    unknown = [float(item.get("unknown_fraction", 0.0) or 0.0) for item in candidates]
    conflict = sum(bool(item.get("route_occ_conflict")) for item in candidates)
    floor_safe = sum(bool(item.get("floor_aligned_known_free")) for item in candidates)
    return {
        "event_count": len(rows),
        "event_coverage": sum(bool(pool) for pool in pools) / max(1, len(rows)),
        "candidate_count": len(candidates),
        "candidate_per_event_mean": len(candidates) / max(1, len(rows)),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "route_occ_conflict_count": int(conflict),
        "floor_aligned_known_free_count": int(floor_safe),
        "unknown_fraction_mean": sum(unknown) / max(1, len(unknown)),
        "unknown_fraction_max": max(unknown) if unknown else 0.0,
        "action_applied_count": sum(bool(row.get("action_applied")) for row in events),
        "gt_fields_used_union": sorted({field for row in events for field in row.get("gt_fields_used", [])}),
    }


def analyze(root: Path, gt_path: Path | None = None) -> Dict[str, Any]:
    events = _load_events(root)
    gt = _load_gt(gt_path)
    reports = {stage: _stage_report(events, stage) for stage in ("route_only", "route_occ", "route_occ_clearance")}
    matched = {
        "all_gt_events": len(gt),
        "route_only": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_only", {}).get("event_has_candidate")),
        "route_occ": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_occ", {}).get("event_has_candidate")),
        "route_occ_clearance": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_occ_clearance", {}).get("event_has_candidate")),
        "label_status": "coverage_overlap_only_not_event_gt",
    }
    return {
        "task": "stage27_m3_candidate_generation_shadow_audit",
        "event_schema": "stage27_m3_candidate_generation_v1",
        "event_count": len(events),
        "reports": reports,
        "gt_overlap_diagnostic": matched,
        "active_recovery_enabled": False,
        "ranker_trained": False,
        "unknown_is_free": False,
        "success_is_event_gt": False,
        "integrity_passed": bool(events) and all(
            bool(row.get("shadow_only")) and not bool(row.get("action_applied")) and not row.get("gt_fields_used")
            for row in events
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root, args.gt_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

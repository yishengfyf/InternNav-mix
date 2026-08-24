#!/usr/bin/env python3
"""Audit Stage27 M3 candidate-generation shadow logs.

The report measures coverage and safety evidence only.  It does not score a
ranker, execute a candidate, or interpret episode success as event ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
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
    events = list(events)
    rows = [row.get("ablation", {}).get(stage, {}) for row in events]
    pools = [list(row.get("candidates") or []) for row in rows]
    candidates = [item for pool in pools for item in pool]
    family_counts = Counter(item.get("source_type", "unknown") for item in candidates)
    unknown = [float(item.get("unknown_fraction", 0.0) or 0.0) for item in candidates]
    conflict = sum(bool(item.get("route_occ_conflict")) for item in candidates)
    floor_safe = sum(bool(item.get("floor_aligned_known_free")) for item in candidates)
    floor_sources = Counter(item.get("floor_z_source", "unspecified") for item in candidates)
    local_free = [float(item.get("local_free_fraction", 0.0) or 0.0) for item in candidates]
    occupied = [float(item.get("occupied_fraction", 0.0) or 0.0) for item in candidates]
    clearance_failures = sum(
        not bool(item.get("floor_aligned_known_free")) for item in candidates
    )
    frontier_candidates = [
        item for item in candidates
        if str(item.get("source_type", "")).startswith("F-local-known-safe-frontier")
    ]
    direction_counts = []
    for event, pool in zip(events, pools):
        trigger = event.get("trigger_grid") or [0, 0]
        direction_counts.append(len({
            round(math.degrees(math.atan2(
                float(item.get("grid", [0, 0])[0]) - float(trigger[0]),
                float(item.get("grid", [0, 0])[1]) - float(trigger[1]),
            )) / 45.0)
            for item in pool
        }))
    return {
        "event_count": len(rows),
        "event_coverage": sum(bool(pool) for pool in pools) / max(1, len(rows)),
        "candidate_count": len(candidates),
        "candidate_per_event_mean": len(candidates) / max(1, len(rows)),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "route_candidate_universe_mean": sum(
            int(row.get("route_candidate_universe_count", 0) or 0) for row in events
        ) / max(1, len(events)),
        "route_path_eligible_candidate_mean": sum(
            int(row.get("route_path_eligible_candidate_count", 0) or 0) for row in events
        ) / max(1, len(events)),
        "candidate_direction_count_mean": sum(direction_counts) / max(1, len(events)),
        "route_occ_conflict_count": int(conflict),
        "floor_aligned_known_free_count": int(floor_safe),
        "floor_z_source_counts": dict(sorted(floor_sources.items())),
        "nonzero_floor_z_count": sum(
            abs(float(item.get("floor_z_m", 0.0) or 0.0)) > 1e-4
            for item in candidates
        ),
        "local_free_fraction_mean": sum(local_free) / max(1, len(local_free)),
        "unknown_fraction_mean": sum(unknown) / max(1, len(unknown)),
        "unknown_fraction_max": max(unknown) if unknown else 0.0,
        "occupied_fraction_mean": sum(occupied) / max(1, len(occupied)),
        "clearance_failure_count": int(clearance_failures),
        "frontier_candidate_count": len(frontier_candidates),
        "frontier_event_count": sum(bool(pool) for pool in pools if any(
            str(item.get("source_type", "")).startswith("F-local-known-safe-frontier")
            for item in pool
        )),
        "action_applied_count": sum(bool(row.get("action_applied")) for row in events),
        "gt_fields_used_union": sorted({field for row in events for field in row.get("gt_fields_used", [])}),
    }


def _candidate_direction_count(
    event: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]
) -> int:
    trigger = event.get("trigger_grid") or [0, 0]
    return len({
        round(math.degrees(math.atan2(
            float(item.get("grid", [0, 0])[0]) - float(trigger[0]),
            float(item.get("grid", [0, 0])[1]) - float(trigger[1]),
        )) / 45.0)
        for item in candidates
    })


def _manifest_group_report(
    expected: Iterable[Mapping[str, Any]],
    events_by_key: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = list(expected)
    stages = (
        "route_only",
        "route_occ",
        "route_occ_clearance",
        "route_occ_clearance_frontier",
    )
    observed = [events_by_key[_event_key(row)] for row in expected if _event_key(row) in events_by_key]
    stage_reports: Dict[str, Any] = {}
    for stage in stages:
        pools = [
            list(events_by_key.get(_event_key(row), {}).get("ablation", {}).get(stage, {}).get("candidates") or [])
            for row in expected
        ]
        direction_counts = [
            _candidate_direction_count(events_by_key[_event_key(row)], pool)
            if _event_key(row) in events_by_key else 0
            for row, pool in zip(expected, pools)
        ]
        counts = [len(pool) for pool in pools]
        denominator = max(1, len(expected))
        stage_reports[stage] = {
            "event_coverage": sum(count >= 1 for count in counts) / denominator,
            "event_coverage_at_least_2": sum(count >= 2 for count in counts) / denominator,
            "event_coverage_at_least_3": sum(count >= 3 for count in counts) / denominator,
            "candidate_count": sum(counts),
            "candidate_per_expected_event_mean": sum(counts) / denominator,
            "multi_candidate_event_count": sum(count >= 2 for count in counts),
            "multi_direction_event_count": sum(count >= 2 for count in direction_counts),
            "candidate_direction_count_mean": sum(direction_counts) / denominator,
        }
    return {
        "expected_event_count": len(expected),
        "observed_exact_event_count": len(observed),
        "emitted_event_recall": len(observed) / max(1, len(expected)),
        "reports": stage_reports,
    }


def _manifest_coverage_report(
    events: Iterable[Mapping[str, Any]], gt: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    events = list(events)
    gt = list(gt)
    events_by_key = {_event_key(row): row for row in events}
    expected_by_key = {_event_key(row): row for row in gt}
    expected = [expected_by_key[key] for key in sorted(expected_by_key)]
    missing_keys = sorted(set(expected_by_key) - set(events_by_key))
    unexpected_keys = sorted(set(events_by_key) - set(expected_by_key))

    def grouped(field: str) -> Dict[str, Any]:
        values = sorted({str(row.get(field) or "unspecified") for row in expected})
        return {
            value: _manifest_group_report(
                [row for row in expected if str(row.get(field) or "unspecified") == value],
                events_by_key,
            )
            for value in values
        }

    return {
        "denominator_contract": "exact_manifest_key_missing_events_count_as_zero_coverage",
        "raw_manifest_event_count": len(gt),
        "unique_expected_event_count": len(expected),
        "duplicate_manifest_event_count": len(gt) - len(expected),
        "observed_event_count": len(events),
        "observed_exact_event_count": len(set(expected_by_key) & set(events_by_key)),
        "missing_expected_event_count": len(missing_keys),
        "unexpected_event_count": len(unexpected_keys),
        "missing_expected_event_keys": [
            {"scene_id": key[0], "episode_id": key[1], "step_id": key[2]}
            for key in missing_keys
        ],
        "unexpected_event_keys": [
            {"scene_id": key[0], "episode_id": key[1], "step_id": key[2]}
            for key in unexpected_keys
        ],
        "all": _manifest_group_report(expected, events_by_key),
        "by_gt_state": grouped("gt_state"),
        "by_gt_split": grouped("gt_split"),
    }


def analyze(root: Path, gt_path: Path | None = None) -> Dict[str, Any]:
    events = _load_events(root)
    gt = _load_gt(gt_path)
    stages = ("route_only", "route_occ", "route_occ_clearance", "route_occ_clearance_frontier")
    reports = {stage: _stage_report(events, stage) for stage in stages}
    frontier_triggered_event_count = sum(bool(row.get("frontier_triggered")) for row in events)
    raw_frontier_candidate_count = sum(
        int(row.get("frontier_candidate_count", 0) or 0) for row in events
    )
    safe_frontier_candidate_count = sum(
        int(row.get("frontier_safe_candidate_count", 0) or 0) for row in events
    )
    frontier_increment_event_count = sum(
        int(row.get("frontier_safe_candidate_count", 0) or 0) > 0
        and not bool(row.get("ablation", {}).get("route_occ_clearance", {}).get("event_has_candidate"))
        for row in events
    )
    cumulative_candidate_count = sum(
        len(row.get("ablation", {}).get("route_occ_clearance_frontier", {}).get("candidates") or [])
        for row in events
    )
    matched = {
        "all_gt_events": len(gt),
        "route_only": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_only", {}).get("event_has_candidate")),
        "route_occ": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_occ", {}).get("event_has_candidate")),
        "route_occ_clearance": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_occ_clearance", {}).get("event_has_candidate")),
        "route_occ_clearance_frontier": sum(any(_overlap(event, row) for row in gt) for event in events if event.get("ablation", {}).get("route_occ_clearance_frontier", {}).get("event_has_candidate")),
        "label_status": "coverage_overlap_only_not_event_gt",
    }
    def _candidate_contract_ok(row: Mapping[str, Any]) -> bool:
        for stage in stages:
            for candidate in row.get("ablation", {}).get(stage, {}).get("candidates", []) or []:
                if (
                    not bool(candidate.get("shadow_only"))
                    or bool(candidate.get("action_applied"))
                    or candidate.get("gt_fields_used")
                    or str(candidate.get("source_type", "")).split("+")[0]
                    not in {"R-route-near", "R-route-open", "F-local-known-safe-frontier"}
                ):
                    return False
        return True

    def _frontier_contract_ok(row: Mapping[str, Any]) -> bool:
        if row.get("frontier_path_mode") != "known_free_geodesic":
            return False
        return all(
            candidate.get("path_geometry") == "known_free_geodesic"
            and candidate.get("route_support") == "local_known_safe_frontier"
            and candidate.get("frontier_boundary_grid") is not None
            and float(candidate.get("frontier_standoff_m", 0.0) or 0.0) > 0.0
            for candidate in row.get("frontier_candidates", []) or []
        )

    schema_ok = all(
        row.get("event_schema_version") == "stage27_m3_candidate_generation_v5"
        for row in events
    )
    route_contract_ok = all(
        row.get("candidate_pool_contract") == "R-route-near_union_R-route-open"
        for row in events
    )
    frontier_pool_contract_ok = all(
        row.get("frontier_pool_contract") == "F-local-known-safe-frontier"
        for row in events
    )
    return {
        "task": "stage27_m3_candidate_generation_shadow_audit",
        "event_schema": "stage27_m3_candidate_generation_v5",
        "event_count": len(events),
        "reports": reports,
        "frontier_metrics": {
            "frontier_triggered_event_count": frontier_triggered_event_count,
            "frontier_triggered_event_rate": frontier_triggered_event_count / max(1, len(events)),
            "raw_frontier_candidate_count": raw_frontier_candidate_count,
            "safe_frontier_candidate_count": safe_frontier_candidate_count,
            "frontier_increment_event_count": frontier_increment_event_count,
            "frontier_incremental_event_coverage": frontier_increment_event_count / max(1, len(events)),
            "cumulative_candidate_count": cumulative_candidate_count,
            "frontier_unknown_fraction_mean": sum(
                float(item.get("unknown_fraction", 0.0) or 0.0)
                for row in events for item in row.get("frontier_candidates", []) or []
            ) / max(1, raw_frontier_candidate_count),
            "frontier_occupied_fraction_mean": sum(
                float(item.get("occupied_fraction", 0.0) or 0.0)
                for row in events for item in row.get("frontier_candidates", []) or []
            ) / max(1, raw_frontier_candidate_count),
            "frontier_clearance_failure_count": sum(
                not bool(item.get("floor_aligned_known_free"))
                for row in events for item in row.get("frontier_candidates", []) or []
            ),
            "frontier_path_clear_candidate_count": sum(
                float(item.get("unknown_fraction", 0.0) or 0.0) == 0.0
                and float(item.get("occupied_fraction", 0.0) or 0.0) == 0.0
                for row in events for item in row.get("frontier_candidates", []) or []
            ),
            "frontier_path_clear_pass_rate": sum(
                float(item.get("unknown_fraction", 0.0) or 0.0) == 0.0
                and float(item.get("occupied_fraction", 0.0) or 0.0) == 0.0
                for row in events for item in row.get("frontier_candidates", []) or []
            ) / max(1, raw_frontier_candidate_count),
            "frontier_joint_safe_pass_rate": safe_frontier_candidate_count / max(1, raw_frontier_candidate_count),
            "frontier_local_cell_count": sum(
                int(row.get("frontier_local_cell_count", 0) or 0) for row in events
            ),
            "frontier_sampled_cell_count": sum(
                int(row.get("frontier_sampled_cell_count", 0) or 0) for row in events
            ),
            "frontier_geodesic_reachable_count": sum(
                int(row.get("frontier_geodesic_reachable_count", 0) or 0) for row in events
            ),
            "frontier_path_modes": sorted({
                str(row.get("frontier_path_mode") or "unspecified") for row in events
            }),
        },
        "manifest_candidate_coverage": _manifest_coverage_report(events, gt),
        "gt_overlap_diagnostic": matched,
        "active_recovery_enabled": False,
        "ranker_trained": False,
        "unknown_is_free": False,
        "success_is_event_gt": False,
        "candidate_pool_contracts": sorted({
            str(row.get("candidate_pool_contract") or "legacy_all_route_nodes")
            for row in events
        }),
        "event_schema_versions": sorted({
            str(row.get("event_schema_version") or "legacy") for row in events
        }),
        "integrity_checks": {
            "schema_v5": schema_ok,
            "route_pool_contract": route_contract_ok,
            "frontier_pool_contract": frontier_pool_contract_ok,
            "frontier_geodesic_standoff_contract": all(
                _frontier_contract_ok(row) for row in events
            ),
            "shadow_only": all(bool(row.get("shadow_only")) for row in events),
            "no_action": all(not bool(row.get("action_applied")) for row in events),
            "no_gt_fields": all(not row.get("gt_fields_used") for row in events),
            "candidate_records_shadow_only": all(_candidate_contract_ok(row) for row in events),
        },
        "integrity_passed": bool(events) and all(
            bool(row.get("shadow_only"))
            and not bool(row.get("action_applied"))
            and not row.get("gt_fields_used")
            and row.get("event_schema_version") == "stage27_m3_candidate_generation_v5"
            and row.get("candidate_pool_contract") == "R-route-near_union_R-route-open"
            and row.get("frontier_pool_contract") == "F-local-known-safe-frontier"
            and _candidate_contract_ok(row)
            and _frontier_contract_ok(row)
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

#!/usr/bin/env python3
"""Audit Stage21 cross-query S2 action-loop shadow events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FORBIDDEN_GT_KEYS = {
    "distance_to_goal",
    "geodesic_distance",
    "ne",
    "oracle_success",
    "success",
    "spl",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_key(row: dict) -> str:
    return f"{row.get('scene_id')}/{int(row.get('episode_id'))}"


def _find_forbidden_keys(value, prefix="") -> list[str]:
    matches = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_GT_KEYS:
                matches.append(path)
            matches.extend(_find_forbidden_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(_find_forbidden_keys(item, f"{prefix}[{index}]"))
    return matches


def build_audit(run_root: Path, expected_episodes: int) -> dict:
    progress_rows = _read_jsonl(run_root / "progress.json")
    event_paths = sorted(run_root.glob("vlmap_safety_debug/**/s2_action_loop_events.jsonl"))
    event_entries = [
        (path, row) for path in event_paths for row in _read_jsonl(path)
    ]
    start_entries = [
        (path, row) for path, row in event_entries if row.get("transition") == "start"
    ]
    start_events = [row for _, row in start_entries]
    progress_by_episode = {_episode_key(row): row for row in progress_rows}
    events_by_episode = Counter(_episode_key(row) for row in start_events)
    tier_counts = Counter(str(row.get("triage_tier") or "missing") for row in start_events)
    failure_counts = Counter(str(row.get("failure_type") or "missing") for row in start_events)
    gt_key_hits = [
        {"episode": _episode_key(row), "step_id": row.get("step_id"), "paths": paths}
        for row in start_events
        if (paths := _find_forbidden_keys(row))
    ]
    non_shadow = [row for row in start_events if not row.get("shadow_only")]
    applied = [row for row in start_events if row.get("applied")]
    missing_schema = [
        row for row in start_events
        if row.get("event_schema_version") != "stage21a_s2_loop_v1"
    ]
    missing_candidate = [row for row in start_events if not row.get("candidate")]
    missing_rgb = [
        {"episode": _episode_key(row), "step_id": row.get("step_id")}
        for path, row in start_entries
        if not row.get("rgb_file") or not (path.parent / str(row["rgb_file"])).is_file()
    ]
    triggered_success_episodes = {
        _episode_key(row)
        for row in start_events
        if float(
            (progress_by_episode.get(_episode_key(row)) or {}).get("success", 0.0)
            or 0.0
        )
        > 0.5
    }
    success_episode_count = sum(
        float(row.get("success", 0.0) or 0.0) > 0.5 for row in progress_rows
    )
    summary = {
        "task": "stage21_s2_action_loop_shadow_audit",
        "run_root": str(run_root),
        "expected_episode_count": int(expected_episodes),
        "progress_episode_count": len(progress_rows),
        "event_file_count": len(event_paths),
        "loop_event_count": len(start_events),
        "loop_episode_count": len(events_by_episode),
        "loop_events_by_episode": dict(events_by_episode),
        "first_loop_step_by_episode": {
            key: min(int(row["step_id"]) for row in start_events if _episode_key(row) == key)
            for key in events_by_episode
        },
        "triage_tier_counts": dict(tier_counts),
        "failure_type_counts": dict(failure_counts),
        "candidate_coverage_rate": (
            (len(start_events) - len(missing_candidate)) / len(start_events)
            if start_events else None
        ),
        "missing_rgb_snapshots": missing_rgb,
        "success_trigger_count": len(triggered_success_episodes),
        "success_trigger_episode_rate": (
            len(triggered_success_episodes) / max(1, success_episode_count)
        ),
        "gt_leakage_scan": {"passed": not gt_key_hits, "hits": gt_key_hits},
        "shadow_safety": {
            "non_shadow_count": len(non_shadow),
            "applied_count": len(applied),
            "passed": not non_shadow and not applied,
        },
        "schema_missing_count": len(missing_schema),
        "passed": bool(
            len(progress_rows) == int(expected_episodes)
            and not gt_key_hits
            and not non_shadow
            and not applied
            and not missing_schema
            and not missing_rgb
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-episode", action="append", default=[])
    parser.add_argument("--forbid-episode", action="append", default=[])
    parser.add_argument("--max-first-step", action="append", default=[])
    parser.add_argument("--min-candidate-coverage", type=float)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    summary = build_audit(args.run_root, args.expected_episodes)
    first_steps = summary["first_loop_step_by_episode"]
    missing_required = [key for key in args.require_episode if key not in first_steps]
    forbidden_present = [key for key in args.forbid_episode if key in first_steps]
    max_step_failures = []
    for value in args.max_first_step:
        key, raw_step = value.rsplit("=", 1)
        if key not in first_steps or int(first_steps[key]) > int(raw_step):
            max_step_failures.append(
                {"episode": key, "actual": first_steps.get(key), "maximum": int(raw_step)}
            )
    summary["missing_required_loop_episodes"] = missing_required
    summary["forbidden_loop_episodes_present"] = forbidden_present
    summary["max_first_step_failures"] = max_step_failures
    candidate_coverage_failure = bool(
        args.min_candidate_coverage is not None
        and (
            summary["candidate_coverage_rate"] is None
            or float(summary["candidate_coverage_rate"])
            < float(args.min_candidate_coverage)
        )
    )
    summary["minimum_candidate_coverage"] = args.min_candidate_coverage
    summary["candidate_coverage_failure"] = candidate_coverage_failure
    summary["passed"] = bool(
        summary["passed"]
        and not missing_required
        and not forbidden_present
        and not max_step_failures
        and not candidate_coverage_failure
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_all and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Stage55 route-OCC baseline versus post-turn collision guard."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from internnav.utils.stage55_occ_2p5d_audit import POST_TURN_FORWARD_SOURCES


METRICS = ("success", "spl", "ne", "steps", "collision_count")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('scene_id')}/{int(row.get('episode_id', -1))}"


def _progress(root: Path) -> dict[str, dict[str, Any]]:
    return {_key(row): row for row in _jsonl(root / "progress.json")}


def _events(root: Path, name: str) -> list[dict[str, Any]]:
    paths = glob.glob(str(root / "vlmap_safety_debug" / "*" / name))
    return [row for path in paths for row in _jsonl(Path(path))]


def _active_integrity(events: list[dict[str, Any]]) -> list[str]:
    violations = []
    for row in events:
        if not row.get("action_applied"):
            continue
        candidate = dict(row.get("candidate") or {})
        planned = list(row.get("reorient_actions") or [])
        executed = list(row.get("reorient_actions_applied") or [])
        if len(planned) != 1 or executed != planned or row.get("pixel_action_applied"):
            violations.append("non_single_turn_active")
        if candidate.get("stage46_safety_derivation") != "route_occ":
            violations.append("candidate_stage_mismatch")
        if candidate.get("stage54_translation_allowed") is not False:
            violations.append("translation_not_disabled")
        if row.get("gt_fields_used") or candidate.get("gt_fields_used"):
            violations.append("active_gt_leakage")
    return violations


def _occ_audits(events: list[dict[str, Any]]) -> dict[str, Any]:
    unique = {}
    violations = []
    for row in events:
        audit = dict(row.get("stage55_occ_2p5d_audit") or {})
        if not audit:
            continue
        key = (
            row.get("scene_id"),
            row.get("episode_id"),
            row.get("trigger_step"),
            audit.get("candidate_id"),
        )
        unique[key] = audit
        if audit.get("decision_applied") is not False or audit.get("unknown_is_free") is not False:
            violations.append("occ_audit_changed_decision")
    reports = list(unique.values())
    return {
        "candidate_audit_count": len(reports),
        "continuous_support_shadow_count": sum(
            bool((row.get("support_2p5d") or {}).get("continuous_support_shadow"))
            for row in reports
        ),
        "legacy_blocked_frame_consensus_free_sum": sum(
            int(row.get("legacy_blocked_frame_consensus_free_count", 0) or 0)
            for row in reports
        ),
        "legacy_blocked_frame_consensus_unknown_sum": sum(
            int(row.get("legacy_blocked_frame_consensus_unknown_count", 0) or 0)
            for row in reports
        ),
        "records": reports,
        "violations": violations,
    }


def analyze(baseline_root: Path, guard_root: Path, manifest: Path) -> dict[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        f"{row['scene_id']}/{int(row['episode_id'])}": int(row["episode_eval_seed"])
        for row in rows
    }
    baseline = _progress(baseline_root)
    guard = _progress(guard_root)
    baseline_active = _events(
        baseline_root, "s2_loop_path_reobserve_active_events.jsonl"
    )
    guard_active = _events(
        guard_root, "s2_loop_path_reobserve_active_events.jsonl"
    )
    baseline_guard = _events(
        baseline_root, "stage55_post_turn_collision_guard_events.jsonl"
    )
    guard_events = _events(
        guard_root, "stage55_post_turn_collision_guard_events.jsonl"
    )
    violations = _active_integrity(baseline_active) + _active_integrity(guard_active)
    if baseline_guard:
        violations.append("baseline_guard_event_present")
    for row in guard_events:
        if (
            row.get("previous_action") != 1
            or row.get("previous_action_source") not in POST_TURN_FORWARD_SOURCES
            or float(row.get("collision_delta", 0.0) or 0.0) <= 0.0
            or row.get("environment_action_applied") is not False
            or row.get("pixel_translation_applied") is not False
            or row.get("gt_fields_used")
        ):
            violations.append("invalid_guard_event")
    seed_mismatch = [
        key
        for key, seed in expected.items()
        if baseline.get(key, {}).get("episode_eval_seed") != seed
        or guard.get(key, {}).get("episode_eval_seed") != seed
    ]
    paired = []
    for key in sorted(expected):
        item = {"scene_episode": key}
        for metric in METRICS:
            base = float(baseline.get(key, {}).get(metric, 0.0) or 0.0)
            active = float(guard.get(key, {}).get(metric, 0.0) or 0.0)
            item[f"baseline_{metric}"] = base
            item[f"guard_{metric}"] = active
            item[f"delta_{metric}"] = active - base
        item["guard_event_count"] = sum(_key(row) == key for row in guard_events)
        paired.append(item)
    baseline_occ = _occ_audits(baseline_active)
    guard_occ = _occ_audits(guard_active)
    violations += baseline_occ["violations"] + guard_occ["violations"]
    integrity = bool(
        len(baseline) == len(expected)
        and len(guard) == len(expected)
        and not seed_mismatch
        and not violations
    )
    return {
        "task": "stage55_post_turn_collision_guard_paired",
        "expected_episode_count": len(expected),
        "baseline_episode_count": len(baseline),
        "guard_episode_count": len(guard),
        "seed_mismatch": seed_mismatch,
        "guard_event_count": len(guard_events),
        "guard_episode_count_with_event": len({_key(row) for row in guard_events}),
        "failed_to_success_count": sum(
            row["baseline_success"] == 0 and row["guard_success"] > 0
            for row in paired
        ),
        "success_to_failed_count": sum(
            row["baseline_success"] > 0 and row["guard_success"] == 0
            for row in paired
        ),
        "collision_delta_sum": sum(row["delta_collision_count"] for row in paired),
        "step_delta_sum": sum(row["delta_steps"] for row in paired),
        "paired_episode_records": paired,
        "baseline_occ_2p5d": baseline_occ,
        "guard_occ_2p5d": guard_occ,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "violation_counts": {name: violations.count(name) for name in set(violations)},
        "integrity_passed": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--guard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.baseline_root, args.guard_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

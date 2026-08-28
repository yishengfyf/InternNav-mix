#!/usr/bin/env python3
"""Audit Stage54 control/strict/relaxed one-turn active arms."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


METRICS = ("success", "spl", "ne", "steps", "collision_count")
ARM_CONTRACTS = {
    "strict": ("route_occ_clearance_frontier", "strict"),
    "route_occ": ("route_occ", "route_occ_turn_only"),
    "route_only": ("route_only", "route_only_turn_only"),
}


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


def _events(root: Path) -> list[dict[str, Any]]:
    paths = glob.glob(
        str(root / "vlmap_safety_debug" / "*" / "s2_loop_path_reobserve_active_events.jsonl")
    )
    return [row for path in paths for row in _jsonl(Path(path))]


def _arm_report(
    name: str,
    root: Path,
    control: dict[str, dict[str, Any]],
    expected: dict[str, int],
) -> dict[str, Any]:
    progress = _progress(root)
    events = _events(root)
    applied = [row for row in events if row.get("action_applied")]
    stage, mode = ARM_CONTRACTS[name]
    violations = []
    for row in applied:
        candidate = dict(row.get("candidate") or {})
        planned = list(row.get("reorient_actions") or [])
        executed = list(row.get("reorient_actions_applied") or [])
        if row.get("pixel_action_applied") or len(planned) != 1 or executed != planned:
            violations.append("non_single_turn_action")
        if row.get("gt_fields_used") or candidate.get("gt_fields_used"):
            violations.append("gt_leakage")
        if candidate.get("stage46_safety_derivation") != stage:
            violations.append("candidate_stage_mismatch")
        if candidate.get("stage54_safety_mode", "strict") != mode:
            violations.append("safety_mode_mismatch")
        if mode != "strict" and candidate.get("stage54_translation_allowed") is not False:
            violations.append("relaxed_translation_not_disabled")
    seed_mismatch = [
        key
        for key, seed in expected.items()
        if progress.get(key, {}).get("episode_eval_seed") != seed
    ]
    paired = []
    applied_episodes = {_key(row) for row in applied}
    for key in sorted(expected):
        item = {"scene_episode": key, "intervened": key in applied_episodes}
        for metric in METRICS:
            c = float(control.get(key, {}).get(metric, 0.0) or 0.0)
            a = float(progress.get(key, {}).get(metric, 0.0) or 0.0)
            item[f"control_{metric}"] = c
            item[f"active_{metric}"] = a
            item[f"delta_{metric}"] = a - c
        paired.append(item)
    return {
        "candidate_stage": stage,
        "safety_mode": mode,
        "episode_count": len(progress),
        "event_count": len(events),
        "event_reason_counts": dict(Counter(str(row.get("reason")) for row in events)),
        "applied_event_count": len(applied),
        "applied_episode_count": len(applied_episodes),
        "frozen_path_bearing_relaxation_count": sum(
            bool(row.get("frozen_candidate_path_bearing_relaxation")) for row in applied
        ),
        "failed_to_success_count": sum(
            row["intervened"] and row["control_success"] == 0 and row["active_success"] > 0
            for row in paired
        ),
        "success_to_failed_count": sum(
            row["intervened"] and row["control_success"] > 0 and row["active_success"] == 0
            for row in paired
        ),
        "collision_delta_sum": sum(row["delta_collision_count"] for row in paired),
        "step_delta_sum": sum(row["delta_steps"] for row in paired),
        "paired_episode_records": paired,
        "seed_mismatch": seed_mismatch,
        "violation_counts": dict(Counter(violations)),
        "integrity_passed": bool(
            len(progress) == len(expected) and not seed_mismatch and not violations
        ),
    }


def analyze(control_root: Path, arm_roots: dict[str, Path], manifest: Path) -> dict[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        f"{row['scene_id']}/{int(row['episode_id'])}": int(row["episode_eval_seed"])
        for row in rows
    }
    control = _progress(control_root)
    control_seed_mismatch = [
        key
        for key, seed in expected.items()
        if control.get(key, {}).get("episode_eval_seed") != seed
    ]
    arms = {
        name: _arm_report(name, root, control, expected)
        for name, root in arm_roots.items()
    }
    integrity = bool(
        len(control) == len(expected)
        and not control_seed_mismatch
        and all(report["integrity_passed"] for report in arms.values())
    )
    return {
        "task": "stage54_relaxed_turn_active_paired",
        "expected_episode_count": len(expected),
        "control_episode_count": len(control),
        "control_seed_mismatch": control_seed_mismatch,
        "arms": arms,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "ranker_trained": False,
        "integrity_passed": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--route-occ-root", type=Path, required=True)
    parser.add_argument("--route-only-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(
        args.control_root,
        {
            "strict": args.strict_root,
            "route_occ": args.route_occ_root,
            "route_only": args.route_only_root,
        },
        args.manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

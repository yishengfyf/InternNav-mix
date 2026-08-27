#!/usr/bin/env python3
"""Audit paired frozen control vs Stage46 one-primitive active recovery."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


METRICS = ("success", "spl", "ne", "steps", "collision_count")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}/{int(row.get('episode_id', -1))}"


def _progress(root: Path) -> dict[str, dict[str, Any]]:
    return {
        _episode_key(row): row
        for row in _jsonl(root / "progress.json")
        if row.get("episode_id") is not None
    }


def _events(root: Path) -> list[dict[str, Any]]:
    paths = glob.glob(
        str(root / "vlmap_safety_debug" / "*" / "s2_loop_path_reobserve_active_events.jsonl")
    )
    return [row for path in paths for row in _jsonl(Path(path))]


def _safe_candidate(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("floor_aligned_known_free")
        and float(candidate.get("unknown_fraction", 1.0) or 0.0) == 0.0
        and float(candidate.get("occupied_fraction", 1.0) or 0.0) == 0.0
        and not candidate.get("route_occ_conflict")
        and not candidate.get("gt_fields_used")
        and candidate.get("stage46_safety_derivation")
        == "route_occ_clearance_frontier"
    )


def analyze(control_root: Path, active_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        f"{row['scene_id']}/{int(row['episode_id'])}": int(row["episode_eval_seed"])
        for row in manifest
    }
    control = _progress(control_root)
    active = _progress(active_root)
    events = _events(active_root)
    common = sorted(set(expected).intersection(control, active))
    applied = [row for row in events if row.get("action_applied")]
    reorient = [row for row in events if row.get("reorient_action_applied")]
    pixel = [row for row in events if row.get("pixel_action_applied")]

    violations: dict[str, list[Any]] = {
        "seed_mismatch": [],
        "non_stage27_candidate_source": [],
        "unsafe_applied_candidate": [],
        "multi_primitive_reorient": [],
        "multi_primitive_nextdit": [],
        "pixel_contract_failure": [],
        "gt_leakage": [],
    }
    for key, seed in expected.items():
        if control.get(key, {}).get("episode_eval_seed") != seed or active.get(key, {}).get("episode_eval_seed") != seed:
            violations["seed_mismatch"].append(key)
    for row in applied:
        if row.get("candidate_source") != "stage27_frozen_m3":
            violations["non_stage27_candidate_source"].append(row)
        if not _safe_candidate(row.get("candidate") or {}):
            violations["unsafe_applied_candidate"].append(row)
        if row.get("gt_fields_used"):
            violations["gt_leakage"].append(row)
    for row in reorient:
        planned = list(row.get("reorient_actions") or [])
        executed = list(row.get("reorient_actions_applied") or [])
        if len(planned) != 1 or executed != planned or planned[0] not in (2, 3):
            violations["multi_primitive_reorient"].append(row)
    for row in pixel:
        trajectory = dict(row.get("trajectory_preflight") or {})
        actions = list(trajectory.get("local_actions") or [])
        if len(actions) != 1:
            violations["multi_primitive_nextdit"].append(row)
        bridge = dict(row.get("post_path_bridge") or row.get("path_bridge") or {})
        selected = dict(bridge.get("selected_probe") or {})
        if (
            not bridge.get("valid")
            or not selected.get("path_eligible")
            or selected.get("goal_state") != "free"
            or not trajectory.get("valid")
            or not trajectory.get("safe")
            or trajectory.get("reject_required")
        ):
            violations["pixel_contract_failure"].append(row)

    paired = []
    intervention_episodes = {_episode_key(row) for row in applied}
    for key in common:
        item = {"scene_episode": key, "intervened": key in intervention_episodes}
        for metric in METRICS:
            c = float(control[key].get(metric, 0.0) or 0.0)
            a = float(active[key].get(metric, 0.0) or 0.0)
            item[f"control_{metric}"] = c
            item[f"active_{metric}"] = a
            item[f"delta_{metric}"] = a - c
        paired.append(item)

    integrity = bool(
        len(control) == len(expected)
        and len(active) == len(expected)
        and len(common) == len(expected)
        and not any(violations.values())
    )
    return {
        "task": "stage46_m3_one_primitive_active_paired",
        "expected_episode_count": len(expected),
        "control_episode_count": len(control),
        "active_episode_count": len(active),
        "event_count": len(events),
        "event_reason_counts": dict(Counter(str(row.get("reason")) for row in events)),
        "applied_intervention_count": len(
            {
                (_episode_key(row), int(row.get("trigger_step", -1)))
                for row in applied
            }
        ),
        "applied_episode_count": len(intervention_episodes),
        "reorient_applied_event_count": len(reorient),
        "pixel_primitive_applied_event_count": len(pixel),
        "failed_to_success_count": sum(
            item["intervened"] and item["control_success"] == 0 and item["active_success"] > 0
            for item in paired
        ),
        "success_to_failed_count": sum(
            item["intervened"] and item["control_success"] > 0 and item["active_success"] == 0
            for item in paired
        ),
        "paired_episode_records": paired,
        "violations": {name: len(rows) for name, rows in violations.items()},
        "integrity_passed": integrity,
        "unknown_is_free": False,
        "ranker_trained": False,
        "one_primitive_per_reaudit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.control_root, args.active_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

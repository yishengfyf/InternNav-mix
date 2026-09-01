#!/usr/bin/env python3
"""Aggregate Stage63 adaptive counterfactual view-sweep shadow events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(run_root: Path) -> dict[str, Any]:
    arms: Counter[str] = Counter()
    arm_visible: Counter[str] = Counter()
    arm_readable: Counter[str] = Counter()
    arm_restored: Counter[str] = Counter()
    event_count = 0
    scene_ids: set[str] = set()
    violations: list[list[Any]] = []
    records = []
    for path in run_root.glob("vlmap_safety_debug/*run_*/s2_loop_path_reobserve_active_events.jsonl"):
        for event in _jsonl(path):
            report = event.get("stage59_productive_onset") or {}
            if not report:
                continue
            anchor = next(
                (row for row in report.get("anchors") or [] if row.get("anchor") == "last_productive_pre_loop"),
                None,
            )
            adaptive = (anchor or {}).get("stage63_adaptive_reobserve") or {}
            if not adaptive:
                continue
            event_count += 1
            scene_ids.add(str(event.get("scene_id")))
            if not (
                adaptive.get("shadow_only") is True
                and adaptive.get("decision_applied") is False
                and adaptive.get("action_applied") is False
                and adaptive.get("pixel_translation_allowed") is False
                and adaptive.get("unknown_is_free") is False
                and adaptive.get("official_memory_mutated") is False
                and adaptive.get("sim_pose_all_restored") is True
                and not adaptive.get("gt_fields_used")
            ):
                violations.append([event.get("scene_id"), event.get("episode_id"), event.get("trigger_step")])
            row = {
                "scene_id": event.get("scene_id"),
                "episode_id": event.get("episode_id"),
                "trigger_step": event.get("trigger_step"),
                "relative_bearing_deg": adaptive.get("relative_bearing_deg"),
                "any_first_edge_visible": bool(adaptive.get("any_first_edge_visible")),
                "probe_count": len(adaptive.get("probes") or []),
                "probes": [],
            }
            for probe in adaptive.get("probes") or []:
                arm_names = list(probe.get("arm_aliases") or [probe.get("arm")])
                for arm in arm_names:
                    arms[str(arm)] += 1
                    arm_readable[str(arm)] += int(bool(probe.get("observation_readable")))
                    arm_restored[str(arm)] += int(bool(probe.get("sim_pose_restored")))
                    arm_visible[str(arm)] += int(bool(probe.get("first_edge_visible_projection")))
                row["probes"].append({
                    "arm": probe.get("arm"),
                    "arm_aliases": arm_names,
                    "observation_readable": bool(probe.get("observation_readable")),
                    "sim_pose_restored": bool(probe.get("sim_pose_restored")),
                    "first_edge_visible_projection": bool(probe.get("first_edge_visible_projection")),
                    "bridge_reason": probe.get("bridge_reason"),
                    "selected_path_progress_m": probe.get("selected_path_progress_m"),
                })
            records.append(row)
    arm_summary = {
        arm: {
            "probe_count": int(count),
            "observation_readable": int(arm_readable[arm]),
            "sim_pose_restored": int(arm_restored[arm]),
            "first_edge_visible": int(arm_visible[arm]),
        }
        for arm, count in sorted(arms.items())
    }
    return {
        "task": "stage63_adaptive_reobserve",
        "schema_version": "stage63_adaptive_reobserve_v1",
        "event_count": event_count,
        "scene_count": len(scene_ids),
        "arm_summary": arm_summary,
        "any_view_visible_count": sum(int(row["any_first_edge_visible"]) for row in records),
        "integrity_passed": bool(event_count and not violations),
        "shadow_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "integrity_violations": violations,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate Stage59 productive-onset shadow events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


POLICY = "known_free_floor_frames2"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1))


def _policy(anchor: dict[str, Any]) -> dict[str, Any]:
    report = anchor.get("stage58_support_policy") or {}
    return next((row for row in report.get("arms") or [] if row.get("policy") == POLICY), {})


def analyze(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {_key(row): row for row in manifest_rows}
    progress = {}
    for path in run_root.glob("vlmap_safety_debug/*run_*/progress.json"):
        for row in _jsonl(path):
            progress[_key(row)] = row

    events = []
    for path in run_root.glob("vlmap_safety_debug/*run_*/s2_loop_path_reobserve_active_events.jsonl"):
        for row in _jsonl(path):
            report = row.get("stage59_productive_onset") or {}
            if report:
                events.append(
                    {
                        "scene_id": row.get("scene_id"),
                        "episode_id": row.get("episode_id"),
                        "trigger_step": row.get("trigger_step"),
                        "estimated_loop_onset_step": report.get("estimated_loop_onset_step"),
                        "anchors": report.get("anchors") or [],
                        "contract": {
                            key: report.get(key)
                            for key in (
                                "shadow_only", "decision_applied", "action_applied",
                                "pixel_translation_allowed", "unknown_is_free",
                                "gt_used_for_navigation",
                            )
                        },
                    }
                )

    errors = []
    for key, expected in manifest.items():
        row = progress.get(key)
        if row is None:
            errors.append(f"missing_progress:{key}")
        elif int(row.get("episode_eval_seed", -1)) != int(expected.get("episode_eval_seed", -2)):
            errors.append(f"seed_mismatch:{key}")

    violations = []
    anchor_counts: dict[str, Counter] = {}
    anchor_scenes: dict[str, set[str]] = {}
    for event in events:
        contract = event["contract"]
        if not (
            contract.get("shadow_only") is True
            and contract.get("decision_applied") is False
            and contract.get("action_applied") is False
            and contract.get("pixel_translation_allowed") is False
            and contract.get("unknown_is_free") is False
            and contract.get("gt_used_for_navigation") is False
        ):
            violations.append([event["scene_id"], event["episode_id"], event["trigger_step"]])
        for anchor in event["anchors"]:
            name = str(anchor.get("anchor"))
            counter = anchor_counts.setdefault(name, Counter())
            anchor_scenes.setdefault(name, set()).add(str(event["scene_id"]))
            policy = _policy(anchor)
            truth = anchor.get("offline_primitive_truth") or {}
            retreat = int(anchor.get("route_edge_count", 0) or 0) > 0
            predicted_safe = bool(policy.get("predicted_first_primitive_safe"))
            visible = bool(anchor.get("first_edge_visible_projection"))
            counter["event_count"] += 1
            counter["valid_anchor"] += int(bool(anchor.get("valid")))
            counter["retreat_edge_nonzero"] += int(retreat)
            counter["current_sparseocc_connected"] += int(bool(anchor.get("current_sparseocc_connectivity")))
            counter["first_edge_visible"] += int(visible)
            counter["first_0p25m_support_safe"] += int(predicted_safe)
            counter["shadow_joint_eligible"] += int(retreat and visible and predicted_safe)
            counter["offline_truth_valid"] += int(bool(truth.get("valid")))
            counter["offline_truth_safe"] += int(bool(truth.get("valid") and truth.get("primitive_safe")))
            counter["false_safe"] += int(bool(truth.get("valid") and predicted_safe and not truth.get("primitive_safe")))
            counter["false_block"] += int(bool(truth.get("valid") and not predicted_safe and truth.get("primitive_safe")))

    summaries = {}
    for name, counter in sorted(anchor_counts.items()):
        summaries[name] = {**dict(counter), "scene_count": len(anchor_scenes.get(name, set()))}
    productive = summaries.get("last_productive_pre_loop", {})
    integrity = bool(
        len(progress) >= len(manifest)
        and not errors
        and events
        and not violations
    )
    return {
        "task": "stage59_productive_onset",
        "schema_version": "stage59_productive_onset_audit_v1",
        "expected_episode_count": len(manifest),
        "completed_episode_count": len(progress),
        "natural_d0_event_count": len(events),
        "natural_d0_scene_count": len({str(row["scene_id"]) for row in events}),
        "support_policy": POLICY,
        "anchor_summaries": summaries,
        "stage59_active_gate": {
            "required_joint_eligible": 12,
            "required_scenes": 4,
            "required_false_safe": 0,
            "observed_joint_eligible": int(productive.get("shadow_joint_eligible", 0) or 0),
            "observed_scenes": int(productive.get("scene_count", 0) or 0),
            "observed_false_safe": int(productive.get("false_safe", 0) or 0),
            "passed": bool(
                int(productive.get("shadow_joint_eligible", 0) or 0) >= 12
                and int(productive.get("scene_count", 0) or 0) >= 4
                and int(productive.get("false_safe", 0) or 0) == 0
            ),
        },
        "integrity_passed": integrity,
        "shadow_only": True,
        "decision_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "errors": errors,
        "integrity_violations": violations,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

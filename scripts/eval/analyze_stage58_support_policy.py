#!/usr/bin/env python3
"""Aggregate Stage58.1 support-policy counterfactuals."""

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


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1))


def analyze(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {_key(row): row for row in manifest_rows}
    progress = {}
    for path in run_root.glob("vlmap_safety_debug/*run_*/progress.json"):
        for row in _jsonl(path):
            progress[_key(row)] = row

    events = []
    for path in run_root.glob(
        "vlmap_safety_debug/*run_*/s2_loop_path_reobserve_active_events.jsonl"
    ):
        for event in _jsonl(path):
            audit = event.get("stage58_support_policy") or {}
            if not audit:
                continue
            events.append(
                {
                    "scene_id": event.get("scene_id"),
                    "episode_id": event.get("episode_id"),
                    "trigger_step": event.get("trigger_step"),
                    "candidate_id": (event.get("candidate") or {}).get(
                        "candidate_id"
                    ),
                    "path_reachable": bool(
                        (event.get("stage57_planned_prefix_audit") or {}).get(
                            "path_reachable"
                        )
                    ),
                    "image_bridge_reason": (
                        event.get("stage57_planned_prefix_audit") or {}
                    ).get("image_bridge_reason"),
                    "audit": audit,
                }
            )

    errors = []
    for key, expected in manifest.items():
        row = progress.get(key)
        if row is None:
            errors.append(f"missing_progress:{key}")
        elif int(row.get("episode_eval_seed", -1)) != int(
            expected.get("episode_eval_seed", -2)
        ):
            errors.append(f"seed_mismatch:{key}")

    policy_counts: dict[str, Counter] = {}
    integrity_violations = []
    truth_valid = 0
    truth_safe = 0
    directional_truth_valid = 0
    directional_truth_safe = 0
    directional_truth_unsafe = 0
    runtime_contracts = Counter()
    for event in events:
        audit = event["audit"]
        truth = audit.get("offline_geometry_truth") or {}
        truth_valid += int(bool(truth.get("valid")))
        truth_safe += int(bool(truth.get("valid") and truth.get("primitive_safe")))
        contract = audit.get("runtime_contract") or {}
        runtime_contracts[
            (
                contract.get("agent_radius_m"),
                contract.get("agent_height_m"),
                contract.get("sensor_height_m"),
                contract.get("forward_step_m"),
            )
        ] += 1
        if not (
            audit.get("decision_applied") is False
            and audit.get("action_applied") is False
            and audit.get("pixel_translation_allowed") is False
            and audit.get("direct_path_primitive_executed") is False
            and audit.get("unknown_is_free") is False
            and audit.get("known_free_floor_fallback_mutates_memory") is False
            and audit.get("gt_used_for_navigation") is False
            and audit.get("directional_controls_used_for_navigation") is False
        ):
            integrity_violations.append(
                [event["scene_id"], event["episode_id"], event["trigger_step"]]
            )
        for arm in audit.get("arms") or []:
            counter = policy_counts.setdefault(str(arm.get("policy")), Counter())
            counter["event_count"] += 1
            counter["predicted_safe"] += int(
                bool(arm.get("predicted_first_primitive_safe"))
            )
            counter["false_safe"] += int(bool(arm.get("false_safe")))
            counter["false_block"] += int(bool(arm.get("false_block")))
            graph = arm.get("graph") or {}
            counter["leading_support_nonzero"] += int(
                float(graph.get("leading_full_footprint_safe_segment_m", 0.0) or 0.0)
                > 0.0
            )
        for control in audit.get("directional_controls") or []:
            control_audit = control.get("audit") or {}
            control_truth = control_audit.get("offline_geometry_truth") or {}
            valid = bool(control_truth.get("valid"))
            safe = bool(control_truth.get("primitive_safe")) if valid else False
            directional_truth_valid += int(valid)
            directional_truth_safe += int(valid and safe)
            directional_truth_unsafe += int(valid and not safe)
            for arm in control_audit.get("arms") or []:
                counter = policy_counts.setdefault(
                    str(arm.get("policy")), Counter()
                )
                counter["directional_control_count"] += 1
                counter["directional_predicted_safe"] += int(
                    bool(arm.get("predicted_first_primitive_safe"))
                )
                counter["directional_false_safe"] += int(
                    bool(arm.get("false_safe"))
                )
                counter["directional_false_block"] += int(
                    bool(arm.get("false_block"))
                )

    contracts = [
        {
            "agent_radius_m": key[0],
            "agent_height_m": key[1],
            "sensor_height_m": key[2],
            "forward_step_m": key[3],
            "event_count": count,
        }
        for key, count in runtime_contracts.items()
    ]
    runtime_ok = bool(contracts) and all(
        abs(float(item["agent_radius_m"]) - 0.10) < 1e-6
        and abs(float(item["agent_height_m"]) - 1.50) < 1e-6
        and abs(float(item["sensor_height_m"]) - 1.25) < 1e-6
        and abs(float(item["forward_step_m"]) - 0.25) < 1e-6
        for item in contracts
    )
    integrity = bool(
        len(progress) >= len(manifest)
        and not errors
        and events
        and not integrity_violations
        and runtime_ok
    )
    return {
        "task": "stage58_support_policy",
        "schema_version": "stage58_support_policy_audit_v1",
        "expected_episode_count": len(manifest),
        "completed_episode_count": len(progress),
        "natural_candidate_event_count": len(events),
        "runtime_contracts": contracts,
        "runtime_contract_matches_r2r": runtime_ok,
        "offline_truth_valid_count": truth_valid,
        "offline_truth_safe_count": truth_safe,
        "directional_truth_valid_count": directional_truth_valid,
        "directional_truth_safe_count": directional_truth_safe,
        "directional_truth_unsafe_count": directional_truth_unsafe,
        "policy_arms": {
            key: dict(value) for key, value in sorted(policy_counts.items())
        },
        "integrity_passed": integrity,
        "shadow_only": True,
        "decision_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "errors": errors,
        "integrity_violations": integrity_violations,
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
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

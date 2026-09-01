#!/usr/bin/env python3
"""Aggregate Stage64 temporary recovery-subtask replacement shadows."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ARMS = {"control", "programmatic_subtask", "self_authored_subtask"}
RECOVERY_ARMS = EXPECTED_ARMS - {"control"}


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(run_root: Path, *, expected_episodes: int) -> dict:
    progress = _jsonl(run_root / "progress.json")
    paths = glob.glob(
        str(run_root / "vlmap_safety_debug" / "*run_*" / "stage64_recovery_subtask_events.jsonl")
    )
    events = [event for path in paths for event in _jsonl(Path(path))]
    by_arm = defaultdict(list)
    for event in events:
        for arm in list(event.get("arms") or []):
            by_arm[str(arm.get("variant") or "missing")].append((event, arm))

    arm_summary = {}
    for variant, rows in sorted(by_arm.items()):
        arms = [arm for _, arm in rows]
        joint_scenes = {
            str(event.get("scene_id"))
            for event, arm in rows
            if arm.get("joint_eligible")
        }
        arm_summary[variant] = {
            "event_count": len(arms),
            "ok_count": sum(arm.get("status") == "ok" for arm in arms),
            "protocol_valid_count": sum(bool(arm.get("protocol_valid")) for arm in arms),
            "pixel_valid_count": sum(bool(arm.get("pixel_valid")) for arm in arms),
            "endpoint_free_count": sum(
                (arm.get("waypoint_preflight") or {}).get("goal_state") == "free"
                for arm in arms
            ),
            "path_consistent_count": sum(bool(arm.get("path_consistent")) for arm in arms),
            "support_safe_count": sum(bool(arm.get("support_safe")) for arm in arms),
            "joint_eligible_count": sum(bool(arm.get("joint_eligible")) for arm in arms),
            "joint_eligible_scene_count": len(joint_scenes),
            "direct_turn_count": sum(bool(arm.get("direct_turn_direction")) for arm in arms),
            "stop_count": sum(bool(arm.get("is_stop")) for arm in arms),
            "output_counts": dict(Counter(str(arm.get("output")) for arm in arms)),
        }

    complete_events = [
        event
        for event in events
        if {str(arm.get("variant")) for arm in list(event.get("arms") or [])}
        == EXPECTED_ARMS
    ]
    violations = []
    for event in events:
        arms = list(event.get("arms") or [])
        reset = dict(event.get("queue_reset_plan") or {})
        valid = bool(
            event.get("reason") == "ok"
            and event.get("shadow_only") is True
            and event.get("decision_applied") is False
            and event.get("action_applied") is False
            and event.get("pixel_translation_allowed") is False
            and event.get("unknown_is_free") is False
            and not event.get("gt_fields_used")
            and event.get("official_memory_mutated") is False
            and event.get("original_instruction_restored") is True
            and event.get("original_instruction_leaked_to_recovery_prompt") is False
            and event.get("recovery_messages_prefix_empty") is True
            and event.get("queue_state_unchanged") is True
            and reset.get("shadow_only") is True
            and reset.get("applied") is False
            and event.get("authoring_prompt_image_binding_valid") is True
            and all(
                arm.get("status") in {"ok", "invalid_instruction"}
                for arm in arms
            )
            and all(arm.get("prompt_image_binding_valid") is True for arm in arms)
            and all(arm.get("action_applied") is False for arm in arms)
            and all(arm.get("pixel_translation_allowed") is False for arm in arms)
            and all(arm.get("unknown_is_free") is False for arm in arms)
        )
        if not valid:
            violations.append(
                [event.get("scene_id"), event.get("episode_id"), event.get("trigger_step")]
            )

    recovery_gate_rows = {
        arm: arm_summary.get(arm, {}) for arm in sorted(RECOVERY_ARMS)
    }
    best_joint = max(
        (int(row.get("joint_eligible_count", 0)) for row in recovery_gate_rows.values()),
        default=0,
    )
    best_scenes = max(
        (
            int(row.get("joint_eligible_scene_count", 0))
            for row in recovery_gate_rows.values()
        ),
        default=0,
    )
    integrity_passed = bool(
        len(progress) == int(expected_episodes)
        and events
        and len(complete_events) == len(events)
        and set(by_arm) == EXPECTED_ARMS
        and not violations
    )
    return {
        "task": "stage64_recovery_subtask",
        "schema_version": "stage64_recovery_subtask_v1",
        "expected_episode_count": int(expected_episodes),
        "completed_episode_count": len(progress),
        "event_count": len(events),
        "scene_count": len({str(event.get("scene_id")) for event in events}),
        "complete_three_arm_event_count": len(complete_events),
        "self_authored_instruction_count": sum(
            bool(event.get("self_authored_instruction_valid")) for event in events
        ),
        "arm_summary": arm_summary,
        "active_gate": {
            "required_joint_eligible": 4,
            "required_scenes": 4,
            "best_recovery_arm_joint_eligible": best_joint,
            "best_recovery_arm_scenes": best_scenes,
            "passed": bool(
                integrity_passed
                and any(
                    int(row.get("joint_eligible_count", 0)) >= 4
                    and int(row.get("joint_eligible_scene_count", 0)) >= 4
                    for row in recovery_gate_rows.values()
                )
            ),
        },
        "integrity_passed": integrity_passed,
        "shadow_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "integrity_violations": violations,
        "records": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_root, expected_episodes=args.expected_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

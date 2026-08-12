#!/usr/bin/env python3
"""Audit and summarize Stage21 representative stuck-decision snapshots."""

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_key(row):
    return str(row["scene_id"]), int(row["episode_id"])


def _diagnosis(snapshot):
    source = str(snapshot.get("dominant_action_source") or snapshot.get("action_source") or "unknown")
    current_action = snapshot.get("current_action")
    pre_safety_action = snapshot.get("pre_safety_action")
    reasons = []
    if "stop" in source:
        reasons.append("stop_handling_or_fallback")
    if source == "system2_action_queue":
        reasons.append("system2_action_queue_loop")
    if source in {"nextdit_local_queue", "nextdit_regenerated_local_queue"}:
        reasons.append("nextdit_local_queue_loop")
    if source == "vlmap_recovery_queue":
        reasons.append("recovery_queue_loop")
    if pre_safety_action is not None and current_action is not None and pre_safety_action != current_action:
        reasons.append("vlmap_safety_action_override")
    if int(snapshot.get("pixel_goal_age_steps") or 0) >= 32:
        reasons.append("stale_pixel_goal")
    if int(snapshot.get("s2_query_age_steps") or 0) >= 32:
        reasons.append("stale_s2_decision")
    if "map_or_pose_stagnation" in snapshot.get("trigger_reasons", []):
        reasons.append("map_or_pose_stagnation")
    if not reasons:
        reasons.append("needs_visual_inspection")
    return reasons


def _parse_expected_seeds(values):
    expected = {}
    for value in values:
        key, separator, raw_seed = value.rpartition("=")
        if not separator or "/" not in key:
            raise ValueError(f"Invalid --expected-seed value: {value!r}")
        scene_id, raw_episode_id = key.rsplit("/", 1)
        expected[(scene_id, int(raw_episode_id))] = int(raw_seed)
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--expected-seed", action="append", default=[])
    args = parser.parse_args()
    expected_seeds = _parse_expected_seeds(args.expected_seed)

    expected_rows = _read_json(args.episode_manifest)
    expected = {_episode_key(row) for row in expected_rows}
    progress_path = args.run_root / "progress.json"
    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    progress_by_key = {_episode_key(row): row for row in progress_rows}

    snapshots = []
    duplicate_keys = Counter()
    required_fields = (
        "action_source",
        "dominant_action_source",
        "environment_step_applied",
        "pixel_goal_age_steps",
        "s2_query_age_steps",
        "local_action_queue_length",
        "system2_action_queue_length",
        "episode_eval_seed",
    )
    for path in sorted(args.run_root.glob("vlmap_safety_debug/**/stuck_snapshots/*.json")):
        row = _read_json(path)
        key = _episode_key(row)
        duplicate_keys[key] += 1
        jpg_path = path.with_name(str(row.get("rgb_file") or path.with_suffix(".jpg").name))
        snapshots.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "step_id": row.get("step_id"),
                "json_path": str(path),
                "rgb_path": str(jpg_path),
                "rgb_exists": jpg_path.is_file(),
                "trigger_reasons": row.get("trigger_reasons", []),
                "current_action": row.get("current_action"),
                "dominant_action": row.get("dominant_action"),
                "dominant_action_ratio": row.get("dominant_action_ratio"),
                "action_source": row.get("action_source"),
                "dominant_action_source": row.get("dominant_action_source"),
                "environment_step_applied": row.get("environment_step_applied"),
                "pixel_goal": row.get("pixel_goal"),
                "pixel_goal_age_steps": row.get("pixel_goal_age_steps"),
                "last_s2_query_step": row.get("last_s2_query_step"),
                "s2_query_age_steps": row.get("s2_query_age_steps"),
                "local_action_queue_length": row.get("local_action_queue_length"),
                "system2_action_queue_length": row.get("system2_action_queue_length"),
                "episode_eval_seed": row.get("episode_eval_seed"),
                "missing_schema_fields": [field for field in required_fields if field not in row],
                "diagnostic_hypotheses": _diagnosis(row),
            }
        )

    snapshot_keys = {(row["scene_id"], row["episode_id"]) for row in snapshots}
    missing_progress = sorted(expected - set(progress_by_key))
    missing_snapshots = sorted(expected - snapshot_keys)
    unexpected_snapshots = sorted(snapshot_keys - expected)
    missing_rgb = [row["json_path"] for row in snapshots if not row["rgb_exists"]]
    duplicate_snapshot_episodes = [
        {"scene_id": key[0], "episode_id": key[1], "count": count}
        for key, count in sorted(duplicate_keys.items())
        if count > 1
    ]
    field_missing = {
        field: sum(field in row["missing_schema_fields"] for row in snapshots)
        for field in required_fields
    }
    seed_mismatches = []
    for row in snapshots:
        key = row["scene_id"], row["episode_id"]
        if key not in expected_seeds:
            continue
        actual_seed = row.get("episode_eval_seed")
        if actual_seed != expected_seeds[key]:
            seed_mismatches.append(
                {
                    "scene_id": key[0],
                    "episode_id": key[1],
                    "expected_seed": expected_seeds[key],
                    "actual_seed": actual_seed,
                }
            )
    episode_results = []
    for key in sorted(expected):
        progress = progress_by_key.get(key, {})
        episode_results.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "success": progress.get("success"),
                "spl": progress.get("spl"),
                "ne": progress.get("ne"),
                "steps": progress.get("steps"),
                "collision_count": progress.get("collision_count"),
                "failure_type": progress.get("stage19_semantic_resilience_episode_failure_type"),
                "snapshot_count": duplicate_keys.get(key, 0),
            }
        )

    passed = (
        not missing_progress
        and not missing_rgb
        and not duplicate_snapshot_episodes
        and not seed_mismatches
        and not unexpected_snapshots
        and (not args.require_all or not missing_snapshots)
        and all(count == 0 for count in field_missing.values())
    )
    summary = {
        "task": "stage21_stuck_snapshot_audit",
        "run_root": str(args.run_root),
        "episode_manifest": str(args.episode_manifest),
        "expected_episode_count": len(expected),
        "progress_episode_count": len(progress_rows),
        "snapshot_episode_count": len(snapshot_keys),
        "snapshot_count": len(snapshots),
        "coverage_rate": len(snapshot_keys & expected) / len(expected) if expected else 0.0,
        "missing_progress": missing_progress,
        "missing_snapshots": missing_snapshots,
        "unexpected_snapshots": unexpected_snapshots,
        "missing_rgb": missing_rgb,
        "duplicate_snapshot_episodes": duplicate_snapshot_episodes,
        "required_field_missing_counts": field_missing,
        "expected_episode_seeds": {
            f"{key[0]}/{key[1]}": seed for key, seed in sorted(expected_seeds.items())
        },
        "seed_mismatches": seed_mismatches,
        "episode_results": episode_results,
        "snapshots": snapshots,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

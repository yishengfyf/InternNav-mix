#!/usr/bin/env python3
"""Audit complete Habitat sensor-pose endpoint and observed-volume OCC metrics."""

import argparse
import json
from pathlib import Path


def _read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _key(row):
    return str(row.get("scene_id")), int(row.get("episode_id"))


def _load_unique(paths, event_type=None):
    rows = {}
    for path in paths:
        for row in _read_jsonl(path):
            if event_type and row.get("event_type") != event_type:
                continue
            rows[_key(row)] = row
    return rows


def _load_manifest(path):
    return {_key(row): row for row in json.loads(path.read_text(encoding="utf-8"))}


def _find_comparison(run_root, key):
    for path in run_root.glob(
        "vlmap_safety_debug/rank*_run_*/stage23a_sensor_occ_comparison/*.json"
    ):
        if path.stem == f"{key[0]}_{key[1]}_comparison":
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def analyze(run_root, manifest, output, require_all):
    expected = _load_manifest(manifest)
    progress = _load_unique(run_root.glob("vlmap_safety_debug/rank*_run_*/progress.json"))
    current = _load_unique(
        run_root.glob("vlmap_safety_debug/rank*_run_*/occ_memory/memory_episode_summary.jsonl"),
        "occ_memory_episode_summary",
    )
    oracle = _load_unique(
        run_root.glob(
            "vlmap_safety_debug/rank*_run_*/stage23a_oracle_sensor_pose/occ_memory/memory_episode_summary.jsonl"
        ),
        "occ_memory_episode_summary",
    )
    errors = []
    episodes = []
    action_fields = (
        "occ_memory_stage21_multitask_shadow_action_applied_count",
        "s2_loop_path_reobserve_applied_count",
        "s2_loop_path_reobserve_intervention_count",
        "s2_loop_path_reobserve_pixel_rewrite_count",
        "s2_loop_strict_active_applied_count",
        "s2_loop_strict_active_rewrite_count",
        "stage19_semantic_resilience_active_applied_count",
    )
    for key, manifest_row in expected.items():
        p, c, o = progress.get(key), current.get(key), oracle.get(key)
        if p is None or c is None or o is None:
            errors.append(f"missing_rows:{key}")
            continue
        if int(p.get("episode_eval_seed", -1)) != int(manifest_row["episode_eval_seed"]):
            errors.append(f"seed_mismatch:{key}")
        if int(c.get("update_count", -1)) != int(o.get("update_count", -2)):
            errors.append(f"branch_update_mismatch:{key}")
        if not p.get("stage23a_oracle_sensor_pose_audit_enabled"):
            errors.append(f"sensor_branch_disabled:{key}")
        if p.get("stage23a_gt_fields_used_for_navigation"):
            errors.append(f"gt_navigation_leakage:{key}")
        violations = sum(int(p.get(name, 0) or 0) for name in action_fields)
        if violations:
            errors.append(f"shadow_action_violation:{key}:{violations}")
        comparison = _find_comparison(run_root, key)
        if comparison is None:
            errors.append(f"missing_sensor_comparison:{key}")
        endpoint = c.get("validation_endpoint_gt_error_stats") or {}
        if not endpoint.get("count"):
            errors.append(f"missing_endpoint_gt_stats:{key}")
        episodes.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "audit_role": manifest_row.get("audit_role"),
                "episode_eval_seed": p.get("episode_eval_seed"),
                "steps": p.get("steps"),
                "success": p.get("success"),
                "current_update_count": c.get("update_count"),
                "oracle_sensor_update_count": o.get("update_count"),
                "pose_xy_error_stats": c.get("validation_pose_gt_xy_error_stats"),
                "pose_z_error_stats": c.get("validation_pose_gt_z_error_stats"),
                "pose_yaw_error_stats": c.get("validation_pose_gt_yaw_error_deg_stats"),
                "endpoint_error_stats": endpoint,
                "endpoint_error_groups": c.get("validation_endpoint_gt_error_groups"),
                "endpoint_mapped_ratio": c.get("validation_endpoint_mapped_ratio"),
                "comparison": comparison,
                "shadow_action_violation_count": violations,
            }
        )
    report = {
        "audit_name": "stage23a_sensor_pose_occ",
        "integrity_passed": not errors,
        "expected_episode_count": len(expected),
        "completed_episode_count": len(episodes),
        "gt_navigation_leakage_count": sum(
            bool(row.get("stage23a_gt_fields_used_for_navigation"))
            for row in progress.values()
        ),
        "shadow_action_violation_count": sum(
            int(row["shadow_action_violation_count"]) for row in episodes
        ),
        "episodes": episodes,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if require_all and errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    analyze(args.run_root, args.manifest, args.output, args.require_all)


if __name__ == "__main__":
    main()

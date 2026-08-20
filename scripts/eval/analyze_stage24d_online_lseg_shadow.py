"""Audit Stage24D online LSeg outputs and Frozen-S2 replay invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _jsonl(path: Path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ledgers(root: Path):
    return {
        (str(meta.get("scene_id")), str(meta.get("episode_id"))): path.parent
        for path in root.glob("**/replay_ledger/*/episode_meta.json")
        for meta in [json.loads(path.read_text(encoding="utf-8"))]
    }


def _semantic_dirs(root: Path):
    return {
        (str(meta.get("scene_id")), str(meta.get("episode_id"))): path.parent
        for path in root.glob("**/online_lseg_shadow/*/episode_meta.json")
        for meta in [json.loads(path.read_text(encoding="utf-8"))]
    }


def _compare_ledgers(current: Path, baseline: Path):
    mismatches = []
    for name in ("queries.jsonl", "actions.jsonl"):
        left = _jsonl(current / name)
        right = _jsonl(baseline / name)
        if len(left) != len(right):
            mismatches.append(f"{name}:count:{len(left)}!={len(right)}")
            continue
        keys = (
            ("step_id", "output", "pixel_goal", "input_steps")
            if name == "queries.jsonl"
            else ("step_id", "action", "action_source", "pre_safety_action", "action_applied")
        )
        for index, (a, b) in enumerate(zip(left, right)):
            for key in keys:
                if a.get(key) != b.get(key):
                    mismatches.append(f"{name}:{index}:{key}")
    left_obs = _jsonl(current / "observations.jsonl")
    right_obs = _jsonl(baseline / "observations.jsonl")
    if len(left_obs) != len(right_obs):
        mismatches.append(f"observations.jsonl:count:{len(left_obs)}!={len(right_obs)}")
    else:
        for index, (a, b) in enumerate(zip(left_obs, right_obs)):
            for key in ("step_id", "previous_action", "previous_action_source", "previous_action_applied"):
                if a.get(key) != b.get(key):
                    mismatches.append(f"observations.jsonl:{index}:{key}")
            for key in ("gps", "compass", "stage23_gt_camera_pose_map"):
                left_value = np.asarray((a.get("pose") or {}).get(key))
                right_value = np.asarray((b.get("pose") or {}).get(key))
                if left_value.shape != right_value.shape or not np.allclose(
                    left_value, right_value, atol=1e-6, rtol=0.0
                ):
                    mismatches.append(f"observations.jsonl:{index}:pose.{key}")
    return mismatches


def analyze(run_root: Path, baseline_root: Path, output: Path, require_all: bool):
    errors = []
    current_ledgers = _ledgers(run_root)
    baseline_ledgers = _ledgers(baseline_root)
    semantic_dirs = _semantic_dirs(run_root)
    episodes = []
    for key in sorted(current_ledgers):
        ledger = current_ledgers[key]
        semantic_dir = semantic_dirs.get(key)
        if semantic_dir is None:
            errors.append(f"missing_online_lseg:{key[0]}/{key[1]}")
            continue
        summary_path = semantic_dir / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing_lseg_summary:{key[0]}/{key[1]}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        queries = _jsonl(ledger / "queries.jsonl")
        if int(summary.get("valid_frame_count", -1)) != len(queries):
            errors.append(f"query_frame_mismatch:{key[0]}/{key[1]}")
        if int(summary.get("error_count", -1)) != 0:
            errors.append(f"lseg_error:{key[0]}/{key[1]}")
        if int(summary.get("action_applied_count", -1)) != 0:
            errors.append(f"lseg_action_violation:{key[0]}/{key[1]}")
        if summary.get("decision_status") != "audit_only_not_navigation_ready":
            errors.append(f"decision_status_violation:{key[0]}/{key[1]}")
        for relative in (summary.get("visualizations") or {}).values():
            if not (semantic_dir / relative).is_file():
                errors.append(f"missing_visualization:{key[0]}/{key[1]}:{relative}")
        mismatch = []
        baseline = baseline_ledgers.get(key)
        if baseline is None:
            errors.append(f"missing_baseline:{key[0]}/{key[1]}")
        else:
            mismatch = _compare_ledgers(ledger, baseline)
            errors.extend(f"trajectory_mismatch:{key[0]}/{key[1]}:{item}" for item in mismatch)
        cuda_baseline = summary.get("cuda_s2_loaded_baseline") or {}
        cuda_before_load = summary.get("cuda_immediately_before_lseg_load") or {}
        cuda_after = summary.get("cuda_after_lseg_load") or {}
        joint_peaks = [
            (record.get("cuda_after") or {}).get("max_allocated_mb")
            for record in _jsonl(semantic_dir / "events.jsonl")
            if record.get("valid")
        ]
        joint_peaks = [float(value) for value in joint_peaks if value is not None]
        episodes.append({
            "scene_id": key[0], "episode_id": key[1],
            "query_count": len(queries),
            "valid_lseg_frame_count": summary.get("valid_frame_count"),
            "trajectory_exact_match": not mismatch and baseline is not None,
            "inference_seconds_mean": summary.get("inference_seconds_mean"),
            "inference_seconds_p95": summary.get("inference_seconds_p95"),
            "stored_surface_sample_count": summary.get("stored_surface_sample_count"),
            "node_count": summary.get("node_count"),
            "multi_view_node_rate": summary.get("multi_view_node_rate"),
            "cross_label_conflict_count": summary.get("cross_label_conflict_count"),
            "gt_audit": summary.get("gt_audit"),
            "cuda_s2_loaded_allocated_mb": cuda_baseline.get("allocated_mb"),
            "cuda_immediately_before_lseg_load_allocated_mb": cuda_before_load.get("allocated_mb"),
            "cuda_after_lseg_load_allocated_mb": cuda_after.get("allocated_mb"),
            "cuda_lseg_load_increment_mb": (
                float(cuda_after["allocated_mb"]) - float(cuda_before_load["allocated_mb"])
                if cuda_after.get("allocated_mb") is not None
                and cuda_before_load.get("allocated_mb") is not None else None
            ),
            "cuda_joint_forward_peak_allocated_mb": max(joint_peaks) if joint_peaks else None,
            "semantic_dir": str(semantic_dir),
        })
    if require_all and set(current_ledgers) != set(baseline_ledgers):
        missing = sorted(set(baseline_ledgers) - set(current_ledgers))
        errors.extend(f"missing_episode:{scene}/{episode}" for scene, episode in missing)
    result = {
        "audit_name": "stage24d_online_lseg_shadow",
        "integrity_passed": not errors,
        "episode_count": len(episodes),
        "all_trajectories_exact_match": bool(episodes) and all(
            item["trajectory_exact_match"] for item in episodes
        ),
        "errors": errors, "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    analyze(args.run_root, args.baseline_root, args.output, args.require_all)


if __name__ == "__main__":
    main()

"""Audit and render recovery-specific LSeg/SparseOcc route attachments."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
import copy
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.analyze_stage24d_online_lseg_shadow import _ledgers
from scripts.eval.visualize_stage38_recovery_bev import render_recovery_bev


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows(root: Path, name: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path_text in glob.glob(str(root / "**" / name), recursive=True):
        if path_text in seen:
            continue
        seen.add(path_text)
        rows.extend(_jsonl(Path(path_text)))
    return rows


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_stage78(value: Any) -> Any:
    """Remove only Stage78 audit payloads before behavior-ledger comparison."""
    if isinstance(value, dict):
        return {
            key: _without_stage78(item)
            for key, item in value.items()
            if not str(key).startswith("stage78_")
        }
    if isinstance(value, list):
        return [_without_stage78(item) for item in value]
    return copy.deepcopy(value)


def _compare_behavior_ledgers(current: Path, baseline: Path) -> list[str]:
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
        for index, (left_row, right_row) in enumerate(zip(left, right)):
            for key in keys:
                left_value = _without_stage78(left_row.get(key))
                right_value = _without_stage78(right_row.get(key))
                if left_value != right_value:
                    mismatches.append(f"{name}:{index}:{key}")
    left_obs = _jsonl(current / "observations.jsonl")
    right_obs = _jsonl(baseline / "observations.jsonl")
    if len(left_obs) != len(right_obs):
        mismatches.append(f"observations.jsonl:count:{len(left_obs)}!={len(right_obs)}")
    else:
        for index, (left_row, right_row) in enumerate(zip(left_obs, right_obs)):
            for key in ("step_id", "previous_action", "previous_action_source", "previous_action_applied"):
                if left_row.get(key) != right_row.get(key):
                    mismatches.append(f"observations.jsonl:{index}:{key}")
            for key in ("gps", "compass", "stage23_gt_camera_pose_map"):
                left_value = np.asarray((left_row.get("pose") or {}).get(key))
                right_value = np.asarray((right_row.get("pose") or {}).get(key))
                if left_value.shape != right_value.shape or not np.allclose(
                    left_value, right_value, atol=1e-6, rtol=0.0
                ):
                    mismatches.append(f"observations.jsonl:{index}:pose.{key}")
    return mismatches


def analyze(
    *, run_root: Path, baseline_root: Path, manifest: Path,
    output: Path, bev_dir: Path,
) -> dict[str, Any]:
    expected_rows = _load_json(manifest)
    expected = {
        (str(row["scene_id"]), str(row["episode_id"])): row
        for row in expected_rows
    }
    errors = []
    progress_path = run_root / "progress.json"
    progress = _jsonl(progress_path)
    if len(progress) != len(expected):
        errors.append(f"progress_count:{len(progress)}!={len(expected)}")

    ledgers = _ledgers(run_root)
    baseline_ledgers = _ledgers(baseline_root)
    trajectory_reports = []
    for key in expected:
        current = ledgers.get(key)
        baseline = baseline_ledgers.get(key)
        mismatch = ["missing_current_ledger"] if current is None else []
        if baseline is None:
            mismatch.append("missing_baseline_ledger")
        elif current is not None:
            mismatch.extend(_compare_behavior_ledgers(current, baseline))
        trajectory_reports.append({
            "scene_id": key[0], "episode_id": key[1],
            "exact_match": not mismatch, "mismatch": mismatch,
        })
        errors.extend(f"{key[0]}/{key[1]}:{item}" for item in mismatch)

    context_rows = _rows(run_root, "s2_recovery_context_events.jsonl")
    route_events = [
        row for row in context_rows
        if row.get("event_type") == "stage75_route_guidance"
    ]
    attached_events = [
        row for row in route_events
        if isinstance(row.get("stage78_semantic_route_attachment"), dict)
    ]
    if not attached_events:
        errors.append("missing_stage78_attachment_events")

    summaries = []
    for path_text in glob.glob(
        str(run_root / "**" / "online_lseg_shadow" / "*" / "summary.json"),
        recursive=True,
    ):
        summaries.append(_load_json(Path(path_text)))
    summary_keys = {
        (str(row.get("scene_id")), str(row.get("episode_id"))) for row in summaries
    }
    missing_summaries = sorted(set(expected) - summary_keys)
    errors.extend(f"{scene}/{episode}:missing_lseg_summary" for scene, episode in missing_summaries)
    for row in summaries:
        label = f"{row.get('scene_id')}/{row.get('episode_id')}"
        if row.get("decision_status") != "audit_only_not_navigation_ready":
            errors.append(f"{label}:decision_status_violation")
        if int(row.get("action_applied_count", -1)) != 0:
            errors.append(f"{label}:semantic_action_consumer_violation")
        if int(row.get("error_count", -1)) != 0:
            errors.append(f"{label}:lseg_error")

    label_counts: Counter[str] = Counter()
    stable_label_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    structural_counts: Counter[str] = Counter()
    episode_stable: defaultdict[tuple[str, str], int] = defaultdict(int)
    event_reports = []
    rendered = []
    rendered_per_episode: Counter[tuple[str, str]] = Counter()
    for event in attached_events:
        key = (str(event.get("scene_id")), str(event.get("episode_id")))
        attachment = dict(event["stage78_semantic_route_attachment"])
        if attachment.get("unknown_is_free") is not False:
            errors.append(f"{key[0]}/{key[1]}:unknown_contract_violation")
        if attachment.get("semantic_can_override_safety") is not False:
            errors.append(f"{key[0]}/{key[1]}:semantic_override_violation")
        if attachment.get("prompt_injected") is not False:
            errors.append(f"{key[0]}/{key[1]}:prompt_injection_violation")
        label_counts.update(attachment.get("label_counts") or {})
        stable_label_counts.update(attachment.get("stable_label_counts") or {})
        state_counts.update(attachment.get("occ_state_at_centroid_counts") or {})
        stable = int(attachment.get("stable_route_bound_node_count", 0) or 0)
        episode_stable[key] += stable
        for node in attachment.get("route_bound_nodes") or []:
            if node.get("structural_label"):
                structural_counts[str(node.get("label") or "other")] += 1
        event_reports.append({
            "scene_id": key[0], "episode_id": key[1],
            "step_id": event.get("current_query_step"),
            "route_valid": bool(event.get("valid")),
            "route_reason": event.get("reason"),
            "semantic_node_count": attachment.get("semantic_node_count"),
            "route_bound_node_count": attachment.get("route_bound_node_count"),
            "stable_route_bound_node_count": stable,
            "structural_route_bound_node_count": attachment.get(
                "structural_route_bound_node_count"
            ),
            "route_bound_nodes": attachment.get("route_bound_nodes") or [],
        })
        spatial = event.get("stage78_recovery_bev_spatial")
        if not isinstance(spatial, dict) or rendered_per_episode[key] >= 2:
            continue
        safe_name = f"{key[0]}_{key[1]}_q{event.get('current_query_step')}"
        png = bev_dir / f"{safe_name}.png"
        anchor = {
            "anchor_id": safe_name,
            "capture": {
                "pose": event.get("start_grid"),
                "candidate_grid": event.get("anchor_grid"),
                "path_cells": event.get("path_preview") or [],
            },
        }
        meta = render_recovery_bev(anchor, spatial, png, scale=8)
        rendered.append(meta)
        rendered_per_episode[key] += 1

    stable_episode_keys = sorted(key for key, count in episode_stable.items() if count > 0)
    result = {
        "task": "stage78_semantic_attachment_shadow",
        "schema_version": "stage78_semantic_attachment_audit_v1",
        "integrity_passed": not errors,
        "errors": errors,
        "expected_episode_count": len(expected),
        "completed_episode_count": len(progress),
        "trajectory_exact_match": bool(trajectory_reports) and all(
            row["exact_match"] for row in trajectory_reports
        ),
        "trajectory_reports": trajectory_reports,
        "lseg_summary_count": len(summaries),
        "lseg_error_count": sum(int(row.get("error_count", 0) or 0) for row in summaries),
        "route_guidance_event_count": len(route_events),
        "attachment_event_count": len(attached_events),
        "attachment_valid_route_count": sum(
            bool((row.get("stage78_semantic_route_attachment") or {}).get("valid"))
            for row in attached_events
        ),
        "stable_route_landmark_episode_count": len(stable_episode_keys),
        "stable_route_landmark_episode_keys": [list(key) for key in stable_episode_keys],
        "semantic_label_counts": dict(sorted(label_counts.items())),
        "stable_semantic_label_counts": dict(sorted(stable_label_counts.items())),
        "semantic_centroid_occ_state_counts": dict(sorted(state_counts.items())),
        "route_bound_structural_label_counts": dict(sorted(structural_counts.items())),
        "event_reports": event_reports,
        "bev_render_count": len(rendered),
        "bev_render_manifest": rendered,
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "semantic_prompt_injected": False,
        "semantic_action_applied_count": 0,
        "gt_used_for_navigation": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bev-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        run_root=args.run_root,
        baseline_root=args.baseline_root,
        manifest=args.manifest,
        output=args.output,
        bev_dir=args.bev_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Audit Stage22A executed pose-chain evidence against ray-derived OCC."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


METRICS = ("success", "spl", "ne", "steps", "collision_count")


def _read_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row):
    return f"{row.get('scene_id')}/{int(row.get('episode_id'))}"


def _progress(run_root: Path):
    return {
        _key(row): row
        for row in _read_jsonl(run_root / "progress.json")
        if row.get("episode_id") is not None
    }


def _events(run_root: Path, filename: str):
    paths = glob.glob(str(run_root / "vlmap_safety_debug" / "*" / filename))
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _loop_signatures(run_root: Path, *, strict_only: bool = False):
    signatures = defaultdict(list)
    for row in _events(run_root, "s2_action_loop_events.jsonl"):
        if row.get("transition") != "start":
            continue
        if strict_only and row.get("triage_tier") != "strict_intervention":
            continue
        signatures[_key(row)].append(
            (
                int(row.get("step_id", -1)),
                int(row.get("start_step", -1)),
                str(row.get("turn_direction")),
                str(row.get("triage_tier")),
            )
        )
    return {key: sorted(values) for key, values in signatures.items()}


def _seed_manifest(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['scene_id']}/{int(row['episode_id'])}": int(
            row["episode_eval_seed"]
        )
        for row in rows
    }


def _mean(values):
    return None if not values else float(sum(values) / len(values))


def analyze(
    run_root: Path,
    expected_episodes: int,
    seed_manifest: Path,
    reference_root: Path,
):
    progress = _progress(run_root)
    reference_progress = _progress(reference_root)
    expected_seeds = _seed_manifest(seed_manifest)
    loops = _loop_signatures(run_root)
    reference_loops = _loop_signatures(reference_root)
    reference_strict = _loop_signatures(reference_root, strict_only=True)
    events = _events(run_root, "s2_loop_executed_route_occ_audit_events.jsonl")
    expected_event_count = sum(
        len(reference_strict.get(key, [])) for key in expected_seeds
    )

    seed_mismatches = [
        {
            "scene_episode": key,
            "expected_episode_eval_seed": seed,
            "actual_episode_eval_seed": progress.get(key, {}).get(
                "episode_eval_seed"
            ),
        }
        for key, seed in expected_seeds.items()
        if progress.get(key, {}).get("episode_eval_seed") != seed
    ]
    reference_metric_mismatches = []
    reference_loop_mismatches = []
    for key in sorted(expected_seeds):
        row = progress.get(key)
        reference = reference_progress.get(key)
        if row is None or reference is None:
            reference_metric_mismatches.append(
                {"scene_episode": key, "reason": "missing_progress_or_reference"}
            )
            continue
        differing = {}
        for field in METRICS:
            actual = float(row.get(field, 0.0) or 0.0)
            expected = float(reference.get(field, 0.0) or 0.0)
            if abs(actual - expected) > 1e-6:
                differing[field] = {"actual": actual, "reference": expected}
        if differing:
            reference_metric_mismatches.append(
                {"scene_episode": key, "fields": differing}
            )
        if loops.get(key, []) != reference_loops.get(key, []):
            reference_loop_mismatches.append(
                {
                    "scene_episode": key,
                    "actual_loop_signatures": loops.get(key, []),
                    "reference_loop_signatures": reference_loops.get(key, []),
                }
            )

    nested_audits = [
        dict(row.get("audit") or {})
        for row in events
        if isinstance(row.get("audit"), dict)
    ]
    action_violations = [
        row
        for row in events
        if bool(row.get("action_applied"))
        or bool((row.get("audit") or {}).get("action_applied"))
        or bool(row.get("output_rewritten"))
        or bool((row.get("audit") or {}).get("output_rewritten"))
    ]
    action_violations.extend(
        row
        for row in progress.values()
        if int(row.get("s2_loop_strict_active_applied_count", 0) or 0) > 0
        or int(row.get("s2_loop_path_reobserve_applied_count", 0) or 0) > 0
        or int(row.get("stage19_semantic_resilience_active_applied_count", 0) or 0)
        > 0
    )
    gt_leakage = [
        row
        for row in events
        if list(row.get("gt_fields_used") or [])
        or list((row.get("audit") or {}).get("gt_fields_used") or [])
    ]
    violations = {
        "seed_replay_mismatch": seed_mismatches,
        "reference_metric_mismatch": reference_metric_mismatches,
        "reference_loop_mismatch": reference_loop_mismatches,
        "action_or_output_applied": action_violations,
        "gt_leakage": gt_leakage,
        "non_shadow_event": [
            row
            for row in events
            if not bool(row.get("shadow_only"))
            or not bool((row.get("audit") or {}).get("shadow_only"))
        ],
        "non_strict_event": [
            row for row in events if row.get("triage_tier") != "strict_intervention"
        ],
        "missing_audit": [
            row for row in events if not isinstance(row.get("audit"), dict)
        ],
        "invalid_audit": [
            row
            for row in events
            if isinstance(row.get("audit"), dict)
            and not bool(row["audit"].get("valid"))
        ],
        "source_anchor_pose_mismatch": [
            row
            for row in events
            if isinstance(row.get("audit"), dict)
            and not bool(row["audit"].get("source_anchor_pose_match"))
        ],
    }
    integrity_passed = bool(
        len(progress) == expected_episodes
        and len(expected_seeds) == expected_episodes
        and len(events) == expected_event_count
        and expected_event_count > 0
        and not any(violations.values())
    )

    route_reason_counts = Counter(
        str(audit.get("reason") or "missing") for audit in nested_audits
    )
    connectivity_reason_counts = Counter(
        str((audit.get("known_free_connectivity") or {}).get("reason") or "missing")
        for audit in nested_audits
    )
    continuous = [
        audit for audit in nested_audits if bool(audit.get("route_chain_continuous"))
    ]
    ray_reachable = [
        audit
        for audit in nested_audits
        if bool((audit.get("known_free_connectivity") or {}).get("reachable"))
    ]
    continuous_disconnected = [
        audit
        for audit in nested_audits
        if bool(audit.get("continuous_but_ray_disconnected"))
    ]
    occupied_conflicts = [
        audit
        for audit in nested_audits
        if int((audit.get("route_cell_state_counts") or {}).get("occupied", 0)) > 0
    ]
    unknown_gaps = [
        audit
        for audit in nested_audits
        if int((audit.get("route_cell_state_counts") or {}).get("unknown", 0)) > 0
    ]
    ratio_values = {
        state: [
            float(audit["route_cell_state_ratios"][state])
            for audit in nested_audits
            if (audit.get("route_cell_state_ratios") or {}).get(state) is not None
        ]
        for state in ("free", "unknown", "occupied")
    }
    height_values = {
        name: [
            int(
                (audit.get("route_occupied_height_diagnostics") or {}).get(
                    name, 0
                )
                or 0
            )
            for audit in nested_audits
        ]
        for name in (
            "occupied_route_cell_count",
            "low_or_ground_conflict_cell_count",
            "lower_obstacle_conflict_cell_count",
            "body_obstacle_conflict_cell_count",
            "obstacle_band_conflict_cell_count",
            "high_conflict_cell_count",
            "no_voxel_conflict_cell_count",
        )
    }

    event_records = []
    for row in events:
        audit = dict(row.get("audit") or {})
        connectivity = dict(audit.get("known_free_connectivity") or {})
        event_records.append(
            {
                "scene_episode": _key(row),
                "step_id": row.get("step_id"),
                "episode_eval_seed": row.get("episode_eval_seed"),
                "source_step": audit.get("source_step"),
                "anchor_grid": audit.get("anchor_grid"),
                "trigger_pose_grid": audit.get("trigger_pose_grid"),
                "source_anchor_pose_match": audit.get("source_anchor_pose_match"),
                "route_raw_pose_count": audit.get("route_raw_pose_count"),
                "route_translation_node_count": audit.get(
                    "route_translation_node_count"
                ),
                "route_movement_edge_count": audit.get(
                    "route_movement_edge_count"
                ),
                "route_length_m": audit.get("route_length_m"),
                "route_max_edge_m": audit.get("route_max_edge_m"),
                "route_chain_continuous": audit.get("route_chain_continuous"),
                "route_cell_state_counts": audit.get("route_cell_state_counts"),
                "route_cell_state_ratios": audit.get("route_cell_state_ratios"),
                "route_occupied_height_diagnostics": audit.get(
                    "route_occupied_height_diagnostics"
                ),
                "route_pitch_observation_count": audit.get(
                    "route_pitch_observation_count"
                ),
                "route_max_camera_pitch_deg": audit.get(
                    "route_max_camera_pitch_deg"
                ),
                "longest_unknown_gap_m": audit.get("longest_unknown_gap_m"),
                "first_occupied_conflict": audit.get("first_occupied_conflict"),
                "first_unknown_gap": audit.get("first_unknown_gap"),
                "ray_free_reachable": connectivity.get("reachable"),
                "ray_free_reason": connectivity.get("reason"),
                "continuous_but_ray_disconnected": audit.get(
                    "continuous_but_ray_disconnected"
                ),
            }
        )

    return {
        "task": "stage22a_executed_route_occ_shadow_audit",
        "expected_episode_count": expected_episodes,
        "completed_episode_count": len(progress),
        "seed_replay_expected_count": len(expected_seeds),
        "seed_replay_verified_count": len(expected_seeds) - len(seed_mismatches),
        "reference_metric_verified_count": expected_episodes
        - len(reference_metric_mismatches),
        "reference_loop_verified_count": expected_episodes
        - len(reference_loop_mismatches),
        "expected_strict_event_count": expected_event_count,
        "route_audit_event_count": len(events),
        "valid_route_audit_count": sum(bool(audit.get("valid")) for audit in nested_audits),
        "source_anchor_pose_match_count": sum(
            bool(audit.get("source_anchor_pose_match")) for audit in nested_audits
        ),
        "route_chain_continuous_count": len(continuous),
        "ray_free_reachable_count": len(ray_reachable),
        "continuous_but_ray_disconnected_count": len(continuous_disconnected),
        "route_with_occupied_conflict_count": len(occupied_conflicts),
        "route_with_unknown_gap_count": len(unknown_gaps),
        "route_reason_counts": dict(route_reason_counts),
        "ray_free_connectivity_reason_counts": dict(connectivity_reason_counts),
        "route_cell_free_ratio_mean": _mean(ratio_values["free"]),
        "route_cell_unknown_ratio_mean": _mean(ratio_values["unknown"]),
        "route_cell_occupied_ratio_mean": _mean(ratio_values["occupied"]),
        "route_occupied_height_diagnostics_sum": {
            name: int(sum(values)) for name, values in height_values.items()
        },
        "route_length_mean_m": _mean(
            [float(audit.get("route_length_m", 0.0)) for audit in nested_audits]
        ),
        "route_max_edge_mean_m": _mean(
            [float(audit.get("route_max_edge_m", 0.0)) for audit in nested_audits]
        ),
        "integrity_passed": integrity_passed,
        "measurement_complete": bool(integrity_passed and len(events) == expected_event_count),
        "violations": {name: len(rows) for name, rows in violations.items()},
        "violation_records": violations,
        "event_records": event_records,
        "interpretation_guard": (
            "A continuous executed pose chain proves historical connectivity evidence, not "
            "current geometric traversability. Unknown or occupied route cells diagnose the "
            "ray-derived OCC gap and must not be promoted to free without an independent gate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        args.run_root,
        args.expected_episodes,
        args.seed_manifest,
        args.reference_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_all and not summary["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

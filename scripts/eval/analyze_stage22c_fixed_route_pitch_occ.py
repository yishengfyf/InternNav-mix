"""Audit the same Stage22A routes on Stage22C pitch-aware OCC."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from scripts.eval.analyze_stage22a_executed_route_occ_audit import (
    METRICS,
    _events,
    _key,
    _mean,
    _progress,
    _read_jsonl,
    _seed_manifest,
)


def _event_key(row):
    return (_key(row), int(row.get("step_id", -1)))


def _audit_events(root: Path, filename: str):
    return {
        _event_key(row): row
        for row in _events(root, filename)
        if isinstance(row.get("audit"), dict)
    }


def _fixed_manifest(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (f"{row['scene_id']}/{int(row['episode_id'])}", int(row["step_id"])): row
        for row in rows
    }


def _loop_starts(root: Path):
    return {
        _event_key(row): row
        for row in _events(root, "s2_action_loop_events.jsonl")
        if row.get("transition") == "start"
    }


def _memory_updates(root: Path):
    return [
        row
        for path in glob.glob(
            str(
                root
                / "vlmap_safety_debug"
                / "*"
                / "occ_memory"
                / "memory_events.jsonl"
            )
        )
        for row in _read_jsonl(Path(path))
        if row.get("event_type") == "occ_memory_update" and row.get("valid")
    ]


def analyze(
    run_root: Path,
    expected_episodes: int,
    seed_manifest: Path,
    navigation_reference_root: Path,
    stage22a_root: Path,
    fixed_route_manifest: Path,
    expected_fixed_routes: int = 12,
):
    progress = _progress(run_root)
    reference_progress = _progress(navigation_reference_root)
    expected_seeds = _seed_manifest(seed_manifest)
    fixed_entries = _fixed_manifest(fixed_route_manifest)
    current_events = _audit_events(
        run_root, "s2_loop_fixed_route_occ_audit_events.jsonl"
    )
    baseline_events = _audit_events(
        stage22a_root, "s2_loop_executed_route_occ_audit_events.jsonl"
    )
    current_loops = _loop_starts(run_root)
    baseline_loops = _loop_starts(stage22a_root)
    memory_updates = _memory_updates(run_root)

    seed_mismatches = [
        {
            "scene_episode": key,
            "expected": seed,
            "actual": progress.get(key, {}).get("episode_eval_seed"),
        }
        for key, seed in expected_seeds.items()
        if progress.get(key, {}).get("episode_eval_seed") != seed
    ]
    metric_mismatches = []
    for key in sorted(expected_seeds):
        actual = progress.get(key)
        reference = reference_progress.get(key)
        if actual is None or reference is None:
            metric_mismatches.append(
                {"scene_episode": key, "reason": "missing_progress_or_reference"}
            )
            continue
        differing = {}
        for field in METRICS:
            lhs = float(actual.get(field, 0.0) or 0.0)
            rhs = float(reference.get(field, 0.0) or 0.0)
            if abs(lhs - rhs) > 1e-6:
                differing[field] = {"actual": lhs, "reference": rhs}
        if differing:
            metric_mismatches.append(
                {"scene_episode": key, "fields": differing}
            )

    raw_loop_mismatches = []
    dynamic_failure_type_transitions = Counter()
    if set(current_loops) != set(baseline_loops):
        raw_loop_mismatches.append(
            {
                "reason": "loop_event_key_mismatch",
                "missing_current": sorted(set(baseline_loops) - set(current_loops)),
                "missing_baseline": sorted(set(current_loops) - set(baseline_loops)),
            }
        )
    for key in sorted(set(current_loops) & set(baseline_loops)):
        current = current_loops[key]
        baseline = baseline_loops[key]
        fields = {
            name: {"stage22a": baseline.get(name), "stage22c": current.get(name)}
            for name in ("start_step", "turn_direction")
            if baseline.get(name) != current.get(name)
        }
        if baseline.get("failure_type") != current.get("failure_type"):
            dynamic_failure_type_transitions[
                f"{baseline.get('failure_type')}->{current.get('failure_type')}"
            ] += 1
        if fields:
            raw_loop_mismatches.append(
                {"scene_episode": key[0], "step_id": key[1], "fields": fields}
            )

    expected_keys = set(fixed_entries)
    missing_current = sorted(expected_keys - set(current_events))
    extra_current = sorted(set(current_events) - expected_keys)
    missing_baseline = sorted(expected_keys - set(baseline_events))
    extra_baseline = sorted(set(baseline_events) - expected_keys)
    route_identity_mismatches = []
    invalid_audits = []
    reference_mismatches = []
    action_or_output_violations = []
    gt_leakage = []
    non_pitch_aware = []
    comparison_records = []
    identity_fields = (
        "source_step",
        "anchor_grid",
        "source_pose_grid",
        "trigger_pose_grid",
        "route_cells",
        "route_chain_continuous",
    )
    for key in sorted(expected_keys & set(current_events) & set(baseline_events)):
        current_row = current_events[key]
        baseline_row = baseline_events[key]
        current = dict(current_row.get("audit") or {})
        baseline = dict(baseline_row.get("audit") or {})
        reference = dict(current_row.get("fixed_reference") or {})
        expected = fixed_entries[key]
        if not current.get("valid") or not current.get("source_anchor_pose_match"):
            invalid_audits.append(
                {"scene_episode": key[0], "step_id": key[1], "audit": current}
            )
        if not current.get("mapping_camera_pitch_aware"):
            non_pitch_aware.append(key)
        expected_reference = {
            name: expected.get(name)
            for name in (
                "scene_id",
                "episode_id",
                "step_id",
                "source_step",
                "anchor_grid",
                "candidate_id",
            )
        }
        actual_reference = {
            name: reference.get(name) for name in expected_reference
        }
        if actual_reference != expected_reference:
            reference_mismatches.append(
                {
                    "scene_episode": key[0],
                    "step_id": key[1],
                    "expected": expected_reference,
                    "actual": actual_reference,
                }
            )
        differing = {
            name: {"stage22a": baseline.get(name), "stage22c": current.get(name)}
            for name in identity_fields
            if baseline.get(name) != current.get(name)
        }
        if differing:
            route_identity_mismatches.append(
                {"scene_episode": key[0], "step_id": key[1], "fields": differing}
            )
        if (
            current_row.get("action_applied")
            or current_row.get("output_rewritten")
            or current.get("action_applied")
            or current.get("output_rewritten")
        ):
            action_or_output_violations.append(current_row)
        if current_row.get("gt_fields_used") or current.get("gt_fields_used"):
            gt_leakage.append(current_row)
        before_ratios = dict(baseline.get("route_cell_state_ratios") or {})
        now_ratios = dict(current.get("route_cell_state_ratios") or {})
        before_reachable = bool(
            (baseline.get("known_free_connectivity") or {}).get("reachable")
        )
        now_reachable = bool(
            (current.get("known_free_connectivity") or {}).get("reachable")
        )
        comparison_records.append(
            {
                "scene_episode": key[0],
                "step_id": key[1],
                "stage22a_ray_reachable": before_reachable,
                "stage22c_ray_reachable": now_reachable,
                "stage22a_free_ratio": before_ratios.get("free"),
                "stage22c_free_ratio": now_ratios.get("free"),
                "free_ratio_delta": float(now_ratios.get("free", 0.0) or 0.0)
                - float(before_ratios.get("free", 0.0) or 0.0),
                "stage22a_occupied_ratio": before_ratios.get("occupied"),
                "stage22c_occupied_ratio": now_ratios.get("occupied"),
                "occupied_ratio_delta": float(
                    now_ratios.get("occupied", 0.0) or 0.0
                )
                - float(before_ratios.get("occupied", 0.0) or 0.0),
                "current_selected_candidate": current_row.get(
                    "current_selected_candidate"
                ),
                "current_triage_tier": current_row.get("current_triage_tier"),
                "height_diagnostics": current.get(
                    "route_occupied_height_diagnostics"
                ),
            }
        )

    progress_action_violations = [
        row
        for row in progress.values()
        if int(row.get("s2_loop_strict_active_applied_count", 0) or 0) > 0
        or int(row.get("s2_loop_path_reobserve_applied_count", 0) or 0) > 0
        or int(row.get("stage19_semantic_resilience_active_applied_count", 0) or 0)
        > 0
    ]
    action_or_output_violations.extend(progress_action_violations)
    pitched_updates = [
        row
        for row in memory_updates
        if abs(float(row.get("requested_camera_pitch_deg", 0.0) or 0.0)) > 1e-4
    ]
    pitch_mismatches = [
        row
        for row in memory_updates
        if abs(
            float(row.get("requested_camera_pitch_deg", 0.0) or 0.0)
            - float(row.get("applied_camera_pitch_deg", 0.0) or 0.0)
        )
        > 1e-4
    ]

    triage_transitions = Counter()
    candidate_identity_changed = 0
    loop_comparison_records = []
    for key in sorted(set(current_loops) & set(baseline_loops)):
        current = current_loops[key]
        baseline = baseline_loops[key]
        before_candidate = dict(baseline.get("candidate") or {})
        now_candidate = dict(current.get("candidate") or {})
        before_identity = (
            before_candidate.get("candidate_id"),
            before_candidate.get("semantic_resilience_source_step_id"),
            before_candidate.get("grid"),
        )
        now_identity = (
            now_candidate.get("candidate_id"),
            now_candidate.get("semantic_resilience_source_step_id"),
            now_candidate.get("grid"),
        )
        changed = before_identity != now_identity
        candidate_identity_changed += int(changed)
        transition = (
            f"{baseline.get('triage_tier')}->{current.get('triage_tier')}"
        )
        triage_transitions[transition] += 1
        loop_comparison_records.append(
            {
                "scene_episode": key[0],
                "step_id": key[1],
                "candidate_identity_changed": changed,
                "stage22a_triage_tier": baseline.get("triage_tier"),
                "stage22c_triage_tier": current.get("triage_tier"),
                "stage22a_failure_type": baseline.get("failure_type"),
                "stage22c_failure_type": current.get("failure_type"),
            }
        )

    occupied_deltas = [row["occupied_ratio_delta"] for row in comparison_records]
    free_deltas = [row["free_ratio_delta"] for row in comparison_records]
    integrity_passed = bool(
        len(progress) == expected_episodes
        and len(expected_seeds) == expected_episodes
        and len(fixed_entries) == expected_fixed_routes
        and not seed_mismatches
        and not metric_mismatches
        and not raw_loop_mismatches
        and not missing_current
        and not extra_current
        and not missing_baseline
        and not extra_baseline
        and not invalid_audits
        and not reference_mismatches
        and not route_identity_mismatches
        and not action_or_output_violations
        and not gt_leakage
        and not non_pitch_aware
        and memory_updates
        and pitched_updates
        and not pitch_mismatches
    )
    return {
        "task": "stage22c_fixed_route_pitch_aware_occ_shadow",
        "expected_episode_count": expected_episodes,
        "completed_episode_count": len(progress),
        "fixed_route_expected_count": expected_fixed_routes,
        "fixed_route_manifest_count": len(fixed_entries),
        "fixed_route_audit_count": len(current_events),
        "seed_replay_verified_count": len(expected_seeds) - len(seed_mismatches),
        "reference_metric_verified_count": expected_episodes
        - len(metric_mismatches),
        "raw_loop_identity_verified": not raw_loop_mismatches,
        "route_identity_verified_count": len(comparison_records)
        - len(route_identity_mismatches),
        "valid_occ_update_count": len(memory_updates),
        "pitched_occ_update_count": len(pitched_updates),
        "pitch_application_mismatch_count": len(pitch_mismatches),
        "stage22a_ray_reachable_count": sum(
            row["stage22a_ray_reachable"] for row in comparison_records
        ),
        "stage22c_ray_reachable_count": sum(
            row["stage22c_ray_reachable"] for row in comparison_records
        ),
        "ray_reachability_gain_count": sum(
            not row["stage22a_ray_reachable"]
            and row["stage22c_ray_reachable"]
            for row in comparison_records
        ),
        "ray_reachability_regression_count": sum(
            row["stage22a_ray_reachable"]
            and not row["stage22c_ray_reachable"]
            for row in comparison_records
        ),
        "occupied_ratio_delta_mean": _mean(occupied_deltas),
        "free_ratio_delta_mean": _mean(free_deltas),
        "occupied_ratio_improved_event_count": sum(
            value < -1e-9 for value in occupied_deltas
        ),
        "occupied_ratio_regressed_event_count": sum(
            value > 1e-9 for value in occupied_deltas
        ),
        "candidate_identity_changed_count": candidate_identity_changed,
        "triage_transition_counts": dict(triage_transitions),
        "dynamic_failure_type_transition_counts": dict(
            dynamic_failure_type_transitions
        ),
        "integrity_passed": integrity_passed,
        "measurement_complete": integrity_passed,
        "violations": {
            "seed_mismatch": len(seed_mismatches),
            "metric_mismatch": len(metric_mismatches),
            "raw_loop_mismatch": len(raw_loop_mismatches),
            "missing_current_fixed_event": len(missing_current),
            "extra_current_fixed_event": len(extra_current),
            "missing_baseline_event": len(missing_baseline),
            "extra_baseline_event": len(extra_baseline),
            "invalid_audit": len(invalid_audits),
            "fixed_reference_mismatch": len(reference_mismatches),
            "route_identity_mismatch": len(route_identity_mismatches),
            "action_or_output_applied": len(action_or_output_violations),
            "gt_leakage": len(gt_leakage),
            "non_pitch_aware_audit": len(non_pitch_aware),
            "pitch_application_mismatch": len(pitch_mismatches),
        },
        "violation_records": {
            "seed_mismatch": seed_mismatches,
            "metric_mismatch": metric_mismatches,
            "raw_loop_mismatch": raw_loop_mismatches,
            "missing_current_fixed_event": missing_current,
            "extra_current_fixed_event": extra_current,
            "missing_baseline_event": missing_baseline,
            "extra_baseline_event": extra_baseline,
            "invalid_audit": invalid_audits,
            "fixed_reference_mismatch": reference_mismatches,
            "route_identity_mismatch": route_identity_mismatches,
        },
        "comparison_records": comparison_records,
        "loop_comparison_records": loop_comparison_records,
        "interpretation_guard": (
            "Only fixed-route OCC deltas measure pitch projection quality. "
            "Dynamic candidate and triage changes are reported separately and "
            "do not fail integrity when the Frozen trajectory is unchanged. "
            "failure_type is also a dynamic OCC-derived diagnostic."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--navigation-reference-root", type=Path, required=True)
    parser.add_argument("--stage22a-root", type=Path, required=True)
    parser.add_argument("--fixed-route-manifest", type=Path, required=True)
    parser.add_argument("--expected-fixed-routes", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        args.run_root,
        args.expected_episodes,
        args.seed_manifest,
        args.navigation_reference_root,
        args.stage22a_root,
        args.fixed_route_manifest,
        args.expected_fixed_routes,
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

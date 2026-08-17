"""Compare Stage22B pitch-aware OCC against the Stage22A horizontal baseline."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from scripts.eval.analyze_stage22a_executed_route_occ_audit import (
    _events,
    _key,
    _read_jsonl,
    analyze as analyze_stage22a,
)


def _audit_events(root: Path):
    rows = _events(root, "s2_loop_executed_route_occ_audit_events.jsonl")
    return {
        (_key(row), int(row.get("step_id", -1))): dict(row.get("audit") or {})
        for row in rows
    }


def _mean(values):
    return None if not values else float(sum(values) / len(values))


def analyze(
    run_root: Path,
    expected_episodes: int,
    seed_manifest: Path,
    navigation_reference_root: Path,
    stage22a_root: Path,
):
    summary = analyze_stage22a(
        run_root,
        expected_episodes,
        seed_manifest,
        navigation_reference_root,
    )
    current = _audit_events(run_root)
    baseline = _audit_events(stage22a_root)
    missing_current = sorted(set(baseline) - set(current))
    missing_baseline = sorted(set(current) - set(baseline))
    identity_fields = (
        "source_step",
        "anchor_grid",
        "source_pose_grid",
        "trigger_pose_grid",
        "route_cells",
        "route_chain_continuous",
    )
    route_identity_mismatches = []
    records = []
    for key in sorted(set(current) & set(baseline)):
        now = current[key]
        before = baseline[key]
        differing = {
            field: {"stage22a": before.get(field), "stage22b": now.get(field)}
            for field in identity_fields
            if before.get(field) != now.get(field)
        }
        if differing:
            route_identity_mismatches.append(
                {"scene_episode": key[0], "step_id": key[1], "fields": differing}
            )
        before_ratios = dict(before.get("route_cell_state_ratios") or {})
        now_ratios = dict(now.get("route_cell_state_ratios") or {})
        before_connectivity = dict(before.get("known_free_connectivity") or {})
        now_connectivity = dict(now.get("known_free_connectivity") or {})
        records.append(
            {
                "scene_episode": key[0],
                "step_id": key[1],
                "stage22a_ray_reachable": bool(before_connectivity.get("reachable")),
                "stage22b_ray_reachable": bool(now_connectivity.get("reachable")),
                "stage22a_occupied_ratio": before_ratios.get("occupied"),
                "stage22b_occupied_ratio": now_ratios.get("occupied"),
                "occupied_ratio_delta": float(now_ratios.get("occupied", 0.0) or 0.0)
                - float(before_ratios.get("occupied", 0.0) or 0.0),
                "stage22a_free_ratio": before_ratios.get("free"),
                "stage22b_free_ratio": now_ratios.get("free"),
                "free_ratio_delta": float(now_ratios.get("free", 0.0) or 0.0)
                - float(before_ratios.get("free", 0.0) or 0.0),
                "route_pitch_observation_count": int(
                    now.get("route_pitch_observation_count", 0) or 0
                ),
                "route_max_camera_pitch_deg": float(
                    now.get("route_max_camera_pitch_deg", 0.0) or 0.0
                ),
                "height_diagnostics": now.get(
                    "route_occupied_height_diagnostics"
                ),
            }
        )

    non_pitch_aware_audits = [
        key for key, audit in current.items() if not audit.get("mapping_camera_pitch_aware")
    ]
    memory_updates = [
        row
        for path in glob.glob(
            str(run_root / "vlmap_safety_debug" / "*" / "occ_memory" / "memory_events.jsonl")
        )
        for row in _read_jsonl(Path(path))
    ]
    valid_updates = [row for row in memory_updates if row.get("event_type") == "occ_memory_update" and row.get("valid")]
    pitched_updates = [
        row
        for row in valid_updates
        if abs(float(row.get("requested_camera_pitch_deg", 0.0) or 0.0)) > 1e-4
    ]
    pitch_application_mismatches = [
        row
        for row in valid_updates
        if abs(
            float(row.get("requested_camera_pitch_deg", 0.0) or 0.0)
            - float(row.get("applied_camera_pitch_deg", 0.0) or 0.0)
        )
        > 1e-4
    ]
    event_sets_match = not missing_current and not missing_baseline
    comparison_integrity_passed = bool(
        summary.get("integrity_passed")
        and event_sets_match
        and not route_identity_mismatches
        and not non_pitch_aware_audits
        and valid_updates
        and pitched_updates
        and not pitch_application_mismatches
    )
    occupied_deltas = [row["occupied_ratio_delta"] for row in records]
    free_deltas = [row["free_ratio_delta"] for row in records]
    summary.update(
        {
            "task": "stage22b_pitch_aware_occ_shadow_comparison",
            "stage22a_reference_root": str(stage22a_root),
            "stage22a_event_count": len(baseline),
            "stage22b_event_count": len(current),
            "event_sets_match": event_sets_match,
            "missing_current_event_keys": missing_current,
            "missing_baseline_event_keys": missing_baseline,
            "route_identity_mismatch_count": len(route_identity_mismatches),
            "route_identity_mismatches": route_identity_mismatches,
            "non_pitch_aware_audit_count": len(non_pitch_aware_audits),
            "valid_occ_update_count": len(valid_updates),
            "pitched_occ_update_count": len(pitched_updates),
            "pitch_application_mismatch_count": len(pitch_application_mismatches),
            "stage22a_ray_reachable_count": sum(
                row["stage22a_ray_reachable"] for row in records
            ),
            "stage22b_ray_reachable_count": sum(
                row["stage22b_ray_reachable"] for row in records
            ),
            "ray_reachability_gain_count": sum(
                (not row["stage22a_ray_reachable"])
                and row["stage22b_ray_reachable"]
                for row in records
            ),
            "ray_reachability_regression_count": sum(
                row["stage22a_ray_reachable"]
                and (not row["stage22b_ray_reachable"])
                for row in records
            ),
            "occupied_ratio_delta_mean": _mean(occupied_deltas),
            "free_ratio_delta_mean": _mean(free_deltas),
            "occupied_ratio_improved_event_count": sum(
                value < -1e-9 for value in occupied_deltas
            ),
            "occupied_ratio_regressed_event_count": sum(
                value > 1e-9 for value in occupied_deltas
            ),
            "comparison_records": records,
            "comparison_integrity_passed": comparison_integrity_passed,
            "measurement_complete": comparison_integrity_passed,
            "interpretation_guard": (
                "This run isolates camera-pitch-aware depth projection in OCC. "
                "A route-map improvement diagnoses projection quality only; no "
                "historical cell is promoted to free and no navigation action is changed."
            ),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--navigation-reference-root", type=Path, required=True)
    parser.add_argument("--stage22a-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        args.run_root,
        args.expected_episodes,
        args.seed_manifest,
        args.navigation_reference_root,
        args.stage22a_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_all and not summary["comparison_integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

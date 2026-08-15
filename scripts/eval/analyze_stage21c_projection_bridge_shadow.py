"""Audit Stage21c map-candidate to visible-free-pixel projection shadow."""

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


def _loop_signatures(run_root: Path):
    signatures = defaultdict(list)
    for row in _events(run_root, "s2_action_loop_events.jsonl"):
        if row.get("transition") != "start":
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
        f"{row['scene_id']}/{int(row['episode_id'])}": int(row["episode_eval_seed"])
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
    reference = _progress(reference_root)
    expected_seeds = _seed_manifest(seed_manifest)
    loops = _loop_signatures(run_root)
    reference_loops = _loop_signatures(reference_root)
    events = _events(run_root, "s2_loop_projection_bridge_events.jsonl")
    strict = [row for row in events if row.get("triage_tier") == "strict_intervention"]
    valid = [row for row in strict if bool(row.get("proposal_valid"))]

    seed_mismatches = [
        {
            "scene_episode": key,
            "expected_episode_eval_seed": seed,
            "actual_episode_eval_seed": progress.get(key, {}).get("episode_eval_seed"),
        }
        for key, seed in expected_seeds.items()
        if progress.get(key, {}).get("episode_eval_seed") != seed
    ]
    reference_metric_mismatches = []
    reference_loop_mismatches = []
    for key in sorted(expected_seeds):
        row = progress.get(key)
        ref = reference.get(key)
        if row is None or ref is None:
            reference_metric_mismatches.append(
                {"scene_episode": key, "reason": "missing_progress_or_reference"}
            )
            continue
        differing = {}
        for field in METRICS:
            actual = float(row.get(field, 0.0) or 0.0)
            expected = float(ref.get(field, 0.0) or 0.0)
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

    expected_loop_event_count = sum(
        len(reference_loops.get(key, [])) for key in expected_seeds
    )
    action_violations = [row for row in events if bool(row.get("action_applied"))]
    action_violations.extend(
        row
        for row in progress.values()
        if int(row.get("s2_loop_strict_active_applied_count", 0) or 0) > 0
        or int(row.get("stage19_semantic_resilience_active_applied_count", 0) or 0) > 0
    )
    gt_leakage = [row for row in events if list(row.get("gt_fields_used") or [])]
    non_shadow = [row for row in events if not bool(row.get("shadow_only"))]
    missing_bridge = [row for row in strict if not isinstance(row.get("bridge"), dict)]
    invalid_selected = []
    for row in valid:
        bridge = dict(row.get("bridge") or {})
        selected = dict(bridge.get("selected_probe") or {})
        if (
            bridge.get("selected_pixel_goal") is None
            or selected.get("goal_state") != "free"
            or not bool(selected.get("eligible"))
            or not bool(selected.get("in_bounds"))
            or not bool(selected.get("projection_valid"))
        ):
            invalid_selected.append(row)

    violations = {
        "seed_replay_mismatch": seed_mismatches,
        "reference_metric_mismatch": reference_metric_mismatches,
        "reference_loop_mismatch": reference_loop_mismatches,
        "action_applied": action_violations,
        "gt_leakage": gt_leakage,
        "non_shadow_event": non_shadow,
        "missing_bridge": missing_bridge,
        "invalid_selected_proposal": invalid_selected,
    }
    integrity_passed = bool(
        len(progress) == expected_episodes
        and len(expected_seeds) == expected_episodes
        and len(events) == expected_loop_event_count
        and strict
        and not any(violations.values())
    )

    fixed_state_counts = Counter()
    bridge_reason_counts = Counter()
    exact_projection_reason_counts = Counter()
    exact_in_bounds_count = 0
    selected_angle_errors = []
    selected_candidate_distances = []
    compact_records = []
    for row in strict:
        bridge = dict(row.get("bridge") or {})
        fixed = dict(bridge.get("baseline_probe") or {})
        exact = dict(bridge.get("exact_projection") or {})
        exact_projection = dict(exact.get("projection") or {})
        exact_probe = dict(exact.get("probe") or {})
        selected = dict(bridge.get("selected_probe") or {})
        fixed_state_counts[str(fixed.get("goal_state") or fixed.get("reason") or "missing")] += 1
        bridge_reason_counts[str(bridge.get("reason") or "missing")] += 1
        exact_projection_reason_counts[
            str(exact_projection.get("reason") or "missing")
        ] += 1
        exact_in_bounds_count += int(bool(exact_probe.get("in_bounds")))
        if selected.get("angle_error_deg") is not None:
            selected_angle_errors.append(float(selected["angle_error_deg"]))
        if selected.get("proxy_to_candidate_distance_m") is not None:
            selected_candidate_distances.append(
                float(selected["proxy_to_candidate_distance_m"])
            )
        compact_records.append(
            {
                "scene_episode": _key(row),
                "step_id": row.get("step_id"),
                "triage_tier": row.get("triage_tier"),
                "candidate_grid": bridge.get("candidate_grid"),
                "candidate_direction_bucket": bridge.get(
                    "candidate_direction_bucket"
                ),
                "candidate_direction_angle_deg": bridge.get(
                    "candidate_direction_angle_deg"
                ),
                "fixed_pixel_goal": fixed.get("pixel_goal"),
                "fixed_goal_state": fixed.get("goal_state"),
                "fixed_reason": fixed.get("reason"),
                "exact_pixel_goal": exact_projection.get("pixel_goal"),
                "exact_projection_reason": exact_projection.get("reason"),
                "exact_in_bounds": exact_probe.get("in_bounds"),
                "sample_count": bridge.get("sample_count"),
                "sample_goal_state_counts": bridge.get(
                    "sample_goal_state_counts"
                ),
                "aligned_free_count": bridge.get(
                    "angle_aligned_free_sample_count"
                ),
                "eligible_probe_count": bridge.get("eligible_probe_count"),
                "proposal_valid": row.get("proposal_valid"),
                "selected_pixel_goal": bridge.get("selected_pixel_goal"),
                "selected_goal_grid": selected.get("goal_grid"),
                "selected_angle_error_deg": selected.get("angle_error_deg"),
                "selected_proxy_to_candidate_distance_m": selected.get(
                    "proxy_to_candidate_distance_m"
                ),
            }
        )

    return {
        "task": "stage21c_projection_bridge_shadow",
        "expected_episode_count": expected_episodes,
        "completed_episode_count": len(progress),
        "seed_replay_expected_count": len(expected_seeds),
        "seed_replay_verified_count": len(expected_seeds) - len(seed_mismatches),
        "reference_metric_verified_count": expected_episodes
        - len(reference_metric_mismatches),
        "reference_loop_verified_count": expected_episodes
        - len(reference_loop_mismatches),
        "expected_loop_event_count": expected_loop_event_count,
        "bridge_event_count": len(events),
        "strict_event_count": len(strict),
        "non_strict_hold_count": len(events) - len(strict),
        "valid_proposal_count": len(valid),
        "valid_proposal_rate_over_strict": (
            None if not strict else float(len(valid)) / len(strict)
        ),
        "fixed_baseline_goal_state_counts": dict(fixed_state_counts),
        "bridge_reason_counts": dict(bridge_reason_counts),
        "exact_projection_reason_counts": dict(exact_projection_reason_counts),
        "exact_projection_in_bounds_count": exact_in_bounds_count,
        "selected_angle_error_mean_deg": _mean(selected_angle_errors),
        "selected_proxy_to_candidate_distance_mean_m": _mean(
            selected_candidate_distances
        ),
        "integrity_passed": integrity_passed,
        "projection_bridge_shadow_gate_passed": bool(
            integrity_passed and valid
        ),
        "interpretation_guard": (
            "A valid proxy proves only that a map recovery intent can be bridged "
            "to a visible known-free pixel. Navigation benefit still requires a "
            "strict-only paired active experiment."
        ),
        "violations": {name: len(rows) for name, rows in violations.items()},
        "violation_records": violations,
        "strict_event_records": compact_records,
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

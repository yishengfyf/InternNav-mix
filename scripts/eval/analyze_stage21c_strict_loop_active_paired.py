"""Audit paired Frozen-S2 control vs Stage21c strict S2-loop active treatment."""

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
    rows = _read_jsonl(run_root / "progress.json")
    return {_key(row): row for row in rows if row.get("episode_id") is not None}


def _events(run_root: Path):
    paths = glob.glob(
        str(run_root / "vlmap_safety_debug" / "*" / "s2_loop_strict_active_events.jsonl")
    )
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _loop_events(run_root: Path):
    paths = glob.glob(
        str(run_root / "vlmap_safety_debug" / "*" / "s2_action_loop_events.jsonl")
    )
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _loop_signatures(run_root: Path):
    signatures = defaultdict(list)
    for row in _loop_events(run_root):
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


def _seed_replay_manifest(path: Path | None):
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {}
    for row in rows:
        key = f"{row['scene_id']}/{int(row['episode_id'])}"
        expected[key] = int(row["episode_eval_seed"])
    return expected


def _mean(rows, field):
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return None if not values else sum(values) / len(values)


def analyze(
    control_root: Path,
    active_root: Path,
    expected_episodes: int,
    seed_manifest: Path | None = None,
    reference_root: Path | None = None,
):
    control = _progress(control_root)
    active = _progress(active_root)
    expected_seeds = _seed_replay_manifest(seed_manifest)
    reference = _progress(reference_root) if reference_root is not None else {}
    reference_loops = _loop_signatures(reference_root) if reference_root is not None else {}
    control_loops = _loop_signatures(control_root)
    events = _events(active_root)
    common = sorted(set(control).intersection(active))
    applied = [row for row in events if bool(row.get("action_applied"))]
    by_episode = defaultdict(list)
    for row in applied:
        by_episode[_key(row)].append(row)

    paired_rows = []
    for key in common:
        base = control[key]
        treatment = active[key]
        record = {"scene_episode": key, "intervention_count": len(by_episode.get(key, []))}
        for field in METRICS:
            base_value = float(base.get(field, 0.0) or 0.0)
            active_value = float(treatment.get(field, 0.0) or 0.0)
            record[f"control_{field}"] = base_value
            record[f"active_{field}"] = active_value
            record[f"delta_{field}"] = active_value - base_value
        paired_rows.append(record)

    applied_rows = [row for row in paired_rows if row["intervention_count"] > 0]
    wins = [
        row for row in applied_rows
        if row["control_success"] <= 0.0 and row["active_success"] > 0.0
    ]
    regressions = [
        row for row in applied_rows
        if row["control_success"] > 0.0 and row["active_success"] <= 0.0
    ]
    reference_metric_mismatches = []
    reference_loop_mismatches = []
    if reference_root is not None:
        for key in common:
            reference_row = reference.get(key)
            if reference_row is None:
                reference_metric_mismatches.append(
                    {"scene_episode": key, "reason": "missing_reference_episode"}
                )
                continue
            differing = {}
            for field in METRICS:
                control_value = float(control[key].get(field, 0.0) or 0.0)
                reference_value = float(reference_row.get(field, 0.0) or 0.0)
                if abs(control_value - reference_value) > 1e-6:
                    differing[field] = {
                        "control": control_value,
                        "reference": reference_value,
                    }
            if differing:
                reference_metric_mismatches.append(
                    {"scene_episode": key, "fields": differing}
                )
            if control_loops.get(key, []) != reference_loops.get(key, []):
                reference_loop_mismatches.append(
                    {
                        "scene_episode": key,
                        "control_loop_signatures": control_loops.get(key, []),
                        "reference_loop_signatures": reference_loops.get(key, []),
                    }
                )

    violations = {
        "non_strict_applied": [
            row for row in applied if row.get("triage_tier") != "strict_intervention"
        ],
        "unsafe_applied": [
            row
            for row in applied
            if not bool((row.get("geometry_preflight") or {}).get("geometry_safe"))
            or not bool((row.get("geometry_preflight") or {}).get("active_gate_safe"))
            or not bool((row.get("waypoint_preflight") or {}).get("valid"))
            or str((row.get("waypoint_preflight") or {}).get("goal_state")) != "free"
            or bool((row.get("trajectory_preflight") or {}).get("reject_required"))
            or int((row.get("trajectory_preflight") or {}).get("first_action", 0) or 0) == 0
        ],
        "budget_exceeded_episodes": [key for key, rows in by_episode.items() if len(rows) > 1],
        "rewrite_failures": [row for row in events if row.get("reason") == "output_rewrite_failed"],
        "gt_leakage": [row for row in events if list(row.get("gt_fields_used") or [])],
        "paired_seed_mismatch": [
            {
                "scene_episode": key,
                "control_episode_eval_seed": control[key].get("episode_eval_seed"),
                "active_episode_eval_seed": active[key].get("episode_eval_seed"),
            }
            for key in common
            if control[key].get("episode_eval_seed")
            != active[key].get("episode_eval_seed")
        ],
        "control_seed_replay_mismatch": [
            {
                "scene_episode": key,
                "expected_episode_eval_seed": seed,
                "actual_episode_eval_seed": control.get(key, {}).get("episode_eval_seed"),
            }
            for key, seed in expected_seeds.items()
            if control.get(key, {}).get("episode_eval_seed") != seed
        ],
        "active_seed_replay_mismatch": [
            {
                "scene_episode": key,
                "expected_episode_eval_seed": seed,
                "actual_episode_eval_seed": active.get(key, {}).get("episode_eval_seed"),
            }
            for key, seed in expected_seeds.items()
            if active.get(key, {}).get("episode_eval_seed") != seed
        ],
        "control_reference_metric_mismatch": reference_metric_mismatches,
        "control_reference_loop_mismatch": reference_loop_mismatches,
    }
    integrity_passed = bool(
        len(control) == expected_episodes
        and len(active) == expected_episodes
        and len(common) == expected_episodes
        and (not seed_manifest or len(expected_seeds) == expected_episodes)
        and applied
        and not any(violations.values())
    )
    aggregate = {"episode_count": len(common)}
    for field in METRICS:
        control_mean = _mean([control[key] for key in common], field)
        active_mean = _mean([active[key] for key in common], field)
        aggregate[f"control_{field}_mean"] = control_mean
        aggregate[f"active_{field}_mean"] = active_mean
        aggregate[f"delta_{field}_mean"] = (
            None if control_mean is None or active_mean is None else active_mean - control_mean
        )
    return {
        "task": "stage21c_strict_loop_active_paired_tiny",
        "scope": {
            "control": "Frozen S2/NextDiT",
            "treatment": "strict S2-loop candidate directional pixel then Frozen NextDiT replan",
            "max_interventions_per_episode": 1,
            "adapter_and_abstain": "hold_s2",
            "performance_regression_blocks_integrity_audit": False,
        },
        "expected_episode_count": expected_episodes,
        "control_episode_count": len(control),
        "active_episode_count": len(active),
        "common_episode_count": len(common),
        "active_event_count": len(events),
        "seed_replay_expected_count": len(expected_seeds),
        "seed_replay_manifest_complete": bool(
            not seed_manifest or len(expected_seeds) == expected_episodes
        ),
        "seed_replay_verified_count": sum(
            control.get(key, {}).get("episode_eval_seed") == seed
            and active.get(key, {}).get("episode_eval_seed") == seed
            for key, seed in expected_seeds.items()
        ),
        "reference_replay_required": bool(reference_root is not None),
        "reference_metric_verified_count": (
            0 if reference_root is None else len(common) - len(reference_metric_mismatches)
        ),
        "reference_loop_verified_count": (
            0 if reference_root is None else len(common) - len(reference_loop_mismatches)
        ),
        "event_reason_counts": dict(Counter(str(row.get("reason")) for row in events)),
        "applied_event_count": len(applied),
        "applied_episode_count": len(by_episode),
        "violations": {name: len(rows) for name, rows in violations.items()},
        "violation_records": violations,
        "integrity_passed": integrity_passed,
        "paired_aggregate": aggregate,
        "failed_to_success_count": len(wins),
        "success_to_failed_count": len(regressions),
        "net_success_flip": len(wins) - len(regressions),
        "failed_to_success_records": wins,
        "success_to_failed_records": regressions,
        "paired_episode_records": paired_rows,
        "interpretation_guard": (
            "Integrity pass only proves the bounded active path executed safely; "
            "navigation benefit requires manual paired outcome and visual review."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        args.control_root,
        args.active_root,
        args.expected_episodes,
        seed_manifest=args.seed_manifest,
        reference_root=args.reference_root,
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

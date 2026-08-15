"""Audit paired Frozen-S2 control vs bounded path reorient/reobserve active."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


METRICS = ("success", "spl", "ne", "steps", "collision_count")


def _read_jsonl(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _key(row):
    return f"{row.get('scene_id')}/{int(row.get('episode_id'))}"


def _intervention_key(row):
    return (_key(row), int(row.get("trigger_step", row.get("step_id", -1))))


def _progress(root: Path):
    return {
        _key(row): row
        for row in _read_jsonl(root / "progress.json")
        if row.get("episode_id") is not None
    }


def _events(root: Path, name: str):
    paths = glob.glob(str(root / "vlmap_safety_debug" / "*" / name))
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _loop_signatures(root: Path):
    signatures = defaultdict(list)
    for row in _events(root, "s2_action_loop_events.jsonl"):
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
    return {key: sorted(value) for key, value in signatures.items()}


def _manifest(path: Path | None):
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['scene_id']}/{int(row['episode_id'])}": int(row["episode_eval_seed"])
        for row in rows
    }


def _mean(rows, field):
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return None if not values else sum(values) / len(values)


def _path_bridge(row):
    return dict(row.get("post_path_bridge") or row.get("path_bridge") or {})


def analyze(
    control_root: Path,
    active_root: Path,
    expected_episodes: int,
    seed_manifest: Path | None,
    reference_root: Path | None,
):
    control = _progress(control_root)
    active = _progress(active_root)
    reference = _progress(reference_root) if reference_root else {}
    expected_seeds = _manifest(seed_manifest)
    control_loops = _loop_signatures(control_root)
    reference_loops = _loop_signatures(reference_root) if reference_root else {}
    events = _events(active_root, "s2_loop_path_reobserve_active_events.jsonl")
    common = sorted(set(control).intersection(active))

    applied_events = [row for row in events if bool(row.get("action_applied"))]
    reorient_events = [
        row for row in events if bool(row.get("reorient_action_applied"))
    ]
    pixel_events = [row for row in events if bool(row.get("pixel_action_applied"))]
    intervention_keys = sorted({_intervention_key(row) for row in applied_events})
    by_episode = defaultdict(set)
    for key in intervention_keys:
        by_episode[key[0]].add(key)

    paired_rows = []
    for key in common:
        base = control[key]
        treatment = active[key]
        record = {
            "scene_episode": key,
            "intervention_count": len(by_episode.get(key, set())),
        }
        for field in METRICS:
            base_value = float(base.get(field, 0.0) or 0.0)
            treatment_value = float(treatment.get(field, 0.0) or 0.0)
            record[f"control_{field}"] = base_value
            record[f"active_{field}"] = treatment_value
            record[f"delta_{field}"] = treatment_value - base_value
        paired_rows.append(record)

    applied_rows = [row for row in paired_rows if row["intervention_count"] > 0]
    wins = [
        row
        for row in applied_rows
        if row["control_success"] <= 0.0 and row["active_success"] > 0.0
    ]
    regressions = [
        row
        for row in applied_rows
        if row["control_success"] > 0.0 and row["active_success"] <= 0.0
    ]

    reference_metric_mismatches = []
    reference_loop_mismatches = []
    if reference_root:
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
                        "control": control_loops.get(key, []),
                        "reference": reference_loops.get(key, []),
                    }
                )

    reorient_violations = []
    for row in reorient_events:
        planned = [int(item) for item in row.get("reorient_actions") or []]
        applied = [int(item) for item in row.get("reorient_actions_applied") or []]
        if (
            not planned
            or len(planned) > 4
            or any(item not in (2, 3) for item in planned)
            or applied != planned
        ):
            reorient_violations.append(row)

    pixel_violations = []
    for row in pixel_events:
        bridge = _path_bridge(row)
        selected = dict(bridge.get("selected_probe") or {})
        trajectory = dict(row.get("trajectory_preflight") or {})
        if (
            not bridge.get("valid")
            or not selected.get("path_eligible")
            or selected.get("goal_state") != "free"
            or not trajectory.get("valid")
            or not trajectory.get("safe")
            or trajectory.get("reject_required")
            or int(trajectory.get("first_action", 0) or 0) == 0
        ):
            pixel_violations.append(row)

    trigger_mismatches = []
    for episode_key, trigger_step in intervention_keys:
        control_strict_steps = {
            signature[0]
            for signature in control_loops.get(episode_key, [])
            if signature[3] == "strict_intervention"
        }
        if trigger_step not in control_strict_steps:
            trigger_mismatches.append(
                {
                    "scene_episode": episode_key,
                    "trigger_step": trigger_step,
                    "control_strict_steps": sorted(control_strict_steps),
                }
            )

    violations = {
        "non_strict_applied": [
            row
            for row in applied_events
            if row.get("triage_tier") != "strict_intervention"
        ],
        "reorient_plan_or_execution": reorient_violations,
        "path_pixel_safety_or_identity": pixel_violations,
        "budget_exceeded_episodes": [
            key for key, values in by_episode.items() if len(values) > 1
        ],
        "active_trigger_not_in_control_strict": trigger_mismatches,
        "gt_leakage": [row for row in events if list(row.get("gt_fields_used") or [])],
        "paired_seed_mismatch": [
            {
                "scene_episode": key,
                "control": control[key].get("episode_eval_seed"),
                "active": active[key].get("episode_eval_seed"),
            }
            for key in common
            if control[key].get("episode_eval_seed")
            != active[key].get("episode_eval_seed")
        ],
        "control_seed_replay_mismatch": [
            {"scene_episode": key, "expected": seed, "actual": control.get(key, {}).get("episode_eval_seed")}
            for key, seed in expected_seeds.items()
            if control.get(key, {}).get("episode_eval_seed") != seed
        ],
        "active_seed_replay_mismatch": [
            {"scene_episode": key, "expected": seed, "actual": active.get(key, {}).get("episode_eval_seed")}
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
        and not any(violations.values())
    )
    aggregate = {"episode_count": len(common)}
    for field in METRICS:
        control_mean = _mean([control[key] for key in common], field)
        active_mean = _mean([active[key] for key in common], field)
        aggregate[f"control_{field}_mean"] = control_mean
        aggregate[f"active_{field}_mean"] = active_mean
        aggregate[f"delta_{field}_mean"] = (
            None
            if control_mean is None or active_mean is None
            else active_mean - control_mean
        )

    return {
        "task": "stage21c_path_reobserve_active_paired",
        "expected_episode_count": expected_episodes,
        "control_episode_count": len(control),
        "active_episode_count": len(active),
        "common_episode_count": len(common),
        "event_count": len(events),
        "event_reason_counts": dict(Counter(str(row.get("reason")) for row in events)),
        "active_experiment_formed": bool(intervention_keys),
        "applied_intervention_count": len(intervention_keys),
        "applied_episode_count": len(by_episode),
        "reorient_completed_event_count": len(reorient_events),
        "path_pixel_applied_event_count": len(pixel_events),
        "seed_replay_verified_count": sum(
            control.get(key, {}).get("episode_eval_seed") == seed
            and active.get(key, {}).get("episode_eval_seed") == seed
            for key, seed in expected_seeds.items()
        ),
        "reference_metric_verified_count": (
            0 if not reference_root else len(common) - len(reference_metric_mismatches)
        ),
        "reference_loop_verified_count": (
            0 if not reference_root else len(common) - len(reference_loop_mismatches)
        ),
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
            "A completed bounded turn is a real treatment even when no path pixel is found. "
            "Integrity is separate from navigation benefit; inspect paired outcomes and images."
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

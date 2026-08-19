#!/usr/bin/env python3
"""Offline Stage25 stuck/failure detector audit.

This script replays already logged shadow data.  It never calls the navigator,
changes an action, creates a candidate, or executes recovery.  D0/D1/D2 are
event detectors only:

* D0: strict S2 turn-loop and repeated-action snapshots.
* D1: D0 plus trajectory revisit and OCC/policy-conflict evidence.
* D2: D1 plus semantic stagnation/dead-zone/drift evidence from the logged
  weak semantic backend.  LSeg is deliberately not treated as available unless
  a future run logs matching episode/step evidence.

Final success/collision values are episode-level outcome labels.  They are
reported as outcome correlation proxies, not as per-step ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Event streams are JSONL; snapshot files are pretty-printed JSON.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def discover(run_root: Path) -> Dict[str, List[Tuple[Path, Dict[str, Any]]]]:
    names = {
        "progress": "progress.json",
        "trajectory": "trajectory_events.jsonl",
        "loops": "s2_action_loop_events.jsonl",
        "semantic": "semantic_events.jsonl",
        "resilience": "stage19_semantic_resilience_active_events.jsonl",
        "occ_audit": "s2_loop_fixed_route_occ_audit_events.jsonl",
        "snapshots": "*.json",
    }
    found: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    for kind, name in names.items():
        if kind == "snapshots":
            paths = sorted(run_root.glob("vlmap_safety_debug/**/stuck_snapshots/*.json"))
        else:
            paths = sorted(run_root.glob(f"vlmap_safety_debug/**/{name}"))
            if run_root.name == "vlmap_safety_debug":
                paths += sorted(run_root.glob(f"**/{name}"))
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            for row in read_records(path):
                found[kind].append((path, row))
    return found


def latest_image(path: Path, row: Mapping[str, Any]) -> Optional[str]:
    raw = row.get("rgb_file")
    candidates = []
    if raw:
        candidates.append(path.parent / str(raw))
        candidates.append(path.parent / ".." / str(raw))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return str(candidate)
    return None


def trajectory_by_episode(rows: Iterable[Tuple[Path, Dict[str, Any]]]):
    output: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    for path, row in rows:
        if row.get("scene_id") is not None and row.get("episode_id") is not None:
            output[episode_key(row)].append((path, row))
    for values in output.values():
        values.sort(key=lambda item: safe_int(item[1].get("eval_step"), -1) or -1)
    return output


def progress_map(rows: Iterable[Tuple[Path, Dict[str, Any]]]):
    output = {}
    for _, row in rows:
        if row.get("scene_id") is not None and row.get("episode_id") is not None:
            output[episode_key(row)] = row
    return output


def self_recovered(
    trajectory: Sequence[Tuple[Path, Mapping[str, Any]]],
    step_id: int,
    *,
    min_later_steps: int = 8,
    min_escape_m: float = 0.60,
) -> Tuple[bool, Optional[int]]:
    current = None
    for _, row in trajectory:
        if safe_int(row.get("eval_step"), -1) == step_id:
            current = row.get("gps")
            break
    if current is None:
        return False, None
    for _, row in trajectory:
        later = safe_int(row.get("eval_step"), -1)
        if later is None or later < step_id + min_later_steps:
            continue
        if distance(current, row.get("gps")) >= min_escape_m:
            return True, later
    return False, None


def route_revisit_events(
    trajectory: Mapping[str, Sequence[Tuple[Path, Mapping[str, Any]]]],
    *,
    radius_m: float,
    min_step_gap: int,
    min_path_m: float,
) -> List[Dict[str, Any]]:
    events = []
    for key, values in trajectory.items():
        for index, (path, row) in enumerate(values):
            step = safe_int(row.get("eval_step"))
            gps = row.get("gps")
            if step is None or not isinstance(gps, (list, tuple)):
                continue
            best = None
            path_m = 0.0
            for prior_index in range(index - 1, -1, -1):
                prior_step = safe_int(values[prior_index][1].get("eval_step"))
                prior_gps = values[prior_index][1].get("gps")
                if prior_step is None or step - prior_step < min_step_gap:
                    continue
                # Sum the observed route between the two poses, so repeated
                # samples while standing still do not become route revisits.
                path_m = 0.0
                for k in range(prior_index + 1, index + 1):
                    path_m += distance(values[k - 1][1].get("gps"), values[k][1].get("gps"))
                if distance(gps, prior_gps) <= radius_m and path_m >= min_path_m:
                    best = (prior_step, prior_gps, path_m)
                    break
            if best is None:
                continue
            prior_step, prior_gps, path_m = best
            recovered, recovery_step = self_recovered(values, step)
            events.append(
                {
                    "event_type": "route_revisit",
                    "scene_id": row.get("scene_id"),
                    "episode_id": row.get("episode_id"),
                    "step_id": step,
                    "prior_step": prior_step,
                    "revisit_distance_m": distance(gps, prior_gps),
                    "route_path_m": path_m,
                    "self_recovered": recovered,
                    "recovery_step": recovery_step,
                    "source": str(path),
                    "reasons": ["gps_revisit_hindsight"],
                }
            )
    return events


def compact_outcome(progress: Mapping[str, Any]) -> Dict[str, Any]:
    success = bool((safe_float(progress.get("success"), 0.0) or 0.0) >= 0.5)
    collision = safe_float(progress.get("collision_count"), 0.0) or 0.0
    failure_type = str(progress.get("stage19_semantic_resilience_episode_failure_type") or "unknown")
    timeout = (safe_int(progress.get("steps"), 0) or 0) >= 500
    return {
        "success": success,
        "collision_count": collision,
        "collision_episode": collision > 0,
        "failure_type": failure_type,
        "timeout": timeout,
        "steps": safe_int(progress.get("steps"), 0) or 0,
        "spl": safe_float(progress.get("spl")),
        "ne": safe_float(progress.get("ne")),
    }


def event_record(
    source_path: Path,
    row: Mapping[str, Any],
    event_type: str,
    trajectories: Mapping[str, Sequence[Tuple[Path, Mapping[str, Any]]]],
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    key = episode_key(row)
    step = safe_int(row.get("step_id"), safe_int(row.get("eval_step"), 0)) or 0
    recovered, recovery_step = self_recovered(trajectories.get(key, []), step)
    result = dict(outcomes.get(key, {}))
    return {
        "event_type": event_type,
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "step_id": step,
        "end_step": safe_int(row.get("step_id"), step) or step,
        "reasons": list(reasons or row.get("trigger_reasons") or []),
        "source": str(source_path),
        "rgb_path": latest_image(source_path, row),
        "self_recovered": recovered,
        "recovery_step": recovery_step,
        "outcome": result,
        "evidence": {},
    }


def dedupe_events(events: Sequence[Dict[str, Any]], min_gap: int = 8):
    output = []
    by_episode_type: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for event in sorted(events, key=lambda item: (str(item.get("scene_id")), int(item.get("episode_id") or 0), int(item.get("step_id") or 0))):
        key = (episode_key(event), str(event.get("event_type")))
        step = int(event.get("step_id") or 0)
        if any(abs(step - prior) < min_gap for prior in by_episode_type[key]):
            continue
        by_episode_type[key].append(step)
        output.append(event)
    return output


def outcome_proxy_metrics(events: Sequence[Mapping[str, Any]], all_outcomes: Mapping[str, Mapping[str, Any]]):
    failed = {key for key, outcome in all_outcomes.items() if not outcome.get("success")}
    collisions = {key for key, outcome in all_outcomes.items() if outcome.get("collision_episode")}
    predicted = {episode_key(event) for event in events}
    true_positive = predicted & failed
    collision_positive = predicted & collisions
    return {
        "predicted_event_count": len(events),
        "predicted_episode_count": len(predicted),
        "failed_episode_count": len(failed),
        "failed_episode_detected_count": len(true_positive),
        "episode_outcome_precision_proxy": len(true_positive) / len(predicted) if predicted else None,
        "episode_outcome_recall_proxy": len(true_positive) / len(failed) if failed else None,
        "success_episode_trigger_count": len(predicted - failed),
        "success_episode_trigger_rate": len(predicted - failed) / max(1, len(predicted)),
        "collision_episode_count": len(collisions),
        "collision_episode_detected_count": len(collision_positive),
        "collision_episode_precision_proxy": (
            len(collision_positive) / len(predicted) if predicted else None
        ),
        "collision_episode_recall_proxy": (
            len(collision_positive) / len(collisions) if collisions else None
        ),
    }


def build_report(
    run_root: Path,
    *,
    output: Path,
    revisit_radius_m: float,
    min_revisit_gap: int,
    min_revisit_path_m: float,
    copy_evidence: bool,
) -> Dict[str, Any]:
    discovered = discover(run_root)
    progress = progress_map(discovered["progress"])
    trajectories = trajectory_by_episode(discovered["trajectory"])
    outcomes = {key: compact_outcome(row) for key, row in progress.items()}

    d0: List[Dict[str, Any]] = []
    for path, row in discovered["loops"]:
        if row.get("transition") == "start":
            item = event_record(path, row, "strict_turn_loop", trajectories, outcomes)
            item["evidence"] = {
                "failure_type": row.get("failure_type"),
                "translation_m": row.get("translation_m"),
                "heading_cycle_error_deg": row.get("heading_cycle_error_deg"),
                "triage_tier": row.get("triage_tier"),
                "shadow_only": row.get("shadow_only"),
            }
            d0.append(item)
    for path, row in discovered["snapshots"]:
        item = event_record(path, row, "repeated_action", trajectories, outcomes)
        item["evidence"] = {
            "dominant_action": row.get("dominant_action"),
            "dominant_action_ratio": row.get("dominant_action_ratio"),
            "action_window": row.get("action_window"),
            "environment_step_applied": row.get("environment_step_applied"),
        }
        d0.append(item)
    # The old runtime does not persist a per-step STOP/collision onset.  Keep
    # timeout and near-goal-no-stop as explicit end-of-episode diagnostics,
    # without pretending they provide an onset label.
    for key, outcome in outcomes.items():
        if not outcome.get("timeout") and outcome.get("failure_type") != "near_goal_no_stop":
            continue
        scene_id, raw_episode = key.split("|", 1)
        row = {
            "scene_id": scene_id,
            "episode_id": safe_int(raw_episode, raw_episode),
            "step_id": outcome.get("steps", 0),
        }
        item = event_record(
            run_root / "progress.json",
            row,
            "timeout_or_wrong_stop",
            trajectories,
            outcomes,
            reasons=["timeout" if outcome.get("timeout") else "near_goal_no_stop"],
        )
        item["evidence"] = {
            "timeout": outcome.get("timeout"),
            "failure_type": outcome.get("failure_type"),
            "step_onset_available": False,
            "stop_action_status_available": False,
        }
        d0.append(item)
    d0 = dedupe_events(d0)

    revisit = route_revisit_events(
        trajectories,
        radius_m=revisit_radius_m,
        min_step_gap=min_revisit_gap,
        min_path_m=min_revisit_path_m,
    )
    for item in revisit:
        key = episode_key(item)
        item["outcome"] = dict(outcomes.get(key, {}))
        item["evidence"] = {
            "revisit_distance_m": item.get("revisit_distance_m"),
            "route_path_m": item.get("route_path_m"),
            "prior_step": item.get("prior_step"),
            "hindsight_geometry_label": True,
        }
    d1 = d0 + revisit

    occ_conflict = []
    for path, row in discovered["resilience"]:
        candidate = row.get("candidate") or {}
        triggers = set(row.get("trigger_reasons") or [])
        tags = set(row.get("recovery_context_tags") or [])
        reasons = sorted(
            triggers & {
                "local_trap",
                "current_waypoint_occupied",
                "current_points_to_revisited_region",
                "policy_memory_conflict",
                "spatial_constriction",
            }
        )
        # A gate rejection alone is not an OCC collision label.  Keep this
        # detector strict: only an explicit geometric unsafe state can add a
        # conflict in the absence of a named route/occupancy trigger.
        candidate_conflict = candidate.get("geometry_safe") is False
        if not reasons and not candidate_conflict:
            continue
        if not reasons and candidate_conflict:
            reasons = ["candidate_safety_conflict"]
        item = event_record(path, row, "occ_policy_conflict", trajectories, outcomes, reasons=reasons)
        item["evidence"] = {
            "failure_type": row.get("failure_type"),
            "current_problem": row.get("current_problem"),
            "geometry_safe": candidate.get("geometry_safe"),
            "active_gate_safe": candidate.get("active_gate_safe"),
            "goal_state": candidate.get("goal_state"),
            "recovery_context_tags": sorted(tags),
        }
        occ_conflict.append(item)
    d1 = dedupe_events(d1 + occ_conflict)

    semantic = []
    for path, row in discovered["resilience"]:
        failure_type = str(row.get("failure_type") or "")
        triggers = set(row.get("trigger_reasons") or [])
        reasons = sorted(
            triggers & {"semantic_dead_zone", "semantic_stagnation", "semantic_drift", "semantic_uncertainty"}
        )
        if failure_type in {"semantic_stagnation", "semantic_drift_revisit"}:
            reasons = reasons or [failure_type]
        if not reasons:
            continue
        item = event_record(path, row, "semantic_progress_drift", trajectories, outcomes, reasons=reasons)
        item["evidence"] = {
            "semantic_backend": "weak_clip_logged_event",
            "semantic_precision_backend_available": False,
            "failure_type": failure_type,
            "trigger_reasons": sorted(triggers),
            "current_problem": row.get("current_problem"),
            "v2_evidence_tier": row.get("v2_evidence_tier"),
        }
        semantic.append(item)
    d2 = dedupe_events(d1 + semantic)

    all_events = {"D0": d0, "D1": d1, "D2": d2}
    summary = {
        "task": "stage25_failure_detector_offline_audit",
        "run_root": str(run_root),
        "detector_contract": {
            "shadow_only": True,
            "actions_changed": False,
            "candidate_generation": False,
            "candidate_ranking": False,
            "recovery_execution": False,
            "unknown_is_free": False,
        },
        "dataset_gt_scope": {
            "episode_outcome": "progress.success/collision_count/steps",
            "event_gt_available": False,
            "event_metrics_are_outcome_proxies": True,
            "route_revisit_is_hindsight_geometry_label": True,
        },
        "semantic_scope": {
            "d2_source": "weak_clip_logged_event",
            "precise_lseg_used": False,
            "reason": "Stage24 LSeg replay covers a separate 5-episode manifest and semantic nodes remain audit-only",
        },
        "counts": {
            "progress_episodes": len(progress),
            "trajectory_rows": len(discovered["trajectory"]),
            "strict_loop_rows": len(discovered["loops"]),
            "stuck_snapshot_rows": len(discovered["snapshots"]),
            "semantic_rows": len(discovered["semantic"]),
            "resilience_rows": len(discovered["resilience"]),
            "occ_audit_rows": len(discovered["occ_audit"]),
        },
        "outcome_summary": {
            "success_episodes": sum(bool(item["success"]) for item in outcomes.values()),
            "failure_episodes": sum(not bool(item["success"]) for item in outcomes.values()),
            "collision_episodes": sum(bool(item["collision_episode"]) for item in outcomes.values()),
            "timeout_episodes": sum(bool(item["timeout"]) for item in outcomes.values()),
            "failure_type_counts": dict(Counter(str(item["failure_type"]) for item in outcomes.values())),
        },
        "detectors": {},
        "event_examples": {},
    }
    for name, events in all_events.items():
        by_type = Counter(str(event["event_type"]) for event in events)
        self_recovered_count = sum(bool(event.get("self_recovered")) for event in events)
        summary["detectors"][name] = {
            "event_count": len(events),
            "episode_count": len({episode_key(event) for event in events}),
            "event_type_counts": dict(by_type),
            "self_recovered_event_count": self_recovered_count,
            "self_recovered_event_rate": self_recovered_count / len(events) if events else None,
            "outcome_correlation_proxy": outcome_proxy_metrics(events, outcomes),
        }
        summary["event_examples"][name] = events[:20]

    output.mkdir(parents=True, exist_ok=True)
    (output / "stage25_detector_events.json").write_text(
        json.dumps(all_events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evidence = []
    for name, events in all_events.items():
        for event in events:
            if event.get("rgb_path"):
                evidence.append(
                    {
                        "detector": name,
                        "event_type": event.get("event_type"),
                        "scene_id": event.get("scene_id"),
                        "episode_id": event.get("episode_id"),
                        "step_id": event.get("step_id"),
                        "self_recovered": event.get("self_recovered"),
                        "success": (event.get("outcome") or {}).get("success"),
                        "rgb_path": event.get("rgb_path"),
                        "source": event.get("source"),
                    }
                )
    if copy_evidence:
        evidence_dir = output / "evidence_images"
        evidence_dir.mkdir(exist_ok=True)
        for item in evidence:
            source = Path(str(item["rgb_path"]))
            if not source.is_file():
                continue
            target = evidence_dir / f"{item['scene_id']}_{item['episode_id']}_step{item['step_id']}_{item['detector']}.jpg"
            shutil.copy2(source, target)
            item["copied_path"] = str(target)
    (output / "stage25_evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "stage25_detector_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revisit-radius-m", type=float, default=0.35)
    parser.add_argument("--min-revisit-gap", type=int, default=12)
    parser.add_argument("--min-revisit-path-m", type=float, default=0.75)
    parser.add_argument("--copy-evidence", action="store_true")
    args = parser.parse_args()
    summary = build_report(
        args.run_root,
        output=args.output,
        revisit_radius_m=args.revisit_radius_m,
        min_revisit_gap=args.min_revisit_gap,
        min_revisit_path_m=args.min_revisit_path_m,
        copy_evidence=args.copy_evidence,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

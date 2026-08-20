#!/usr/bin/env python3
"""Audit Stage25 replay contracts and mine causal stuck-event candidates.

The script is offline-only. It treats final navigation outcome as metadata, not
event ground truth, and never reads future observations when deciding an onset.
Future observations are used only to label self-recovery and persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distance(a: Any, b: Any) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def sha256_array(path: Path, *, rgb: bool) -> str:
    if rgb:
        array = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    else:
        with np.load(path) as payload:
            array = np.ascontiguousarray(payload["depth_m"], dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def discover_episodes(run_root: Path) -> List[Path]:
    return sorted({path.parent for path in run_root.glob("**/replay_ledger/*/observations.jsonl")})


def progress_by_episode(run_root: Path) -> Dict[str, Dict[str, Any]]:
    output = {}
    for path in run_root.glob("**/progress.json"):
        for row in jsonl(path):
            output[episode_key(row)] = row
    return output


def loop_events_by_episode(run_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in run_root.glob("**/s2_action_loop_events.jsonl"):
        for row in jsonl(path):
            if row.get("transition") == "start":
                output[episode_key(row)].append(row)
    return output


def lseg_events(episode_dir: Path) -> List[Dict[str, Any]]:
    run_dir = episode_dir.parent.parent
    prefix = episode_dir.name.rsplit("_r", 1)[0]
    candidates = sorted((run_dir / "online_lseg_shadow").glob(f"{prefix}_r*/events.jsonl"))
    return jsonl(candidates[-1]) if candidates else []


def compact_observation(row: Mapping[str, Any]) -> Dict[str, Any]:
    pose = row.get("pose") or {}
    occ = row.get("occ_summary") or {}
    audit = row.get("audit_metrics") or {}
    return {
        "record_index": int(row.get("record_index", 0)),
        "step_id": int(row.get("step_id", 0)),
        "observation_key": row.get("observation_key"),
        "gps": pose.get("gps"),
        "compass": pose.get("compass"),
        "previous_action": row.get("previous_action"),
        "previous_action_applied": row.get("previous_action_applied"),
        "collision_count": audit.get("collision_count"),
        "collision_delta": audit.get("collision_delta"),
        "distance_to_goal": audit.get("distance_to_goal"),
        "success_so_far": audit.get("success"),
        "occupied_added": occ.get("occupied_added"),
        "free_added": occ.get("free_added"),
        "occupied_voxel_count": occ.get("occupied_voxel_count"),
        "free_voxel_count": occ.get("free_voxel_count"),
        "frontier_count": occ.get("frontier_count"),
        "rgb_path": row.get("rgb_path"),
        "depth_path": row.get("depth_path"),
    }


def window_displacement(rows: Sequence[Mapping[str, Any]]) -> float:
    finite = [row.get("gps") for row in rows if isinstance(row.get("gps"), (list, tuple))]
    return 0.0 if len(finite) < 2 else distance(finite[0], finite[-1])


def path_length(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(distance(a.get("gps"), b.get("gps")) for a, b in zip(rows, rows[1:]))


def recovery_label(rows: Sequence[Mapping[str, Any]], index: int) -> Tuple[str, Optional[int], Optional[float]]:
    start = rows[index].get("gps")
    if not isinstance(start, (list, tuple)):
        return "unknown", None, None
    limit = min(len(rows), index + 33)
    for later in range(index + 8, limit):
        moved = distance(start, rows[later].get("gps"))
        if moved >= 0.60:
            return "self_recovered", int(rows[later]["step_id"]), moved
    return "persistent_in_observed_horizon", None, None


def route_revisit(rows: Sequence[Mapping[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    current = rows[index]
    for prior in range(index - 12, -1, -1):
        segment = rows[prior:index + 1]
        route_m = path_length(segment)
        revisit_m = distance(current.get("gps"), rows[prior].get("gps"))
        if route_m >= 0.75 and revisit_m <= 0.35:
            return {
                "prior_step": int(rows[prior]["step_id"]),
                "route_path_m": route_m,
                "revisit_distance_m": revisit_m,
            }
    return None


def semantic_context(events: Sequence[Mapping[str, Any]], step: int) -> Dict[str, Any]:
    past = [event for event in events if event.get("valid") and int(event.get("step_id", -1)) <= step]
    recent = past[-4:]
    labels = [set((event.get("class_surface_counts") or {}).keys()) for event in recent]
    union = sorted(set().union(*labels)) if labels else []
    repeated = bool(len(labels) >= 4 and labels[-1] and all(item == labels[-1] for item in labels))
    return {
        "available": bool(past),
        "recent_query_count": len(recent),
        "recent_labels": union,
        "repeated_label_set": repeated,
        "last_query_step": int(past[-1]["step_id"]) if past else None,
    }


def mine_events(
    observations: Sequence[Mapping[str, Any]],
    loops: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    rows = [compact_observation(row) for row in observations]
    candidates: List[Dict[str, Any]] = []
    loop_steps = {int(row.get("step_id", -1)): row for row in loops}
    last_event_step: Dict[str, int] = {}
    for index, row in enumerate(rows):
        step = int(row["step_id"] or 0)
        recent8 = rows[max(0, index - 7):index + 1]
        recent12 = rows[max(0, index - 11):index + 1]
        collision_burst = sum(float(item.get("collision_delta") or 0.0) for item in recent8)
        forward_count = sum(
            int(item.get("previous_action") == 1 and item.get("previous_action_applied") is not False)
            for item in recent8
        )
        displacement = window_displacement(recent8)
        occ_growth = sum(
            int(item.get("occupied_added") or 0) + int(item.get("free_added") or 0)
            for item in recent12
        )
        evidence: List[str] = []
        family = None
        if step in loop_steps:
            family = "G2_policy_loop"
            evidence.append("strict_s2_turn_loop")
        if collision_burst >= 2 and displacement <= 0.25:
            family = family or "G1_geometry_execution"
            evidence.append("collision_burst_low_displacement")
        if forward_count >= 3 and displacement <= 0.15:
            family = family or "G1_geometry_execution"
            evidence.append("commanded_forward_not_realized")
        revisit = route_revisit(rows, index)
        if revisit is not None:
            family = family or "G3_route_topology"
            evidence.append("route_revisit")
        if family is None:
            continue
        if step - last_event_step.get(family, -1000) < 8:
            continue
        last_event_step[family] = step
        recovery, recovery_step, recovery_m = recovery_label(rows, index)
        sem = semantic_context(semantic, step)
        candidates.append({
            "step_id": step,
            "observation_index": int(row["record_index"]),
            "event_family": family,
            "evidence": evidence,
            "window": {
                "collision_delta_8": collision_burst,
                "forward_count_8": forward_count,
                "displacement_8_m": displacement,
                "path_length_8_m": path_length(recent8),
                "occ_new_voxels_12": occ_growth,
                **(revisit or {}),
            },
            "semantic_confirmation": {
                **sem,
                "supports_existing_suspicion": bool(sem.get("repeated_label_set")),
            },
            "recoverability_proxy": recovery,
            "recovery_step": recovery_step,
            "recovery_displacement_m": recovery_m,
            "rgb_path": row.get("rgb_path"),
        })
    return {
        "D0": [event for event in candidates if "strict_s2_turn_loop" in event["evidence"]],
        "D1": [event for event in candidates if event["event_family"] in {"G1_geometry_execution", "G2_policy_loop"}],
        "D2": candidates,
        "D3Q_confirmed": [event for event in candidates if event["semantic_confirmation"]["supports_existing_suspicion"]],
    }


def audit_episode(episode_dir: Path, loops: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    observations = jsonl(episode_dir / "observations.jsonl")
    queries = jsonl(episode_dir / "queries.jsonl")
    actions = jsonl(episode_dir / "actions.jsonl")
    keys = {row.get("observation_key") for row in observations}
    prior_collision = 0.0
    for row in observations:
        key = row.get("observation_key")
        pose = row.get("pose") or {}
        audit = row.get("audit_metrics") or {}
        for field in ("distance_to_goal", "collision_count", "collision_delta"):
            if audit.get(field) is None:
                errors.append(f"missing_{field}:{key}")
        current_collision = float(audit.get("collision_count", prior_collision) or 0.0)
        expected_delta = max(0.0, current_collision - prior_collision)
        if abs(float(audit.get("collision_delta", expected_delta) or 0.0) - expected_delta) > 1e-6:
            errors.append(f"collision_delta_mismatch:{key}")
        if current_collision < prior_collision:
            errors.append(f"collision_not_monotonic:{key}")
        prior_collision = current_collision
        if pose.get("gps") is None or pose.get("stage23_gt_camera_pose_map") is None:
            errors.append(f"pose_missing:{key}")
        for field, rgb in (("rgb_path", True), ("depth_path", False)):
            relative = row.get(field)
            path = episode_dir / str(relative) if relative else None
            if path is None or not path.is_file():
                errors.append(f"{field}_missing:{key}")
            elif sha256_array(path, rgb=rgb) != row.get("rgb_sha256" if rgb else "depth_sha256"):
                errors.append(f"{field}_hash_mismatch:{key}")
    if any(row.get("observation_key") not in keys for row in queries + actions):
        errors.append("invalid_observation_reference")
    prior_action_collision = 0.0
    for row in actions:
        audit = row.get("audit_metrics") or {}
        action_index = row.get("action_index")
        for field in ("distance_to_goal", "collision_count", "collision_delta"):
            if audit.get(field) is None:
                errors.append(f"action_missing_{field}:{action_index}")
        current_collision = float(
            audit.get("collision_count", prior_action_collision) or 0.0
        )
        expected_delta = max(0.0, current_collision - prior_action_collision)
        if abs(float(audit.get("collision_delta", expected_delta) or 0.0) - expected_delta) > 1e-6:
            errors.append(f"action_collision_delta_mismatch:{action_index}")
        if current_collision < prior_action_collision:
            errors.append(f"action_collision_not_monotonic:{action_index}")
        prior_action_collision = current_collision
    if not (meta.get("semantic_scene_gt") or {}).get("available"):
        errors.append("semantic_scene_gt_missing")
    final_collision = (summary.get("final_metrics") or {}).get("collision_count")
    if final_collision is None or abs(float(final_collision) - prior_action_collision) > 1e-6:
        errors.append("final_collision_mismatch")
    return {
        "scene_id": meta.get("scene_id"),
        "episode_id": meta.get("episode_id"),
        "observation_count": len(observations),
        "query_count": len(queries),
        "action_count": len(actions),
        "loop_count": len(loops),
        "final_metrics": summary.get("final_metrics") or {},
        "ledger_dir": str(episode_dir),
    }, errors


def analyze(run_root: Path, output: Path, require_all: bool) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    progress = progress_by_episode(run_root)
    loops = loop_events_by_episode(run_root)
    contract_errors = []
    episode_reports = []
    all_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    annotation = []
    for episode_dir in discover_episodes(run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        key = episode_key(meta)
        report, errors = audit_episode(episode_dir, loops.get(key, []))
        episode_reports.append(report)
        contract_errors.extend(f"{key}:{error}" for error in errors)
        observations = jsonl(episode_dir / "observations.jsonl")
        semantic = lseg_events(episode_dir)
        variants = mine_events(observations, loops.get(key, []), semantic)
        outcome = progress.get(key, {})
        for variant, events in variants.items():
            for event in events:
                event.update({
                    "scene_id": meta.get("scene_id"),
                    "episode_id": meta.get("episode_id"),
                    "outcome": {
                        "success": outcome.get("success"),
                        "spl": outcome.get("spl"),
                        "steps": outcome.get("steps"),
                    },
                })
                all_events[variant].append(event)
                if variant == "D2":
                    annotation.append({
                        **event,
                        "annotation": {
                            "state": None,
                            "type": None,
                            "onset_step": None,
                            "end_step": None,
                            "recoverability": None,
                            "failure_link": None,
                            "intervention_likely_needed": None,
                            "confidence": None,
                            "notes": "",
                        },
                    })
    detector_summary = {}
    for variant in ("D0", "D1", "D2", "D3Q_confirmed"):
        events = all_events.get(variant, [])
        detector_summary[variant] = {
            "event_count": len(events),
            "episode_count": len({episode_key(event) for event in events}),
            "event_family_counts": dict(Counter(event["event_family"] for event in events)),
            "self_recovered_count": sum(event["recoverability_proxy"] == "self_recovered" for event in events),
            "persistent_proxy_count": sum(event["recoverability_proxy"].startswith("persistent") for event in events),
        }
    report = {
        "task": "stage25_gt_detector_contract_and_candidate_mining",
        "contract_passed": not contract_errors and bool(episode_reports),
        "event_gt_status": "objective_proxy_pending_manual_interval_annotation",
        "outcome_is_event_gt": False,
        "future_used_by_detector": False,
        "future_used_for_recoverability_label_only": True,
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
        "detectors": detector_summary,
        "errors": contract_errors,
    }
    (output / "stage25_contract_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "stage25_event_candidates.json").write_text(json.dumps(all_events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "stage25_annotation_manifest.json").write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if require_all and not report["contract_passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    analyze(args.run_root, args.output, args.require_all)


if __name__ == "__main__":
    main()

"""Build a Stage21 candidate-recoverability dataset from sharded shadow runs.

The wrapper reuses the Stage17 GT alignment and route-progress label builders,
then adds multi-rank discovery, deterministic deduplication, episode outcomes,
triage audits, and a manifest suitable for reproducible offline training.
Episode success is retained only as context and is never used as a candidate
label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_route_progress_ranker_dataset import (  # noqa: E402
    _candidate_status,
    _candidate_xy,
    _is_repeated_candidate,
    _project_to_polyline,
    _reference_path_for_row,
    build_dataset,
)
from collect_gt_candidate_labels import (  # noqa: E402
    _load_reference_paths,
    _read_json_records,
    _safe_float,
    _scene_token,
    collect_labels,
)


def _event_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        _scene_token(row.get("scene_id")),
        str(row.get("episode_id")),
        int(row.get("step_id", -1)),
    )


def _episode_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (_scene_token(row.get("scene_id")), str(row.get("episode_id")))


def discover_run_dirs(run_root: Path) -> List[Path]:
    """Return directories containing both candidate and trajectory logs."""
    root = Path(run_root)
    candidates = []
    if (root / "occ_memory" / "memory_events.jsonl").is_file():
        candidates.append(root)
    if root.exists():
        for path in root.rglob("memory_events.jsonl"):
            if path.parent.name != "occ_memory":
                continue
            run_dir = path.parent.parent
            if (run_dir / "trajectory_events.jsonl").is_file():
                candidates.append(run_dir)
    return sorted(set(path.resolve() for path in candidates), key=str)


def _load_outcomes(run_dirs: Sequence[Path]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Counter]:
    outcomes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    counts = Counter()
    for run_dir in run_dirs:
        for row in _read_json_records(run_dir / "progress.json"):
            counts["progress_records"] += 1
            key = _episode_key(row)
            if key in outcomes:
                counts["duplicate_progress_records"] += 1
                continue
            outcomes[key] = {
                "success": row.get("success"),
                "spl": row.get("spl"),
                "ne": row.get("ne"),
                "steps": row.get("steps"),
                "collision_count": row.get("collision_count"),
                "failure_type": row.get("stage19_semantic_resilience_episode_failure_type"),
            }
    return outcomes, counts


def _load_triage(run_dirs: Sequence[Path]) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Counter]:
    triage_by_event: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    counts = Counter()
    for run_dir in run_dirs:
        path = run_dir / "stage19_semantic_resilience_active_events.jsonl"
        for row in _read_json_records(path):
            counts["events"] += 1
            tier = str(row.get("v2_evidence_tier") or "unknown")
            counts[f"tier={tier}"] += 1
            if bool(row.get("applied")):
                counts["applied"] += 1
            key = _event_key(row)
            triage_by_event.setdefault(
                key,
                {
                    "tier": tier,
                    "reason": row.get("v2_evidence_reason") or row.get("reason"),
                    "considered": bool(row.get("considered")),
                    "applied": bool(row.get("applied")),
                    "failure_type": row.get("failure_type"),
                    "recommended_primitive": row.get("recommended_primitive"),
                },
            )
    return triage_by_event, counts


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_route_value(
    candidate: Dict[str, Any],
    *,
    current_arc: float,
    current_route_dist: float,
    path: Sequence[Tuple[float, float]],
) -> Optional[Dict[str, float]]:
    xy = _candidate_xy(candidate)
    if xy is None:
        return None
    projection = _project_to_polyline(xy, path)
    if projection is None:
        return None
    candidate_arc, candidate_route_dist, candidate_segment = projection
    progress_m = candidate_arc - current_arc
    offroute_delta_m = max(0.0, candidate_route_dist - current_route_dist)
    value_m = progress_m - 0.50 * offroute_delta_m
    if _candidate_status(candidate) == "completed":
        value_m -= 1.25
    if _is_repeated_candidate(candidate):
        value_m -= 0.25
    value_m += 0.20 * max(0.0, _safe_float(candidate.get("next_landmark_relevance")))
    if bool(candidate.get("target_frontier_candidate")):
        value_m += 0.10
    return {
        "route_progress_m": float(progress_m),
        "route_value_m": float(value_m),
        "route_distance_m": float(candidate_route_dist),
        "route_segment_index": int(candidate_segment),
    }


def _build_candidate_rows(
    labels: Sequence[Dict[str, Any]],
    *,
    episodes_file: Optional[Path],
    reference_frame: str,
    quaternion_order: str,
    coordinate_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Separate deployable online inputs from GT-only relative-value labels."""
    if episodes_file is None:
        return [], {"status": "not_built_without_episodes_file"}
    reference_metadata = _load_reference_paths(episodes_file)
    prepared_paths: Dict[str, Any] = {}
    rows = []
    counts = Counter()
    source_counts = Counter()
    direction_counts = Counter()
    tier_counts = Counter()
    preferences_by_event: Dict[Tuple[str, str, int], Counter] = {}

    for event in labels:
        counts["input_events"] += 1
        if event.get("label_status") != "ok":
            counts[f"drop_status={event.get('label_status')}"] += 1
            continue
        current_xy = event.get("current_xy")
        if not isinstance(current_xy, (list, tuple)) or len(current_xy) < 2:
            counts["drop_missing_current_xy"] += 1
            continue
        path, error = _reference_path_for_row(
            event,
            reference_metadata,
            prepared_paths,
            reference_frame=reference_frame,
            quaternion_order=quaternion_order,
            coordinate_mode=coordinate_mode,
        )
        if path is None:
            counts[f"drop_{error}"] += 1
            continue
        current_projection = _project_to_polyline(
            (_safe_float(current_xy[0]), _safe_float(current_xy[1])), path
        )
        if current_projection is None:
            counts["drop_current_projection_failed"] += 1
            continue
        current_arc, current_route_dist, _ = current_projection
        s2_candidate = event.get("current_policy_candidate") or {}
        s2_route = _candidate_route_value(
            s2_candidate,
            current_arc=current_arc,
            current_route_dist=current_route_dist,
            path=path,
        )
        if s2_route is None:
            counts["events_without_s2_route_value"] += 1
            continue
        event_key = _event_key(event)
        event_preferences = Counter()
        triage = event.get("triage_context") or {}
        tier = str(triage.get("tier") or "not_considered")
        for candidate in event.get("candidates") or []:
            route = _candidate_route_value(
                candidate,
                current_arc=current_arc,
                current_route_dist=current_route_dist,
                path=path,
            )
            if route is None:
                counts["drop_candidate_without_route_value"] += 1
                continue
            status = _candidate_status(candidate)
            online_safe = bool(candidate.get("geometry_safe", True)) and bool(
                candidate.get("active_gate_safe", True)
            )
            online_safe = online_safe and status != "completed" and not _is_repeated_candidate(candidate)
            advantage = route["route_value_m"] - s2_route["route_value_m"]
            if not online_safe or advantage <= -0.20:
                preference = "negative"
            elif advantage >= 0.20:
                preference = "positive"
            else:
                preference = "tie_or_ambiguous"
            event_preferences[preference] += 1
            source = str(candidate.get("candidate_type") or candidate.get("source") or "unknown")
            direction = str(candidate.get("direction_bucket") or "unknown")
            source_counts[source] += 1
            direction_counts[direction] += 1
            tier_counts[tier] += 1
            rows.append(
                {
                    "identity": {
                        "split": event.get("split"),
                        "rank": event.get("rank"),
                        "eval_random_seed": event.get("eval_random_seed"),
                        "episode_eval_seed": event.get("episode_eval_seed"),
                        "scene_id": event.get("scene_id"),
                        "episode_id": event.get("episode_id"),
                        "step_id": event.get("step_id"),
                        "candidate_id": candidate.get("candidate_id"),
                    },
                    "online_inputs": {
                        "current_policy_candidate": s2_candidate,
                        "candidate": candidate,
                        "triage_context": triage,
                    },
                    "offline_labels": {
                        **route,
                        "s2_route_progress_m": s2_route["route_progress_m"],
                        "s2_route_value_m": s2_route["route_value_m"],
                        "advantage_vs_s2_m": float(advantage),
                        "preference_vs_s2": preference,
                        "online_safe": bool(online_safe),
                    },
                    "episode_outcome_context_only": event.get(
                        "episode_outcome_context_only"
                    ),
                }
            )
        preferences_by_event[event_key] = event_preferences

    pair_count = 0
    paired_event_count = 0
    for event_preferences in preferences_by_event.values():
        positive = event_preferences.get("positive", 0)
        negative = event_preferences.get("negative", 0)
        if positive and negative:
            paired_event_count += 1
            pair_count += positive * negative
    counts["candidate_rows"] = len(rows)
    counts["events_with_positive_and_negative"] = paired_event_count
    counts["positive_vs_negative_pairs"] = pair_count
    counts.update(
        Counter(row["offline_labels"]["preference_vs_s2"] for row in rows)
    )
    return rows, {
        "status": "ok",
        "counts": dict(counts),
        "candidate_source_counts": dict(source_counts),
        "candidate_direction_counts": dict(direction_counts),
        "triage_tier_candidate_counts": dict(tier_counts),
        "label_margin_m": 0.20,
        "online_offline_field_separation": True,
        "final_episode_success_used_as_candidate_label": False,
    }


def build_stage21_dataset(
    *,
    run_root: Path,
    episodes_file: Optional[Path],
    output_dir: Path,
    reference_frame: str = "episodic_gps",
    reference_coordinate_mode: str = "x_neg_y",
    gps_coordinate_mode: str = "x_neg_y",
    quaternion_order: str = "xyzw",
    lookahead_m: float = 1.5,
    max_angle_deg: float = 60.0,
    step_min: int = 0,
    step_max: int = 500,
    val_ratio: float = 0.15,
    split_seed: int = 21,
    split_key: str = "scene",
) -> Dict[str, Any]:
    run_dirs = discover_run_dirs(run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No Stage21-compatible run directories found under {run_root}")

    outcomes, outcome_counts = _load_outcomes(run_dirs)
    triage_by_event, triage_counts = _load_triage(run_dirs)
    merged: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    collect_summaries = []
    duplicate_rows = 0

    for run_dir in run_dirs:
        rows, collect_summary = collect_labels(
            run_dir=run_dir,
            episodes_file=episodes_file,
            reference_frame=reference_frame,
            reference_coordinate_mode=reference_coordinate_mode,
            gps_coordinate_mode=gps_coordinate_mode,
            quaternion_order=quaternion_order,
            lookahead_m=lookahead_m,
            max_angle_deg=max_angle_deg,
            step_min=step_min,
            step_max=step_max,
        )
        collect_summaries.append(collect_summary)
        for row in rows:
            key = _event_key(row)
            if key in merged:
                duplicate_rows += 1
                continue
            enriched = dict(row)
            enriched["source_run_dir"] = str(run_dir)
            enriched["episode_outcome_context_only"] = outcomes.get(key[:2])
            enriched["triage_context"] = triage_by_event.get(key)
            merged[key] = enriched

    labels = [merged[key] for key in sorted(merged)]
    status_counts = Counter(str(row.get("label_status")) for row in labels)
    scene_count = len({_scene_token(row.get("scene_id")) for row in labels})
    episode_count = len({_episode_key(row) for row in labels})
    candidate_count = sum(len(row.get("candidates") or []) for row in labels)
    multi_candidate_rows = sum(len(row.get("candidates") or []) >= 2 for row in labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "labels.jsonl", labels)

    candidate_rows, candidate_row_audit = _build_candidate_rows(
        labels,
        episodes_file=episodes_file,
        reference_frame=reference_frame,
        quaternion_order=quaternion_order,
        coordinate_mode=gps_coordinate_mode,
    )
    _write_jsonl(output_dir / "rows.jsonl", candidate_rows)

    dataset_outputs = {"train": [], "val": []}
    dataset_summary: Dict[str, Any] = {"not_built": True}
    if episodes_file is not None:
        dataset_outputs, dataset_summary = build_dataset(
            labels,
            episodes_file=episodes_file,
            reference_frame=reference_frame,
            quaternion_order=quaternion_order,
            coordinate_mode=gps_coordinate_mode,
            label_source="stage21_route_progress_value_v1",
            val_ratio=val_ratio,
            split_seed=split_seed,
            split_key=split_key,
            min_positive_progress_m=0.20,
            offroute_penalty_weight=0.50,
            completed_penalty_m=1.25,
            repeated_penalty_m=0.25,
            next_landmark_bonus_m=0.20,
            target_frontier_bonus_m=0.10,
            label_clip_m=3.0,
            drop_completed_only_positive=True,
            hard_only=False,
            hard_heuristics=("front_bucket", "intent_alignment"),
        )
    for split in ("train", "val"):
        _write_jsonl(output_dir / f"{split}.jsonl", dataset_outputs[split])

    train_scenes = {str(row.get("scene_id")) for row in dataset_outputs["train"]}
    val_scenes = {str(row.get("scene_id")) for row in dataset_outputs["val"]}
    outcome_distribution = Counter(
        "missing" if row.get("success") is None else f"success={int(float(row['success']) > 0.5)}"
        for row in outcomes.values()
    )
    summary = {
        "task": "stage21_candidate_recoverability_dataset",
        "event_schema_version": "stage21a_v1",
        "run_root": str(Path(run_root).resolve()),
        "run_dirs": [str(path) for path in run_dirs],
        "run_dir_count": len(run_dirs),
        "episodes_file": None if episodes_file is None else str(episodes_file),
        "candidate_labels_are_route_progress_not_episode_success": True,
        "counts": {
            "collected_rows_before_dedup": len(labels) + duplicate_rows,
            "duplicate_rows_removed": duplicate_rows,
            "label_rows": len(labels),
            "scenes": scene_count,
            "episodes": episode_count,
            "candidates": candidate_count,
            "multi_candidate_rows": multi_candidate_rows,
            "reference_joined_rows": int(status_counts.get("ok", 0)),
        },
        "label_status_counts": dict(status_counts),
        "reference_join_rate": status_counts.get("ok", 0) / max(1, len(labels)),
        "outcome_audit": {
            **dict(outcome_counts),
            "unique_episode_outcomes": len(outcomes),
            "distribution": dict(outcome_distribution),
        },
        "triage_audit": dict(triage_counts),
        "active_safety_check": {
            "applied_count": int(triage_counts.get("applied", 0)),
            "expected_for_shadow": 0,
            "passed": int(triage_counts.get("applied", 0)) == 0,
        },
        "split_audit": {
            "split_key": split_key,
            "train_scene_count": len(train_scenes),
            "val_scene_count": len(val_scenes),
            "scene_overlap_count": len(train_scenes & val_scenes),
        },
        "collect_summaries": collect_summaries,
        "route_progress_dataset": dataset_summary,
        "candidate_recoverability_rows": candidate_row_audit,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_root": summary["run_root"],
        "run_dirs": summary["run_dirs"],
        "episodes_file": summary["episodes_file"],
        "output_files": [
            "labels.jsonl",
            "rows.jsonl",
            "train.jsonl",
            "val.jsonl",
            "summary.json",
        ],
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "split_key": split_key,
    }
    (output_dir / "episode_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-frame", choices=["world", "episodic_gps"], default="episodic_gps")
    parser.add_argument("--reference-coordinate-mode", choices=["xy", "x_neg_y", "xz", "x_neg_z"], default="x_neg_y")
    parser.add_argument("--gps-coordinate-mode", choices=["xy", "x_neg_y"], default="x_neg_y")
    parser.add_argument("--quaternion-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--lookahead-m", type=float, default=1.5)
    parser.add_argument("--max-angle-deg", type=float, default=60.0)
    parser.add_argument("--step-min", type=int, default=0)
    parser.add_argument("--step-max", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=21)
    parser.add_argument("--split-key", choices=["episode", "scene"], default="scene")
    args = parser.parse_args()
    summary = build_stage21_dataset(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

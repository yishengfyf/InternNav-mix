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
    _split_for_row,
    build_dataset,
)
from collect_gt_candidate_labels import (  # noqa: E402
    _load_reference_paths,
    _read_json_records,
    _safe_float,
    _scene_token,
    collect_labels,
)


RECOVERY_CANDIDATE_TYPES = {"resilience_backtrack", "backtrack_reobserve"}

ONLINE_CANDIDATE_FIELDS = {
    "candidate_id",
    "candidate_index",
    "candidate_type",
    "source",
    "grid",
    "xy",
    "goal_state",
    "geometry_safe",
    "active_gate_safe",
    "direction_bucket",
    "direction_angle_deg",
    "distance_m",
    "frontier_distance_m",
    "frontier_progress_score",
    "topology_novelty_score",
    "nearby_visit_count",
    "revisit_risk",
    "points_to_revisited_region",
    "angle_to_current_waypoint_deg",
    "intent_alignment_score",
    "aligned_with_current_waypoint",
    "distance_to_current_waypoint_m",
    "semanticized_candidate",
    "instruction_relevant",
    "semantic_relevance_score",
    "semantic_novelty_score",
    "semantic_confidence_score",
    "semantic_bind_score",
    "goal_progress_enabled",
    "goal_progress_next_landmark",
    "matched_landmark",
    "landmark_status",
    "next_landmark_relevance",
    "completed_landmark_penalty",
    "repeated_semantic_penalty",
    "semantic_progress_score",
    "unknown_target_frontier_bonus",
    "goal_progress_score",
    "target_frontier_enabled",
    "target_frontier_score",
    "target_frontier_candidate",
    "target_frontier_escape_candidate",
    "target_frontier_cluster_count",
    "target_frontier_cluster_score",
    "target_frontier_doorway_like_score",
    "target_frontier_corridor_continuation_score",
    "target_frontier_transition_prior",
    "target_frontier_intent_deviation_penalty",
    "target_frontier_intent_safe",
    "target_frontier_local_free_count",
    "target_frontier_local_occupied_count",
    "target_frontier_local_unknown_count",
    "score",
    "semantic_resilience_candidate",
    "semantic_resilience_recommended",
    "semantic_resilience_reason",
    "semantic_resilience_source",
    "semantic_resilience_source_step_id",
    "semantic_resilience_step_gap",
    "semantic_resilience_backtrack_distance_m",
    "semantic_resilience_open_score",
    "semantic_resilience_distance_score",
    "semantic_resilience_score",
    "semantic_resilience_trigger_reasons",
    "semantic_resilience_recovery_context_tags",
    "semantic_resilience_local_trap",
    "semantic_resilience_recovery_trigger",
    "semantic_resilience_active_safe",
    "semantic_resilience_obstacle_term_count",
    "semantic_resilience_passage_term_count",
    "semantic_resilience_nearest_obstacle_term",
    "semantic_resilience_nearest_obstacle_distance_m",
    "semantic_resilience_nearest_passage_term",
    "semantic_resilience_nearest_passage_distance_m",
    "recovery_feature_schema_version",
    "anchor_source_is_keyframe",
    "anchor_visible_free_ratio",
    "anchor_occupied_ratio_observed",
    "anchor_frontier_count",
    "anchor_branch_count",
    "anchor_executable_exit_count",
    "anchor_connected_component_count",
    "anchor_branch_depth_mean",
    "anchor_direction_entropy",
    "anchor_semantic_unique_count",
    "anchor_instruction_relevant_count",
    "anchor_high_conf_landmark_count",
    "anchor_next_landmark_count",
    "anchor_passage_semantic_count",
    "anchor_outgoing_trace_direction_count",
    "anchor_last_visit_step",
    "anchor_last_visit_age_steps",
    "anchor_recent_return_count",
    "anchor_recent_cycle_count",
    "anchor_short_cycle_risk",
    "current_visible_free_ratio",
    "current_frontier_count",
    "current_branch_count",
    "current_executable_exit_count",
    "current_connected_component_count",
    "current_branch_depth_mean",
    "current_direction_entropy",
    "current_to_anchor_free_ratio_gain",
    "current_to_anchor_frontier_gain",
    "current_to_anchor_branch_gain",
    "current_to_anchor_direction_entropy_gain",
    "anchor_semantic_top_match",
    "anchor_semantic_top_score",
    "anchor_high_conf_semantic",
}

ONLINE_TRIAGE_FIELDS = {
    "tier",
    "reason",
    "considered",
    "applied",
    "failure_type",
    "recommended_primitive",
}


def _online_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Return a strict deployable allowlist; GT annotations never enter inputs."""
    return {
        key: candidate.get(key)
        for key in sorted(ONLINE_CANDIDATE_FIELDS)
        if key in candidate
    }


def _online_triage(triage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    value = dict(triage or {})
    return {
        key: value.get(key)
        for key in sorted(ONLINE_TRIAGE_FIELDS)
        if key in value
    }


def _identity_for_candidate(
    event: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "split": event.get("split"),
        "rank": event.get("rank"),
        "eval_random_seed": event.get("eval_random_seed"),
        "episode_eval_seed": event.get("episode_eval_seed"),
        "scene_id": event.get("scene_id"),
        "episode_id": event.get("episode_id"),
        "step_id": event.get("step_id"),
        "candidate_id": candidate.get("candidate_id"),
    }


def _contains_gt_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).startswith("gt_") or _contains_gt_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_gt_field(item) for item in value)
    return False


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
        loop_path = run_dir / "s2_action_loop_events.jsonl"
        for row in _read_json_records(loop_path):
            if row.get("transition") != "start":
                continue
            counts["s2_action_loop_events"] += 1
            tier = str(row.get("triage_tier") or "unknown")
            counts[f"s2_action_loop_tier={tier}"] += 1
            if bool(row.get("applied")):
                counts["applied"] += 1
            key = _event_key(row)
            triage_by_event[key] = {
                "tier": tier,
                "reason": row.get("triage_reason"),
                "considered": True,
                "applied": bool(row.get("applied")),
                "failure_type": row.get("failure_type"),
                "recommended_primitive": row.get("recommended_primitive"),
            }
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


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _safe_float(value, 0.0)))


def _positive_gain(value: Any, scale: float) -> float:
    return _clamp(_safe_float(value, 0.0) / max(1e-6, scale))


def _is_recovery_candidate(candidate: Dict[str, Any]) -> bool:
    candidate_type = str(candidate.get("candidate_type") or candidate.get("source") or "")
    return bool(candidate.get("semantic_resilience_candidate")) or candidate_type in RECOVERY_CANDIDATE_TYPES


def _recovery_proxy(
    candidate: Dict[str, Any], *, route_progress_advantage_m: float
) -> Dict[str, Any]:
    """Shadow-only proxy for restoring a safe, informative decision state."""
    geometry_safe = bool(candidate.get("geometry_safe", True))
    distance_m = _safe_float(
        candidate.get("semantic_resilience_backtrack_distance_m", candidate.get("distance_m")),
        0.0,
    )
    occupied_ratio = _clamp(candidate.get("anchor_occupied_ratio_observed"))
    distance_safe = (
        0.0 if distance_m <= 0.25 else
        1.0 if 0.75 <= distance_m <= 4.0 else
        max(0.0, 1.0 - abs(distance_m - 2.25) / 5.75)
    )
    anchor_age = _safe_float(candidate.get("anchor_last_visit_age_steps"), 0.0)
    freshness = 1.0 if anchor_age <= 48.0 else _clamp(1.0 - (anchor_age - 48.0) / 96.0)
    collision_risk = _clamp(candidate.get("revisit_risk"))
    unknown_ratio = _clamp(
        _safe_float(candidate.get("target_frontier_local_unknown_count"), 0.0),
        0.0,
        50.0,
    ) / 50.0
    executability = _clamp(
        0.35 * float(geometry_safe)
        + 0.20 * (1.0 - occupied_ratio)
        + 0.15 * distance_safe
        + 0.15 * freshness
        + 0.10 * (1.0 - collision_risk)
        + 0.05 * (1.0 - unknown_ratio)
    )
    free_gain = _positive_gain(candidate.get("current_to_anchor_free_ratio_gain"), 0.20)
    frontier_gain = _positive_gain(candidate.get("current_to_anchor_frontier_gain"), 12.0)
    branch_gain = _positive_gain(candidate.get("current_to_anchor_branch_gain"), 2.0)
    entropy_gain = _positive_gain(candidate.get("current_to_anchor_direction_entropy_gain"), 0.35)
    observability = 0.35 * free_gain + 0.25 * frontier_gain + 0.25 * branch_gain + 0.15 * entropy_gain
    trigger = max(
        float(bool(candidate.get("semantic_resilience_local_trap"))),
        0.60 * float(bool(candidate.get("semantic_resilience_recovery_trigger"))),
    )
    trap_exit = _clamp(0.55 * trigger + 0.45 * max(free_gain, branch_gain))
    semantic_keyframe = _clamp(
        0.18 * float(bool(candidate.get("anchor_source_is_keyframe")))
        + 0.18 * _positive_gain(candidate.get("anchor_semantic_unique_count"), 4.0)
        + 0.22 * _positive_gain(candidate.get("anchor_instruction_relevant_count"), 2.0)
        + 0.18 * _positive_gain(candidate.get("anchor_high_conf_landmark_count"), 2.0)
        + 0.12 * _positive_gain(candidate.get("anchor_passage_semantic_count"), 2.0)
        + 0.12 * _clamp(candidate.get("anchor_semantic_top_score"))
    )
    replanning = _clamp(
        0.35 * _positive_gain(
            candidate.get("anchor_executable_exit_count", candidate.get("anchor_branch_count")),
            3.0,
        )
        + 0.25 * _positive_gain(candidate.get("anchor_connected_component_count"), 2.0)
        + 0.25 * _positive_gain(candidate.get("anchor_outgoing_trace_direction_count"), 3.0)
        + 0.15 * _clamp(candidate.get("anchor_direction_entropy"))
    )
    stage_consistency = _clamp(
        0.45 * _clamp(candidate.get("next_landmark_relevance"))
        + 0.25 * float(bool(candidate.get("instruction_relevant")))
        + 0.20 * _positive_gain(candidate.get("anchor_next_landmark_count"), 2.0)
        + 0.10 * _positive_gain(candidate.get("anchor_passage_semantic_count"), 2.0)
        - 0.08 * _clamp(candidate.get("completed_landmark_penalty"))
    )
    oscillation = max(
        _clamp(candidate.get("anchor_short_cycle_risk")),
        _positive_gain(candidate.get("anchor_recent_cycle_count"), 2.0),
    )
    route_tiebreak = _clamp(route_progress_advantage_m / 3.0, -1.0, 1.0)
    base = _clamp(
        0.24 * executability
        + 0.18 * trap_exit
        + 0.20 * observability
        + 0.14 * semantic_keyframe
        + 0.14 * replanning
        + 0.10 * stage_consistency
        - 0.25 * oscillation
    )
    score_w005 = _clamp(base + 0.05 * route_tiebreak)
    hard_safe = bool(geometry_safe and occupied_ratio <= 0.55 and distance_m <= 8.0)
    proxy_class = (
        "unsafe" if not hard_safe else "promising" if score_w005 >= 0.65
        else "weak" if score_w005 < 0.35 else "ambiguous"
    )
    return {
        "proxy_definition": "safe_decision_state_restoration_v2",
        "proxy_is_causal_success_label": False,
        "short_horizon_executability_proxy": float(executability),
        "trap_exit_proxy": float(trap_exit),
        "future_observability_proxy": float(observability),
        "semantic_keyframe_proxy": float(semantic_keyframe),
        "replanning_affordance_proxy": float(replanning),
        "instruction_stage_consistency_proxy": float(stage_consistency),
        "oscillation_penalty": float(oscillation),
        "route_progress_tiebreak": float(route_tiebreak),
        "recovery_proxy_route_w0": float(base),
        "recovery_proxy_route_w005": float(score_w005),
        "recovery_proxy_route_w010": float(_clamp(base + 0.10 * route_tiebreak)),
        "recovery_proxy_class": proxy_class,
        "hard_safe_proxy": hard_safe,
    }


def _split_task_rows(
    rows: Sequence[Dict[str, Any]], val_ratio: float, split_seed: int, split_key: str
) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for row in rows:
        outputs[_split_for_row(row.get("identity") or {}, val_ratio, split_seed, split_key)].append(row)
    return outputs


def _build_task_rows(
    labels: Sequence[Dict[str, Any]],
    *,
    episodes_file: Optional[Path],
    reference_frame: str,
    quaternion_order: str,
    coordinate_mode: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Build disjoint progress, recovery-proxy, and safety task rows."""
    if episodes_file is None:
        return {"progress": [], "recovery_proxy": [], "safety": []}, {
            "status": "not_built_without_episodes_file"
        }
    reference_metadata = _load_reference_paths(episodes_file)
    prepared_paths: Dict[str, Any] = {}
    task_rows: Dict[str, List[Dict[str, Any]]] = {
        "progress": [],
        "recovery_proxy": [],
        "safety": [],
    }
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
            recovery_candidate = _is_recovery_candidate(candidate)
            source = str(candidate.get("candidate_type") or candidate.get("source") or "unknown")
            direction = str(candidate.get("direction_bucket") or "unknown")
            source_counts[source] += 1
            direction_counts[direction] += 1
            tier_counts[tier] += 1
            common = {
                "identity": _identity_for_candidate(event, candidate),
                "online_inputs": {
                    "current_policy_candidate": _online_candidate(s2_candidate),
                    "candidate": _online_candidate(candidate),
                    "triage_context": _online_triage(triage),
                },
                "episode_outcome_context_only": event.get("episode_outcome_context_only"),
            }
            safety_proxy = _recovery_proxy(
                candidate, route_progress_advantage_m=0.0
            )["short_horizon_executability_proxy"]
            task_rows["safety"].append(
                {
                    **common,
                    "offline_labels": {
                        "safety_proxy_definition": "geometry_map_executability_v2",
                        "geometry_safe_target": bool(candidate.get("geometry_safe", True)),
                        "short_horizon_executability_proxy": float(safety_proxy),
                        "active_gate_safe_is_target": False,
                    },
                }
            )
            if recovery_candidate:
                route_advantage = (
                    route["route_progress_m"] - s2_route["route_progress_m"]
                    if route is not None and s2_route is not None
                    else 0.0
                )
                proxy = _recovery_proxy(
                    candidate, route_progress_advantage_m=route_advantage
                )
                task_rows["recovery_proxy"].append(
                    {
                        **common,
                        "offline_labels": {
                            **proxy,
                            "route_progress_m_context_only": (
                                None if route is None else route["route_progress_m"]
                            ),
                            "route_progress_advantage_vs_s2_m_context_only": (
                                None if route is None or s2_route is None else float(route_advantage)
                            ),
                            "points_to_revisited_region_context_only": bool(
                                candidate.get("points_to_revisited_region")
                            ),
                            "landmark_status_context_only": _candidate_status(candidate),
                        },
                    }
                )
                counts["recovery_candidate_rows"] += 1
                counts[f"recovery_class={proxy['recovery_proxy_class']}"] += 1
                counts[f"recovery_direction={direction}"] += 1
                counts["recovery_revisited_rows"] += int(
                    bool(candidate.get("points_to_revisited_region"))
                )
                counts["recovery_completed_landmark_rows"] += int(
                    _candidate_status(candidate) == "completed"
                )
                counts["recovery_oscillation_rows"] += int(
                    proxy["oscillation_penalty"] > 0.0
                )
            else:
                if route is None or s2_route is None:
                    counts["drop_progress_missing_route_or_s2"] += 1
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
                task_rows["progress"].append(
                    {
                        **common,
                        "offline_labels": {
                            **route,
                            "s2_route_progress_m": s2_route["route_progress_m"],
                            "s2_route_value_m": s2_route["route_value_m"],
                            "advantage_vs_s2_m": float(advantage),
                            "preference_vs_s2": preference,
                            "online_safe": bool(online_safe),
                        },
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
    progress_rows = task_rows["progress"]
    recovery_rows = task_rows["recovery_proxy"]
    counts["candidate_rows"] = len(progress_rows)
    counts["progress_rows"] = len(progress_rows)
    counts["safety_rows"] = len(task_rows["safety"])
    counts["events_with_positive_and_negative"] = paired_event_count
    counts["positive_vs_negative_pairs"] = pair_count
    counts.update(
        Counter(row["offline_labels"]["preference_vs_s2"] for row in progress_rows)
    )
    leakage_by_task = {
        name: sum(_contains_gt_field(row.get("online_inputs")) for row in rows)
        for name, rows in task_rows.items()
    }
    component_names = (
        "short_horizon_executability_proxy", "trap_exit_proxy",
        "future_observability_proxy", "semantic_keyframe_proxy",
        "replanning_affordance_proxy", "instruction_stage_consistency_proxy",
        "oscillation_penalty", "recovery_proxy_route_w0",
        "recovery_proxy_route_w005", "recovery_proxy_route_w010",
    )
    component_means = {
        name: sum(_safe_float(row["offline_labels"].get(name)) for row in recovery_rows)
        / max(1, len(recovery_rows))
        for name in component_names
    }
    safety_values = [
        _safe_float(row["offline_labels"].get("short_horizon_executability_proxy"))
        for row in task_rows["safety"]
    ]
    recovery_candidates = [
        (row.get("online_inputs") or {}).get("candidate", {})
        for row in recovery_rows
    ]
    revisit_intervals = [
        _safe_float(candidate.get("anchor_revisit_interval_min_steps"))
        for candidate in recovery_candidates
        if candidate.get("anchor_revisit_interval_min_steps") is not None
    ]
    cycle_nonzero_count = sum(
        _safe_float(candidate.get("anchor_recent_cycle_count")) > 0.0
        for candidate in recovery_candidates
    )
    cycle_interval_inconsistent_count = sum(
        (
            _safe_float(candidate.get("anchor_recent_cycle_count")) > 0.0
            and candidate.get("anchor_revisit_interval_min_steps") is None
        )
        or (
            _safe_float(candidate.get("anchor_recent_cycle_count")) <= 0.0
            and candidate.get("anchor_revisit_interval_min_steps") is not None
        )
        for candidate in recovery_candidates
    )
    recovery_feature_fields = (
        "recovery_feature_schema_version",
        "anchor_visible_free_ratio",
        "anchor_frontier_count",
        "anchor_branch_count",
        "anchor_executable_exit_count",
        "anchor_connected_component_count",
        "anchor_branch_depth_mean",
        "anchor_direction_entropy",
        "anchor_semantic_unique_count",
        "anchor_recent_cycle_count",
        "anchor_short_cycle_risk",
        "anchor_revisit_interval_min_steps",
        "anchor_revisit_interval_mean_steps",
        "current_to_anchor_free_ratio_gain",
        "current_to_anchor_branch_gain",
    )
    feature_non_null_counts = {
        name: sum(
            (row.get("online_inputs") or {}).get("candidate", {}).get(name) is not None
            for row in recovery_rows
        )
        for name in recovery_feature_fields
    }
    feature_coverage = {
        name: count / max(1, len(recovery_rows))
        for name, count in feature_non_null_counts.items()
    }
    return task_rows, {
        "status": "ok",
        "counts": dict(counts),
        "candidate_source_counts": dict(source_counts),
        "candidate_direction_counts": dict(direction_counts),
        "triage_tier_candidate_counts": dict(tier_counts),
        "label_margin_m": 0.20,
        "online_offline_field_separation": True,
        "gt_leakage_scan": {
            "rows_with_gt_field_in_online_inputs": leakage_by_task,
            "passed": all(value == 0 for value in leakage_by_task.values()),
        },
        "recovery_proxy_component_means": component_means,
        "safety_proxy_audit": {
            "row_count": len(safety_values),
            "mean": sum(safety_values) / max(1, len(safety_values)),
            "min": min(safety_values) if safety_values else None,
            "max": max(safety_values) if safety_values else None,
            "exact_one_rate": sum(value >= 0.999999 for value in safety_values)
            / max(1, len(safety_values)),
            "geometry_safe_target_distribution": dict(
                Counter(
                    bool(row["offline_labels"].get("geometry_safe_target"))
                    for row in task_rows["safety"]
                )
            ),
        },
        "cycle_feature_audit": {
            "recovery_row_count": len(recovery_candidates),
            "nonzero_cycle_count": int(cycle_nonzero_count),
            "interval_observed_count": len(revisit_intervals),
            "interval_min_steps": min(revisit_intervals) if revisit_intervals else None,
            "interval_mean_steps": (
                sum(revisit_intervals) / len(revisit_intervals)
                if revisit_intervals else None
            ),
            "interval_at_or_below_one_count": sum(
                interval <= 1.0 for interval in revisit_intervals
            ),
            "cycle_interval_inconsistent_count": int(
                cycle_interval_inconsistent_count
            ),
            "passed": bool(
                not any(interval <= 1.0 for interval in revisit_intervals)
                and cycle_interval_inconsistent_count == 0
            ),
        },
        "recovery_feature_coverage": {
            "row_count": len(recovery_rows),
            "non_null_counts": feature_non_null_counts,
            "rates": feature_coverage,
        },
        "recovery_route_weight_ablation": [0.0, 0.05, 0.10],
        "active_gate_safe_used_as_recovery_target": False,
        "revisit_or_back_direction_is_automatic_recovery_negative": False,
        "completed_landmark_is_automatic_recovery_negative": False,
        "final_episode_success_used_as_candidate_label": False,
    }


def _build_candidate_rows(
    labels: Sequence[Dict[str, Any]],
    *,
    episodes_file: Optional[Path],
    reference_frame: str,
    quaternion_order: str,
    coordinate_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compatibility wrapper: legacy callers receive ordinary progress rows."""
    task_rows, audit = _build_task_rows(
        labels,
        episodes_file=episodes_file,
        reference_frame=reference_frame,
        quaternion_order=quaternion_order,
        coordinate_mode=coordinate_mode,
    )
    return task_rows["progress"], audit


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

    task_rows, candidate_row_audit = _build_task_rows(
        labels,
        episodes_file=episodes_file,
        reference_frame=reference_frame,
        quaternion_order=quaternion_order,
        coordinate_mode=gps_coordinate_mode,
    )
    progress_task_rows = task_rows["progress"]
    recovery_task_rows = task_rows["recovery_proxy"]
    safety_task_rows = task_rows["safety"]
    task_splits = {
        name: _split_task_rows(rows, val_ratio, split_seed, split_key)
        for name, rows in task_rows.items()
    }
    # rows.jsonl remains a compatibility alias for the ordinary progress task.
    _write_jsonl(output_dir / "rows.jsonl", progress_task_rows)
    for name, rows in (
        ("progress_rows", progress_task_rows),
        ("recovery_proxy_rows", recovery_task_rows),
        ("safety_rows", safety_task_rows),
    ):
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
        split_name = name[:-5] if name.endswith("_rows") else name
        for split in ("train", "val"):
            _write_jsonl(
                output_dir / f"{name}_{split}.jsonl",
                task_splits[split_name][split],
            )

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

    train_scenes = {
        str((row.get("identity") or {}).get("scene_id"))
        for rows in task_splits.values() for row in rows["train"]
    }
    val_scenes = {
        str((row.get("identity") or {}).get("scene_id"))
        for rows in task_splits.values() for row in rows["val"]
    }
    outcome_distribution = Counter(
        "missing" if row.get("success") is None else f"success={int(float(row['success']) > 0.5)}"
        for row in outcomes.values()
    )
    summary = {
        "task": "stage21_candidate_recoverability_dataset",
        "event_schema_version": "stage21a_r3_v3",
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
        "task_rows": {
            "progress": len(progress_task_rows),
            "recovery_proxy": len(recovery_task_rows),
            "safety": len(safety_task_rows),
            "legacy_rows_alias": "progress_rows.jsonl",
        },
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
            "progress_rows.jsonl",
            "progress_rows_train.jsonl",
            "progress_rows_val.jsonl",
            "recovery_proxy_rows.jsonl",
            "recovery_proxy_rows_train.jsonl",
            "recovery_proxy_rows_val.jsonl",
            "safety_rows.jsonl",
            "safety_rows_train.jsonl",
            "safety_rows_val.jsonl",
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

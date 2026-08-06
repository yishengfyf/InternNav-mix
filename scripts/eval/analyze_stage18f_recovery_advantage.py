"""Analyze Stage18f offline recovery advantage.

This is an offline diagnostic tool. It never runs Habitat and never changes
navigation actions. It reads Stage18e-style ``memory_events.jsonl`` logs and
compares the frozen S2/current waypoint against the proposed
``resilience_backtrack`` recovery candidate.

The goal is not to prove causality. The goal is to answer a more practical
question before any active intervention:

* when the memory triggers, is the backtrack candidate more like a useful
  re-observation anchor than the frozen current policy waypoint?
* do the backtrack candidates consistently look like "safe recovery" moves
  according to the geometry/semantic proxies already present in the logs?
* are the strongest recovery events concentrated in failures and semantic
  dead-zones / stagnation / revisit loops?

The script can optionally export one JSONL row per query event for later
dataset construction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if isinstance(row, dict):
                yield row


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return list(_read_jsonl(path))
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _event_key(row: Mapping[str, Any]) -> str:
    return f"{_episode_key(row)}|step={row.get('step_id')}"


def _find_memory_event_files(paths: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for path in paths:
        candidates: List[Path] = []
        if path.is_file() and path.name == "memory_events.jsonl":
            candidates.append(path)
        if path.is_dir():
            candidates.extend(
                [
                    path / "occ_memory" / "memory_events.jsonl",
                    path / "memory_events.jsonl",
                    path / "vlmap_safety_debug" / "run_001" / "occ_memory" / "memory_events.jsonl",
                ]
            )
            candidates.extend(path.rglob("memory_events.jsonl"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.exists():
                continue
            seen.add(resolved)
            found.append(candidate)
    return sorted(found)


def _find_progress_files(paths: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for path in paths:
        candidates: List[Path] = []
        if path.is_file() and path.name == "progress.json":
            candidates.append(path)
        if path.is_dir():
            candidates.extend([path / "progress.json", path.parent / "progress.json"])
            candidates.extend(path.rglob("progress.json"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.exists():
                continue
            seen.add(resolved)
            found.append(candidate)
    return sorted(found)


def _progress_by_episode(progress_files: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for path in progress_files:
        for row in _read_json_records(path):
            if row.get("episode_id") is None:
                continue
            by_key[_episode_key(row)] = row
    return by_key


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _candidate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_type": candidate.get("candidate_type"),
        "direction_bucket": candidate.get("direction_bucket"),
        "geometry_safe": bool(candidate.get("geometry_safe")),
        "active_gate_safe": bool(candidate.get("active_gate_safe")),
        "score": candidate.get("score"),
        "semantic_resilience_score": candidate.get("semantic_resilience_score"),
        "semantic_resilience_open_score": candidate.get("semantic_resilience_open_score"),
        "semantic_resilience_backtrack_distance_m": candidate.get(
            "semantic_resilience_backtrack_distance_m"
        ),
        "semantic_resilience_source": candidate.get("semantic_resilience_source"),
        "semantic_resilience_step_gap": candidate.get("semantic_resilience_step_gap"),
        "semantic_resilience_obstacle_term_count": candidate.get(
            "semantic_resilience_obstacle_term_count"
        ),
        "semantic_resilience_passage_term_count": candidate.get(
            "semantic_resilience_passage_term_count"
        ),
        "semantic_resilience_nearest_obstacle_term": candidate.get(
            "semantic_resilience_nearest_obstacle_term"
        ),
        "semantic_resilience_nearest_passage_term": candidate.get(
            "semantic_resilience_nearest_passage_term"
        ),
        "grid": candidate.get("grid"),
        "xy": candidate.get("xy"),
    }


def _current_risk_proxy(current: Mapping[str, Any], state: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    occupied = 1.0 if str(current.get("goal_state") or "").lower() == "occupied" else 0.0
    unknown = 1.0 if str(current.get("goal_state") or "").lower() == "unknown" else 0.0
    unsafe = 1.0 if bool(state.get("current_policy_unsafe")) else 0.0
    dead_zone = 1.0 if bool(state.get("current_policy_dead_zone")) else 0.0
    stagnation = 1.0 if bool(state.get("current_policy_stagnation")) else 0.0
    revisited = 1.0 if bool(state.get("current_policy_revisited")) else 0.0
    problem = 1.0 if bool(state.get("current_policy_problem")) else 0.0
    not_active_safe = 1.0 if bool(state.get("current_policy_not_active_safe")) else 0.0
    frontier_alignment = 1.0 if bool(current.get("waypoint_aligns_with_dominant_frontier")) else 0.0
    keyframe_alignment = 1.0 if bool(current.get("waypoint_aligns_with_high_conf_keyframe")) else 0.0
    distance = _safe_float(current.get("frontier_distance_m"), 0.0)
    frontier_close = 1.0 - min(1.0, distance / 1.0)

    risk = (
        0.24 * occupied
        + 0.08 * unknown
        + 0.18 * unsafe
        + 0.16 * dead_zone
        + 0.12 * stagnation
        + 0.08 * revisited
        + 0.08 * problem
        + 0.04 * not_active_safe
        + 0.02 * (1.0 - frontier_alignment)
        + 0.02 * (1.0 - keyframe_alignment)
        + 0.02 * frontier_close
    )
    risk = max(0.0, min(1.0, risk))
    return risk, {
        "occupied": occupied,
        "unknown": unknown,
        "unsafe": unsafe,
        "dead_zone": dead_zone,
        "stagnation": stagnation,
        "revisited": revisited,
        "problem": problem,
        "not_active_safe": not_active_safe,
        "frontier_alignment": frontier_alignment,
        "high_conf_alignment": keyframe_alignment,
        "frontier_close": frontier_close,
    }


def _backtrack_utility_proxy(candidate: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    geometry_safe = 1.0 if bool(candidate.get("geometry_safe")) else 0.0
    open_score = _clamp01(candidate.get("semantic_resilience_open_score"))
    semantic_score = _clamp01(candidate.get("semantic_resilience_score"))
    source = str(candidate.get("semantic_resilience_source") or "")
    if source == "keyframe":
        source_bonus = 1.0
    elif source == "pose_trace":
        source_bonus = 0.85
    else:
        source_bonus = 0.65
    step_gap = _safe_float(candidate.get("semantic_resilience_step_gap"), 60.0)
    recency_score = 1.0 - min(1.0, max(0.0, step_gap / 60.0))
    distance = _safe_float(candidate.get("semantic_resilience_backtrack_distance_m"), 0.0)
    # Prefer a short-but-not-zero recovery hop.
    distance_score = 1.0 - min(1.0, abs(distance - 2.0) / 2.0)
    passage_terms = min(1.0, _safe_float(candidate.get("semantic_resilience_passage_term_count"), 0.0) / 4.0)
    obstacle_terms = min(1.0, _safe_float(candidate.get("semantic_resilience_obstacle_term_count"), 0.0) / 3.0)
    direction_back = 1.0 if str(candidate.get("direction_bucket") or "") == "back" else 0.0

    utility = (
        0.30 * open_score
        + 0.25 * semantic_score
        + 0.15 * recency_score
        + 0.10 * source_bonus
        + 0.10 * passage_terms
        + 0.05 * distance_score
        + 0.03 * geometry_safe
        + 0.02 * direction_back
        - 0.02 * obstacle_terms
    )
    utility = max(0.0, min(1.0, utility))
    return utility, {
        "geometry_safe": geometry_safe,
        "open_score": open_score,
        "semantic_score": semantic_score,
        "source_bonus": source_bonus,
        "recency_score": recency_score,
        "distance_score": distance_score,
        "passage_terms": passage_terms,
        "obstacle_terms": obstacle_terms,
        "direction_back": direction_back,
    }


def _select_best_backtrack(candidates: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    valid = [
        candidate
        for candidate in candidates
        if bool(candidate.get("semantic_resilience_candidate"))
        or candidate.get("candidate_type") == "resilience_backtrack"
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda candidate: (
            _safe_float(candidate.get("semantic_resilience_score"), 0.0),
            _safe_float(candidate.get("semantic_resilience_open_score"), 0.0),
            _safe_float(candidate.get("score"), 0.0),
        ),
    )


def _episode_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    episodes = {_episode_key(row) for row in rows}
    triggered = [row for row in rows if row.get("trigger")]
    positive = [row for row in rows if row.get("advantage_margin_proxy", 0.0) > 0.0]
    strong_positive = [row for row in rows if row.get("strong_recovery_proxy")]
    return {
        "episodes": len(episodes),
        "trigger_episode_count": len({_episode_key(row) for row in triggered}),
        "positive_margin_episode_count": len({_episode_key(row) for row in positive}),
        "strong_positive_episode_count": len({_episode_key(row) for row in strong_positive}),
        "trigger_rate": _rate(len(triggered), len(rows)),
        "positive_margin_rate": _rate(len(positive), len(rows)),
        "strong_positive_rate": _rate(len(strong_positive), len(rows)),
    }


def analyze(
    paths: Sequence[Path],
    *,
    utility_threshold: float,
    advantage_margin_threshold: float,
) -> Dict[str, Any]:
    memory_files = _find_memory_event_files(paths)
    if not memory_files:
        raise FileNotFoundError(
            "No memory_events.jsonl found. Pass the Stage18e run root, "
            "vlmap_safety_debug dir, run dir, or memory_events.jsonl itself."
        )

    progress = _progress_by_episode(_find_progress_files(paths))
    query_events: List[Dict[str, Any]] = []
    summary_events: List[Dict[str, Any]] = []
    trigger_reasons = Counter()
    context_tags = Counter()
    backtrack_sources = Counter()
    backtrack_directions = Counter()
    current_problem_reasons = Counter()

    rows: List[Dict[str, Any]] = []
    positive_examples: List[Dict[str, Any]] = []
    negative_examples: List[Dict[str, Any]] = []

    for path in memory_files:
        for event in _read_jsonl(path):
            event_type = event.get("event_type")
            if event_type == "occ_memory_episode_summary":
                item = dict(event)
                item["_memory_file"] = str(path)
                summary_events.append(item)
                continue
            if event_type != "occ_memory_query_candidates":
                continue

            state = event.get("semantic_resilience_state") or {}
            current = event.get("current_policy_candidate") or {}
            candidates = [candidate for candidate in event.get("candidates") or [] if isinstance(candidate, dict)]
            best_backtrack = _select_best_backtrack(candidates)
            episode_progress = progress.get(_episode_key(event))
            success = None
            spl = None
            ne = None
            if episode_progress is not None:
                success = _safe_float(episode_progress.get("success")) >= 0.5
                spl = _safe_float(episode_progress.get("spl"))
                ne = _safe_float(episode_progress.get("ne"))

            trigger = bool(event.get("semantic_resilience_recovery_trigger"))
            trigger_reasons.update(str(reason) for reason in event.get("semantic_resilience_trigger_reasons") or [])
            context_tags.update(
                str(tag)
                for tag in event.get("semantic_resilience_recovery_context_tags")
                or state.get("recovery_context_tags")
                or []
            )
            if bool(state.get("current_policy_problem")):
                current_problem_reasons.update(
                    str(reason) for reason in event.get("semantic_resilience_trigger_reasons") or []
                )

            current_risk, current_detail = _current_risk_proxy(current, state)
            best_utility = None
            best_detail = None
            if best_backtrack is not None:
                best_utility, best_detail = _backtrack_utility_proxy(best_backtrack)
                backtrack_sources.update([str(best_backtrack.get("semantic_resilience_source") or "unknown")])
                backtrack_directions.update([str(best_backtrack.get("direction_bucket") or "unknown")])

            best_candidate_summary = _candidate_summary(best_backtrack) if best_backtrack is not None else None
            current_summary = {
                "candidate_id": current.get("candidate_id"),
                "candidate_type": current.get("candidate_type"),
                "direction_bucket": current.get("direction_bucket"),
                "geometry_safe": bool(current.get("geometry_safe")),
                "active_gate_safe": bool(current.get("active_gate_safe")),
                "goal_state": current.get("goal_state"),
                "frontier_distance_m": current.get("frontier_distance_m"),
                "distance_m": current.get("distance_m"),
                "semantic_dead_zone": bool(state.get("current_policy_dead_zone")),
                "semantic_stagnation": bool(state.get("current_policy_stagnation")),
                "revisited": bool(state.get("current_policy_revisited")),
                "unsafe": bool(state.get("current_policy_unsafe")),
                "not_active_safe": bool(state.get("current_policy_not_active_safe")),
                "waypoint_aligns_with_dominant_frontier": current.get(
                    "waypoint_aligns_with_dominant_frontier"
                ),
                "waypoint_aligns_with_high_conf_keyframe": current.get(
                    "waypoint_aligns_with_high_conf_keyframe"
                ),
            }

            margin = None
            strong_recovery_proxy = False
            recovery_ready = False
            if best_utility is not None:
                margin = float(best_utility - current_risk)
                recovery_ready = bool(
                    bool(best_backtrack.get("geometry_safe"))
                    and best_utility >= float(utility_threshold)
                    and _safe_float(best_backtrack.get("semantic_resilience_backtrack_distance_m")) >= 0.75
                )
                strong_recovery_proxy = bool(
                    trigger
                    and recovery_ready
                    and (
                        bool(state.get("current_policy_problem"))
                        or bool(state.get("current_policy_dead_zone"))
                        or bool(state.get("current_policy_stagnation"))
                        or bool(state.get("current_policy_revisited"))
                    )
                    and margin >= float(advantage_margin_threshold)
                )

            row = {
                "scene_id": event.get("scene_id"),
                "episode_id": event.get("episode_id"),
                "step_id": event.get("step_id"),
                "event_key": _event_key(event),
                "trigger": trigger,
                "trigger_reasons": event.get("semantic_resilience_trigger_reasons") or [],
                "success": success,
                "spl": spl,
                "ne": ne,
                "query_count": int(event.get("candidate_count", 0) or 0),
                "current_problem": bool(state.get("current_policy_problem")),
                "current_risk_proxy": float(current_risk),
                "current_risk_detail": current_detail,
                "current_policy_candidate": current_summary,
                "best_backtrack_candidate": best_candidate_summary,
                "backtrack_present": bool(best_backtrack is not None),
                "backtrack_utility_proxy": None if best_utility is None else float(best_utility),
                "backtrack_utility_detail": best_detail,
                "advantage_margin_proxy": margin,
                "recovery_ready": recovery_ready,
                "strong_recovery_proxy": strong_recovery_proxy,
                "candidate_count": int(event.get("candidate_count", 0) or 0),
                "geometry_safe_candidate_count": int(event.get("candidate_geometry_safe_count", 0) or 0),
                "active_gate_safe_candidate_count": int(event.get("candidate_active_gate_safe_count", 0) or 0),
                "recovery_candidate_count": int(event.get("candidate_semantic_resilience_count", 0) or 0),
                "recovery_recommended_count": int(
                    event.get("candidate_semantic_resilience_recommended_count", 0) or 0
                ),
                "recovery_source": None if best_backtrack is None else best_backtrack.get("semantic_resilience_source"),
            }
            query_events.append(dict(row))
            rows.append(row)

            if margin is not None and margin > 0.0 and len(positive_examples) < 20:
                positive_examples.append(
                    {
                        "event_key": row["event_key"],
                        "trigger": trigger,
                        "current_problem": row["current_problem"],
                        "current_risk_proxy": row["current_risk_proxy"],
                        "backtrack_utility_proxy": row["backtrack_utility_proxy"],
                        "advantage_margin_proxy": row["advantage_margin_proxy"],
                        "current_policy_candidate": current_summary,
                        "best_backtrack_candidate": best_candidate_summary,
                    }
                )
            if (margin is None or margin <= 0.0) and len(negative_examples) < 20:
                negative_examples.append(
                    {
                        "event_key": row["event_key"],
                        "trigger": trigger,
                        "current_problem": row["current_problem"],
                        "current_risk_proxy": row["current_risk_proxy"],
                        "backtrack_utility_proxy": row["backtrack_utility_proxy"],
                        "advantage_margin_proxy": row["advantage_margin_proxy"],
                        "current_policy_candidate": current_summary,
                        "best_backtrack_candidate": best_candidate_summary,
                    }
                )

    by_success: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success") is None:
            by_success["success_unknown"].append(row)
        elif row.get("success"):
            by_success["success_true"].append(row)
        else:
            by_success["success_false"].append(row)

    event_margins = [row["advantage_margin_proxy"] for row in rows if row["advantage_margin_proxy"] is not None]
    current_risks = [row["current_risk_proxy"] for row in rows]
    backtrack_utilities = [row["backtrack_utility_proxy"] for row in rows if row["backtrack_utility_proxy"] is not None]
    trigger_rows = [row for row in rows if row["trigger"]]
    backtrack_rows = [row for row in rows if row["backtrack_present"]]
    recovery_ready_rows = [row for row in rows if row["recovery_ready"]]
    positive_margin_rows = [row for row in rows if (row["advantage_margin_proxy"] or 0.0) > 0.0]
    strong_rows = [row for row in rows if row["strong_recovery_proxy"]]

    result = {
        "task": "stage18f_recovery_advantage",
        "memory_event_files": [str(path) for path in memory_files],
        "progress_episode_count": len(progress),
        "episode_summary_count": len(summary_events),
        "query_summary": {
            "query_events": len(rows),
            "episodes": len({_episode_key(row) for row in rows}),
            "trigger_count": len(trigger_rows),
            "trigger_rate": _rate(len(trigger_rows), len(rows)),
            "backtrack_event_count": len(backtrack_rows),
            "backtrack_event_rate": _rate(len(backtrack_rows), len(rows)),
            "backtrack_present_given_trigger_rate": _rate(len([row for row in trigger_rows if row["backtrack_present"]]), len(trigger_rows)),
            "current_problem_count": len([row for row in rows if row["current_problem"]]),
            "current_problem_rate": _rate(len([row for row in rows if row["current_problem"]]), len(rows)),
            "recovery_ready_count": len(recovery_ready_rows),
            "recovery_ready_rate": _rate(len(recovery_ready_rows), len(rows)),
            "positive_margin_count": len(positive_margin_rows),
            "positive_margin_rate": _rate(len(positive_margin_rows), len(rows)),
            "strong_recovery_count": len(strong_rows),
            "strong_recovery_rate": _rate(len(strong_rows), len(rows)),
            "mean_current_risk_proxy": _stats(current_risks).get("mean"),
            "mean_backtrack_utility_proxy": _stats(backtrack_utilities).get("mean"),
            "mean_advantage_margin_proxy": _stats(event_margins).get("mean"),
            "median_advantage_margin_proxy": _stats(event_margins).get("median"),
            "backtrack_geometry_safe_rate": _rate(
                sum(
                    1
                    for row in rows
                    if row["best_backtrack_candidate"]
                    and bool(row["best_backtrack_candidate"].get("geometry_safe"))
                ),
                len(backtrack_rows),
            ),
            "backtrack_active_gate_safe_rate": _rate(
                sum(
                    1
                    for row in rows
                    if row["best_backtrack_candidate"]
                    and bool(row["best_backtrack_candidate"].get("active_gate_safe"))
                ),
                len(backtrack_rows),
            ),
            "backtrack_direction_back_rate": _rate(
                sum(
                    1
                    for row in rows
                    if row["best_backtrack_candidate"]
                    and str(row["best_backtrack_candidate"].get("direction_bucket") or "") == "back"
                ),
                len(backtrack_rows),
            ),
        },
        "by_success": {
            name: {
                "query_events": len(group),
                "episodes": len({_episode_key(row) for row in group}),
                "trigger_rate": _rate(len([row for row in group if row["trigger"]]), len(group)),
                "backtrack_present_given_trigger_rate": _rate(
                    len([row for row in group if row["trigger"] and row["backtrack_present"]]),
                    len([row for row in group if row["trigger"]]),
                ),
                "recovery_ready_rate": _rate(len([row for row in group if row["recovery_ready"]]), len(group)),
                "positive_margin_rate": _rate(
                    len([row for row in group if (row["advantage_margin_proxy"] or 0.0) > 0.0]),
                    len(group),
                ),
                "strong_recovery_rate": _rate(len([row for row in group if row["strong_recovery_proxy"]]), len(group)),
                "mean_advantage_margin_proxy": _stats(
                    [row["advantage_margin_proxy"] for row in group if row["advantage_margin_proxy"] is not None]
                ).get("mean"),
                "mean_current_risk_proxy": _stats([row["current_risk_proxy"] for row in group]).get("mean"),
                "mean_backtrack_utility_proxy": _stats(
                    [row["backtrack_utility_proxy"] for row in group if row["backtrack_utility_proxy"] is not None]
                ).get("mean"),
            }
            for name, group in sorted(by_success.items())
        },
        "trigger_reason_counts": dict(trigger_reasons),
        "recovery_context_tag_counts": dict(context_tags),
        "backtrack_source_counts": dict(backtrack_sources),
        "backtrack_direction_counts": dict(backtrack_directions),
        "current_problem_reason_counts": dict(current_problem_reasons),
        "backtrack_utility_stats": _stats(backtrack_utilities),
        "current_risk_stats": _stats(current_risks),
        "advantage_margin_stats": _stats(event_margins),
        "recovery_ready_examples": positive_examples,
        "non_positive_examples": negative_examples,
    }

    all_episode_keys = {_episode_key(row) for row in rows}
    positive_episode_keys = {_episode_key(row) for row in positive_margin_rows}
    strong_episode_keys = {_episode_key(row) for row in strong_rows}
    non_positive_episode_keys = all_episode_keys - positive_episode_keys
    non_strong_episode_keys = all_episode_keys - strong_episode_keys

    result["episode_summary"] = {
        "episodes_with_trigger": len({_episode_key(row) for row in trigger_rows}),
        "episodes_with_backtrack": len({_episode_key(row) for row in backtrack_rows}),
        "episodes_with_positive_margin": len(positive_episode_keys),
        "episodes_with_strong_recovery": len(strong_episode_keys),
        "positive_margin_episode_success_rate": _rate(
            sum(
                1
                for ep in positive_episode_keys
                if progress.get(ep) is not None and _safe_float(progress[ep].get("success")) >= 0.5
            ),
            len(positive_episode_keys),
        ),
        "non_positive_episode_success_rate": _rate(
            sum(
                1
                for ep in non_positive_episode_keys
                if progress.get(ep) is not None and _safe_float(progress[ep].get("success")) >= 0.5
            ),
            len(non_positive_episode_keys),
        ),
        "strong_recovery_episode_success_rate": _rate(
            sum(
                1
                for ep in strong_episode_keys
                if progress.get(ep) is not None and _safe_float(progress[ep].get("success")) >= 0.5
            ),
            len(strong_episode_keys),
        ),
        "non_strong_episode_success_rate": _rate(
            sum(
                1
                for ep in non_strong_episode_keys
                if progress.get(ep) is not None and _safe_float(progress[ep].get("success")) >= 0.5
            ),
            len(non_strong_episode_keys),
        ),
    }

    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        nargs="+",
        required=True,
        help="Stage18e/Stage18f run root(s), debug dir(s), run dir(s), or memory_events.jsonl file(s).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows-output",
        type=Path,
        help="Optional JSONL output with one row per query event for later offline dataset building.",
    )
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=0.60,
        help="Minimum backtrack utility proxy to call a candidate recovery-ready.",
    )
    parser.add_argument(
        "--advantage-margin-threshold",
        type=float,
        default=0.0,
        help="Minimum utility-minus-risk margin to call an event a positive recovery example.",
    )
    args = parser.parse_args()

    result, rows = analyze(
        args.run_root,
        utility_threshold=float(args.utility_threshold),
        advantage_margin_threshold=float(args.advantage_margin_threshold),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(result["query_summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    if args.rows_output is not None:
        print(f"Wrote {args.rows_output}")


if __name__ == "__main__":
    main()

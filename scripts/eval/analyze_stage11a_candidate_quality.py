"""
Post-hoc Stage11a OccMem candidate quality audit.

Research question:
  Do Stage11a OccMem candidates contain useful alternatives that the frozen
  S2 waypoint ignored, especially in failure / Mode-C episodes?

Default input:
  ../../compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed/
    vlmap_safety_debug/run_001

Outputs:
  analysis_stage11a_candidate_quality.json

Important caveat:
  Habitat GT trajectory is not available in these logs. "correct_candidate"
  below is therefore a proxy:
    target frontier + safe + next-landmark/progress-positive signal +
    sufficiently different from the current S2 waypoint direction.
  This is a go/no-go signal for information content, not a final correctness
  label for training.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return data
    records: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
    return records


def _episode_key(record: Dict[str, Any]) -> str:
    return f"{record.get('scene_id')}|{record.get('episode_id')}"


def _default_run_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed"
        / "vlmap_safety_debug"
        / "run_001"
    )


def _resolve_run_dir(path: Path) -> Path:
    candidates = [
        path,
        path / "vlmap_safety_debug" / "run_001",
        path / "run_001",
    ]
    for candidate in candidates:
        if (candidate / "occ_memory" / "memory_events.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"occ_memory/memory_events.jsonl not found under {path}")


def _progress_path(run_dir: Path) -> Path:
    for candidate in [
        run_dir / "progress.json",
        run_dir.parents[1] / "progress.json" if len(run_dir.parents) > 1 else run_dir / "progress.json",
        run_dir / "semantic_episode_summary.jsonl",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"progress.json not found for {run_dir}")


def _failure_mode(progress: Dict[str, Any]) -> str:
    if _safe_float(progress.get("success")) >= 0.5:
        return "success"
    steps = _safe_int(progress.get("steps"))
    ne = _safe_float(progress.get("ne"))
    collision = _safe_float(progress.get("collision_count"))
    if collision >= 50 or (steps >= 400 and collision > 0):
        return "mode_a_stuck_collision"
    if steps < 60:
        return "mode_b_early"
    if ne > 8.0:
        return "mode_c_navigation_lost"
    if ne > 3.0:
        return "mode_d_mid_fail"
    return "other_failure"


def _group_progress(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = {}
    for record in records:
        if record.get("event_type") not in (None, "episode_summary"):
            continue
        grouped[_episode_key(record)] = record
    return grouped


def _selection_index(memory_events: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    index = {}
    for event in memory_events:
        if event.get("event_type") != "occ_memory_candidate_selection":
            continue
        step = _safe_int(event.get("step_id"), -1)
        index[(_episode_key(event), step)] = event
    return index


def _is_safe_candidate(candidate: Dict[str, Any]) -> bool:
    if candidate.get("active_gate_safe") is not None:
        return bool(candidate.get("active_gate_safe"))
    if candidate.get("geometry_safe") is not None:
        return bool(candidate.get("geometry_safe"))
    return candidate.get("goal_state") in ("free", "unknown")


def _is_target_frontier(candidate: Dict[str, Any]) -> bool:
    return bool(
        candidate.get("target_frontier_candidate")
        or candidate.get("target_frontier_escape_candidate")
        or candidate.get("candidate_type") in ("frontier", "semantic_frontier")
    )


def _has_instruction_signal(candidate: Dict[str, Any]) -> bool:
    if candidate.get("instruction_relevant"):
        return True
    if _safe_float(candidate.get("next_landmark_relevance")) > 0.0:
        return True
    if _safe_float(candidate.get("semantic_relevance_score")) >= 0.5:
        return True
    if _safe_float(candidate.get("semantic_progress_score")) > 0.0:
        return True
    semantic_evidence = candidate.get("semantic_evidence") or {}
    if _safe_float(semantic_evidence.get("instruction_relevance")) >= 0.5:
        return True
    return False


def _has_progress_signal(candidate: Dict[str, Any]) -> bool:
    return bool(
        _has_instruction_signal(candidate)
        or candidate.get("target_frontier_escape_candidate")
        or _safe_float(candidate.get("unknown_target_frontier_bonus")) > 0.0
        or _safe_float(candidate.get("target_frontier_score")) >= 0.75
    )


def _has_next_landmark_signal(candidate: Dict[str, Any]) -> bool:
    if _safe_float(candidate.get("next_landmark_relevance")) > 0.0:
        return True
    if _safe_float(candidate.get("semantic_progress_score")) > 0.0:
        return True
    if str(candidate.get("landmark_status") or "").lower() in ("next", "future", "pending"):
        return True
    next_landmark = candidate.get("goal_progress_next_landmark")
    matched_landmark = candidate.get("matched_landmark")
    if next_landmark and matched_landmark and str(next_landmark).lower() == str(matched_landmark).lower():
        return True
    semantic_evidence = candidate.get("semantic_evidence") or {}
    if _safe_float(semantic_evidence.get("next_landmark_relevance")) > 0.0:
        return True
    if _safe_float(semantic_evidence.get("semantic_progress_score")) > 0.0:
        return True
    if str(semantic_evidence.get("landmark_status") or "").lower() in ("next", "future", "pending"):
        return True
    return False


def _ignored_by_current_waypoint(candidate: Dict[str, Any], min_angle_deg: float) -> bool:
    if candidate.get("aligned_with_current_waypoint") is False:
        return True
    angle = candidate.get("angle_to_current_waypoint_deg")
    if angle is not None:
        return abs(_safe_float(angle)) >= min_angle_deg
    return False


def _candidate_flags(candidate: Dict[str, Any], min_angle_deg: float) -> Dict[str, bool]:
    safe = _is_safe_candidate(candidate)
    target = _is_target_frontier(candidate)
    instruction = _has_instruction_signal(candidate)
    next_landmark = _has_next_landmark_signal(candidate)
    progress = _has_progress_signal(candidate)
    ignored = _ignored_by_current_waypoint(candidate, min_angle_deg)
    strict = bool(safe and target and next_landmark and ignored)
    medium = bool(safe and target and progress and ignored)
    broad = bool(safe and target and ignored)
    return {
        "safe": safe,
        "target_frontier": target,
        "instruction_signal": instruction,
        "next_landmark_signal": next_landmark,
        "progress_signal": progress,
        "ignored_by_current_waypoint": ignored,
        "strict_correct_proxy": strict,
        "medium_correct_proxy": medium,
        "broad_correct_proxy": broad,
    }


def _candidate_summary(
    event: Dict[str, Any],
    selection_event: Optional[Dict[str, Any]],
    *,
    min_angle_deg: float,
) -> Dict[str, Any]:
    candidates = event.get("candidates") or []
    selected_id = None
    if selection_event:
        selected = selection_event.get("selected_candidate") or {}
        selected_id = selected.get("candidate_id") or selection_event.get("selection_choice")

    candidate_rows = []
    counts = Counter()
    direction_counts = Counter()
    current_bucket = event.get("current_waypoint_direction_bucket")
    for candidate in candidates:
        flags = _candidate_flags(candidate, min_angle_deg)
        for key, value in flags.items():
            if value:
                counts[key] += 1
        direction = candidate.get("direction_bucket") or "unknown"
        if flags["strict_correct_proxy"]:
            direction_counts[direction] += 1
        candidate_id = candidate.get("candidate_id")
        candidate_rows.append({
            "candidate_id": candidate_id,
            "candidate_type": candidate.get("candidate_type"),
            "direction_bucket": direction,
            "direction_angle_deg": candidate.get("direction_angle_deg"),
            "angle_to_current_waypoint_deg": candidate.get("angle_to_current_waypoint_deg"),
            "score": candidate.get("score"),
            "target_frontier_score": candidate.get("target_frontier_score"),
            "semantic_progress_score": candidate.get("semantic_progress_score"),
            "goal_progress_next_landmark": candidate.get("goal_progress_next_landmark"),
            "matched_landmark": candidate.get("matched_landmark"),
            "selected_by_candidate_selector": bool(selected_id and candidate_id == selected_id),
            **flags,
        })

    return {
        "step_id": event.get("step_id"),
        "valid": bool(event.get("valid")),
        "candidate_count": len(candidates),
        "current_waypoint_direction_bucket": current_bucket,
        "current_waypoint_goal_state": event.get("current_waypoint_goal_state"),
        "current_waypoint_semantic_dead_zone": event.get("current_waypoint_semantic_dead_zone"),
        "candidate_type_counts": event.get("candidate_type_counts") or {},
        "candidate_direction_counts": event.get("candidate_direction_counts") or {},
        "selected_candidate_id": selected_id,
        "strict_correct_proxy_count": counts["strict_correct_proxy"],
        "medium_correct_proxy_count": counts["medium_correct_proxy"],
        "broad_correct_proxy_count": counts["broad_correct_proxy"],
        "safe_count": counts["safe"],
        "target_frontier_count": counts["target_frontier"],
        "instruction_signal_count": counts["instruction_signal"],
        "next_landmark_signal_count": counts["next_landmark_signal"],
        "progress_signal_count": counts["progress_signal"],
        "ignored_by_current_waypoint_count": counts["ignored_by_current_waypoint"],
        "strict_proxy_direction_counts": dict(direction_counts),
        "candidates": candidate_rows,
    }


def _summarize_group(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "group": name,
            "episode_count": 0,
        }
    episode_count = len(rows)
    def rate(key: str) -> float:
        return sum(1 for row in rows if row[key]) / max(1, episode_count)

    total_events = sum(row["candidate_event_count"] for row in rows)
    direction_counts = Counter()
    current_direction_counts = Counter()
    for row in rows:
        direction_counts.update(row["strict_proxy_direction_counts"])
        current_direction_counts.update(row["current_waypoint_direction_counts"])

    return {
        "group": name,
        "episode_count": episode_count,
        "candidate_event_count": total_events,
        "mean_candidate_events_per_episode": total_events / max(1, episode_count),
        "episodes_with_any_candidate": sum(1 for row in rows if row["candidate_event_count"] > 0),
        "episodes_with_strict_correct_proxy": sum(1 for row in rows if row["has_strict_correct_proxy"]),
        "episodes_with_medium_correct_proxy": sum(1 for row in rows if row["has_medium_correct_proxy"]),
        "episodes_with_broad_correct_proxy": sum(1 for row in rows if row["has_broad_correct_proxy"]),
        "strict_correct_proxy_rate": rate("has_strict_correct_proxy"),
        "medium_correct_proxy_rate": rate("has_medium_correct_proxy"),
        "broad_correct_proxy_rate": rate("has_broad_correct_proxy"),
        "mean_strict_proxy_count": sum(row["strict_correct_proxy_count"] for row in rows) / max(1, episode_count),
        "mean_medium_proxy_count": sum(row["medium_correct_proxy_count"] for row in rows) / max(1, episode_count),
        "mean_broad_proxy_count": sum(row["broad_correct_proxy_count"] for row in rows) / max(1, episode_count),
        "strict_proxy_direction_counts": dict(direction_counts),
        "current_waypoint_direction_counts": dict(current_direction_counts),
    }


def _decision(rate_value: float, success_rate: float) -> str:
    enrichment = rate_value - success_rate
    if rate_value > 0.30 and enrichment >= 0.10:
        return "pass_candidate_training_go"
    if rate_value > 0.30 and enrichment < 0.10:
        return "candidate_common_not_failure_discriminative"
    if rate_value < 0.15:
        return "fail_candidate_training_pause"
    return "gray_zone_needs_manual_case_review"


def analyze(
    run_dir: Path,
    *,
    step_min: int,
    step_max: int,
    min_angle_deg: float,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_dir)
    progress = _read_json_records(_progress_path(run_dir))
    memory_events = _read_json_records(run_dir / "occ_memory" / "memory_events.jsonl")

    progress_by_key = _group_progress(progress)
    modes = {key: _failure_mode(item) for key, item in progress_by_key.items()}
    mode_counts = Counter(modes.values())
    selection_by_step = _selection_index(memory_events)

    query_events = [
        event for event in memory_events
        if event.get("event_type") == "occ_memory_query_candidates"
        and step_min <= _safe_int(event.get("step_id"), -1) <= step_max
    ]
    events_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in query_events:
        events_by_key[_episode_key(event)].append(event)
    for items in events_by_key.values():
        items.sort(key=lambda event: _safe_int(event.get("step_id"), -1))

    episode_rows = []
    for key, progress_item in sorted(progress_by_key.items(), key=lambda item: _safe_int(item[1].get("episode_id"))):
        events = events_by_key.get(key, [])
        event_summaries = []
        current_waypoint_direction_counts = Counter()
        strict_proxy_direction_counts = Counter()
        for event in events:
            step = _safe_int(event.get("step_id"), -1)
            selection = selection_by_step.get((key, step))
            summary = _candidate_summary(event, selection, min_angle_deg=min_angle_deg)
            event_summaries.append(summary)
            current_waypoint_direction_counts.update([summary.get("current_waypoint_direction_bucket") or "unknown"])
            strict_proxy_direction_counts.update(summary.get("strict_proxy_direction_counts") or {})

        strict_count = sum(item["strict_correct_proxy_count"] for item in event_summaries)
        medium_count = sum(item["medium_correct_proxy_count"] for item in event_summaries)
        broad_count = sum(item["broad_correct_proxy_count"] for item in event_summaries)
        target_count = sum(item["target_frontier_count"] for item in event_summaries)
        instruction_count = sum(item["instruction_signal_count"] for item in event_summaries)
        ignored_count = sum(item["ignored_by_current_waypoint_count"] for item in event_summaries)
        candidate_count = sum(item["candidate_count"] for item in event_summaries)

        episode_rows.append({
            "key": key,
            "scene_id": progress_item.get("scene_id"),
            "episode_id": progress_item.get("episode_id"),
            "success": _safe_float(progress_item.get("success")) >= 0.5,
            "mode": modes.get(key),
            "steps": _safe_int(progress_item.get("steps")),
            "ne": _safe_float(progress_item.get("ne")),
            "candidate_event_count": len(event_summaries),
            "candidate_count": candidate_count,
            "target_frontier_count": target_count,
            "instruction_signal_count": instruction_count,
            "ignored_by_current_waypoint_count": ignored_count,
            "strict_correct_proxy_count": strict_count,
            "medium_correct_proxy_count": medium_count,
            "broad_correct_proxy_count": broad_count,
            "has_strict_correct_proxy": strict_count > 0,
            "has_medium_correct_proxy": medium_count > 0,
            "has_broad_correct_proxy": broad_count > 0,
            "strict_proxy_direction_counts": dict(strict_proxy_direction_counts),
            "current_waypoint_direction_counts": dict(current_waypoint_direction_counts),
            "event_summaries": event_summaries,
        })

    groups = {
        "all_episodes": episode_rows,
        "successes": [row for row in episode_rows if row["success"]],
        "all_failures": [row for row in episode_rows if not row["success"]],
        "mode_c_navigation_lost": [row for row in episode_rows if row["mode"] == "mode_c_navigation_lost"],
        "mode_d_mid_fail": [row for row in episode_rows if row["mode"] == "mode_d_mid_fail"],
        "mode_b_early": [row for row in episode_rows if row["mode"] == "mode_b_early"],
    }
    group_summaries = {name: _summarize_group(name, rows) for name, rows in groups.items()}
    all_fail_rate = group_summaries["all_failures"].get("strict_correct_proxy_rate", 0.0)
    mode_c_rate = group_summaries["mode_c_navigation_lost"].get("strict_correct_proxy_rate", 0.0)

    return {
        "run_dir": str(run_dir),
        "analysis_window": {"step_min": step_min, "step_max": step_max},
        "min_angle_deg_for_ignored_proxy": min_angle_deg,
        "episode_count": len(episode_rows),
        "mode_counts": dict(mode_counts),
        "proxy_definition": {
            "strict_correct_proxy": (
                "safe candidate AND target_frontier AND next-landmark/progress-positive signal "
                "AND angle_to_current_waypoint >= min_angle_deg or not aligned"
            ),
            "medium_correct_proxy": (
                "safe candidate AND target_frontier AND progress signal "
                "AND ignored by current waypoint"
            ),
            "broad_correct_proxy": (
                "safe candidate AND target_frontier AND ignored by current waypoint"
            ),
            "caveat": (
                "No Habitat GT trajectory is available in these logs; proxy rates "
                "estimate candidate information content, not final correctness."
            ),
        },
        "group_summaries": group_summaries,
        "go_no_go": {
            "all_failures_strict_rate": all_fail_rate,
            "successes_strict_rate": group_summaries["successes"].get("strict_correct_proxy_rate", 0.0),
            "all_failures_enrichment_over_success": (
                all_fail_rate - group_summaries["successes"].get("strict_correct_proxy_rate", 0.0)
            ),
            "all_failures_decision": _decision(
                all_fail_rate,
                group_summaries["successes"].get("strict_correct_proxy_rate", 0.0),
            ),
            "mode_c_strict_rate": mode_c_rate,
            "mode_c_enrichment_over_success": (
                mode_c_rate - group_summaries["successes"].get("strict_correct_proxy_rate", 0.0)
            ),
            "mode_c_decision": _decision(
                mode_c_rate,
                group_summaries["successes"].get("strict_correct_proxy_rate", 0.0),
            ),
            "thresholds": {
                "pass": "> 0.30 and at least +0.10 over success rate",
                "fail": "< 0.15",
                "gray_zone": "0.15 - 0.30",
                "common_not_discriminative": "> 0.30 but < +0.10 over success rate",
            },
        },
        "episode_rows": episode_rows,
    }


def _print_summary(result: Dict[str, Any]) -> None:
    print(f"\nStage11a candidate quality audit: {result['run_dir']}")
    print(f"Window: {result['analysis_window']}  min_angle={result['min_angle_deg_for_ignored_proxy']}")
    print(f"Mode counts: {result['mode_counts']}\n")
    for name in [
        "all_failures",
        "mode_c_navigation_lost",
        "mode_d_mid_fail",
        "mode_b_early",
        "successes",
    ]:
        summary = result["group_summaries"].get(name, {})
        if not summary or summary.get("episode_count", 0) == 0:
            continue
        print(
            f"{name:<24} n={summary['episode_count']:<3} "
            f"strict={summary['strict_correct_proxy_rate']:.3f} "
            f"medium={summary['medium_correct_proxy_rate']:.3f} "
            f"broad={summary['broad_correct_proxy_rate']:.3f} "
            f"events/ep={summary['mean_candidate_events_per_episode']:.2f}"
        )
    print("\nGo/no-go:")
    print(json.dumps(result["go_no_go"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage11a OccMem candidate information content.")
    parser.add_argument("--run-dir", type=Path, default=_default_run_dir())
    parser.add_argument("--step-min", type=int, default=20)
    parser.add_argument("--step-max", type=int, default=80)
    parser.add_argument("--min-angle-deg", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = analyze(
        args.run_dir,
        step_min=args.step_min,
        step_max=args.step_max,
        min_angle_deg=args.min_angle_deg,
    )
    if not args.quiet:
        _print_summary(result)

    output = args.output
    if output is None:
        output = _resolve_run_dir(args.run_dir) / "analysis_stage11a_candidate_quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

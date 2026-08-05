"""Analyze Stage18e semantic-resilience shadow logs.

Stage18e is deliberately not an active navigation policy.  It asks whether the
online sparse metric-semantic memory can detect broad recovery contexts
(spatial constriction, semantic stagnation, revisit loops, policy-memory
conflict, or limited frontier escape) and propose a short re-observation
candidate while frozen S2 remains in control.

This script reads one or more ``memory_events.jsonl`` files, including 4-GPU
merged/eval folders, and summarizes:

* how often semantic-resilience triggers fire;
* whether a ``resilience_backtrack`` candidate is present when triggered;
* which broad recovery context tags dominate;
* which trigger reasons dominate;
* whether triggers/candidates are enriched in failed episodes when progress
  metrics are available under the same run root.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _event_key(row: Mapping[str, Any]) -> str:
    return f"{_episode_key(row)}|step={row.get('step_id')}"


def _find_memory_event_files(paths: Sequence[Path]) -> List[Path]:
    found = []
    seen = set()
    for path in paths:
        candidates = []
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
    found = []
    seen = set()
    for path in paths:
        candidates = []
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


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def _candidate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_type": candidate.get("candidate_type"),
        "direction_bucket": candidate.get("direction_bucket"),
        "distance_m": candidate.get("distance_m"),
        "score": candidate.get("score"),
        "geometry_safe": bool(candidate.get("geometry_safe")),
        "active_gate_safe": bool(candidate.get("active_gate_safe")),
        "semantic_resilience_score": candidate.get("semantic_resilience_score"),
        "semantic_resilience_open_score": candidate.get("semantic_resilience_open_score"),
        "semantic_resilience_backtrack_distance_m": candidate.get(
            "semantic_resilience_backtrack_distance_m"
        ),
        "semantic_resilience_source": candidate.get("semantic_resilience_source"),
        "semantic_resilience_source_step_id": candidate.get(
            "semantic_resilience_source_step_id"
        ),
        "semantic_resilience_recovery_context_tags": candidate.get(
            "semantic_resilience_recovery_context_tags"
        ),
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


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "query_events": 0,
            "episodes": 0,
            "trigger_rate": None,
            "local_trap_rate": None,
            "backtrack_present_given_trigger_rate": None,
        }
    trigger_rows = [row for row in rows if row.get("semantic_resilience_recovery_trigger")]
    candidate_rows = [row for row in rows if row.get("candidate_semantic_resilience_count", 0)]
    trigger_with_candidate = [
        row
        for row in trigger_rows
        if int(row.get("candidate_semantic_resilience_count", 0) or 0) > 0
    ]
    local_traps = [row for row in rows if row.get("semantic_resilience_local_trap")]
    raw_counts = [int(row.get("semantic_resilience_raw_candidate_count", 0) or 0) for row in rows]
    selected_counts = [int(row.get("candidate_semantic_resilience_count", 0) or 0) for row in rows]
    current_problem = [
        row
        for row in rows
        if (row.get("semantic_resilience_state") or {}).get("current_policy_problem")
    ]
    obstacle_events = [
        row
        for row in rows
        if int((row.get("semantic_resilience_state") or {}).get("semantic_obstacle_term_count", 0) or 0)
        > 0
    ]
    passage_events = [
        row
        for row in rows
        if int((row.get("semantic_resilience_state") or {}).get("semantic_passage_term_count", 0) or 0)
        > 0
    ]
    return {
        "query_events": len(rows),
        "episodes": len({_episode_key(row) for row in rows}),
        "trigger_count": len(trigger_rows),
        "trigger_rate": _rate(len(trigger_rows), len(rows)),
        "local_trap_count": len(local_traps),
        "local_trap_rate": _rate(len(local_traps), len(rows)),
        "current_policy_problem_count": len(current_problem),
        "current_policy_problem_rate": _rate(len(current_problem), len(rows)),
        "semantic_obstacle_event_count": len(obstacle_events),
        "semantic_obstacle_event_rate": _rate(len(obstacle_events), len(rows)),
        "semantic_passage_event_count": len(passage_events),
        "semantic_passage_event_rate": _rate(len(passage_events), len(rows)),
        "backtrack_event_count": len(candidate_rows),
        "backtrack_event_rate": _rate(len(candidate_rows), len(rows)),
        "backtrack_present_given_trigger_rate": _rate(
            len(trigger_with_candidate),
            len(trigger_rows),
        ),
        "raw_backtrack_candidate_count": int(sum(raw_counts)),
        "selected_backtrack_candidate_count": int(sum(selected_counts)),
        "mean_raw_backtrack_candidates_per_event": _mean_or_none([float(v) for v in raw_counts]),
        "mean_selected_backtrack_candidates_per_event": _mean_or_none(
            [float(v) for v in selected_counts]
        ),
    }


def analyze(paths: Sequence[Path]) -> Dict[str, Any]:
    memory_files = _find_memory_event_files(paths)
    if not memory_files:
        raise FileNotFoundError(
            "No memory_events.jsonl found. Pass the Stage18e run root, "
            "vlmap_safety_debug dir, run dir, or memory_events.jsonl itself."
        )
    progress = _progress_by_episode(_find_progress_files(paths))

    query_events = []
    summary_events = []
    reason_counts = Counter()
    context_tag_counts = Counter()
    candidate_type_counts = Counter()
    candidate_direction_counts = Counter()
    source_counts = Counter()
    obstacle_terms = Counter()
    passage_terms = Counter()
    examples_with_backtrack = []
    examples_trigger_without_backtrack = []

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
            item = dict(event)
            item["_memory_file"] = str(path)
            episode_progress = progress.get(_episode_key(event))
            if episode_progress is not None:
                item["_success"] = _safe_float(episode_progress.get("success")) >= 0.5
                item["_spl"] = _safe_float(episode_progress.get("spl"))
                item["_ne"] = _safe_float(episode_progress.get("ne"))
            query_events.append(item)

            reason_counts.update(str(reason) for reason in event.get("semantic_resilience_trigger_reasons") or [])
            context_tag_counts.update(
                str(tag)
                for tag in event.get("semantic_resilience_recovery_context_tags")
                or (event.get("semantic_resilience_state") or {}).get("recovery_context_tags")
                or []
            )
            candidate_type_counts.update(event.get("candidate_type_counts") or {})
            candidate_direction_counts.update(event.get("candidate_direction_counts") or {})
            resilience_candidates = [
                candidate
                for candidate in event.get("candidates") or []
                if candidate.get("semantic_resilience_candidate")
                or candidate.get("candidate_type") == "resilience_backtrack"
            ]
            for candidate in resilience_candidates:
                source_counts.update([str(candidate.get("semantic_resilience_source") or "unknown")])
                obstacle = candidate.get("semantic_resilience_nearest_obstacle_term")
                passage = candidate.get("semantic_resilience_nearest_passage_term")
                if obstacle:
                    obstacle_terms.update([str(obstacle)])
                if passage:
                    passage_terms.update([str(passage)])
            if resilience_candidates and len(examples_with_backtrack) < 20:
                examples_with_backtrack.append(
                    {
                        "event_key": _event_key(event),
                        "memory_file": str(path),
                        "success": item.get("_success"),
                        "trigger": bool(event.get("semantic_resilience_recovery_trigger")),
                        "trigger_reasons": event.get("semantic_resilience_trigger_reasons"),
                        "current_policy_candidate": event.get("current_policy_candidate"),
                        "semantic_resilience_state": event.get("semantic_resilience_state"),
                        "resilience_candidates": [
                            _candidate_summary(candidate) for candidate in resilience_candidates
                        ],
                    }
                )
            if (
                event.get("semantic_resilience_recovery_trigger")
                and not resilience_candidates
                and len(examples_trigger_without_backtrack) < 20
            ):
                examples_trigger_without_backtrack.append(
                    {
                        "event_key": _event_key(event),
                        "memory_file": str(path),
                        "success": item.get("_success"),
                        "trigger_reasons": event.get("semantic_resilience_trigger_reasons"),
                        "semantic_resilience_state": event.get("semantic_resilience_state"),
                        "current_policy_candidate": event.get("current_policy_candidate"),
                    }
                )

    grouped_by_success: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in query_events:
        if "_success" not in event:
            grouped_by_success["success_unknown"].append(event)
        elif event["_success"]:
            grouped_by_success["success_true"].append(event)
        else:
            grouped_by_success["success_false"].append(event)

    step_values = [
        _safe_int(event.get("step_id"))
        for event in query_events
        if _safe_int(event.get("step_id")) is not None
    ]
    result = {
        "task": "stage18e_semantic_resilience_shadow",
        "memory_event_files": [str(path) for path in memory_files],
        "progress_episode_count": len(progress),
        "episode_summary_count": len(summary_events),
        "query_summary": _summarize_group(query_events),
        "by_success": {
            name: _summarize_group(rows)
            for name, rows in sorted(grouped_by_success.items())
        },
        "trigger_reason_counts": dict(reason_counts),
        "recovery_context_tag_counts": dict(context_tag_counts),
        "candidate_type_counts": dict(candidate_type_counts),
        "candidate_direction_counts": dict(candidate_direction_counts),
        "resilience_source_counts": dict(source_counts),
        "nearest_obstacle_term_counts": dict(obstacle_terms),
        "nearest_passage_term_counts": dict(passage_terms),
        "step_min": min(step_values) if step_values else None,
        "step_max": max(step_values) if step_values else None,
        "examples_with_backtrack": examples_with_backtrack,
        "examples_trigger_without_backtrack": examples_trigger_without_backtrack,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        nargs="+",
        required=True,
        help="Stage18e run root(s), debug dir(s), run dir(s), or memory_events.jsonl file(s).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["query_summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

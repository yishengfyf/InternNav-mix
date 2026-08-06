"""Analyze Stage19b semantic-resilience shadow taxonomy logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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


def _summarize_events(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "query_events": 0,
            "episodes": 0,
            "considered_count": 0,
            "applied_count": 0,
            "shadow_only_rate": None,
        }
    considered_rows = [row for row in rows if row.get("considered")]
    applied_rows = [row for row in rows if row.get("applied")]
    shadow_rows = [row for row in rows if row.get("shadow_only")]
    reason_counts = Counter(str(row.get("reason") or "unknown") for row in considered_rows)
    failure_type_counts = Counter(str(row.get("failure_type") or "unknown") for row in considered_rows)
    primitive_counts = Counter(
        str(row.get("recommended_primitive") or "hold_s2") for row in considered_rows
    )
    action_counts = [len(list(row.get("actions") or [])) for row in applied_rows]
    utility_values = [
        _safe_float(row.get("utility_threshold_used"))
        for row in considered_rows
        if row.get("utility_threshold_used") is not None
    ]
    return {
        "query_events": len(rows),
        "episodes": len({_episode_key(row) for row in rows}),
        "considered_count": len(considered_rows),
        "considered_rate": _rate(len(considered_rows), len(rows)),
        "applied_count": len(applied_rows),
        "applied_rate": _rate(len(applied_rows), len(rows)),
        "shadow_only_count": len(shadow_rows),
        "shadow_only_rate": _rate(len(shadow_rows), len(rows)),
        "reason_counts": dict(reason_counts),
        "failure_type_counts": dict(failure_type_counts),
        "recommended_primitive_counts": dict(primitive_counts),
        "mean_actions_per_applied": _mean_or_none([float(v) for v in action_counts]),
        "mean_utility_threshold_used": _mean_or_none(utility_values),
    }


def analyze(paths: Sequence[Path]) -> Dict[str, Any]:
    memory_files = _find_memory_event_files(paths)
    if not memory_files:
        raise FileNotFoundError(
            "No memory_events.jsonl found. Pass the Stage19b run root, "
            "vlmap_safety_debug dir, run dir, or memory_events.jsonl itself."
        )
    progress = _progress_by_episode(_find_progress_files(paths))

    query_events = []
    episode_summary_events = []
    trigger_reason_counts = Counter()
    context_tag_counts = Counter()
    failure_type_counts = Counter()
    recommended_primitive_counts = Counter()
    active_reason_counts = Counter()
    episode_failure_type_counts = Counter()
    episode_recommended_primitive_counts = Counter()

    for path in memory_files:
        for event in _read_jsonl(path):
            event_type = event.get("event_type")
            if event_type == "occ_memory_episode_summary":
                episode_summary_events.append(dict(event))
                continue
            if event_type != "stage19_semantic_resilience_active":
                continue
            item = dict(event)
            query_events.append(item)
            trigger_reason_counts.update(
                str(reason) for reason in event.get("trigger_reasons") or []
            )
            context_tag_counts.update(str(tag) for tag in event.get("recovery_context_tags") or [])
            failure_type_counts.update([str(event.get("failure_type") or "unknown")])
            recommended_primitive_counts.update(
                [str(event.get("recommended_primitive") or "hold_s2")]
            )
            active_reason_counts.update([str(event.get("reason") or "unknown")])

    for row in progress.values():
        ft = row.get("stage19_semantic_resilience_episode_failure_type")
        rp = row.get("stage19_semantic_resilience_episode_recommended_primitive")
        if ft:
            episode_failure_type_counts.update([str(ft)])
        if rp:
            episode_recommended_primitive_counts.update([str(rp)])

    grouped_by_shadow = defaultdict(list)
    for event in query_events:
        grouped_by_shadow["shadow_only" if event.get("shadow_only") else "active"].append(event)

    step_values = [
        _safe_int(event.get("step_id"))
        for event in query_events
        if _safe_int(event.get("step_id")) is not None
    ]
    result = {
        "task": "stage19b_semantic_resilience_shadow_taxonomy",
        "memory_event_files": [str(path) for path in memory_files],
        "progress_episode_count": len(progress),
        "episode_summary_count": len(episode_summary_events),
        "query_summary": _summarize_events(query_events),
        "by_shadow_mode": {
            name: _summarize_events(rows)
            for name, rows in sorted(grouped_by_shadow.items())
        },
        "trigger_reason_counts": dict(trigger_reason_counts),
        "recovery_context_tag_counts": dict(context_tag_counts),
        "failure_type_counts": dict(failure_type_counts),
        "recommended_primitive_counts": dict(recommended_primitive_counts),
        "active_reason_counts": dict(active_reason_counts),
        "episode_failure_type_counts": dict(episode_failure_type_counts),
        "episode_recommended_primitive_counts": dict(episode_recommended_primitive_counts),
        "step_min": min(step_values) if step_values else None,
        "step_max": max(step_values) if step_values else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        nargs="+",
        required=True,
        help="Stage19b run root(s), debug dir(s), run dir(s), or memory_events.jsonl file(s).",
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

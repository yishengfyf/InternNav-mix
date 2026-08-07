"""Analyze Stage20d semantic-recovery active smoke logs."""

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


def _has_episode_identity(row: Mapping[str, Any]) -> bool:
    return row.get("scene_id") is not None and row.get("episode_id") is not None


def _find_active_event_files(paths: Sequence[Path]) -> List[Path]:
    found = []
    seen = set()
    for path in paths:
        candidates = []
        if path.is_file() and path.name == "stage19_semantic_resilience_active_events.jsonl":
            candidates.append(path)
        if path.is_dir():
            candidates.extend(
                [
                    path / "stage19_semantic_resilience_active_events.jsonl",
                    path / "vlmap_safety_debug" / "run_001" / "stage19_semantic_resilience_active_events.jsonl",
                ]
            )
            candidates.extend(path.rglob("stage19_semantic_resilience_active_events.jsonl"))
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
        "episodes": len({_episode_key(row) for row in rows if _has_episode_identity(row)}),
        "events_with_episode_identity": sum(1 for row in rows if _has_episode_identity(row)),
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


def _summarize_progress(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"episodes": 0}

    def _values(key: str) -> List[float]:
        return [_safe_float(row.get(key)) for row in rows if row.get(key) is not None]

    collision_values = _values("collision_count")
    return {
        "episodes": len(rows),
        "success_mean": _mean_or_none(_values("success")),
        "spl_mean": _mean_or_none(_values("spl")),
        "ne_mean": _mean_or_none(_values("ne")),
        "steps_mean": _mean_or_none(_values("steps")),
        "collision_count_sum": float(sum(collision_values)) if collision_values else None,
        "collision_count_mean": _mean_or_none(collision_values),
    }


def analyze(paths: Sequence[Path]) -> Dict[str, Any]:
    event_files = _find_active_event_files(paths)
    if not event_files:
        raise FileNotFoundError(
            "No stage19_semantic_resilience_active_events.jsonl found. Pass the "
            "Stage20d run root, vlmap_safety_debug dir, run dir, or event file itself."
        )
    progress = _progress_by_episode(_find_progress_files(paths))

    query_events = []
    trigger_reason_counts = Counter()
    context_tag_counts = Counter()
    failure_type_counts = Counter()
    recommended_primitive_counts = Counter()
    active_reason_counts = Counter()
    episode_failure_type_counts = Counter()
    episode_recommended_primitive_counts = Counter()

    for path in event_files:
        for event in _read_jsonl(path):
            if event.get("event_type") != "stage19_semantic_resilience_active":
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

    progress_by_failure_type = defaultdict(list)
    for row in progress.values():
        progress_by_failure_type[
            str(row.get("stage19_semantic_resilience_episode_failure_type") or "missing")
        ].append(row)

    step_values = [
        _safe_int(event.get("step_id"))
        for event in query_events
        if _safe_int(event.get("step_id")) is not None
    ]
    result = {
        "task": "stage20d_sparse_semantic_recovery_active",
        "stage19_event_files": [str(path) for path in event_files],
        "progress_episode_count": len(progress),
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
        "progress_by_episode_failure_type": {
            name: _summarize_progress(rows)
            for name, rows in sorted(progress_by_failure_type.items())
        },
        "step_id_min": min(step_values) if step_values else None,
        "step_id_max": max(step_values) if step_values else None,
        "examples_by_failure_type": {},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run root, vlmap_safety_debug dir, or active events JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    args = parser.parse_args()

    summary = analyze(args.paths)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

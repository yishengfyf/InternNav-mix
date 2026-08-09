"""Analyze Stage20g strict semantic-recovery gate calibration logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_stage20d_sparse_semantic_recovery_active import analyze as _analyze_stage20d


def _read_json_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _episode_key(row):
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _active_event_files(paths):
    files = []
    seen = set()
    for path in paths:
        candidates = []
        if path.is_file() and path.name == "stage19_semantic_resilience_active_events.jsonl":
            candidates.append(path)
        if path.is_dir():
            candidates.extend(path.rglob("stage19_semantic_resilience_active_events.jsonl"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen and candidate.exists():
                seen.add(resolved)
                files.append(candidate)
    return sorted(files)


def _progress_files(paths):
    files = []
    seen = set()
    for path in paths:
        candidates = []
        if path.is_file() and path.name == "progress.json":
            candidates.append(path)
        if path.is_dir():
            candidates.extend([path / "progress.json"])
            candidates.extend(path.rglob("progress.json"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen and candidate.exists():
                seen.add(resolved)
                files.append(candidate)
    return sorted(files)


def analyze(paths):
    summary = _analyze_stage20d(paths)
    summary["task"] = "stage20g_sparse_semantic_recovery_gate"

    progress = {}
    for path in _progress_files(paths):
        for row in _read_json_records(path):
            if row.get("episode_id") is not None:
                progress[_episode_key(row)] = row

    pass_events = []
    all_events = []
    for path in _active_event_files(paths):
        for row in _read_json_records(path):
            if row.get("event_type") != "stage19_semantic_resilience_active":
                continue
            all_events.append(row)
            if row.get("reason") in {"shadow_gate_pass", "applied"} or row.get("would_apply"):
                pass_events.append(row)

    outcome_counts = Counter()
    progress_failure_counts = Counter()
    primitive_counts = Counter()
    event_failure_counts = Counter()
    for row in pass_events:
        progress_row = progress.get(_episode_key(row), {})
        outcome = "success" if float(progress_row.get("success", 0.0) or 0.0) > 0.0 else "failed"
        outcome_counts[outcome] += 1
        progress_failure_counts[
            str(progress_row.get("stage19_semantic_resilience_episode_failure_type") or "missing")
        ] += 1
        primitive_counts[str(row.get("recommended_primitive") or "unknown")] += 1
        event_failure_counts[str(row.get("failure_type") or "unknown")] += 1

    summary["gate_pass_summary"] = {
        "event_count": len(all_events),
        "pass_event_count": len(pass_events),
        "pass_outcome_counts": dict(outcome_counts),
        "pass_progress_failure_type_counts": dict(progress_failure_counts),
        "pass_event_failure_type_counts": dict(event_failure_counts),
        "pass_primitive_counts": dict(primitive_counts),
    }
    return summary


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

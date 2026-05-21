import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    records = []
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


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def summarize(events: List[Dict[str, Any]], progress: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    valid_events = [event for event in events if event.get("decision", {}).get("valid")]
    would_reject_events = [
        event for event in valid_events if bool(event.get("decision", {}).get("would_reject"))
    ]
    active_reject_events = [
        event for event in valid_events if bool(event.get("decision", {}).get("reject_required"))
    ]

    all_episode_keys = {_episode_key(event) for event in events}
    reject_episode_keys = {_episode_key(event) for event in would_reject_events}

    per_episode = defaultdict(
        lambda: {
            "events": 0,
            "would_reject_events": 0,
            "active_reject_events": 0,
            "checked_forward_steps": 0,
            "blocked_steps": 0,
            "max_blocked_steps": 0,
            "max_checked_forward_steps": 0,
        }
    )
    reason_counts = Counter()

    for event in valid_events:
        key = _episode_key(event)
        decision = event.get("decision", {})
        item = per_episode[key]
        item["scene_id"] = event.get("scene_id")
        item["episode_id"] = event.get("episode_id")
        item["events"] += 1
        checked = int(decision.get("checked_forward_steps") or 0)
        blocked = int(decision.get("blocked_steps") or 0)
        item["checked_forward_steps"] += checked
        item["blocked_steps"] += blocked
        item["max_checked_forward_steps"] = max(item["max_checked_forward_steps"], checked)
        item["max_blocked_steps"] = max(item["max_blocked_steps"], blocked)
        if decision.get("would_reject"):
            item["would_reject_events"] += 1
            reason_counts[str(decision.get("reject_reason") or "would_reject")] += 1
        if decision.get("reject_required"):
            item["active_reject_events"] += 1

    progress_by_key = {}
    if progress:
        for item in progress:
            progress_by_key[_episode_key(item)] = item

    reject_success_values = []
    non_reject_success_values = []
    failed_keys = set()
    failed_with_reject = set()

    for key, item in progress_by_key.items():
        success = float(item.get("success", 0.0))
        if success < 0.5:
            failed_keys.add(key)
        if key in reject_episode_keys:
            reject_success_values.append(success)
            if success < 0.5:
                failed_with_reject.add(key)
        else:
            non_reject_success_values.append(success)

    top_reject_episodes = sorted(
        per_episode.values(),
        key=lambda item: (
            int(item["would_reject_events"]),
            int(item["blocked_steps"]),
            int(item["events"]),
        ),
        reverse=True,
    )[:20]

    summary = {
        "total_events": len(events),
        "valid_events": len(valid_events),
        "would_reject_events": len(would_reject_events),
        "active_reject_events": len(active_reject_events),
        "episode_count_from_events": len(all_episode_keys),
        "would_reject_episode_count": len(reject_episode_keys),
        "would_reject_episode_rate": (
            len(reject_episode_keys) / len(all_episode_keys) if all_episode_keys else 0.0
        ),
        "reason_counts": dict(reason_counts),
        "top_reject_episodes": top_reject_episodes,
    }

    if progress_by_key:
        failed_count = len(failed_keys)
        summary["progress_episode_count"] = len(progress_by_key)
        summary["reject_episode_success_rate"] = _safe_mean(reject_success_values)
        summary["non_reject_episode_success_rate"] = _safe_mean(non_reject_success_values)
        summary["failed_episode_count"] = failed_count
        summary["failed_with_reject_count"] = len(failed_with_reject)
        summary["failed_with_reject_rate"] = (
            len(failed_with_reject) / failed_count if failed_count else 0.0
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize VLMap trajectory validation shadow logs."
    )
    parser.add_argument("--events", type=Path, help="Path to trajectory_events.jsonl")
    parser.add_argument("--run-dir", type=Path, help="Directory containing trajectory_events.jsonl")
    parser.add_argument("--progress", type=Path, help="Optional progress.json/jsonl for success correlation")
    parser.add_argument("--output", type=Path, help="Optional path to write summary JSON")
    args = parser.parse_args()

    events_path = args.events
    if events_path is None:
        if args.run_dir is None:
            parser.error("Provide --events or --run-dir")
        events_path = args.run_dir / "trajectory_events.jsonl"

    progress_path = args.progress
    if progress_path is None and args.run_dir is not None:
        candidate = args.run_dir / "progress.json"
        if candidate.exists():
            progress_path = candidate

    events = _read_json_records(events_path)
    progress = _read_json_records(progress_path) if progress_path and progress_path.exists() else None
    summary = summarize(events, progress)

    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

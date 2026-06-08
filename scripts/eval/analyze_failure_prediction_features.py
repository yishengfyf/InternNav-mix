import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def _group_by_episode(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_episode_key(record)].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: _safe_int(item.get("step_id"), -1))
    return grouped


def _resolve_run_dir(path: Path) -> Path:
    candidates = [
        path,
        path / "vlmap_safety_debug" / "run_001",
        path / "run_001",
    ]
    for candidate in candidates:
        if (candidate / "occ_memory_recovery_events.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find occ_memory_recovery_events.jsonl under {path}")


def _displacement_total(events: List[Dict[str, Any]]) -> float:
    total = 0.0
    prev_xy = None
    for event in events:
        xy = event.get("pose_xy")
        if not xy or len(xy) < 2:
            continue
        xy = [_safe_float(xy[0]), _safe_float(xy[1])]
        if prev_xy is not None:
            dx = xy[0] - prev_xy[0]
            dy = xy[1] - prev_xy[1]
            total += (dx * dx + dy * dy) ** 0.5
        prev_xy = xy
    return total


def _latest_before(events: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    latest = None
    for event in events:
        event_step = event.get("step_id")
        if event_step is None:
            continue
        if _safe_int(event_step) <= step:
            latest = event
        else:
            break
    return latest


def _semantic_coverage(semantic_event: Optional[Dict[str, Any]], step: int) -> float:
    if not semantic_event:
        return 0.0
    terms = semantic_event.get("landmark_terms") or []
    if not terms:
        landmarks = semantic_event.get("landmarks") or []
        terms = [item.get("term") for item in landmarks if item.get("term")]
    if not terms:
        return 0.0
    first_seen = semantic_event.get("first_seen_step_by_term") or {}
    seen = 0
    for term in terms:
        value = first_seen.get(term)
        if value is not None and _safe_int(value, 10**9) <= step:
            seen += 1
    return float(seen / max(1, len(terms)))


def _extract_features_at(
    *,
    recovery_events: List[Dict[str, Any]],
    semantic_events: List[Dict[str, Any]],
    progress: Dict[str, Any],
    step: int,
    window: int,
    max_steps: int,
) -> Optional[Dict[str, Any]]:
    past = [event for event in recovery_events if _safe_int(event.get("step_id"), -1) <= step]
    if not past:
        return None
    current = past[-1]
    min_step = max(0, step - window)
    window_events = [
        event
        for event in recovery_events
        if min_step <= _safe_int(event.get("step_id"), -1) <= step
    ]
    if not window_events:
        window_events = [current]
    first = window_events[0]
    occ_growth = _safe_int(current.get("occupied_cell_count")) - _safe_int(first.get("occupied_cell_count"))
    free_growth = _safe_int(current.get("free_cell_count")) - _safe_int(first.get("free_cell_count"))
    total_growth = occ_growth + free_growth
    displacement = _displacement_total(window_events)
    collision_sum = sum(_safe_float(event.get("collision_delta")) for event in window_events)
    collision_rate = collision_sum / max(1, len(window_events))

    sem_past = [
        event
        for event in semantic_events
        if event.get("event_type") == "semantic_match"
        and event.get("step_id") is not None
        and _safe_int(event.get("step_id")) <= step
    ]
    sem_window = [
        event
        for event in sem_past
        if _safe_int(event.get("step_id")) >= min_step
    ]
    latest_sem = sem_past[-1] if sem_past else None
    semantic_new_events = len(sem_window)
    high_conf_count = sum(1 for event in sem_past if event.get("high_conf_semantic"))
    semantic_stagnation = bool(latest_sem and latest_sem.get("stagnation_would_requery"))
    semantic_recent_unique = (
        None if latest_sem is None else latest_sem.get("stagnation_recent_unique_count")
    )
    semantic_coverage = _semantic_coverage(latest_sem, step)
    semantic_top_score = 0.0 if latest_sem is None else _safe_float(latest_sem.get("top_score"))
    semantic_top_margin = 0.0 if latest_sem is None else _safe_float(
        latest_sem.get("top_margin_to_second") or latest_sem.get("rank1_margin_to_second")
    )

    explore_efficiency = float(max(0, occ_growth) / max(0.01, displacement))
    steps = _safe_int(progress.get("steps"), max_steps)
    return {
        "occ_growth_last_w": float(occ_growth),
        "free_growth_last_w": float(free_growth),
        "total_growth_last_w": float(total_growth),
        "displacement_total_w": float(displacement),
        "stagnation_streak": float(_safe_int(current.get("occupied_stagnation_streak"))),
        "total_stagnation_streak": float(_safe_int(current.get("total_stagnation_streak"))),
        "collision_sum_w": float(collision_sum),
        "collision_rate_w": float(collision_rate),
        "semantic_new_events_w": float(semantic_new_events),
        "high_conf_count_t": float(high_conf_count),
        "semantic_coverage_t": float(semantic_coverage),
        "semantic_stagnation_t": 1.0 if semantic_stagnation else 0.0,
        "semantic_recent_unique_count_t": float(
            _safe_int(semantic_recent_unique, 0) if semantic_recent_unique is not None else 0
        ),
        "semantic_top_score_t": float(semantic_top_score),
        "semantic_top_margin_t": float(semantic_top_margin),
        "step_fraction": float(step / max(1, max_steps)),
        "episode_progress_fraction": float(step / max(1, steps)),
        "explore_efficiency": float(explore_efficiency),
    }


def _failure_mode(progress: Dict[str, Any]) -> str:
    if _safe_float(progress.get("success")) >= 0.5:
        return "success"
    steps = _safe_int(progress.get("steps"))
    ne = _safe_float(progress.get("ne"))
    collision = _safe_float(progress.get("collision_count"))
    if steps >= 400 or collision > 200:
        return "stuck_wall_hugging"
    if steps < 60 and ne > 5.0 and collision <= 0.0:
        return "early_lost"
    if ne > 3.0:
        return "navigation_failure"
    return "other_failure"


def _auc(labels: List[int], values: List[float]) -> Optional[float]:
    positives = [value for label, value in zip(labels, values) if label == 1]
    negatives = [value for label, value in zip(labels, values) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / max(1, total)


def _best_threshold(
    labels: List[int],
    values: List[float],
    *,
    direction: str,
    max_success_fpr: float,
) -> Dict[str, Any]:
    thresholds = sorted(set(values))
    best = None
    success_total = sum(1 for label in labels if label == 0)
    failure_total = sum(1 for label in labels if label == 1)
    for threshold in thresholds:
        if direction == "high":
            preds = [value >= threshold for value in values]
        else:
            preds = [value <= threshold for value in values]
        tp = sum(1 for pred, label in zip(preds, labels) if pred and label == 1)
        fp = sum(1 for pred, label in zip(preds, labels) if pred and label == 0)
        predicted = sum(1 for pred in preds if pred)
        recall = tp / max(1, failure_total)
        fpr = fp / max(1, success_total)
        precision = tp / max(1, predicted)
        item = {
            "threshold": threshold,
            "direction": direction,
            "tp": tp,
            "fp": fp,
            "predicted": predicted,
            "failure_recall": recall,
            "success_fpr": fpr,
            "precision": precision,
        }
        if fpr > max_success_fpr:
            continue
        if best is None:
            best = item
            continue
        key = (item["failure_recall"], item["precision"], -item["success_fpr"], item["predicted"])
        best_key = (best["failure_recall"], best["precision"], -best["success_fpr"], best["predicted"])
        if key > best_key:
            best = item
    return best or {
        "threshold": None,
        "direction": direction,
        "tp": 0,
        "fp": 0,
        "predicted": 0,
        "failure_recall": 0.0,
        "success_fpr": 0.0,
        "precision": None,
    }


def analyze(
    run_dir: Path,
    *,
    timepoints: List[int],
    window: int,
    max_success_fpr: float,
    max_steps: int,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_dir)
    progress = _read_json_records(run_dir / "progress.json")
    recovery_events = _read_json_records(run_dir / "occ_memory_recovery_events.jsonl")
    semantic_events = _read_json_records(run_dir / "semantic_events.jsonl")

    progress_by_key = {_episode_key(item): item for item in progress}
    recovery_by_key = _group_by_episode(recovery_events)
    semantic_by_key = _group_by_episode(
        event for event in semantic_events if event.get("event_type") == "semantic_match"
    )
    modes = {
        key: _failure_mode(item)
        for key, item in progress_by_key.items()
    }
    mode_counts = defaultdict(int)
    for mode in modes.values():
        mode_counts[mode] += 1

    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "episode_count": len(progress_by_key),
        "mode_counts": dict(mode_counts),
        "window": int(window),
        "timepoints": {},
    }
    for step in timepoints:
        rows = []
        for key, progress_item in progress_by_key.items():
            steps = _safe_int(progress_item.get("steps"))
            if steps < step:
                continue
            features = _extract_features_at(
                recovery_events=recovery_by_key.get(key, []),
                semantic_events=semantic_by_key.get(key, []),
                progress=progress_item,
                step=step,
                window=window,
                max_steps=max_steps,
            )
            if features is None:
                continue
            rows.append(
                {
                    "key": key,
                    "episode_id": progress_item.get("episode_id"),
                    "success": _safe_float(progress_item.get("success")),
                    "failure": 1 if _safe_float(progress_item.get("success")) < 0.5 else 0,
                    "mode": modes.get(key),
                    "features": features,
                }
            )
        labels = [int(row["failure"]) for row in rows]
        feature_names = sorted(rows[0]["features"].keys()) if rows else []
        feature_summaries = {}
        for name in feature_names:
            values = [_safe_float(row["features"].get(name)) for row in rows]
            auc_high = _auc(labels, values)
            auc_low = None if auc_high is None else 1.0 - auc_high
            best_direction = "high"
            best_auc = auc_high
            if auc_low is not None and (best_auc is None or auc_low > best_auc):
                best_auc = auc_low
                best_direction = "low"
            best_rule = _best_threshold(
                labels,
                values,
                direction=best_direction,
                max_success_fpr=max_success_fpr,
            )
            no_stuck_rows = [row for row in rows if row.get("mode") != "stuck_wall_hugging"]
            no_stuck_labels = [int(row["failure"]) for row in no_stuck_rows]
            no_stuck_values = [_safe_float(row["features"].get(name)) for row in no_stuck_rows]
            no_stuck_auc_high = _auc(no_stuck_labels, no_stuck_values)
            no_stuck_best_auc = None
            if no_stuck_auc_high is not None:
                no_stuck_best_auc = max(no_stuck_auc_high, 1.0 - no_stuck_auc_high)
            feature_summaries[name] = {
                "auc": best_auc,
                "auc_direction": best_direction,
                "auc_without_stuck": no_stuck_best_auc,
                "best_rule": best_rule,
            }
        ranked = sorted(
            feature_summaries.items(),
            key=lambda item: (
                -1.0 if item[1]["auc"] is None else -float(item[1]["auc"]),
                item[0],
            ),
        )
        result["timepoints"][str(step)] = {
            "sample_count": len(rows),
            "failure_count": sum(labels),
            "success_count": len(labels) - sum(labels),
            "feature_auc_rank": [
                {"feature": name, **summary}
                for name, summary in ranked
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Stage14a failure/stuck prediction features from Stage13a logs."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timepoints", type=str, default="20,30,40,50,60")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--max-success-fpr", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    timepoints = [int(item.strip()) for item in args.timepoints.split(",") if item.strip()]
    result = analyze(
        args.run_dir,
        timepoints=timepoints,
        window=args.window,
        max_success_fpr=args.max_success_fpr,
        max_steps=args.max_steps,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

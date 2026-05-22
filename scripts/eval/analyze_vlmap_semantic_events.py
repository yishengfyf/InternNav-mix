import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _success_split_mean(
    summaries_by_key: Dict[str, Dict[str, Any]],
    progress_by_key: Dict[str, Dict[str, Any]],
    field: str,
) -> Dict[str, Optional[float]]:
    all_values = []
    success_values = []
    failure_values = []
    for key, semantic in summaries_by_key.items():
        value = semantic.get(field)
        all_values.append(value)
        progress_item = progress_by_key.get(key, {})
        success = progress_item.get("success", semantic.get("success"))
        if success is None:
            continue
        if float(success) >= 0.5:
            success_values.append(value)
        else:
            failure_values.append(value)
    return {
        "mean": _safe_mean(all_values),
        "success_mean": _safe_mean(success_values),
        "failure_mean": _safe_mean(failure_values),
    }


def _success_split_values(
    values_by_key: Dict[str, Optional[float]],
    summaries_by_key: Dict[str, Dict[str, Any]],
    progress_by_key: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    all_values = []
    success_values = []
    failure_values = []
    for key, semantic in summaries_by_key.items():
        value = values_by_key.get(key)
        all_values.append(value)
        progress_item = progress_by_key.get(key, {})
        success = progress_item.get("success", semantic.get("success"))
        if success is None:
            continue
        if float(success) >= 0.5:
            success_values.append(value)
        else:
            failure_values.append(value)
    return {
        "mean": _safe_mean(all_values),
        "success_mean": _safe_mean(success_values),
        "failure_mean": _safe_mean(failure_values),
    }


def summarize(
    events: List[Dict[str, Any]],
    episode_summaries: Optional[List[Dict[str, Any]]] = None,
    progress: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    match_events = [event for event in events if event.get("event_type") == "semantic_match"]
    ok_events = [event for event in match_events if event.get("status") == "ok"]
    no_landmark_events = [event for event in match_events if event.get("status") == "no_landmarks"]
    init_errors = [event for event in events if event.get("event_type") == "semantic_init_error"]

    per_episode = defaultdict(
        lambda: {
            "events": 0,
            "ok_events": 0,
            "hit_events": 0,
            "top_matches": Counter(),
            "threshold_hits": Counter(),
            "rank1_terms": Counter(),
            "rank1_confident_terms": Counter(),
            "relative_hits": Counter(),
        }
    )
    top_counter = Counter()
    hit_counter = Counter()
    status_counter = Counter()
    backend_counter = Counter()
    device_counter = Counter()
    rank1_counter = Counter()
    rank1_confident_counter = Counter()
    relative_counter = Counter()

    for event in match_events:
        key = _episode_key(event)
        item = per_episode[key]
        item["scene_id"] = event.get("scene_id")
        item["episode_id"] = event.get("episode_id")
        item["events"] += 1
        status_counter[event.get("status")] += 1
        if event.get("status") != "ok":
            continue
        item["ok_events"] += 1
        backend_counter[event.get("backend")] += 1
        device_counter[event.get("device")] += 1
        top_match = event.get("top_match")
        if top_match:
            item["top_matches"][top_match] += 1
            top_counter[top_match] += 1
        rank1_term = event.get("rank1_term") or top_match
        if rank1_term:
            item["rank1_terms"][rank1_term] += 1
            rank1_counter[rank1_term] += 1
            if event.get("rank1_confident"):
                item["rank1_confident_terms"][rank1_term] += 1
                rank1_confident_counter[rank1_term] += 1
        hits = event.get("threshold_hits") or []
        if hits:
            item["hit_events"] += 1
        for hit in hits:
            item["threshold_hits"][hit] += 1
            hit_counter[hit] += 1
        for hit in event.get("relative_hits") or []:
            item["relative_hits"][hit] += 1
            relative_counter[hit] += 1

    summaries_by_key = {}
    for summary in episode_summaries or []:
        summaries_by_key[_episode_key(summary)] = summary

    progress_by_key = {}
    for item in progress or []:
        progress_by_key[_episode_key(item)] = item

    summary_keys = set(summaries_by_key)
    progress_keys = set(progress_by_key)
    success_coverages = []
    failure_coverages = []
    success_first_seen = []
    failure_first_seen = []
    success_seen_counts = []
    failure_seen_counts = []

    for key, semantic in summaries_by_key.items():
        progress_item = progress_by_key.get(key, {})
        success = progress_item.get("success", semantic.get("success"))
        coverage = semantic.get("coverage")
        seen_count = semantic.get("seen_count")
        first_seen_step = semantic.get("first_seen_step")
        if success is None:
            continue
        if float(success) >= 0.5:
            success_coverages.append(coverage)
            success_seen_counts.append(seen_count)
            success_first_seen.append(first_seen_step)
        else:
            failure_coverages.append(coverage)
            failure_seen_counts.append(seen_count)
            failure_first_seen.append(first_seen_step)

    top_episodes = []
    for key, item in per_episode.items():
        top_episodes.append(
            {
                "scene_id": item.get("scene_id"),
                "episode_id": item.get("episode_id"),
                "events": item["events"],
                "ok_events": item["ok_events"],
                "hit_events": item["hit_events"],
                "top_matches": dict(item["top_matches"].most_common(5)),
                "threshold_hits": dict(item["threshold_hits"].most_common(5)),
                "rank1_terms": dict(item["rank1_terms"].most_common(5)),
                "rank1_confident_terms": dict(item["rank1_confident_terms"].most_common(5)),
                "relative_hits": dict(item["relative_hits"].most_common(5)),
            }
        )
    top_episodes.sort(key=lambda item: (item["hit_events"], item["ok_events"]), reverse=True)

    tracked_summary_fields = [
        "rank1_coverage",
        "rank1_confident_coverage",
        "relative_coverage",
        "rank1_room_coverage",
        "rank1_object_coverage",
        "rank1_confident_room_coverage",
        "rank1_confident_object_coverage",
        "relative_room_coverage",
        "relative_object_coverage",
        "mean_top_score",
        "max_top_score",
        "mean_top_margin",
        "max_top_margin",
        "top1_stability",
        "top1_diversity",
        "top1_entropy",
        "top1_transition_count",
        "high_conf_seen",
        "high_conf_event_count",
        "high_conf_step_fraction",
        "first_high_conf_step",
        "max_low_conf_streak",
        "confidence_would_requery",
        "confidence_would_requery_count",
        "first_confidence_would_requery_step",
        "stagnation_would_requery",
        "stagnation_would_requery_count",
        "first_stagnation_would_requery_step",
        "stagnation_min_recent_unique_count",
        "stagnation_low_diversity_window_count",
        "rank1_first_seen_step",
        "rank1_confident_first_seen_step",
        "relative_first_seen_step",
    ]
    metric_splits = {
        field: _success_split_mean(summaries_by_key, progress_by_key, field)
        for field in tracked_summary_fields
    }

    threshold_keys = sorted(
        {
            threshold
            for summary in summaries_by_key.values()
            for threshold in (summary.get("coverage_by_threshold") or {}).keys()
        },
        key=lambda item: float(item),
    )
    coverage_by_threshold = {}
    for threshold in threshold_keys:
        field_values = []
        success_values = []
        failure_values = []
        for key, semantic in summaries_by_key.items():
            value = (semantic.get("coverage_by_threshold") or {}).get(threshold)
            field_values.append(value)
            progress_item = progress_by_key.get(key, {})
            success = progress_item.get("success", semantic.get("success"))
            if success is None:
                continue
            if float(success) >= 0.5:
                success_values.append(value)
            else:
                failure_values.append(value)
        coverage_by_threshold[threshold] = {
            "mean": _safe_mean(field_values),
            "success_mean": _safe_mean(success_values),
            "failure_mean": _safe_mean(failure_values),
        }

    relative_coverage_by_z_threshold = {}
    for z_threshold in (0.5, 1.0, 1.5):
        seen_by_key = defaultdict(set)
        for event in ok_events:
            key = _episode_key(event)
            scores = event.get("scores") or []
            for score_item in scores:
                score_z = score_item.get("score_z")
                is_single_rank1 = len(scores) == 1 and score_item.get("rank") == 1
                if is_single_rank1 or (
                    score_z is not None and float(score_z) >= z_threshold
                ):
                    seen_by_key[key].add(score_item.get("term"))
        coverage_values = {}
        for key, semantic in summaries_by_key.items():
            terms = set(semantic.get("landmark_terms") or [])
            seen = {term for term in seen_by_key.get(key, set()) if term}
            coverage_values[key] = (len(terms & seen) / len(terms)) if terms else 0.0
        relative_coverage_by_z_threshold[f"{z_threshold:.1f}"] = _success_split_values(
            coverage_values, summaries_by_key, progress_by_key
        )

    transition_rate_values = {}
    for key, semantic in summaries_by_key.items():
        event_count = semantic.get("semantic_event_count") or 0
        transition_count = semantic.get("top1_transition_count")
        if transition_count is None:
            transition_rate_values[key] = None
        elif event_count and event_count > 1:
            transition_rate_values[key] = float(transition_count) / float(event_count - 1)
        else:
            transition_rate_values[key] = 0.0
    transition_rate_split = _success_split_values(
        transition_rate_values, summaries_by_key, progress_by_key
    )

    def _policy_stats(trigger_field: str) -> Dict[str, Any]:
        success_total = 0
        failure_total = 0
        triggered_success = 0
        triggered_failure = 0
        triggered_total = 0
        severe_failure_total = 0
        triggered_severe_failure = 0
        high_quality_success_total = 0
        triggered_high_quality_success = 0
        for key, semantic in summaries_by_key.items():
            progress_item = progress_by_key.get(key, {})
            success = progress_item.get("success", semantic.get("success"))
            if success is None:
                continue
            triggered = bool(semantic.get(trigger_field))
            spl = progress_item.get("spl", semantic.get("spl"))
            ne = progress_item.get("ne", semantic.get("ne"))
            steps = progress_item.get("steps", semantic.get("steps"))
            is_success = float(success) >= 0.5
            is_severe_failure = False
            if not is_success:
                ne_severe = ne is not None and float(ne) > 8.0
                step_severe = steps is not None and float(steps) > 120.0
                is_severe_failure = ne_severe or step_severe
            is_high_quality_success = (
                is_success and spl is not None and float(spl) >= 0.85
            )

            if is_success:
                success_total += 1
                if triggered:
                    triggered_success += 1
                if is_high_quality_success:
                    high_quality_success_total += 1
                    if triggered:
                        triggered_high_quality_success += 1
            else:
                failure_total += 1
                if triggered:
                    triggered_failure += 1
                if is_severe_failure:
                    severe_failure_total += 1
                    if triggered:
                        triggered_severe_failure += 1
            if triggered:
                triggered_total += 1

        return {
            "triggered_episode_count": triggered_total,
            "triggered_success_count": triggered_success,
            "triggered_failure_count": triggered_failure,
            "success_episode_count": success_total,
            "failure_episode_count": failure_total,
            "failure_precision": (
                triggered_failure / triggered_total if triggered_total else None
            ),
            "failure_recall": (
                triggered_failure / failure_total if failure_total else None
            ),
            "success_false_positive_rate": (
                triggered_success / success_total if success_total else None
            ),
            "severe_failure_episode_count": severe_failure_total,
            "triggered_severe_failure_count": triggered_severe_failure,
            "severe_failure_recall": (
                triggered_severe_failure / severe_failure_total
                if severe_failure_total
                else None
            ),
            "high_quality_success_episode_count": high_quality_success_total,
            "triggered_high_quality_success_count": triggered_high_quality_success,
            "high_quality_success_false_positive_rate": (
                triggered_high_quality_success / high_quality_success_total
                if high_quality_success_total
                else None
            ),
        }

    confidence_policy_stats = _policy_stats("confidence_would_requery")
    stagnation_policy_stats = _policy_stats("stagnation_would_requery")

    return {
        "total_events": len(events),
        "semantic_match_events": len(match_events),
        "ok_match_events": len(ok_events),
        "no_landmark_events": len(no_landmark_events),
        "init_error_count": len(init_errors),
        "status_counts": dict(status_counter.most_common()),
        "backend_counts": dict(backend_counter.most_common()),
        "device_counts": dict(device_counter.most_common()),
        "episode_count_from_events": len(per_episode),
        "episode_summary_count": len(summary_keys),
        "progress_episode_count": len(progress_keys),
        "matched_progress_episode_count": len(summary_keys & progress_keys),
        "mean_coverage": _safe_mean(item.get("coverage") for item in summaries_by_key.values()),
        "mean_seen_count": _safe_mean(item.get("seen_count") for item in summaries_by_key.values()),
        "success_mean_coverage": _safe_mean(success_coverages),
        "failure_mean_coverage": _safe_mean(failure_coverages),
        "success_mean_seen_count": _safe_mean(success_seen_counts),
        "failure_mean_seen_count": _safe_mean(failure_seen_counts),
        "success_mean_first_seen_step": _safe_mean(success_first_seen),
        "failure_mean_first_seen_step": _safe_mean(failure_first_seen),
        "coverage_by_threshold": coverage_by_threshold,
        "rank1_mean_coverage": metric_splits["rank1_coverage"]["mean"],
        "rank1_success_mean_coverage": metric_splits["rank1_coverage"]["success_mean"],
        "rank1_failure_mean_coverage": metric_splits["rank1_coverage"]["failure_mean"],
        "rank1_confident_mean_coverage": metric_splits["rank1_confident_coverage"]["mean"],
        "rank1_confident_success_mean_coverage": metric_splits["rank1_confident_coverage"]["success_mean"],
        "rank1_confident_failure_mean_coverage": metric_splits["rank1_confident_coverage"]["failure_mean"],
        "relative_mean_coverage": metric_splits["relative_coverage"]["mean"],
        "relative_success_mean_coverage": metric_splits["relative_coverage"]["success_mean"],
        "relative_failure_mean_coverage": metric_splits["relative_coverage"]["failure_mean"],
        "relative_coverage_by_z_threshold": relative_coverage_by_z_threshold,
        "transition_rate": transition_rate_split,
        "confidence_policy_stats": confidence_policy_stats,
        "stagnation_policy_stats": stagnation_policy_stats,
        "summary_metric_splits": metric_splits,
        "top_match_counts": dict(top_counter.most_common(20)),
        "threshold_hit_counts": dict(hit_counter.most_common(20)),
        "rank1_counts": dict(rank1_counter.most_common(20)),
        "rank1_confident_counts": dict(rank1_confident_counter.most_common(20)),
        "relative_hit_counts": dict(relative_counter.most_common(20)),
        "top_hit_episodes": top_episodes[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VLMap semantic shadow logs.")
    parser.add_argument("--events", type=Path, help="Path to semantic_events.jsonl")
    parser.add_argument("--summary", type=Path, help="Path to semantic_episode_summary.jsonl")
    parser.add_argument("--run-dir", type=Path, help="Directory containing semantic logs")
    parser.add_argument("--progress", type=Path, help="Optional progress.json/jsonl for success correlation")
    parser.add_argument("--output", type=Path, help="Optional path to write summary JSON")
    args = parser.parse_args()

    events_path = args.events
    if events_path is None:
        if args.run_dir is None:
            parser.error("Provide --events or --run-dir")
        events_path = args.run_dir / "semantic_events.jsonl"

    summary_path = args.summary
    if summary_path is None and args.run_dir is not None:
        summary_path = args.run_dir / "semantic_episode_summary.jsonl"

    progress_path = args.progress
    if progress_path is None and args.run_dir is not None:
        candidate = args.run_dir / "progress.json"
        if candidate.exists():
            progress_path = candidate

    events = _read_json_records(events_path)
    episode_summaries = _read_json_records(summary_path) if summary_path else []
    progress = _read_json_records(progress_path) if progress_path and progress_path.exists() else []
    payload = json.dumps(summarize(events, episode_summaries, progress), indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

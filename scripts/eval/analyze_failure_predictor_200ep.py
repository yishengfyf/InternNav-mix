"""
Post-hoc 200-episode failure predictor audit.

This script re-checks whether early directional / semantic signals can
separate failures from successes on the 200ep baseline safety run.

Default input:
  ../../compare_baseline_safety_epseed_200/vlmap_safety_debug/run_001

Outputs:
  analysis_200ep_failure_predictor.json

Notes:
  - Uses only events at or before the selected analysis step.
  - Episodes shorter than the analysis step are included by default using
    their last available event, so early failures are not dropped.
  - If failure_prediction_events.jsonl is absent, semantic score is computed
    as an explicit proxy from semantic_events.jsonl.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 480.0
IMAGE_HALF_DIAG = math.sqrt((IMAGE_WIDTH / 2.0) ** 2 + (IMAGE_HEIGHT / 2.0) ** 2)


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


def _group_by_episode(records: Iterable[Dict[str, Any]], step_field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_episode_key(record)].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: _safe_int(item.get(step_field), -1))
    return grouped


def _default_run_dir() -> Path:
    # scripts/eval/analyze_failure_predictor_200ep.py -> InternNav -> vln
    return (
        Path(__file__).resolve().parents[3]
        / "compare_baseline_safety_epseed_200"
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
        if (candidate / "trajectory_events.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"trajectory_events.jsonl not found under {path}")


def _progress_path(run_dir: Path) -> Path:
    for candidate in [
        run_dir / "progress.json",
        run_dir.parents[1] / "progress.json" if len(run_dir.parents) > 1 else run_dir / "progress.json",
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


def _latest_before(events: List[Dict[str, Any]], step: int, step_field: str) -> Optional[Dict[str, Any]]:
    latest = None
    for event in events:
        if _safe_int(event.get(step_field), -1) <= step:
            latest = event
        else:
            break
    return latest


def _events_before(events: List[Dict[str, Any]], step: int, step_field: str) -> List[Dict[str, Any]]:
    return [event for event in events if _safe_int(event.get(step_field), -1) <= step]


def _angle_diff_rad(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def _circular_variance(angles_rad: List[float]) -> float:
    if len(angles_rad) < 2:
        return 0.0
    sin_mean = sum(math.sin(a) for a in angles_rad) / len(angles_rad)
    cos_mean = sum(math.cos(a) for a in angles_rad) / len(angles_rad)
    r_bar = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    return max(0.0, min(1.0, 1.0 - r_bar))


def _pixel_eccentricity(pixel_goal: Any) -> Optional[float]:
    if not pixel_goal or len(pixel_goal) < 2:
        return None
    x = _safe_float(pixel_goal[0], float("nan"))
    y = _safe_float(pixel_goal[1], float("nan"))
    if math.isnan(x) or math.isnan(y):
        return None
    dx = x - IMAGE_WIDTH / 2.0
    dy = y - IMAGE_HEIGHT / 2.0
    return math.sqrt(dx * dx + dy * dy) / IMAGE_HALF_DIAG


def _extract_direction_features(traj_events: List[Dict[str, Any]], step: int) -> Dict[str, float]:
    past = _events_before(traj_events, step, "eval_step")
    if not past:
        return {
            "pg_ecc_mean": 0.0,
            "pg_ecc_max": 0.0,
            "heading_var": 0.0,
            "heading_consistency": 1.0,
            "compass_reversal_count": 0.0,
            "compass_reversal_rate": 0.0,
        }

    eccs = []
    compass_vals = []
    gps_points: List[Tuple[int, List[float]]] = []
    for event in past:
        ecc = _pixel_eccentricity(event.get("pixel_goal"))
        if ecc is not None:
            eccs.append(ecc)
        compass = event.get("compass")
        if compass and len(compass) > 0:
            compass_vals.append(_safe_float(compass[0]))
        gps = event.get("gps")
        if gps and len(gps) >= 2:
            gps_points.append((
                _safe_int(event.get("eval_step"), -1),
                [_safe_float(gps[0]), _safe_float(gps[1])],
            ))

    gps_points.sort(key=lambda item: item[0])
    headings = []
    for idx in range(1, len(gps_points)):
        prev = gps_points[idx - 1][1]
        cur = gps_points[idx][1]
        dx = cur[0] - prev[0]
        dz = cur[1] - prev[1]
        if math.sqrt(dx * dx + dz * dz) < 1e-4:
            continue
        headings.append(math.atan2(dx, dz))
    heading_var = _circular_variance(headings)

    reversals = 0
    for idx in range(1, len(compass_vals)):
        if _angle_diff_rad(compass_vals[idx], compass_vals[idx - 1]) >= (math.pi * 0.75):
            reversals += 1

    return {
        "pg_ecc_mean": sum(eccs) / len(eccs) if eccs else 0.0,
        "pg_ecc_max": max(eccs) if eccs else 0.0,
        "heading_var": heading_var,
        "heading_consistency": 1.0 - heading_var,
        "compass_reversal_count": float(reversals),
        "compass_reversal_rate": float(reversals / max(1, len(compass_vals) - 1)),
    }


def _semantic_coverage(event: Optional[Dict[str, Any]], step: int) -> float:
    if not event:
        return 0.0
    terms = event.get("landmark_terms") or []
    if not terms:
        return 0.0
    first_seen = event.get("first_seen_step_by_term") or {}
    seen = 0
    for term in terms:
        value = first_seen.get(term)
        if value is not None and _safe_int(value, 10**9) <= step:
            seen += 1
    return seen / max(1, len(terms))


def _semantic_proxy_score(semantic_events: List[Dict[str, Any]], step: int) -> Dict[str, float]:
    sem_past = [
        event for event in semantic_events
        if event.get("event_type") == "semantic_match"
        and _safe_int(event.get("step_id"), -1) <= step
    ]
    if not sem_past:
        return {
            "fp_semantic_score": 0.0,
            "semantic_stagnation": 0.0,
            "semantic_coverage": 0.0,
            "semantic_top_score": 0.0,
            "semantic_top_margin": 0.0,
        }
    latest = sem_past[-1]
    coverage = _semantic_coverage(latest, step)
    top_score = _safe_float(latest.get("top_score"))
    margin = _safe_float(latest.get("top_margin_to_second") or latest.get("rank1_margin_to_second"))
    stagnation = 1.0 if latest.get("stagnation_would_requery") else 0.0
    low_conf = max(0.0, (0.31 - top_score) / 0.31)
    low_margin = max(0.0, (0.02 - margin) / 0.02)
    low_diversity = 0.0
    unique_count = latest.get("stagnation_recent_unique_count")
    if unique_count is not None:
        low_diversity = 1.0 if _safe_int(unique_count) <= 1 else 0.0
    score = (
        0.35 * stagnation
        + 0.25 * low_conf
        + 0.20 * (1.0 - coverage)
        + 0.10 * low_margin
        + 0.10 * low_diversity
    )
    return {
        "fp_semantic_score": max(0.0, min(1.0, score)),
        "semantic_stagnation": stagnation,
        "semantic_coverage": coverage,
        "semantic_top_score": top_score,
        "semantic_top_margin": margin,
    }


def _failure_prediction_score(fp_events: List[Dict[str, Any]], step: int) -> Optional[float]:
    latest = _latest_before(fp_events, step, "step_id")
    if not latest:
        return None
    for key in ("fp_semantic_score", "semantic_score", "failure_score"):
        value = latest.get(key)
        if value is not None:
            return _safe_float(value)
    breakdown = latest.get("signal_breakdown") or {}
    for key in ("semantic_score", "fp_semantic_score"):
        value = breakdown.get(key)
        if value is not None:
            return _safe_float(value)
    return None


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
    max_fpr: float,
) -> Dict[str, Any]:
    thresholds = sorted(set(values))
    success_total = sum(1 for label in labels if label == 0)
    failure_total = sum(1 for label in labels if label == 1)
    best: Optional[Dict[str, Any]] = None
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
        precision = tp / max(1, predicted) if predicted else None
        if fpr > max_fpr:
            continue
        item = {
            "threshold": threshold,
            "direction": direction,
            "tp": tp,
            "fp": fp,
            "predicted": predicted,
            "recall": recall,
            "success_fpr": fpr,
            "precision": precision,
        }
        if best is None:
            best = item
            continue
        item_key = (item["recall"], item["precision"] or 0.0, -item["success_fpr"], item["predicted"])
        best_key = (best["recall"], best["precision"] or 0.0, -best["success_fpr"], best["predicted"])
        if item_key > best_key:
            best = item
    return best or {
        "threshold": None,
        "direction": direction,
        "tp": 0,
        "fp": 0,
        "predicted": 0,
        "recall": 0.0,
        "success_fpr": 0.0,
        "precision": None,
    }


def _feature_summary(rows: List[Dict[str, Any]], feature: str, max_fpr: float) -> Dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    values = [_safe_float(row["features"].get(feature)) for row in rows]
    auc_high = _auc(labels, values)
    auc_low = None if auc_high is None else 1.0 - auc_high
    if auc_high is None:
        return {
            "feature": feature,
            "auc": None,
            "auc_direction": None,
            "best_rule": None,
        }
    direction = "high"
    auc_value = auc_high
    if auc_low is not None and auc_low > auc_value:
        direction = "low"
        auc_value = auc_low
    return {
        "feature": feature,
        "auc": auc_value,
        "auc_direction": direction,
        "auc_high": auc_high,
        "auc_low": auc_low,
        "best_rule": _best_threshold(labels, values, direction=direction, max_fpr=max_fpr),
    }


def analyze(
    run_dir: Path,
    *,
    timepoints: List[int],
    max_fpr: float,
    include_ended: bool,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_dir)
    progress = _read_json_records(_progress_path(run_dir))
    trajectory_events = _read_json_records(run_dir / "trajectory_events.jsonl")
    semantic_events = _read_json_records(run_dir / "semantic_events.jsonl")
    fp_events = _read_json_records(run_dir / "failure_prediction_events.jsonl")

    progress_by_key = {_episode_key(item): item for item in progress}
    traj_by_key = _group_by_episode(trajectory_events, "eval_step")
    sem_by_key = _group_by_episode(semantic_events, "step_id")
    fp_by_key = _group_by_episode(fp_events, "step_id")

    modes = {key: _failure_mode(item) for key, item in progress_by_key.items()}
    mode_counts: Dict[str, int] = defaultdict(int)
    for mode in modes.values():
        mode_counts[mode] += 1

    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "episode_count": len(progress_by_key),
        "mode_counts": dict(mode_counts),
        "max_fpr": max_fpr,
        "include_ended": include_ended,
        "semantic_score_source": "failure_prediction_events if present, otherwise semantic_events proxy",
        "timepoints": {},
    }

    for requested_step in timepoints:
        episode_rows: List[Dict[str, Any]] = []
        for key, progress_item in progress_by_key.items():
            steps = _safe_int(progress_item.get("steps"))
            if steps < requested_step and not include_ended:
                continue
            effective_step = min(requested_step, steps) if include_ended else requested_step
            dir_features = _extract_direction_features(traj_by_key.get(key, []), effective_step)
            sem_features = _semantic_proxy_score(sem_by_key.get(key, []), effective_step)
            fp_semantic = _failure_prediction_score(fp_by_key.get(key, []), effective_step)
            if fp_semantic is not None:
                sem_features["fp_semantic_score"] = fp_semantic
            features = {**dir_features, **sem_features}
            success = _safe_float(progress_item.get("success")) >= 0.5
            episode_rows.append({
                "key": key,
                "episode_id": progress_item.get("episode_id"),
                "success": success,
                "mode": modes.get(key),
                "requested_step": requested_step,
                "effective_step": effective_step,
                "episode_steps": steps,
                "features": features,
            })

        groups = {
            "all_failures": {"positive_modes": None},
            "mode_c_navigation_lost": {"positive_modes": {"mode_c_navigation_lost"}},
            "mode_d_mid_fail": {"positive_modes": {"mode_d_mid_fail"}},
            "mode_b_early": {"positive_modes": {"mode_b_early"}},
        }
        group_results: Dict[str, Any] = {}
        feature_names = sorted(episode_rows[0]["features"].keys()) if episode_rows else []
        for group_name, cfg in groups.items():
            positive_modes = cfg["positive_modes"]
            rows = []
            for row in episode_rows:
                if row["success"]:
                    rows.append({**row, "label": 0})
                    continue
                is_positive = (
                    True if positive_modes is None
                    else row["mode"] in positive_modes
                )
                if is_positive:
                    rows.append({**row, "label": 1})
            if not rows:
                continue
            summaries = [_feature_summary(rows, feature, max_fpr) for feature in feature_names]
            summaries.sort(key=lambda item: -1.0 if item["auc"] is None else -float(item["auc"]))
            labels = [int(row["label"]) for row in rows]
            group_results[group_name] = {
                "sample_count": len(rows),
                "positive_count": sum(labels),
                "success_negative_count": len(labels) - sum(labels),
                "feature_auc_rank": summaries,
            }

        result["timepoints"][str(requested_step)] = {
            "episode_count": len(episode_rows),
            "group_results": group_results,
            "per_episode_features": [
                {
                    "episode_id": row["episode_id"],
                    "success": row["success"],
                    "mode": row["mode"],
                    "requested_step": row["requested_step"],
                    "effective_step": row["effective_step"],
                    "episode_steps": row["episode_steps"],
                    "features": row["features"],
                }
                for row in episode_rows
            ],
        }
    return result


def _print_summary(result: Dict[str, Any]) -> None:
    print(f"\n200ep failure predictor audit: {result['run_dir']}")
    print(f"Episodes: {result['episode_count']}  max_fpr={result['max_fpr']}")
    print(f"Mode counts: {result['mode_counts']}\n")
    for step, step_result in result["timepoints"].items():
        print(f"=== step={step} ===")
        for group_name, group in step_result["group_results"].items():
            print(
                f"  {group_name}: n={group['sample_count']} "
                f"pos={group['positive_count']} succ_neg={group['success_negative_count']}"
            )
            print(f"  {'feature':<28} {'auc':>6} {'dir':>4} {'recall':>7} {'fpr':>6} {'thr':>9}")
            for item in group["feature_auc_rank"][:10]:
                auc = item.get("auc")
                rule = item.get("best_rule") or {}
                thr = rule.get("threshold")
                print(
                    f"  {item['feature']:<28} "
                    f"{auc:>6.3f} "
                    f"{str(item.get('auc_direction')):>4} "
                    f"{_safe_float(rule.get('recall')):>7.3f} "
                    f"{_safe_float(rule.get('success_fpr')):>6.3f} "
                    f"{thr if thr is not None else 'None':>9}"
                )
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 200ep failure predictor features.")
    parser.add_argument("--run-dir", type=Path, default=_default_run_dir())
    parser.add_argument("--timepoints", default="60")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument(
        "--exclude-ended",
        action="store_true",
        help="Drop episodes that ended before each requested timepoint.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    timepoints = [int(item.strip()) for item in args.timepoints.split(",") if item.strip()]
    result = analyze(
        args.run_dir,
        timepoints=timepoints,
        max_fpr=args.max_fpr,
        include_ended=not args.exclude_ended,
    )
    if not args.quiet:
        _print_summary(result)

    output = args.output
    if output is None:
        output = _resolve_run_dir(args.run_dir) / "analysis_200ep_failure_predictor.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

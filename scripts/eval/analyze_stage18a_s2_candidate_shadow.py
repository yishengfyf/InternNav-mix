"""Audit Stage18a S2/current-policy candidate labels.

Stage18a is a logging and diagnosis stage.  It compares the frozen policy's
current waypoint against OccMem candidates under the same GT route-progress
direction label, but it never trains a model and never changes navigation.

The key question is whether a future S2-aware adapter has intervention
headroom:

* keep S2 when the current policy is already good;
* intervene only when an OccMem candidate is meaningfully better;
* avoid completed/revisited/unsafe changes that would damage success episodes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _read_json_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list: {path}")
        return [item for item in data if isinstance(item, dict)]
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _episode_key(record: Mapping[str, Any]) -> str:
    return f"{record.get('scene_id')}|{record.get('episode_id')}"


def _load_success_by_episode(progress_path: Optional[Path]) -> Dict[str, Optional[bool]]:
    by_key: Dict[str, Optional[bool]] = {}
    for record in _read_json_records(progress_path):
        metrics = record.get("metrics") or {}
        success = metrics.get("success", record.get("success"))
        if success is None:
            by_key[_episode_key(record)] = None
        else:
            by_key[_episode_key(record)] = bool(success)
    return by_key


def _candidate_by_id(candidates: Sequence[Mapping[str, Any]], candidate_id: Any) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return dict(candidate)
    return None


def _select_best(candidates: Sequence[Mapping[str, Any]], score_key: str) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return dict(max(candidates, key=lambda item: _safe_float(item.get(score_key), -1e9)))


def _select_best_angle(candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [item for item in candidates if item.get("gt_angle_diff_deg") is not None]
    if not valid:
        return None
    return dict(min(valid, key=lambda item: _safe_float(item.get("gt_angle_diff_deg"), 1e9)))


def _shadow_selected(row: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    shadow = row.get("progress_ranker_shadow") or {}
    selected = shadow.get(key) or {}
    candidate_id = selected.get("candidate_id")
    if candidate_id is None:
        return None
    return _candidate_by_id(row.get("candidates") or [], candidate_id)


def _is_correct(candidate: Optional[Mapping[str, Any]]) -> bool:
    return bool(candidate and candidate.get("gt_correct"))


def _angle(candidate: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not candidate or candidate.get("gt_angle_diff_deg") is None:
        return None
    return _safe_float(candidate.get("gt_angle_diff_deg"))


def _is_risky(candidate: Optional[Mapping[str, Any]]) -> bool:
    if not candidate:
        return True
    if not candidate.get("geometry_safe", True):
        return True
    if not candidate.get("active_gate_safe", True):
        return True
    if candidate.get("points_to_revisited_region"):
        return True
    if str(candidate.get("landmark_status") or "") == "completed":
        return True
    if _safe_float(candidate.get("completed_landmark_penalty")) > 0.0:
        return True
    if _safe_float(candidate.get("repeated_semantic_penalty")) > 0.0:
        return True
    return False


def _summarize_policy(selections: Sequence[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    valid = [item for item in selections if item]
    angles = [_angle(item) for item in valid]
    angles = [value for value in angles if value is not None]
    return {
        "valid_count": len(valid),
        "gt_correct_count": sum(_is_correct(item) for item in valid),
        "gt_correct_rate": float(sum(_is_correct(item) for item in valid) / max(1, len(valid))),
        "mean_angle_diff_deg": float(sum(angles) / len(angles)) if angles else None,
        "risky_count": sum(_is_risky(item) for item in valid),
        "risky_rate": float(sum(_is_risky(item) for item in valid) / max(1, len(valid))),
    }


def analyze(
    *,
    labels_path: Path,
    progress_path: Optional[Path],
    advantage_margin_deg: float,
) -> Dict[str, Any]:
    rows = [
        row
        for row in _read_json_records(labels_path)
        if row.get("label_status") == "ok"
    ]
    success_by_episode = _load_success_by_episode(progress_path)
    policies: Dict[str, List[Optional[Dict[str, Any]]]] = defaultdict(list)
    change_counts = Counter()
    split_stats: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        current = row.get("current_policy_candidate") or None
        if current and not current.get("valid"):
            current = None
        candidates = row.get("candidates") or []
        candidate_score = _select_best(candidates, "score")
        target_frontier = _select_best(candidates, "target_frontier_score")
        oracle_best_angle = _select_best_angle(candidates)
        ranker = _shadow_selected(row, "ranker_selected")
        ranker_resilience = _shadow_selected(row, "ranker_resilience_selected")

        selections = {
            "current_policy": current,
            "candidate_score": candidate_score,
            "target_frontier": target_frontier,
            "oracle_best_angle": oracle_best_angle,
            "ranker": ranker,
            "ranker_resilience": ranker_resilience,
        }
        for name, selected in selections.items():
            policies[name].append(selected)

        current_angle = _angle(current)
        oracle_angle = _angle(oracle_best_angle)
        target_angle = _angle(target_frontier)
        if current_angle is not None and oracle_angle is not None:
            change_counts["comparable_current_oracle"] += 1
            if oracle_angle + float(advantage_margin_deg) < current_angle and not _is_risky(oracle_best_angle):
                change_counts["oracle_safe_intervention_headroom"] += 1
            if current_angle + float(advantage_margin_deg) < oracle_angle:
                change_counts["current_policy_should_keep"] += 1
        if current and target_frontier:
            change_counts["comparable_current_target_frontier"] += 1
            if str(current.get("candidate_id")) != str(target_frontier.get("candidate_id")):
                change_counts["target_frontier_would_change_current"] += 1
                if _is_correct(target_frontier) and not _is_correct(current):
                    change_counts["target_frontier_would_win"] += 1
                if _is_correct(current) and not _is_correct(target_frontier):
                    change_counts["target_frontier_would_lose"] += 1
                if _is_risky(target_frontier):
                    change_counts["target_frontier_would_change_to_risky"] += 1
        if current_angle is not None and target_angle is not None:
            if target_angle + float(advantage_margin_deg) < current_angle and not _is_risky(target_frontier):
                change_counts["target_frontier_safe_intervention_headroom"] += 1

        success = success_by_episode.get(_episode_key(row))
        split = "success_unknown" if success is None else f"success_{bool(success)}"
        split_stats[split]["rows"] += 1
        if current:
            split_stats[split]["current_valid"] += 1
            split_stats[split]["current_correct"] += int(_is_correct(current))
        if oracle_best_angle:
            split_stats[split]["oracle_correct"] += int(_is_correct(oracle_best_angle))
        if current_angle is not None and oracle_angle is not None:
            split_stats[split]["current_oracle_comparable"] += 1
            if oracle_angle + float(advantage_margin_deg) < current_angle and not _is_risky(oracle_best_angle):
                split_stats[split]["oracle_safe_headroom"] += 1
            if current_angle + float(advantage_margin_deg) < oracle_angle:
                split_stats[split]["current_should_keep"] += 1

    summary: Dict[str, Any] = {
        "labels_path": str(labels_path),
        "progress_path": None if progress_path is None else str(progress_path),
        "advantage_margin_deg": float(advantage_margin_deg),
        "row_count": len(rows),
        "policy_metrics": {
            name: _summarize_policy(values)
            for name, values in sorted(policies.items())
        },
        "intervention_headroom": {
            key: int(value)
            for key, value in sorted(change_counts.items())
        },
        "intervention_headroom_rates": {
            "oracle_safe_intervention_headroom_rate": float(
                change_counts["oracle_safe_intervention_headroom"]
                / max(1, change_counts["comparable_current_oracle"])
            ),
            "current_policy_should_keep_rate": float(
                change_counts["current_policy_should_keep"]
                / max(1, change_counts["comparable_current_oracle"])
            ),
            "target_frontier_safe_intervention_headroom_rate": float(
                change_counts["target_frontier_safe_intervention_headroom"]
                / max(1, change_counts["comparable_current_target_frontier"])
            ),
            "target_frontier_would_win_rate": float(
                change_counts["target_frontier_would_win"]
                / max(1, change_counts["target_frontier_would_change_current"])
            ),
            "target_frontier_would_lose_rate": float(
                change_counts["target_frontier_would_lose"]
                / max(1, change_counts["target_frontier_would_change_current"])
            ),
            "target_frontier_would_change_to_risky_rate": float(
                change_counts["target_frontier_would_change_to_risky"]
                / max(1, change_counts["target_frontier_would_change_current"])
            ),
        },
        "success_split": {
            split: {
                **{key: int(value) for key, value in counter.items()},
                "current_correct_rate": float(
                    counter["current_correct"] / max(1, counter["current_valid"])
                ),
                "oracle_safe_headroom_rate": float(
                    counter["oracle_safe_headroom"]
                    / max(1, counter["current_oracle_comparable"])
                ),
                "current_should_keep_rate": float(
                    counter["current_should_keep"]
                    / max(1, counter["current_oracle_comparable"])
                ),
            }
            for split, counter in sorted(split_stats.items())
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage18a S2-aware candidate headroom.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--advantage-margin-deg", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = analyze(
        labels_path=args.labels,
        progress_path=args.progress,
        advantage_margin_deg=args.advantage_margin_deg,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

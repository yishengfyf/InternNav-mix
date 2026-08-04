"""Build Stage18c S2-aware candidate-advantage supervision.

Stage18b tried to learn ``keep/intervene/abstain`` directly.  On a strong S2
train subset that made positives too sparse.  This builder separates the first
question instead:

    "Does this online-safe OccMem candidate have a directional advantage over
    the frozen S2/current waypoint?"

The target is still privileged offline supervision.  GT angle, final success,
and reference paths are never encoded into features; they are used only to
derive labels and diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_DIR = SCRIPT_DIR.parents[0] / "train"
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from progress_ranker_common import encode_candidate, feature_names  # noqa: E402


EVENT_CONTEXT_NAMES = (
    "current_policy_present",
    "current_policy_valid",
    "current_geometry_safe",
    "current_active_gate_safe",
    "current_revisited",
    "current_semantic_dead_zone",
    "current_semantic_dead_zone_score",
    "current_stagnation_active",
    "candidate_count_norm",
    "safe_candidate_count_norm",
)
PAIR_EXTRA_NAMES = (
    "candidate_minus_current_angle_norm",
    "candidate_minus_current_distance_norm",
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _event_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}|{row.get('step_id')}"


def _load_success_by_episode(progress_path: Optional[Path]) -> Dict[str, bool]:
    if progress_path is None:
        return {}
    result: Dict[str, bool] = {}
    for row in _read_jsonl(progress_path):
        success = row.get("success")
        if success is None:
            metrics = row.get("metrics") or {}
            success = metrics.get("success")
        if success is None:
            continue
        result[_episode_key(row)] = bool(success)
    return result


def _split_for_row(row: Mapping[str, Any], val_ratio: float, seed: int, split_key: str) -> str:
    if split_key == "scene":
        key_text = f"scene|{row.get('scene_id')}|{seed}"
    elif split_key == "episode":
        key_text = f"episode|{row.get('scene_id')}|{row.get('episode_id')}|{seed}"
    else:
        raise ValueError("--split-key must be scene or episode")
    digest = hashlib.sha256(key_text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if bucket < val_ratio else "train"


def _candidate_is_safe(candidate: Mapping[str, Any]) -> bool:
    if not bool(candidate.get("geometry_safe", True)):
        return False
    if not bool(candidate.get("active_gate_safe", True)):
        return False
    if bool(candidate.get("points_to_revisited_region")):
        return False
    if str(candidate.get("landmark_status") or "").lower() == "completed":
        return False
    if _safe_float(candidate.get("completed_landmark_penalty")) > 0.0:
        return False
    if _safe_float(candidate.get("repeated_semantic_penalty")) > 0.0:
        return False
    return True


def _candidate_angle(candidate: Mapping[str, Any]) -> Optional[float]:
    value = candidate.get("gt_angle_diff_deg")
    if value is None:
        return None
    value = _safe_float(value, float("nan"))
    return None if math.isnan(value) else value


def _event_context(
    current: Mapping[str, Any],
    *,
    current_present: bool,
    candidate_count: int,
    safe_candidate_count: int,
) -> List[float]:
    return [
        float(current_present),
        float(bool(current.get("valid"))),
        float(bool(current.get("geometry_safe", True))),
        float(bool(current.get("active_gate_safe", True))),
        float(bool(current.get("points_to_revisited_region"))),
        float(bool(current.get("semantic_dead_zone"))),
        min(1.0, max(0.0, _safe_float(current.get("semantic_dead_zone_score")))),
        float(bool(current.get("semantic_stagnation_active"))),
        min(1.0, float(candidate_count) / 4.0),
        min(1.0, float(safe_candidate_count) / 4.0),
    ]


def _pair_features(
    candidate: Mapping[str, Any],
    current: Mapping[str, Any],
    context: Sequence[float],
) -> List[float]:
    candidate_features = encode_candidate(dict(candidate))
    current_features = encode_candidate(dict(current))
    difference = [
        float(candidate_value - current_value)
        for candidate_value, current_value in zip(candidate_features, current_features)
    ]
    candidate_angle = _safe_float(candidate.get("direction_angle_deg"))
    current_angle = _safe_float(current.get("direction_angle_deg"))
    angle_delta = abs((candidate_angle - current_angle + 180.0) % 360.0 - 180.0) / 180.0
    distance_delta = (
        _safe_float(candidate.get("distance_m")) - _safe_float(current.get("distance_m"))
    ) / 4.0
    return [
        *candidate_features,
        *current_features,
        *difference,
        float(angle_delta),
        float(distance_delta),
        *[float(value) for value in context],
    ]


def _feature_names() -> List[str]:
    base_names = feature_names()
    return (
        [f"candidate::{name}" for name in base_names]
        + [f"current::{name}" for name in base_names]
        + [f"delta::{name}" for name in base_names]
        + list(PAIR_EXTRA_NAMES)
        + [f"context::{name}" for name in EVENT_CONTEXT_NAMES]
    )


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "negative": sum(int(row.get("label", 0)) == 0 for row in rows),
        "positive": sum(int(row.get("label", 0)) == 1 for row in rows),
    }


def _event_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_event: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[str(row.get("event_key"))].append(row)
    positive_events = [
        items for items in by_event.values() if any(int(item.get("label", 0)) == 1 for item in items)
    ]
    multi_candidate_positive_events = [
        items for items in positive_events if len(items) > 1
    ]
    return {
        "candidate_rows": len(rows),
        "events": len(by_event),
        "positive_events": len(positive_events),
        "multi_candidate_positive_events": len(multi_candidate_positive_events),
        "positive_candidate_rows": sum(int(row.get("label", 0)) == 1 for row in rows),
        "mean_candidates_per_event": (
            sum(len(items) for items in by_event.values()) / max(1, len(by_event))
        ),
    }


def build_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    progress_path: Optional[Path],
    val_ratio: float,
    split_seed: int,
    split_key: str,
    positive_advantage_margin_deg: float,
    max_positive_candidate_angle_deg: float,
    label_clip_deg: float,
    min_train_positives: int,
    min_val_positives: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    if split_key not in {"scene", "episode"}:
        raise ValueError("--split-key must be scene or episode")
    if positive_advantage_margin_deg < 0.0:
        raise ValueError("--positive-advantage-margin-deg must be non-negative")
    if not 0.0 < max_positive_candidate_angle_deg <= 180.0:
        raise ValueError("--max-positive-candidate-angle-deg must be in (0, 180]")
    if label_clip_deg <= 0.0:
        raise ValueError("--label-clip-deg must be positive")

    success_by_episode = _load_success_by_episode(progress_path)
    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    counts = Counter()
    success_split_counts: Dict[str, Counter] = defaultdict(Counter)
    pair_feature_names = _feature_names()

    for row in rows:
        counts["input_rows"] += 1
        if row.get("label_status") != "ok":
            counts[f"drop_status={row.get('label_status')}"] += 1
            continue
        raw_current = row.get("current_policy_candidate")
        current_present = isinstance(raw_current, dict) and bool(raw_current)
        current = dict(raw_current or {})
        if not current_present or not bool(current.get("valid")):
            counts["drop_invalid_current_policy"] += 1
            continue
        current_angle = _candidate_angle(current)
        if current_angle is None:
            counts["drop_missing_current_angle"] += 1
            continue

        raw_candidates = [
            dict(candidate)
            for candidate in row.get("candidates") or []
            if isinstance(candidate, dict)
        ]
        safe_candidates = [
            candidate
            for candidate in raw_candidates
            if _candidate_is_safe(candidate) and _candidate_angle(candidate) is not None
        ]
        if not safe_candidates:
            counts["drop_no_safe_candidates"] += 1
            continue

        context = _event_context(
            current,
            current_present=current_present,
            candidate_count=len(raw_candidates),
            safe_candidate_count=len(safe_candidates),
        )
        final_success = success_by_episode.get(_episode_key(row))
        success_key = f"success_{final_success}"
        event_rows = []
        for candidate in safe_candidates:
            candidate_angle = _candidate_angle(candidate)
            if candidate_angle is None:
                continue
            raw_advantage = float(current_angle - candidate_angle)
            positive = bool(
                raw_advantage >= float(positive_advantage_margin_deg)
                and candidate_angle <= float(max_positive_candidate_angle_deg)
            )
            advantage_label = min(
                float(label_clip_deg),
                max(0.0, raw_advantage - float(positive_advantage_margin_deg)),
            )
            item = {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "event_key": _event_key(row),
                "candidate_id": candidate.get("candidate_id"),
                "final_success": final_success,
                "label": int(positive),
                "advantage_label_deg": float(advantage_label),
                "raw_advantage_deg": float(raw_advantage),
                "current_policy_gt_angle_diff_deg": float(current_angle),
                "candidate_gt_angle_diff_deg": float(candidate_angle),
                "current_policy_gt_correct": bool(current.get("gt_correct")),
                "candidate_gt_correct": bool(candidate.get("gt_correct")),
                "candidate_count": len(raw_candidates),
                "safe_candidate_count": len(safe_candidates),
                "features": _pair_features(candidate, current, context),
            }
            event_rows.append(item)
            counts["candidate_rows"] += 1
            counts["positive_candidate_rows"] += int(positive)
            success_split_counts[success_key]["candidate_rows"] += 1
            success_split_counts[success_key]["positive_candidate_rows"] += int(positive)

        if not event_rows:
            counts["drop_no_encoded_candidates"] += 1
            continue
        split = _split_for_row(row, val_ratio, split_seed, split_key)
        outputs[split].extend(event_rows)
        counts[f"kept_events_{split}"] += 1

    train_counts = _class_counts(outputs["train"])
    val_counts = _class_counts(outputs["val"])
    if train_counts["positive"] < int(min_train_positives):
        raise ValueError(
            "Too few train positives: "
            f"{train_counts['positive']} < {int(min_train_positives)}. "
            "Use harder episodes, lower the advantage margin, or collect more events."
        )
    if val_counts["positive"] < int(min_val_positives):
        raise ValueError(
            "Too few val positives: "
            f"{val_counts['positive']} < {int(min_val_positives)}. "
            "Use a different split seed, harder episodes, or collect more events."
        )

    split_membership: Dict[str, set] = defaultdict(set)
    for split, items in outputs.items():
        for item in items:
            key = str(item.get("scene_id")) if split_key == "scene" else str(item.get("event_key"))
            split_membership[key].add(split)

    summary = {
        "label_source": "stage18c_s2_candidate_advantage_v1",
        "task": "candidate_advantage_not_active_gate",
        "feature_names": pair_feature_names,
        "feature_dim": len(pair_feature_names),
        "base_feature_names": feature_names(),
        "event_context_feature_names": list(EVENT_CONTEXT_NAMES),
        "positive_definition": (
            "current_policy_gt_angle - candidate_gt_angle >= margin and "
            "candidate_gt_angle <= max_positive_candidate_angle"
        ),
        "progress_path": str(progress_path) if progress_path is not None else None,
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "split_key": split_key,
        "positive_advantage_margin_deg": float(positive_advantage_margin_deg),
        "max_positive_candidate_angle_deg": float(max_positive_candidate_angle_deg),
        "label_clip_deg": float(label_clip_deg),
        "counts": dict(counts),
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "all": _class_counts(outputs["train"] + outputs["val"]),
        },
        "event_diagnostics": {
            "train": _event_diagnostics(outputs["train"]),
            "val": _event_diagnostics(outputs["val"]),
            "all": _event_diagnostics(outputs["train"] + outputs["val"]),
        },
        "success_split_counts": {
            key: dict(counter) for key, counter in sorted(success_split_counts.items())
        },
        "split_overlap": {
            "keys_total": len(split_membership),
            "keys_in_both_train_val": sum(len(value) > 1 for value in split_membership.values()),
        },
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=18)
    parser.add_argument("--split-key", choices=("scene", "episode"), default="scene")
    parser.add_argument("--positive-advantage-margin-deg", type=float, default=10.0)
    parser.add_argument("--max-positive-candidate-angle-deg", type=float, default=120.0)
    parser.add_argument("--label-clip-deg", type=float, default=90.0)
    parser.add_argument("--min-train-positives", type=int, default=20)
    parser.add_argument("--min-val-positives", type=int, default=5)
    args = parser.parse_args()

    outputs, summary = build_dataset(
        _read_jsonl(args.labels),
        progress_path=args.progress,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_key=args.split_key,
        positive_advantage_margin_deg=args.positive_advantage_margin_deg,
        max_positive_candidate_angle_deg=args.max_positive_candidate_angle_deg,
        label_clip_deg=args.label_clip_deg,
        min_train_positives=args.min_train_positives,
        min_val_positives=args.min_val_positives,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in outputs.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

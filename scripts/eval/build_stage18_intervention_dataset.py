"""Build scene-split Stage18b keep/intervene/abstain supervision.

This builder turns Stage18a shadow labels into a conservative event-level
dataset for a frozen-S2 resilience adapter.  The adapter never sees GT route
labels, final success, or reference paths at inference time; those signals are
used only here to derive privileged offline supervision.

Each row contains:

* current S2 waypoint features;
* a variable-size set of online-safe OccMem candidate pair features;
* a decision label: keep, intervene, or abstain;
* an oracle candidate index only for high-confidence intervene rows.
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


DECISION_NAMES = ("keep", "intervene", "abstain")
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


def _split_for_row(row: Mapping[str, Any], val_ratio: float, seed: int) -> str:
    key_text = f"scene|{row.get('scene_id')}|{seed}"
    digest = hashlib.sha256(key_text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if bucket < val_ratio else "train"


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _load_success_by_episode(progress_path: Path) -> Dict[str, bool]:
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


def _best_safe_candidate(candidates: Sequence[Mapping[str, Any]]) -> Optional[int]:
    valid = [
        index
        for index, candidate in enumerate(candidates)
        if _candidate_angle(candidate) is not None
    ]
    if not valid:
        return None
    return min(valid, key=lambda index: float(_candidate_angle(candidates[index]) or 1e9))


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


def _decision_label(
    *,
    current: Mapping[str, Any],
    current_present: bool,
    final_success: Optional[bool],
    safe_candidates: Sequence[Mapping[str, Any]],
    safe_oracle_index: Optional[int],
    advantage_margin_deg: float,
    max_intervention_candidate_angle_deg: float,
) -> Tuple[str, int]:
    """Return a conservative decision label and candidate target index.

    Intervene positives are intentionally strict: final episode failure,
    current S2 not GT-safe-correct, and a safe candidate that is GT-correct
    with a sizeable directional advantage.  Successful episode rows with a
    correct S2 waypoint are explicit keep hard negatives.
    """
    if not current_present or not bool(current.get("valid")):
        return "abstain", -1
    current_angle = _candidate_angle(current)
    if current_angle is None:
        return "abstain", -1
    current_correct = bool(current.get("gt_correct"))
    if safe_oracle_index is None:
        return ("keep", -1) if current_correct else ("abstain", -1)

    candidate = safe_candidates[safe_oracle_index]
    candidate_angle = _candidate_angle(candidate)
    candidate_recovery_aligned = bool(
        candidate_angle <= float(max_intervention_candidate_angle_deg)
    )
    if candidate_angle is None:
        return "abstain", -1

    has_large_candidate_advantage = bool(
        candidate_angle + float(advantage_margin_deg) < current_angle
    )
    has_large_current_advantage = bool(
        current_angle + float(advantage_margin_deg) < candidate_angle
    )
    if (
        final_success is False
        and not current_correct
        and candidate_recovery_aligned
        and has_large_candidate_advantage
    ):
        return "intervene", int(safe_oracle_index)
    if final_success is True and current_correct:
        return "keep", -1
    if current_correct and (
        not candidate_recovery_aligned
        or has_large_current_advantage
        or not has_large_candidate_advantage
    ):
        return "keep", -1
    return "abstain", -1


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        name: sum(str(row.get("decision_name")) == name for row in rows)
        for name in DECISION_NAMES
    }


def build_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    progress_path: Path,
    val_ratio: float,
    split_seed: int,
    advantage_margin_deg: float,
    max_intervention_candidate_angle_deg: float,
    min_train_intervene: int,
    min_val_intervene: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    if advantage_margin_deg <= 0.0:
        raise ValueError("--advantage-margin-deg must be positive")
    if not 0.0 < max_intervention_candidate_angle_deg <= 180.0:
        raise ValueError("--max-intervention-candidate-angle-deg must be in (0, 180]")

    final_success = _load_success_by_episode(progress_path)
    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    counts = Counter()
    base_feature_names = feature_names()
    pair_feature_names = (
        [f"candidate::{name}" for name in base_feature_names]
        + [f"current::{name}" for name in base_feature_names]
        + [f"delta::{name}" for name in base_feature_names]
        + list(PAIR_EXTRA_NAMES)
        + [f"context::{name}" for name in EVENT_CONTEXT_NAMES]
    )

    for row in rows:
        counts["input_rows"] += 1
        if row.get("label_status") != "ok":
            counts[f"drop_status={row.get('label_status')}"] += 1
            continue
        raw_current = row.get("current_policy_candidate")
        current_present = isinstance(raw_current, dict) and bool(raw_current)
        current = dict(raw_current or {})
        candidates = [
            dict(candidate)
            for candidate in row.get("candidates") or []
            if isinstance(candidate, dict)
        ]
        safe_candidates = [
            candidate
            for candidate in candidates
            if _candidate_is_safe(candidate) and _candidate_angle(candidate) is not None
        ]
        safe_oracle_index = _best_safe_candidate(safe_candidates)
        context = _event_context(
            current,
            current_present=current_present,
            candidate_count=len(candidates),
            safe_candidate_count=len(safe_candidates),
        )
        decision_name, target_index = _decision_label(
            current=current,
            current_present=current_present,
            final_success=final_success.get(_episode_key(row)),
            safe_candidates=safe_candidates,
            safe_oracle_index=safe_oracle_index,
            advantage_margin_deg=advantage_margin_deg,
            max_intervention_candidate_angle_deg=max_intervention_candidate_angle_deg,
        )
        split = _split_for_row(row, val_ratio, split_seed)
        item = {
            "scene_id": row.get("scene_id"),
            "episode_id": row.get("episode_id"),
            "step_id": row.get("step_id"),
            "final_success": final_success.get(_episode_key(row)),
            "decision_name": decision_name,
            "decision_label": int(DECISION_NAMES.index(decision_name)),
            "target_candidate_index": int(target_index),
            "target_candidate_id": (
                safe_candidates[target_index].get("candidate_id")
                if target_index >= 0
                else None
            ),
            "current_policy_present": bool(current_present),
            "current_policy_valid": bool(current.get("valid")),
            "current_policy_gt_correct": bool(current.get("gt_correct")),
            "current_policy_gt_angle_diff_deg": _candidate_angle(current),
            "event_context_features": context,
            "candidate_ids": [candidate.get("candidate_id") for candidate in safe_candidates],
            "candidate_pair_features": [
                _pair_features(candidate, current, context)
                for candidate in safe_candidates
            ],
            "candidate_gt_angle_diff_deg": [
                _candidate_angle(candidate) for candidate in safe_candidates
            ],
            "candidate_gt_correct": [
                bool(candidate.get("gt_correct")) for candidate in safe_candidates
            ],
        }
        outputs[split].append(item)
        counts[f"kept_{split}"] += 1
        counts[f"decision_{decision_name}"] += 1
        if safe_candidates:
            counts["safe_candidate_available"] += 1
        if target_index >= 0:
            counts["intervention_target_rows"] += 1

    train_counts = _class_counts(outputs["train"])
    val_counts = _class_counts(outputs["val"])
    if train_counts["intervene"] < int(min_train_intervene):
        raise ValueError(
            "Too few train intervene labels: "
            f"{train_counts['intervene']} < {int(min_train_intervene)}. "
            "Choose a different --split-seed or collect more train episodes."
        )
    if val_counts["intervene"] < int(min_val_intervene):
        raise ValueError(
            "Too few val intervene labels: "
            f"{val_counts['intervene']} < {int(min_val_intervene)}. "
            "Choose a different --split-seed or collect more train episodes."
        )

    scene_splits: Dict[str, set] = defaultdict(set)
    for split, items in outputs.items():
        for item in items:
            scene_splits[str(item.get("scene_id"))].add(split)
    summary = {
        "label_source": "stage18b_s2_aware_safe_oracle_v1",
        "decision_names": list(DECISION_NAMES),
        "base_feature_names": base_feature_names,
        "pair_feature_names": pair_feature_names,
        "base_feature_dim": len(base_feature_names),
        "pair_feature_dim": len(pair_feature_names),
        "event_context_feature_names": list(EVENT_CONTEXT_NAMES),
        "event_context_dim": len(EVENT_CONTEXT_NAMES),
        "progress_path": str(progress_path),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "split_key": "scene",
        "advantage_margin_deg": float(advantage_margin_deg),
        "max_intervention_candidate_angle_deg": float(max_intervention_candidate_angle_deg),
        "label_definition": {
            "intervene": (
                "final_success=False, current S2 GT-safe-correct=False, and the "
                "best online-safe candidate is within the recovery-angle limit "
                "with the configured angle advantage"
            ),
            "keep": (
                "S2 is GT-safe-correct and either final episode success preserves it "
                "or no safe candidate has a large GT advantage"
            ),
            "abstain": (
                "current S2/candidate evidence is invalid, unavailable, or directionally ambiguous"
            ),
            "online_safe_filter": (
                "geometry_safe, active_gate_safe, non-revisited, non-completed, non-repeated"
            ),
        },
        "counts": dict(counts),
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "all": _class_counts(outputs["train"] + outputs["val"]),
        },
        "split_overlap": {
            "scenes_total": len(scene_splits),
            "scenes_in_both_train_val": sum(len(value) > 1 for value in scene_splits.values()),
        },
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Stage18b S2-aware intervention dataset.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=18)
    parser.add_argument("--advantage-margin-deg", type=float, default=10.0)
    parser.add_argument("--max-intervention-candidate-angle-deg", type=float, default=90.0)
    parser.add_argument("--min-train-intervene", type=int, default=20)
    parser.add_argument("--min-val-intervene", type=int, default=5)
    args = parser.parse_args()

    outputs, summary = build_dataset(
        _read_jsonl(args.labels),
        progress_path=args.progress,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        advantage_margin_deg=args.advantage_margin_deg,
        max_intervention_candidate_angle_deg=args.max_intervention_candidate_angle_deg,
        min_train_intervene=args.min_train_intervene,
        min_val_intervene=args.min_val_intervene,
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

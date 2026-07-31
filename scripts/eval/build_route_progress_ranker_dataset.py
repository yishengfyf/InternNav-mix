"""Build Stage17 listwise datasets from route-progress candidate values.

This is the first non-angle Stage17 label builder. It projects the current
agent position and each OccMem candidate onto the corrected GT reference path,
then uses arc-length progress along that path as a soft listwise label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_DIR = SCRIPT_DIR.parents[0] / "train"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from collect_gt_candidate_labels import (  # noqa: E402
    _episode_key,
    _load_reference_paths,
    _point_to_xy,
    _prepare_reference_path,
    _safe_float,
)
from progress_ranker_common import encode_candidate, feature_names  # noqa: E402


Point2D = Tuple[float, float]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
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


def _split_for_episode(row: Dict[str, Any], val_ratio: float, seed: int) -> str:
    key = f"{row.get('scene_id')}|{row.get('episode_id')}|{seed}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return "val" if bucket < val_ratio else "train"


def _candidate_status(candidate: Dict[str, Any]) -> str:
    return str(
        candidate.get("landmark_status")
        or (candidate.get("semantic_evidence") or {}).get("landmark_status")
        or ""
    ).lower()


def _candidate_xy(candidate: Dict[str, Any]) -> Optional[Point2D]:
    xy = candidate.get("xy")
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    x = _safe_float(xy[0], float("nan"))
    y = _safe_float(xy[1], float("nan"))
    if math.isnan(x) or math.isnan(y):
        return None
    return (x, y)


def _path_xy(reference_path: Sequence[Any], coordinate_mode: str) -> List[Point2D]:
    points = []
    for item in reference_path:
        xy = _point_to_xy(item, coordinate_mode)
        if xy is not None:
            points.append((float(xy[0]), float(xy[1])))
    return points


def _project_to_polyline(point: Point2D, path: Sequence[Point2D]) -> Optional[Tuple[float, float, int]]:
    if len(path) < 2:
        return None
    best_arc = 0.0
    best_dist = float("inf")
    best_segment = 0
    prefix = 0.0
    px, py = point
    for idx in range(len(path) - 1):
        ax, ay = path[idx]
        bx, by = path[idx + 1]
        vx = bx - ax
        vy = by - ay
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1e-10:
            continue
        seg_len = math.sqrt(seg_len_sq)
        t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        qx = ax + t * vx
        qy = ay + t * vy
        dist = math.hypot(px - qx, py - qy)
        if dist < best_dist:
            best_dist = dist
            best_arc = prefix + t * seg_len
            best_segment = idx
        prefix += seg_len
    if not math.isfinite(best_dist):
        return None
    return best_arc, best_dist, best_segment


def _reference_path_for_row(
    row: Dict[str, Any],
    reference_metadata: Dict[str, Dict[str, Any]],
    prepared_paths: Dict[str, Tuple[Optional[List[Any]], Optional[str]]],
    *,
    reference_frame: str,
    quaternion_order: str,
    coordinate_mode: str,
) -> Tuple[Optional[List[Point2D]], Optional[str]]:
    key = _episode_key(row)
    metadata_key = key
    metadata = reference_metadata.get(metadata_key)
    if metadata is None:
        metadata_key = f"|{row.get('episode_id')}"
        metadata = reference_metadata.get(metadata_key)
    if metadata is None:
        return None, "no_reference_path"
    if metadata_key not in prepared_paths:
        prepared_paths[metadata_key] = _prepare_reference_path(
            metadata,
            reference_frame,
            quaternion_order,
        )
    reference_path, error = prepared_paths[metadata_key]
    if reference_path is None:
        return None, error or "invalid_reference_path"
    points = _path_xy(reference_path, coordinate_mode)
    if len(points) < 2:
        return None, "reference_path_too_short"
    return points, None


def build_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    episodes_file: Path,
    reference_frame: str,
    quaternion_order: str,
    coordinate_mode: str,
    val_ratio: float,
    split_seed: int,
    min_positive_progress_m: float,
    offroute_penalty_weight: float,
    completed_penalty_m: float,
    repeated_penalty_m: float,
    next_landmark_bonus_m: float,
    target_frontier_bonus_m: float,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    reference_metadata = _load_reference_paths(episodes_file)
    prepared_paths: Dict[str, Tuple[Optional[List[Any]], Optional[str]]] = {}
    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    counts = Counter()
    feature_dim = len(feature_names())

    for row in rows:
        counts["input_rows"] += 1
        if row.get("label_status") != "ok":
            counts[f"drop_status={row.get('label_status')}"] += 1
            continue
        current_xy_raw = row.get("current_xy")
        if not isinstance(current_xy_raw, (list, tuple)) or len(current_xy_raw) < 2:
            counts["drop_missing_current_xy"] += 1
            continue
        current_xy = (
            _safe_float(current_xy_raw[0], float("nan")),
            _safe_float(current_xy_raw[1], float("nan")),
        )
        if any(math.isnan(value) for value in current_xy):
            counts["drop_invalid_current_xy"] += 1
            continue

        path, error = _reference_path_for_row(
            row,
            reference_metadata,
            prepared_paths,
            reference_frame=reference_frame,
            quaternion_order=quaternion_order,
            coordinate_mode=coordinate_mode,
        )
        if path is None:
            counts[f"drop_{error}"] += 1
            continue
        current_projection = _project_to_polyline(current_xy, path)
        if current_projection is None:
            counts["drop_current_projection_failed"] += 1
            continue
        current_arc, current_route_dist, current_segment = current_projection

        encoded = []
        labels = []
        raw_values = []
        candidate_ids = []
        route_progress = []
        route_distances = []
        for candidate in row.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            xy = _candidate_xy(candidate)
            if xy is None:
                continue
            projection = _project_to_polyline(xy, path)
            if projection is None:
                continue
            candidate_arc, candidate_route_dist, candidate_segment = projection
            progress_m = candidate_arc - current_arc
            offroute_delta = max(0.0, candidate_route_dist - current_route_dist)
            value = progress_m - offroute_penalty_weight * offroute_delta
            status = _candidate_status(candidate)
            if status == "completed":
                value -= completed_penalty_m
            if bool(candidate.get("repeated_semantic_penalty")):
                value -= repeated_penalty_m
            value += next_landmark_bonus_m * max(0.0, _safe_float(candidate.get("next_landmark_relevance")))
            if bool(candidate.get("target_frontier_candidate")):
                value += target_frontier_bonus_m

            label = max(0.0, value) if progress_m >= min_positive_progress_m else 0.0
            encoded.append(encode_candidate(candidate))
            labels.append(float(label))
            raw_values.append(float(value))
            candidate_ids.append(candidate.get("candidate_id"))
            route_progress.append(
                {
                    "progress_m": float(progress_m),
                    "value_m": float(value),
                    "route_distance_m": float(candidate_route_dist),
                    "segment_index": int(candidate_segment),
                }
            )

        if len(encoded) < 2:
            counts["drop_too_few_candidates"] += 1
            continue
        if not any(label > 0.0 for label in labels):
            counts["drop_no_positive_progress"] += 1
            continue
        if any(len(item) != feature_dim for item in encoded):
            raise RuntimeError("Feature schema produced inconsistent dimensions")

        split = _split_for_episode(row, val_ratio, split_seed)
        outputs[split].append(
            {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "label_source": "route_progress_value_v1",
                "candidate_ids": candidate_ids,
                "features": encoded,
                "labels": labels,
                "raw_values": raw_values,
                "current_route_arc_m": float(current_arc),
                "current_route_distance_m": float(current_route_dist),
                "current_segment_index": int(current_segment),
                "route_progress": route_progress,
            }
        )
        counts[f"kept_{split}"] += 1
        counts["positive_candidates"] += sum(1 for label in labels if label > 0.0)

    summary = {
        "label_source": "route_progress_value_v1",
        "feature_names": feature_names(),
        "feature_dim": feature_dim,
        "episodes_file": str(episodes_file),
        "reference_frame": reference_frame,
        "quaternion_order": quaternion_order,
        "coordinate_mode": coordinate_mode,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "min_positive_progress_m": min_positive_progress_m,
        "offroute_penalty_weight": offroute_penalty_weight,
        "completed_penalty_m": completed_penalty_m,
        "repeated_penalty_m": repeated_penalty_m,
        "next_landmark_bonus_m": next_landmark_bonus_m,
        "target_frontier_bonus_m": target_frontier_bonus_m,
        "counts": dict(counts),
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a route-progress Stage17 ranker dataset.")
    parser.add_argument("--labels", type=Path, required=True, help="Corrected Stage17 candidate labels JSONL.")
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-frame", choices=["world", "episodic_gps"], default="episodic_gps")
    parser.add_argument("--quaternion-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--coordinate-mode", choices=["xy", "x_neg_y", "xz", "x_neg_z"], default="x_neg_y")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--min-positive-progress-m", type=float, default=0.20)
    parser.add_argument("--offroute-penalty-weight", type=float, default=0.50)
    parser.add_argument("--completed-penalty-m", type=float, default=0.50)
    parser.add_argument("--repeated-penalty-m", type=float, default=0.25)
    parser.add_argument("--next-landmark-bonus-m", type=float, default=0.20)
    parser.add_argument("--target-frontier-bonus-m", type=float, default=0.10)
    args = parser.parse_args()

    outputs, summary = build_dataset(
        _read_jsonl(args.labels),
        episodes_file=args.episodes_file,
        reference_frame=args.reference_frame,
        quaternion_order=args.quaternion_order,
        coordinate_mode=args.coordinate_mode,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        min_positive_progress_m=args.min_positive_progress_m,
        offroute_penalty_weight=args.offroute_penalty_weight,
        completed_penalty_m=args.completed_penalty_m,
        repeated_penalty_m=args.repeated_penalty_m,
        next_landmark_bonus_m=args.next_landmark_bonus_m,
        target_frontier_bonus_m=args.target_frontier_bonus_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in outputs.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

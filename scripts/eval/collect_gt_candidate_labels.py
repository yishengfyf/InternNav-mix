"""
Build candidate-value training labels by aligning OccMem candidates to GT paths.

This is an offline data-construction helper for the Stage17 candidate value
model. It reads a Stage11a-style shadow run with OccMem candidate events and an
episode file that contains `reference_path`, then labels the candidate whose
relative direction best matches the GT lookahead direction.

The script is deliberately conservative:
  - If no reference_path is available for an episode, it writes
    label_status="no_reference_path" rather than inventing a label.
  - Coordinate conventions are configurable because Habitat/R2R exports differ
    across loaders.

Example:
  python scripts/eval/collect_gt_candidate_labels.py \
    --run-dir ./logs/habitat/compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed/vlmap_safety_debug/run_001 \
    --episodes-file /path/to/val_seen.jsonl.gz \
    --reference-frame episodic_gps \
    --gps-coordinate-mode x_neg_y
"""

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _read_text(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = _read_text(path).strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("episodes", "data", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
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
    return f"{_scene_token(record.get('scene_id'))}|{record.get('episode_id')}"


def _scene_token(scene_id: Any) -> str:
    text = str(scene_id or "")
    if not text:
        return ""
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if not parts:
        return text
    if len(parts) >= 2 and parts[-1].endswith((".glb", ".basis", ".navmesh")):
        return parts[-2]
    stem = Path(parts[-1]).stem
    return stem or parts[-1]


def _default_run_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "habitat"
        / "compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed"
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
        if (
            (candidate / "occ_memory" / "memory_events.jsonl").exists()
            or (candidate / "memory_events.jsonl").exists()
        ):
            return candidate
    raise FileNotFoundError(f"occ_memory/memory_events.jsonl not found under {path}")


def _memory_events_path(run_dir: Path) -> Path:
    nested = run_dir / "occ_memory" / "memory_events.jsonl"
    if nested.exists():
        return nested
    flat = run_dir / "memory_events.jsonl"
    if flat.exists():
        return flat
    return nested


def _group_by_episode(records: Iterable[Dict[str, Any]], step_field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_episode_key(record)].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: _safe_int(item.get(step_field), -1))
    return grouped


def _extract_reference_path(record: Dict[str, Any]) -> Optional[List[Any]]:
    candidates = [
        record.get("reference_path"),
        record.get("reference_paths"),
        record.get("path"),
        record.get("trajectory"),
    ]
    episode = record.get("episode")
    if isinstance(episode, dict):
        candidates.extend([
            episode.get("reference_path"),
            episode.get("reference_paths"),
            episode.get("path"),
        ])
    for item in candidates:
        if not item:
            continue
        if isinstance(item, list) and item and isinstance(item[0], list):
            # Some datasets store multiple reference paths: take the first path.
            if item and item[0] and isinstance(item[0][0], list):
                return item[0]
            return item
    return None


def _episode_reference_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    episode = record.get("episode")
    nested = episode if isinstance(episode, dict) else {}
    return {
        "reference_path": _extract_reference_path(record),
        "start_position": record.get("start_position", nested.get("start_position")),
        "start_rotation": record.get("start_rotation", nested.get("start_rotation")),
    }


def _load_reference_paths(episodes_file: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not episodes_file:
        return {}
    records = _read_json_records(episodes_file)
    by_key: Dict[str, Dict[str, Any]] = {}
    by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = _episode_reference_metadata(record)
        if not metadata["reference_path"]:
            continue
        key = _episode_key(record)
        by_key[key] = metadata
        by_ep[str(record.get("episode_id"))].append(metadata)
    # Fallback for files whose scene_id convention does not match logs.
    for ep_id, metadata_items in by_ep.items():
        if len(metadata_items) == 1:
            by_key.setdefault(f"|{ep_id}", metadata_items[0])
    return by_key


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate_vector_by_inverse_quaternion(
    vector: Tuple[float, float, float],
    rotation: Any,
    quaternion_order: str,
) -> Optional[Tuple[float, float, float]]:
    if not isinstance(rotation, (list, tuple)) or len(rotation) < 4:
        return None
    values = [_safe_float(value, float("nan")) for value in rotation[:4]]
    if any(math.isnan(value) for value in values):
        return None
    if quaternion_order == "xyzw":
        qx, qy, qz, qw = values
    else:
        qw, qx, qy, qz = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-8:
        return None
    # Habitat computes inverse(start_rotation) * (point - start_position).
    inverse_vector = (-qx / norm, -qy / norm, -qz / norm)
    inverse_scalar = qw / norm
    first_cross = _cross(inverse_vector, vector)
    second_cross = _cross(inverse_vector, first_cross)
    return tuple(
        vector[idx] + 2.0 * inverse_scalar * first_cross[idx] + 2.0 * second_cross[idx]
        for idx in range(3)
    )


def _world_path_to_episodic_gps(
    metadata: Dict[str, Any],
    quaternion_order: str,
) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    reference_path = metadata.get("reference_path")
    start_position = metadata.get("start_position")
    start_rotation = metadata.get("start_rotation")
    if not isinstance(start_position, (list, tuple)) or len(start_position) < 3:
        return None, "missing_start_position"
    if not isinstance(start_rotation, (list, tuple)) or len(start_rotation) < 4:
        return None, "missing_start_rotation"
    start = [_safe_float(value, float("nan")) for value in start_position[:3]]
    if any(math.isnan(value) for value in start):
        return None, "invalid_start_position"
    episodic_path = []
    for point in reference_path or []:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        world = [_safe_float(value, float("nan")) for value in point[:3]]
        if any(math.isnan(value) for value in world):
            continue
        delta = tuple(world[idx] - start[idx] for idx in range(3))
        local = _rotate_vector_by_inverse_quaternion(delta, start_rotation, quaternion_order)
        if local is None:
            return None, "invalid_start_rotation"
        # This is the exact 2-D convention used by Habitat's EpisodicGPSSensor.
        episodic_path.append([-float(local[2]), float(local[0])])
    if len(episodic_path) < 2:
        return None, "reference_path_too_short"
    return episodic_path, None


def _prepare_reference_path(
    metadata: Dict[str, Any],
    reference_frame: str,
    quaternion_order: str,
) -> Tuple[Optional[List[Any]], Optional[str]]:
    if reference_frame == "episodic_gps":
        return _world_path_to_episodic_gps(metadata, quaternion_order)
    path = metadata.get("reference_path")
    if not path:
        return None, "no_reference_path"
    return path, None


def _point_to_xy(point: Any, mode: str) -> Optional[Tuple[float, float]]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    x = _safe_float(point[0], float("nan"))
    if math.isnan(x):
        return None
    if len(point) >= 3:
        y = _safe_float(point[1], float("nan"))
        z = _safe_float(point[2], float("nan"))
        if mode == "xy":
            return (x, y)
        if mode == "x_neg_y":
            return (x, -y)
        if mode == "xz":
            return (x, z)
        if mode == "x_neg_z":
            return (x, -z)
    y2 = _safe_float(point[1], float("nan"))
    if math.isnan(y2):
        return None
    return (x, -y2) if mode == "x_neg_y" else (x, y2)


def _gps_to_xy(gps: Any, mode: str) -> Optional[Tuple[float, float]]:
    if not isinstance(gps, (list, tuple)) or len(gps) < 2:
        return None
    x = _safe_float(gps[0], float("nan"))
    y = _safe_float(gps[1], float("nan"))
    if math.isnan(x) or math.isnan(y):
        return None
    if mode == "x_neg_y":
        return (x, -y)
    return (x, y)


def _wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(_wrap_deg(float(a) - float(b)))


def _direction_bucket(angle_deg: float) -> str:
    angle = float(angle_deg)
    if -45.0 <= angle <= 45.0:
        return "front"
    if 45.0 < angle < 135.0:
        return "left"
    if -135.0 < angle < -45.0:
        return "right"
    return "back"


def _nearest_traj_event(events: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    if not events:
        return None
    return min(events, key=lambda item: abs(_safe_int(item.get("eval_step"), -1) - int(step)))


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, float(percentile)))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _gt_direction_from_path(
    *,
    reference_path: List[Any],
    current_xy: Tuple[float, float],
    yaw_rad: float,
    lookahead_m: float,
    coordinate_mode: str,
) -> Dict[str, Any]:
    path_xy = []
    for item in reference_path:
        xy = _point_to_xy(item, coordinate_mode)
        if xy is not None:
            path_xy.append(xy)
    if len(path_xy) < 2:
        return {"valid": False, "reason": "reference_path_too_short"}
    nearest_idx = min(
        range(len(path_xy)),
        key=lambda idx: math.hypot(path_xy[idx][0] - current_xy[0], path_xy[idx][1] - current_xy[1]),
    )
    nearest_dist = math.hypot(
        path_xy[nearest_idx][0] - current_xy[0],
        path_xy[nearest_idx][1] - current_xy[1],
    )
    target_idx = nearest_idx
    accum = 0.0
    for idx in range(nearest_idx + 1, len(path_xy)):
        prev = path_xy[idx - 1]
        cur = path_xy[idx]
        accum += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        target_idx = idx
        if accum >= float(lookahead_m):
            break
    target_xy = path_xy[target_idx]
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    if math.hypot(dx, dy) < 1e-4:
        return {
            "valid": False,
            "reason": "target_too_close",
            "nearest_reference_index": nearest_idx,
            "nearest_reference_distance_m": nearest_dist,
        }
    world_angle = math.atan2(dy, dx)
    rel_angle = _wrap_deg(math.degrees(world_angle - float(yaw_rad)))
    return {
        "valid": True,
        "reason": "ok",
        "nearest_reference_index": nearest_idx,
        "target_reference_index": target_idx,
        "nearest_reference_distance_m": nearest_dist,
        "lookahead_m": float(lookahead_m),
        "gt_target_xy": [float(target_xy[0]), float(target_xy[1])],
        "gt_direction_angle_deg": float(rel_angle),
        "gt_direction_bucket": _direction_bucket(rel_angle),
    }


def _label_candidate_direction(
    candidate: Dict[str, Any],
    gt_angle_deg: float,
    max_angle_deg: float,
) -> Dict[str, Any]:
    item = dict(candidate or {})
    cand_angle = item.get("direction_angle_deg")
    diff = None
    direction_correct = False
    safe_correct = False
    if cand_angle is not None:
        diff = _angle_diff_deg(_safe_float(cand_angle), _safe_float(gt_angle_deg))
        direction_correct = bool(diff <= float(max_angle_deg))
        safe_correct = bool(
            direction_correct
            and bool(item.get("geometry_safe", True))
            and bool(item.get("active_gate_safe", True))
        )
    item["gt_angle_diff_deg"] = diff
    item["gt_direction_correct"] = bool(direction_correct)
    item["gt_correct"] = bool(safe_correct)
    return item


def collect_labels(
    *,
    run_dir: Path,
    episodes_file: Optional[Path],
    reference_frame: str,
    reference_coordinate_mode: str,
    gps_coordinate_mode: str,
    quaternion_order: str,
    lookahead_m: float,
    max_angle_deg: float,
    step_min: int,
    step_max: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    run_dir = _resolve_run_dir(run_dir)
    memory_events = _read_json_records(_memory_events_path(run_dir))
    trajectory_events = _read_json_records(run_dir / "trajectory_events.jsonl")
    progress_records = _read_json_records(run_dir / "progress.json")
    reference_metadata = _load_reference_paths(episodes_file)

    # Also accept reference_path embedded in progress logs.
    for record in progress_records:
        path = _extract_reference_path(record)
        if path:
            reference_metadata.setdefault(_episode_key(record), _episode_reference_metadata(record))

    traj_by_key = _group_by_episode(trajectory_events, "eval_step")
    label_rows = []
    status_counts = Counter()
    correct_candidate_events = 0
    candidate_event_count = 0
    angle_diffs = []
    nearest_reference_distances = []
    waypoint_gt_angle_diffs = []
    current_policy_gt_angle_diffs = []
    gt_positive_candidate_count = 0
    completed_gt_positive_count = 0
    current_policy_candidate_valid_count = 0
    current_policy_candidate_observed_count = 0
    current_policy_candidate_missing_count = 0
    current_policy_candidate_reason_counts = Counter()
    current_policy_candidate_direction_correct_count = 0
    current_policy_candidate_correct_count = 0
    current_policy_better_than_best_candidate_count = 0
    best_candidate_better_than_current_policy_count = 0
    current_policy_tied_best_candidate_count = 0
    events_with_current_policy_and_candidates = 0
    nearest_index_transition_count = 0
    nearest_index_backward_count = 0
    last_nearest_index_by_episode: Dict[str, Tuple[int, int]] = {}
    prepared_paths: Dict[str, Tuple[Optional[List[Any]], Optional[str]]] = {}

    for event in memory_events:
        if event.get("event_type") != "occ_memory_query_candidates":
            continue
        step = _safe_int(event.get("step_id"), -1)
        if step < step_min or step > step_max:
            continue
        candidate_event_count += 1
        key = _episode_key(event)
        metadata_key = key
        metadata = reference_metadata.get(metadata_key)
        if metadata is None:
            metadata_key = f"|{event.get('episode_id')}"
            metadata = reference_metadata.get(metadata_key)
        traj_event = _nearest_traj_event(traj_by_key.get(key, []), step)
        base_row = {
            "event_schema_version": event.get("event_schema_version"),
            "split": event.get("split"),
            "rank": event.get("rank"),
            "local_rank": event.get("local_rank"),
            "world_size": event.get("world_size"),
            "eval_random_seed": event.get("eval_random_seed"),
            "episode_eval_seed": event.get("episode_eval_seed"),
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "episode_index": event.get("episode_index"),
            "step_id": step,
            "start_xy": event.get("start_xy"),
            "candidate_count": _safe_int(event.get("candidate_count")),
            "current_waypoint_direction_angle_deg": event.get("current_waypoint_direction_angle_deg"),
            "current_waypoint_direction_bucket": event.get("current_waypoint_direction_bucket"),
            "current_policy_candidate": event.get("current_policy_candidate"),
            "progress_ranker_shadow": event.get("progress_ranker_shadow"),
            "label_status": None,
        }
        if metadata is None:
            row = {**base_row, "label_status": "no_reference_path", "candidates": event.get("candidates") or []}
            label_rows.append(row)
            status_counts[row["label_status"]] += 1
            continue
        if metadata_key not in prepared_paths:
            prepared_paths[metadata_key] = _prepare_reference_path(
                metadata,
                reference_frame,
                quaternion_order,
            )
        reference_path, reference_error = prepared_paths[metadata_key]
        if reference_path is None:
            row = {
                **base_row,
                "label_status": reference_error or "invalid_reference_path",
                "candidates": event.get("candidates") or [],
            }
            label_rows.append(row)
            status_counts[row["label_status"]] += 1
            continue
        if traj_event is None:
            row = {**base_row, "label_status": "no_trajectory_event", "candidates": event.get("candidates") or []}
            label_rows.append(row)
            status_counts[row["label_status"]] += 1
            continue
        current_xy = _gps_to_xy(traj_event.get("gps"), gps_coordinate_mode)
        compass = traj_event.get("compass")
        if current_xy is None or not compass:
            row = {**base_row, "label_status": "missing_pose", "candidates": event.get("candidates") or []}
            label_rows.append(row)
            status_counts[row["label_status"]] += 1
            continue
        gt = _gt_direction_from_path(
            reference_path=reference_path,
            current_xy=current_xy,
            yaw_rad=_safe_float(compass[0]),
            lookahead_m=lookahead_m,
            coordinate_mode=(gps_coordinate_mode if reference_frame == "episodic_gps" else reference_coordinate_mode),
        )
        nearest_distance = gt.get("nearest_reference_distance_m")
        if nearest_distance is not None:
            nearest_reference_distances.append(_safe_float(nearest_distance))
        if not gt.get("valid"):
            row = {
                **base_row,
                "label_status": str(gt.get("reason") or "invalid_gt_direction"),
                "gt": gt,
                "candidates": event.get("candidates") or [],
            }
            label_rows.append(row)
            status_counts[row["label_status"]] += 1
            continue

        waypoint_angle = event.get("current_waypoint_direction_angle_deg")
        if waypoint_angle is not None:
            waypoint_gt_angle_diffs.append(
                _angle_diff_deg(_safe_float(waypoint_angle), _safe_float(gt["gt_direction_angle_deg"]))
            )
        nearest_index = _safe_int(gt.get("nearest_reference_index"), -1)
        previous_step_index = last_nearest_index_by_episode.get(key)
        if nearest_index >= 0 and previous_step_index is not None and step > previous_step_index[0]:
            nearest_index_transition_count += 1
            if nearest_index < previous_step_index[1]:
                nearest_index_backward_count += 1
        if nearest_index >= 0:
            last_nearest_index_by_episode[key] = (step, nearest_index)

        gt_angle = _safe_float(gt["gt_direction_angle_deg"])
        best_id = None
        best_diff = None
        labeled_candidates = []
        for candidate in event.get("candidates") or []:
            item = _label_candidate_direction(candidate, gt_angle, max_angle_deg)
            diff = item.get("gt_angle_diff_deg")
            if diff is not None and (best_diff is None or diff < best_diff):
                best_diff = diff
                best_id = item.get("candidate_id")
            labeled_candidates.append(item)
        raw_current_policy_candidate = event.get("current_policy_candidate")
        current_policy_candidate = dict(raw_current_policy_candidate or {})
        labeled_current_policy_candidate = current_policy_candidate
        current_policy_diff = None
        current_policy_candidate_observed = isinstance(raw_current_policy_candidate, dict) and bool(
            raw_current_policy_candidate
        )
        if current_policy_candidate_observed:
            current_policy_candidate_observed_count += 1
            labeled_current_policy_candidate = _label_candidate_direction(
                current_policy_candidate,
                gt_angle,
                max_angle_deg,
            )
            if labeled_current_policy_candidate.get("valid"):
                current_policy_candidate_valid_count += 1
                current_policy_diff = labeled_current_policy_candidate.get("gt_angle_diff_deg")
                if current_policy_diff is not None:
                    current_policy_gt_angle_diffs.append(float(current_policy_diff))
                    if labeled_current_policy_candidate.get("gt_direction_correct"):
                        current_policy_candidate_direction_correct_count += 1
                    if labeled_current_policy_candidate.get("gt_correct"):
                        current_policy_candidate_correct_count += 1
        else:
            current_policy_candidate_missing_count += 1
            current_policy_candidate_reason_counts[
                str(event.get("reason") or "missing")
            ] += 1
        if current_policy_diff is not None and best_diff is not None:
            events_with_current_policy_and_candidates += 1
            # A small tolerance keeps numeric jitter from being interpreted as
            # a meaningful Stage18 intervention headroom signal.
            tolerance = 1e-6
            if current_policy_diff + tolerance < best_diff:
                current_policy_better_than_best_candidate_count += 1
            elif best_diff + tolerance < current_policy_diff:
                best_candidate_better_than_current_policy_count += 1
            else:
                current_policy_tied_best_candidate_count += 1
        correct_ids = [item.get("candidate_id") for item in labeled_candidates if item.get("gt_correct")]
        positive_candidates = [item for item in labeled_candidates if item.get("gt_correct")]
        gt_positive_candidate_count += len(positive_candidates)
        completed_gt_positive_count += sum(
            item.get("landmark_status") == "completed" for item in positive_candidates
        )
        if correct_ids:
            correct_candidate_events += 1
        if best_diff is not None:
            angle_diffs.append(float(best_diff))
        row = {
            **base_row,
            "label_status": "ok",
            "trajectory_event_step": traj_event.get("eval_step"),
            "current_xy": [float(current_xy[0]), float(current_xy[1])],
            "current_compass": _safe_float(compass[0]),
            "gt": gt,
            "current_policy_candidate": labeled_current_policy_candidate,
            "current_policy_candidate_observed": bool(current_policy_candidate_observed),
            "current_policy_candidate_missing_reason": (
                None
                if current_policy_candidate_observed
                else str(event.get("reason") or "missing")
            ),
            "current_policy_gt_angle_diff_deg": current_policy_diff,
            "current_policy_gt_direction_correct": bool(
                labeled_current_policy_candidate.get("gt_direction_correct")
            ),
            "current_policy_gt_correct": bool(labeled_current_policy_candidate.get("gt_correct")),
            "correct_candidate_ids": correct_ids,
            "correct_candidate_id": correct_ids[0] if correct_ids else None,
            "best_angle_candidate_id": best_id,
            "best_angle_diff_deg": best_diff,
            "max_angle_deg": float(max_angle_deg),
            "candidates": labeled_candidates,
        }
        label_rows.append(row)
        status_counts["ok"] += 1

    summary = {
        "run_dir": str(run_dir),
        "episodes_file": None if episodes_file is None else str(episodes_file),
        "reference_frame": reference_frame,
        "reference_coordinate_mode": reference_coordinate_mode,
        "gps_coordinate_mode": gps_coordinate_mode,
        "quaternion_order": quaternion_order,
        "lookahead_m": float(lookahead_m),
        "max_angle_deg": float(max_angle_deg),
        "step_min": int(step_min),
        "step_max": int(step_max),
        "candidate_event_count": int(candidate_event_count),
        "label_row_count": int(len(label_rows)),
        "label_status_counts": dict(status_counts),
        "events_with_correct_candidate": int(correct_candidate_events),
        "correct_candidate_event_rate": (
            float(correct_candidate_events / max(1, status_counts.get("ok", 0)))
        ),
        "best_angle_diff_mean_deg": (
            float(sum(angle_diffs) / len(angle_diffs)) if angle_diffs else None
        ),
        "best_angle_diff_min_deg": min(angle_diffs) if angle_diffs else None,
        "best_angle_diff_max_deg": max(angle_diffs) if angle_diffs else None,
        "nearest_reference_distance_mean_m": (
            float(sum(nearest_reference_distances) / len(nearest_reference_distances))
            if nearest_reference_distances
            else None
        ),
        "nearest_reference_distance_median_m": _percentile(nearest_reference_distances, 0.5),
        "nearest_reference_distance_p90_m": _percentile(nearest_reference_distances, 0.9),
        "nearest_reference_distance_max_m": max(nearest_reference_distances) if nearest_reference_distances else None,
        "nearest_reference_distance_le_1m_count": sum(
            distance <= 1.0 for distance in nearest_reference_distances
        ),
        "current_waypoint_gt_angle_diff_mean_deg": (
            float(sum(waypoint_gt_angle_diffs) / len(waypoint_gt_angle_diffs))
            if waypoint_gt_angle_diffs
            else None
        ),
        "current_policy_candidate_valid_count": int(current_policy_candidate_valid_count),
        "current_policy_candidate_observed_count": int(
            current_policy_candidate_observed_count
        ),
        "current_policy_candidate_missing_count": int(
            current_policy_candidate_missing_count
        ),
        "current_policy_candidate_invalid_count": int(
            max(0, current_policy_candidate_observed_count - current_policy_candidate_valid_count)
        ),
        "current_policy_candidate_reason_counts": dict(
            current_policy_candidate_reason_counts
        ),
        "current_policy_candidate_valid_rate": float(
            current_policy_candidate_valid_count
            / max(1, current_policy_candidate_observed_count)
        ),
        "current_policy_gt_angle_diff_mean_deg": (
            float(sum(current_policy_gt_angle_diffs) / len(current_policy_gt_angle_diffs))
            if current_policy_gt_angle_diffs
            else None
        ),
        "current_policy_gt_angle_diff_median_deg": _percentile(
            current_policy_gt_angle_diffs,
            0.5,
        ),
        "current_policy_gt_direction_correct_count": int(
            current_policy_candidate_direction_correct_count
        ),
        "current_policy_gt_direction_correct_rate": float(
            current_policy_candidate_direction_correct_count
            / max(1, current_policy_candidate_valid_count)
        ),
        "current_policy_gt_correct_count": int(current_policy_candidate_correct_count),
        "current_policy_gt_correct_rate": float(
            current_policy_candidate_correct_count / max(1, current_policy_candidate_valid_count)
        ),
        "events_with_current_policy_and_candidates": int(
            events_with_current_policy_and_candidates
        ),
        "current_policy_better_than_best_candidate_count": int(
            current_policy_better_than_best_candidate_count
        ),
        "current_policy_better_than_best_candidate_rate": float(
            current_policy_better_than_best_candidate_count
            / max(1, events_with_current_policy_and_candidates)
        ),
        "best_candidate_better_than_current_policy_count": int(
            best_candidate_better_than_current_policy_count
        ),
        "best_candidate_better_than_current_policy_rate": float(
            best_candidate_better_than_current_policy_count
            / max(1, events_with_current_policy_and_candidates)
        ),
        "current_policy_tied_best_candidate_count": int(
            current_policy_tied_best_candidate_count
        ),
        "nearest_index_transition_count": int(nearest_index_transition_count),
        "nearest_index_backward_count": int(nearest_index_backward_count),
        "nearest_index_backward_rate": float(
            nearest_index_backward_count / max(1, nearest_index_transition_count)
        ),
        "gt_positive_candidate_count": int(gt_positive_candidate_count),
        "completed_gt_positive_count": int(completed_gt_positive_count),
        "completed_gt_positive_rate": float(
            completed_gt_positive_count / max(1, gt_positive_candidate_count)
        ),
    }
    return label_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GT-aligned candidate labels for Stage17.")
    parser.add_argument("--run-dir", type=Path, default=_default_run_dir())
    parser.add_argument("--episodes-file", type=Path)
    parser.add_argument(
        "--reference-frame",
        choices=["world", "episodic_gps"],
        default="world",
        help="Frame of reference_path before comparison with trajectory GPS.",
    )
    parser.add_argument(
        "--reference-coordinate-mode",
        choices=["xy", "x_neg_y", "xz", "x_neg_z"],
        default="x_neg_y",
    )
    parser.add_argument(
        "--gps-coordinate-mode",
        choices=["xy", "x_neg_y"],
        default="x_neg_y",
    )
    parser.add_argument(
        "--quaternion-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Component order of episode start_rotation; Habitat R2R JSON uses xyzw.",
    )
    parser.add_argument("--lookahead-m", type=float, default=1.5)
    parser.add_argument("--max-angle-deg", type=float, default=60.0)
    parser.add_argument("--step-min", type=int, default=0)
    parser.add_argument("--step-max", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    rows, summary = collect_labels(
        run_dir=run_dir,
        episodes_file=args.episodes_file,
        reference_frame=args.reference_frame,
        reference_coordinate_mode=args.reference_coordinate_mode,
        gps_coordinate_mode=args.gps_coordinate_mode,
        quaternion_order=args.quaternion_order,
        lookahead_m=args.lookahead_m,
        max_angle_deg=args.max_angle_deg,
        step_min=args.step_min,
        step_max=args.step_max,
    )
    output = args.output or (run_dir / "gt_candidate_labels.jsonl")
    summary_output = args.summary_output or (run_dir / "gt_candidate_labels_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved labels: {output}")
    print(f"Saved summary: {summary_output}")


if __name__ == "__main__":
    main()

"""Read-only HSGM-inspired route bridge and local BEV audit.

This module consumes frozen Stage27 candidate events and replay-ledger pose
snapshots.  It never creates actions, changes candidate safety, or treats
unknown as free.  The first version deliberately performs a horizontal
field-of-view test only when depth pixels are unavailable; the report names
that limitation instead of claiming pixel-level visibility.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SAFE_STAGE = "route_occ_clearance_frontier"
DEFAULT_FOV_DEG = 135.0
DEFAULT_MAX_VISIBLE_M = 5.0


def _event_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id")), int(row.get("episode_id", -1)), int(row.get("step_id", -1)))


def _as_xy(value: Sequence[float]) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError("xy value must contain at least two numbers")
    return float(value[0]), float(value[1])


def grid_to_xy(grid: Sequence[float], *, origin: Sequence[float], cell_size_m: float) -> tuple[float, float]:
    """Convert the map grid convention used by SparseOcc to metric map XY."""
    row, col = _as_xy(grid)
    origin_row, origin_col = _as_xy(origin)
    return ((origin_row - row) * float(cell_size_m), (origin_col - col) * float(cell_size_m))


def angular_delta_deg(a_deg: float, b_deg: float) -> float:
    """Small signed angle from b to a in degrees."""
    return (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0


def bearing_deg(origin_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    ox, oy = _as_xy(origin_xy)
    tx, ty = _as_xy(target_xy)
    return math.degrees(math.atan2(ty - oy, tx - ox))


def _pose_from_observation(row: Mapping[str, Any]) -> tuple[tuple[float, float], float] | None:
    pose = row.get("pose") or {}
    gps = pose.get("gps")
    compass = pose.get("compass")
    if not isinstance(gps, Sequence) or len(gps) < 2:
        return None
    if not isinstance(compass, Sequence) or not compass:
        return None
    # The map uses x=gps[0], y=-gps[1], while compass=0 points along +map X.
    return (float(gps[0]), -float(gps[1])), math.degrees(float(compass[0]))


def _path_contiguous(path_cells: Sequence[Sequence[float]]) -> bool:
    for first, second in zip(path_cells, path_cells[1:]):
        r0, c0 = _as_xy(first)
        r1, c1 = _as_xy(second)
        if max(abs(r1 - r0), abs(c1 - c0)) > 1.0:
            return False
    return True


def _candidate_safety(candidate: Mapping[str, Any]) -> tuple[bool, str]:
    if candidate.get("route_occ_conflict"):
        return False, "route_occ_conflict"
    if candidate.get("unknown_fraction", 0.0) not in (0, 0.0, None):
        return False, "unknown_path_evidence"
    if candidate.get("occupied_fraction", 0.0) not in (0, 0.0, None):
        return False, "occupied_path_evidence"
    if not bool(candidate.get("floor_aligned_known_free")):
        return False, "clearance_or_floor_not_safe"
    if candidate.get("gt_fields_used"):
        return False, "gt_fields_used"
    if candidate.get("action_applied"):
        return False, "action_applied"
    return True, "safe_candidate"


def _visibility(
    current_xy: Sequence[float],
    current_yaw_deg: float,
    target_xy: Sequence[float],
    *,
    fov_deg: float,
    max_distance_m: float,
    depth_available: bool = False,
) -> dict[str, Any]:
    cx, cy = _as_xy(current_xy)
    tx, ty = _as_xy(target_xy)
    distance = math.hypot(tx - cx, ty - cy)
    bearing = bearing_deg((cx, cy), (tx, ty))
    relative = angular_delta_deg(bearing, current_yaw_deg)
    in_horizontal_fov = abs(relative) <= float(fov_deg) / 2.0 and distance <= float(max_distance_m)
    return {
        "distance_m": distance,
        "bearing_deg": bearing,
        "relative_bearing_deg": relative,
        "in_horizontal_fov": bool(in_horizontal_fov),
        "depth_occlusion_checked": bool(depth_available),
        "visibility_mode": "rgb_depth_occlusion" if depth_available else "pose_bearing_only_no_depth_occlusion",
        "visible": bool(in_horizontal_fov) if not depth_available else None,
    }


def bridge_candidate(
    event: Mapping[str, Any],
    candidate: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    *,
    cell_size_m: float = 0.05,
    grid_origin: Sequence[float] = (500.0, 500.0),
    fov_deg: float = DEFAULT_FOV_DEG,
    max_visible_distance_m: float = DEFAULT_MAX_VISIBLE_M,
) -> dict[str, Any]:
    """Build a causal, non-executing bridge record for one candidate."""
    safe, safety_reason = _candidate_safety(candidate)
    path = list(candidate.get("path_cells") or [])
    path_current_to_candidate = list(reversed(path))
    record: dict[str, Any] = {
        "event_key": {
            "scene_id": str(event.get("scene_id")),
            "episode_id": int(event.get("episode_id", -1)),
            "step_id": int(event.get("step_id", -1)),
        },
        "candidate_id": candidate.get("candidate_id"),
        "source_type": candidate.get("source_type"),
        "source_step": candidate.get("source_step"),
        "route_support": candidate.get("route_support"),
        "route_support_edge_count": candidate.get("route_support_edge_count"),
        "path_cells": path,
        "path_cells_current_to_candidate": path_current_to_candidate,
        "path_length_m": candidate.get("path_length_m"),
        "route_occ_conflict": bool(candidate.get("route_occ_conflict")),
        "unknown_fraction": candidate.get("unknown_fraction"),
        "occupied_fraction": candidate.get("occupied_fraction"),
        "floor_aligned_known_free": bool(candidate.get("floor_aligned_known_free")),
        "floor_z_source": candidate.get("floor_z_source"),
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
        "safety_ok": bool(safe),
        "safety_reason": safety_reason,
    }
    if not safe:
        record.update({"bridge_status": "rejected_before_visibility", "offscreen_reason": safety_reason})
        return record
    if len(path_current_to_candidate) < 2:
        record.update({"bridge_status": "invalid_route_path", "offscreen_reason": "path_too_short"})
        return record
    if not _path_contiguous(path_current_to_candidate):
        record.update({"bridge_status": "invalid_route_path", "offscreen_reason": "non_contiguous_path_cells"})
        return record
    pose = _pose_from_observation(observation or {})
    if pose is None:
        record.update({"bridge_status": "missing_current_pose", "offscreen_reason": "missing_causal_pose"})
        return record
    current_xy, current_yaw_deg = pose
    record["current_pose"] = {"xy": list(current_xy), "yaw_deg": current_yaw_deg}
    if observation and isinstance(observation.get("route_node"), Mapping):
        record["current_grid"] = list(observation["route_node"].get("pose_grid") or [])
    edge_reports = []
    for edge_index, cell in enumerate(path_current_to_candidate[1:], start=1):
        target_xy = grid_to_xy(cell, origin=grid_origin, cell_size_m=cell_size_m)
        visibility = _visibility(
            current_xy,
            current_yaw_deg,
            target_xy,
            fov_deg=fov_deg,
            max_distance_m=max_visible_distance_m,
            depth_available=False,
        )
        edge_reports.append({"edge_index": edge_index, "grid": list(cell), "xy": list(target_xy), **visibility})
    first_edge = edge_reports[0]
    record["edge_reports"] = edge_reports
    record["first_visible_subgoal"] = first_edge["grid"] if first_edge["in_horizontal_fov"] else None
    if first_edge["in_horizontal_fov"]:
        record["bridge_status"] = "first_edge_horizontally_visible"
        record["offscreen_reason"] = None
    else:
        record["bridge_status"] = "offscreen_requires_turn_reobserve"
        record["offscreen_reason"] = "first_edge_outside_horizontal_fov"
    return record


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_events(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/stage27_m3_candidate_events.jsonl")):
        rows.extend(_load_jsonl(path))
    return [{key: value for key, value in row.items()} for key, row in sorted(
        { _event_key(row): row for row in rows }.items()
    )]


def load_observations(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in sorted(root.glob("**/replay_ledger/*/observations.jsonl")):
        for row in _load_jsonl(path):
            key = _event_key(row)
            indexed.setdefault(key, row)
    return indexed


def _safe_candidates(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    pool = event.get("ablation", {}).get(SAFE_STAGE, {}).get("candidates") or []
    unique: dict[str, dict[str, Any]] = {}
    for candidate in pool:
        key = str(candidate.get("candidate_id") or repr(candidate.get("path_cells")))
        unique.setdefault(key, candidate)
    return list(unique.values())


def manifest_audit(expected: Iterable[Mapping[str, Any]], observed_events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    expected = list(expected)
    expected_by_key = {_event_key(row): row for row in expected}
    observed_by_key = {_event_key(row) for row in observed_events}
    missing = sorted(set(expected_by_key) - observed_by_key)
    unexpected = sorted(observed_by_key - set(expected_by_key))
    by_split: dict[str, dict[str, int]] = {}
    for key, row in expected_by_key.items():
        split = str(row.get("gt_split") or row.get("audit_selection", {}).get("gt_split") or "unspecified")
        bucket = by_split.setdefault(split, {"expected": 0, "observed": 0, "missing": 0})
        bucket["expected"] += 1
        if key in observed_by_key:
            bucket["observed"] += 1
        else:
            bucket["missing"] += 1
    return {
        "denominator_contract": "exact_manifest_key_missing_events_count_as_zero_bridge_coverage",
        "expected_event_count": len(expected_by_key),
        "observed_event_count": len(observed_by_key),
        "missing_event_count": len(missing),
        "unexpected_event_count": len(unexpected),
        "missing_event_keys": [
            {"scene_id": key[0], "episode_id": key[1], "step_id": key[2]} for key in missing
        ],
        "unexpected_event_keys": [
            {"scene_id": key[0], "episode_id": key[1], "step_id": key[2]} for key in unexpected
        ],
        "by_gt_split": by_split,
    }


def audit(
    root: Path,
    *,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_visible_distance_m: float = DEFAULT_MAX_VISIBLE_M,
    expected_events: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    events = load_events(root)
    observations = load_observations(root)
    expected_events = list(expected_events) if expected_events is not None else None
    if expected_events is not None:
        expected_keys = {_event_key(row) for row in expected_events}
        events = [row for row in events if _event_key(row) in expected_keys]
        observations = {
            key: row for key, row in observations.items() if key in expected_keys
        }
    records = []
    for event in events:
        for candidate in _safe_candidates(event):
            records.append(bridge_candidate(
                event,
                candidate,
                observations.get(_event_key(event)),
                fov_deg=fov_deg,
                max_visible_distance_m=max_visible_distance_m,
            ))
    status_counts = Counter(str(row.get("bridge_status")) for row in records)
    safe_records = [row for row in records if row.get("safety_ok")]
    report = {
        "task": "stage29_hsgm_inspired_route_bridge_audit",
        "schema_version": "stage29_hsgm_route_bridge_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "visibility_mode": "pose_bearing_only_no_depth_occlusion",
        "fov_deg": float(fov_deg),
        "max_visible_distance_m": float(max_visible_distance_m),
        "event_count": len(events),
        "candidate_count": len(records),
        "safe_candidate_count": len(safe_records),
        "bridgeable_first_edge_count": sum(row.get("bridge_status") == "first_edge_horizontally_visible" for row in safe_records),
        "offscreen_requires_reobserve_count": sum(row.get("bridge_status") == "offscreen_requires_turn_reobserve" for row in safe_records),
        "status_counts": dict(sorted(status_counts.items())),
        "event_zero_count": sum(not _safe_candidates(event) for event in events),
        "event_one_count": sum(len(_safe_candidates(event)) == 1 for event in events),
        "event_multi_count": sum(len(_safe_candidates(event)) >= 2 for event in events),
        "records": records,
    }
    if expected_events is not None:
        report["manifest_audit"] = manifest_audit(expected_events, events)
    return report


def write_bev_ppm(path: Path, records: Iterable[Mapping[str, Any]], *, scale_px: int = 8, margin_px: int = 24) -> None:
    """Write a dependency-free local BEV audit image for one event."""
    records = list(records)
    grids: list[tuple[float, float]] = []
    for record in records:
        current = record.get("current_grid") or []
        if len(current) >= 2:
            grids.append(_as_xy(current))
        for edge in record.get("edge_reports") or []:
            grid = edge.get("grid") or []
            if len(grid) >= 2:
                grids.append(_as_xy(grid))
    if not grids:
        return
    min_row = math.floor(min(value[0] for value in grids))
    max_row = math.ceil(max(value[0] for value in grids))
    min_col = math.floor(min(value[1] for value in grids))
    max_col = math.ceil(max(value[1] for value in grids))
    width = max(64, (max_col - min_col + 1) * int(scale_px) + 2 * int(margin_px))
    height = max(64, (max_row - min_row + 1) * int(scale_px) + 2 * int(margin_px))
    pixels = bytearray([255] * (width * height * 3))

    def put(row: float, col: float, color: tuple[int, int, int], radius: int = 2) -> None:
        x = int(margin_px + (col - min_col) * scale_px)
        y = int(margin_px + (row - min_row) * scale_px)
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                offset = (yy * width + xx) * 3
                pixels[offset:offset + 3] = bytes(color)

    for record in records:
        status = str(record.get("bridge_status"))
        route_color = (70, 150, 70) if record.get("safety_ok") else (180, 180, 180)
        for edge in record.get("edge_reports") or []:
            grid = edge.get("grid") or []
            if len(grid) >= 2:
                put(float(grid[0]), float(grid[1]), route_color, radius=2)
        current = record.get("current_grid") or []
        if len(current) >= 2:
            put(float(current[0]), float(current[1]), (220, 40, 40), radius=4)
        first = (record.get("edge_reports") or [{}])[0].get("grid") or []
        if len(first) >= 2:
            candidate_color = (30, 100, 220) if status == "first_edge_horizontally_visible" else (220, 120, 20)
            put(float(first[0]), float(first[1]), candidate_color, radius=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)

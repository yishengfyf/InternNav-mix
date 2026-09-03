#!/usr/bin/env python3
"""Offline audit of causal RGB-D-only relative-height odometry.

The estimator deliberately receives only fields available to the online agent:
depth, intrinsics, GPS/compass and the commanded camera pitch. Habitat height is
read only after all estimates have been produced and is used exclusively for
scoring. No output is consumed by navigation or SparseOcc.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - server/runtime contract
    raise RuntimeError("Stage80 requires scipy.spatial.cKDTree") from exc


DEPTH_STRIDE = 16
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 5.0
XY_RADIUS_M = 0.10
XY_NEIGHBORS = 4
MAX_DELTA_M = 0.30
HISTOGRAM_BIN_M = 0.02
REFINE_RADIUS_M = 0.03
MIN_CANDIDATES = 40
MIN_PEAK_INLIERS = 24
MIN_PEAK_INLIER_RATE = 0.08
MAX_MAD_M = 0.08
MIN_PEAK_RATIO = 1.10
CAMERA_HEIGHT_M = 1.25


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _yaw_to_tf(position: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _camera_to_base(pitch_down_deg: float) -> np.ndarray:
    pitch = math.radians(float(pitch_down_deg))
    c, s = math.cos(pitch), math.sin(pitch)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (0.0, -s, c),
        (-1.0, 0.0, 0.0),
        (0.0, -c, -s),
    )
    result[:3, 3] = (0.0, 0.0, CAMERA_HEIGHT_M)
    return result


def _base_pose(pose: dict[str, Any]) -> np.ndarray:
    gps = np.asarray(pose.get("gps"), dtype=np.float64).reshape(-1)
    compass = np.asarray(pose.get("compass"), dtype=np.float64).reshape(-1)
    if gps.size < 2 or compass.size < 1:
        raise ValueError("missing_gps_compass")
    return _yaw_to_tf(
        np.asarray((float(gps[0]), -float(gps[1]), 0.0)), float(compass[0])
    )


def project_depth(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    relative_base_tf: np.ndarray,
    pitch_down_deg: float,
    *,
    stride: int = DEPTH_STRIDE,
) -> np.ndarray:
    """Project a causal depth frame with the same 2-D pose convention as SparseOcc."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"invalid_depth_shape:{depth.shape}")
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    values = depth[ys, xs]
    keep = np.isfinite(values) & (values >= MIN_DEPTH_M) & (values <= MAX_DEPTH_M)
    ys, xs, values = ys[keep], xs[keep], values[keep]
    if not values.size:
        return np.zeros((0, 3), dtype=np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera = np.column_stack(
        ((xs - cx) * values / fx, (ys - cy) * values / fy, values, np.ones_like(values))
    )
    camera_pose = np.asarray(relative_base_tf, dtype=np.float64) @ _camera_to_base(
        pitch_down_deg
    )
    return (camera_pose @ camera.T).T[:, :3]


def estimate_pair_delta(previous: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """Estimate current_height - previous_height without semantic labels or GT."""
    previous = np.asarray(previous, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(current, dtype=np.float64).reshape(-1, 3)
    base = {
        "valid": False,
        "reason": None,
        "previous_point_count": int(len(previous)),
        "current_point_count": int(len(current)),
        "xy_radius_m": XY_RADIUS_M,
        "xy_neighbors": XY_NEIGHBORS,
    }
    if len(previous) < MIN_CANDIDATES or len(current) < MIN_CANDIDATES:
        return {**base, "reason": "insufficient_points", "candidate_count": 0}

    distances, indices = cKDTree(previous[:, :2]).query(
        current[:, :2], k=XY_NEIGHBORS, distance_upper_bound=XY_RADIUS_M
    )
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid_neighbor = np.isfinite(distances) & (indices < len(previous))
    current_ids = np.broadcast_to(np.arange(len(current))[:, None], indices.shape)
    dz = previous[indices[valid_neighbor], 2] - current[current_ids[valid_neighbor], 2]
    dz = dz[np.isfinite(dz) & (np.abs(dz) <= MAX_DELTA_M + 1e-12)]
    base["candidate_count"] = int(dz.size)
    base["current_points_with_xy_overlap"] = int(np.count_nonzero(np.any(valid_neighbor, axis=1)))
    base["current_xy_overlap_rate"] = float(
        np.count_nonzero(np.any(valid_neighbor, axis=1)) / len(current)
    )
    if dz.size < MIN_CANDIDATES:
        return {**base, "reason": "insufficient_xy_overlap"}

    edges = np.arange(
        -MAX_DELTA_M, MAX_DELTA_M + HISTOGRAM_BIN_M * 1.01, HISTOGRAM_BIN_M
    )
    counts, edges = np.histogram(dz, bins=edges)
    peak_index = int(np.argmax(counts))
    peak_count = int(counts[peak_index])
    competitors = counts.copy()
    competitors[max(0, peak_index - 1) : peak_index + 2] = 0
    second_count = int(np.max(competitors)) if competitors.size else 0
    peak_ratio = float(peak_count / max(1, second_count))
    peak_center = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
    peak_values = dz[np.abs(dz - peak_center) <= REFINE_RADIUS_M]
    if not peak_values.size:
        return {**base, "reason": "empty_peak"}
    estimate = float(np.median(peak_values))
    residual = np.abs(peak_values - estimate)
    mad = float(np.median(residual))
    inliers = int(peak_values.size)
    inlier_rate = float(inliers / dz.size)
    valid = (
        inliers >= MIN_PEAK_INLIERS
        and inlier_rate >= MIN_PEAK_INLIER_RATE
        and mad <= MAX_MAD_M
        and peak_ratio >= MIN_PEAK_RATIO
    )
    if inliers < MIN_PEAK_INLIERS:
        reason = "insufficient_peak_inliers"
    elif inlier_rate < MIN_PEAK_INLIER_RATE:
        reason = "low_peak_inlier_rate"
    elif mad > MAX_MAD_M:
        reason = "high_peak_mad"
    elif peak_ratio < MIN_PEAK_RATIO:
        reason = "ambiguous_peak"
    else:
        reason = "ok"
    return {
        **base,
        "valid": bool(valid),
        "reason": reason,
        "estimated_delta_m": estimate,
        "peak_center_m": peak_center,
        "peak_inlier_count": inliers,
        "peak_inlier_rate": inlier_rate,
        "peak_ratio": peak_ratio,
        "mad_m": mad,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_observations(path: Path) -> list[dict[str, Any]]:
    """Stream the large ledger and retain no Habitat semantic-scene payload."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append({
                "observation_index": raw.get("observation_index"),
                "step_id": raw.get("step_id"),
                "camera_pitch_deg": raw.get("camera_pitch_deg"),
                "camera_model": {"intrinsic": (raw.get("camera_model") or {}).get("intrinsic")},
                "depth_path": raw.get("depth_path"),
                "pose": raw.get("pose") or {},
            })
    return rows


def _episode_key(path: Path) -> tuple[str, str, str]:
    match = re.match(r"(.+)_([^_]+)_r(\d+)$", path.name)
    if match is None:
        raise ValueError(f"invalid_episode_directory:{path}")
    return match.group(1), match.group(2), match.group(3)


def _stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _height(pose: dict[str, Any]) -> float | None:
    value = pose.get("stage23a_gt_relative_height_m")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _render_curve(report: dict[str, Any], output: Path) -> None:
    frames = report["frames"]
    gt = [(i, row["offline_gt_height_m"]) for i, row in enumerate(frames) if row["offline_gt_height_m"] is not None]
    est = [(i, row["chain_height_m"]) for i, row in enumerate(frames) if row["chain_height_m"] is not None]
    width, height, left, top, bottom = 1100, 420, 72, 42, 72
    image = Image.new("RGB", (width, height), (24, 28, 34))
    draw = ImageDraw.Draw(image)
    plot_w, plot_h = width - left - 28, height - top - bottom
    values = [v for _, v in gt + est]
    lo, hi = (min(values), max(values)) if values else (-0.5, 0.5)
    margin = max(0.15, 0.10 * max(1e-6, hi - lo))
    lo, hi = lo - margin, hi + margin
    count = max(2, len(frames))

    def xy(index: int, value: float) -> tuple[int, int]:
        x = left + int(index * plot_w / (count - 1))
        y = top + int((hi - value) * plot_h / max(1e-9, hi - lo))
        return x, y

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(100, 110, 120))
    for points, color in ((gt, (70, 190, 255)), (est, (255, 185, 50))):
        segments: list[tuple[int, int]] = []
        last_index = None
        for index, value in points:
            point = xy(index, value)
            if last_index is not None and index != last_index + 1:
                if len(segments) > 1:
                    draw.line(segments, fill=color, width=2)
                segments = []
            segments.append(point)
            last_index = index
        if len(segments) > 1:
            draw.line(segments, fill=color, width=2)
    for index, row in enumerate(frames):
        if index and not row["pair_registration_valid"]:
            x, _ = xy(index, lo)
            draw.line((x, top, x, top + plot_h), fill=(120, 55, 55), width=1)
    draw.text((left, 12), f"{report['scene_id']}/{report['episode_id']}  blue=GT audit  orange=causal RGB-D estimate  red=abstain", fill=(235, 235, 235))
    draw.text((left, height - 52), f"coverage={report['pair_valid_coverage']:.3f}  delta_MAE={report['valid_delta_abs_error_m']['mean']}  chain_end={report['chain_end_height_m']}  GT_end={report['offline_gt_end_height_m']}", fill=(220, 220, 220))
    draw.text((left, height - 30), "GT is drawn only for post-hoc scoring; it is never an estimator input.", fill=(150, 165, 180))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _semantic_coverage(semantic_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in semantic_root.glob("**/online_lseg_shadow/*/events.jsonl"):
        key = _episode_key(path.parent)
        rows = _read_jsonl(path)
        valid = [row for row in rows if row.get("valid")]
        fractions = [
            float(row.get("surface_sample_count", 0)) / max(1, int(row.get("sampled_depth_count", 0)))
            for row in valid
        ]
        result[key] = {
            "frame_count": len(rows),
            "valid_frame_count": len(valid),
            "mean_high_confidence_surface_sampling_fraction": (
                float(np.mean(fractions)) if fractions else None
            ),
            "note": "LSeg high-confidence surface coverage is diagnostic only and is not used by the height estimator.",
        }
    return result


def _route_query_steps(semantic_root: Path) -> dict[tuple[str, str], list[int]]:
    result: dict[tuple[str, str], list[int]] = {}
    for path in semantic_root.glob("**/s2_recovery_context_events.jsonl"):
        for row in _read_jsonl(path):
            if row.get("event_type") == "stage75_route_guidance":
                key = (str(row.get("scene_id")), str(row.get("episode_id")))
                result.setdefault(key, []).append(int(row.get("current_query_step")))
    return {key: sorted(set(value)) for key, value in result.items()}


def analyze(*, semantic_root: Path, replay_root: Path, output: Path, viz_dir: Path) -> dict[str, Any]:
    semantic = _semantic_coverage(semantic_root)
    query_steps = _route_query_steps(semantic_root)
    episode_reports = []
    errors = []
    all_delta_errors: list[float] = []
    all_valid_flags: list[bool] = []
    sign_correct = 0
    sign_total = 0

    ledgers = sorted(replay_root.glob("**/replay_ledger/*/observations.jsonl"))
    if not ledgers:
        errors.append("missing_replay_ledgers")
    for observations_path in ledgers:
        episode_dir = observations_path.parent
        key = _episode_key(episode_dir)
        rows = _read_observations(observations_path)
        if len(rows) < 2:
            errors.append(f"insufficient_observations:{episode_dir.name}")
            continue
        first_pose = _base_pose(rows[0].get("pose") or {})
        inverse_first = np.linalg.inv(first_pose)
        clouds: list[np.ndarray] = []
        frame_reports: list[dict[str, Any]] = []
        chain_height: float | None = 0.0
        chain_broken_at: int | None = None
        for index, row in enumerate(rows):
            pose = row.get("pose") or {}
            depth_path = episode_dir / str(row.get("depth_path"))
            depth = np.load(depth_path)["depth_m"]
            relative_base = inverse_first @ _base_pose(pose)
            pitch = float(row.get("camera_pitch_deg", pose.get("camera_pitch_deg", 0.0)) or 0.0)
            intrinsic = np.asarray(row.get("camera_model", {}).get("intrinsic"), dtype=np.float64)
            cloud = project_depth(depth, intrinsic, relative_base, pitch)
            clouds.append(cloud)
            gt = _height(pose)
            report = {
                "observation_index": int(row.get("observation_index", index)),
                "step_id": int(row.get("step_id", index)),
                "camera_pitch_deg": pitch,
                "point_count": int(len(cloud)),
                "offline_gt_height_m": gt,
                "pair_registration_valid": index == 0,
                "pair_registration_reason": "initial_frame" if index == 0 else None,
                "estimated_delta_m": 0.0 if index == 0 else None,
                "offline_gt_delta_m": 0.0 if index == 0 and gt is not None else None,
                "offline_delta_abs_error_m": 0.0 if index == 0 and gt is not None else None,
                "chain_height_m": 0.0 if index == 0 else None,
                "gt_used_by_estimator": False,
            }
            if index:
                pair = estimate_pair_delta(clouds[index - 1], cloud)
                report.update({f"registration_{name}": value for name, value in pair.items()})
                report["pair_registration_valid"] = bool(pair["valid"])
                report["pair_registration_reason"] = pair["reason"]
                report["estimated_delta_m"] = pair.get("estimated_delta_m")
                previous_gt = _height(rows[index - 1].get("pose") or {})
                gt_delta = None if gt is None or previous_gt is None else gt - previous_gt
                report["offline_gt_delta_m"] = gt_delta
                if pair["valid"] and gt_delta is not None:
                    delta_error = abs(float(pair["estimated_delta_m"]) - gt_delta)
                    report["offline_delta_abs_error_m"] = delta_error
                    all_delta_errors.append(delta_error)
                    if abs(gt_delta) >= 0.02:
                        sign_total += 1
                        sign_correct += int(np.sign(pair["estimated_delta_m"]) == np.sign(gt_delta))
                all_valid_flags.append(bool(pair["valid"]))
                if chain_height is not None and pair["valid"]:
                    chain_height += float(pair["estimated_delta_m"])
                    report["chain_height_m"] = chain_height
                else:
                    if chain_broken_at is None:
                        chain_broken_at = index
                    chain_height = None
            frame_reports.append(report)

        pair_errors = [
            row["offline_delta_abs_error_m"]
            for row in frame_reports[1:]
            if row.get("offline_delta_abs_error_m") is not None
        ]
        valid_pairs = sum(row["pair_registration_valid"] for row in frame_reports[1:])
        chain_rows = [row for row in frame_reports if row["chain_height_m"] is not None]
        chain_errors = [
            abs(row["chain_height_m"] - row["offline_gt_height_m"])
            for row in chain_rows
            if row["offline_gt_height_m"] is not None
        ]
        episode_query_steps = query_steps.get((key[0], key[1]), [])
        query_audits = []
        for step in episode_query_steps:
            candidates = [row for row in frame_reports if row["step_id"] == step]
            horizon = [row for row in candidates if abs(row["camera_pitch_deg"]) < 1e-6]
            selected = (horizon or candidates)[0] if candidates else None
            query_audits.append({
                "step_id": step,
                "observation_index": selected.get("observation_index") if selected else None,
                "chain_height_m": selected.get("chain_height_m") if selected else None,
                "offline_gt_height_m": selected.get("offline_gt_height_m") if selected else None,
                "offline_abs_error_m": (
                    abs(selected["chain_height_m"] - selected["offline_gt_height_m"])
                    if selected and selected.get("chain_height_m") is not None and selected.get("offline_gt_height_m") is not None
                    else None
                ),
            })
        report = {
            "scene_id": key[0],
            "episode_id": key[1],
            "rank": int(key[2]),
            "frame_count": len(frame_reports),
            "pair_count": len(frame_reports) - 1,
            "valid_pair_count": valid_pairs,
            "pair_valid_coverage": float(valid_pairs / max(1, len(frame_reports) - 1)),
            "pair_reason_counts": dict(sorted(Counter(row["pair_registration_reason"] for row in frame_reports[1:]).items())),
            "valid_delta_abs_error_m": _stats(pair_errors),
            "chain_broken_at_observation_index": chain_broken_at,
            "chain_frame_count": len(chain_rows),
            "chain_height_abs_error_m": _stats(chain_errors),
            "chain_end_height_m": chain_rows[-1]["chain_height_m"] if chain_rows else None,
            "offline_gt_end_height_m": frame_reports[-1]["offline_gt_height_m"],
            "recovery_query_height_audits": query_audits,
            "semantic_surface_sampling": semantic.get(key),
            "frames": frame_reports,
        }
        episode_reports.append(report)
        _render_curve(report, viz_dir / f"{key[0]}_{key[1]}_height_curve.png")

    vertical_keys = {("PX4nDJXEHrG", "9891"), ("SN83YJsR3w2", "3316")}
    flat_keys = {("8WUmhLawc2A", "8579"), ("HxpKQynjfin", "4863"), ("VVfe2KiqLaN", "4976")}
    vertical_query_pass = True
    flat_drift_pass = True
    vertical_query_details = []
    flat_drift_details = []
    for report in episode_reports:
        key2 = (report["scene_id"], report["episode_id"])
        if key2 in vertical_keys:
            query_valid = bool(report["recovery_query_height_audits"]) and all(
                row["offline_abs_error_m"] is not None and row["offline_abs_error_m"] <= 0.25
                for row in report["recovery_query_height_audits"]
            )
            vertical_query_pass &= query_valid
            vertical_query_details.append({"episode_key": list(key2), "passed": query_valid, "queries": report["recovery_query_height_audits"]})
        if key2 in flat_keys:
            errors_for_flat = [
                abs(row["chain_height_m"])
                for row in report["frames"]
                if row["chain_height_m"] is not None
            ]
            max_drift = max(errors_for_flat) if errors_for_flat else None
            drift_valid = max_drift is not None and max_drift <= 0.15 and report["chain_broken_at_observation_index"] is None
            flat_drift_pass &= drift_valid
            flat_drift_details.append({"episode_key": list(key2), "passed": drift_valid, "max_abs_chain_height_m": max_drift})

    coverage = float(np.mean(all_valid_flags)) if all_valid_flags else 0.0
    delta_stats = _stats(all_delta_errors)
    release = {
        "min_pair_valid_coverage": 0.80,
        "pair_valid_coverage": coverage,
        "pair_coverage_passed": coverage >= 0.80,
        "max_delta_mae_m": 0.05,
        "delta_mae_passed": delta_stats["mean"] is not None and delta_stats["mean"] <= 0.05,
        "max_delta_p95_m": 0.12,
        "delta_p95_passed": delta_stats["p95"] is not None and delta_stats["p95"] <= 0.12,
        "vertical_query_passed": vertical_query_pass and len(vertical_query_details) == 2,
        "vertical_query_details": vertical_query_details,
        "flat_drift_passed": flat_drift_pass and len(flat_drift_details) == 3,
        "flat_drift_details": flat_drift_details,
    }
    release["passed"] = all(
        release[name]
        for name in (
            "pair_coverage_passed", "delta_mae_passed", "delta_p95_passed",
            "vertical_query_passed", "flat_drift_passed",
        )
    )
    result = {
        "task": "stage80_causal_rgbd_height_odometry_offline_audit",
        "schema_version": "stage80_causal_rgbd_height_odometry_v1",
        "integrity_passed": not errors and len(episode_reports) == 6,
        "errors": errors,
        "episode_count": len(episode_reports),
        "frame_count": sum(row["frame_count"] for row in episode_reports),
        "pair_count": len(all_valid_flags),
        "valid_pair_count": sum(all_valid_flags),
        "pair_valid_coverage": coverage,
        "valid_delta_abs_error_m": delta_stats,
        "nontrivial_delta_sign_agreement": (
            float(sign_correct / sign_total) if sign_total else None
        ),
        "nontrivial_delta_sign_sample_count": sign_total,
        "release_gate": release,
        "episode_reports": episode_reports,
        "contract": {
            "offline_audit_only": True,
            "causal_inputs_only": True,
            "estimator_inputs": ["depth", "camera_intrinsic", "gps", "compass", "commanded_camera_pitch"],
            "semantic_labels_used_by_estimator": False,
            "habitat_gt_used_by_estimator": False,
            "habitat_gt_used_for_posthoc_scoring_only": True,
            "unknown_is_free": False,
            "sparseocc_modified": False,
            "prompt_injected": False,
            "action_applied": False,
        },
        "parameters": {
            "depth_stride": DEPTH_STRIDE,
            "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
            "xy_radius_m": XY_RADIUS_M,
            "xy_neighbors": XY_NEIGHBORS,
            "max_pair_delta_m": MAX_DELTA_M,
            "histogram_bin_m": HISTOGRAM_BIN_M,
            "refine_radius_m": REFINE_RADIUS_M,
            "min_candidates": MIN_CANDIDATES,
            "min_peak_inliers": MIN_PEAK_INLIERS,
            "min_peak_inlier_rate": MIN_PEAK_INLIER_RATE,
            "max_mad_m": MAX_MAD_M,
            "min_peak_ratio": MIN_PEAK_RATIO,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viz-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        semantic_root=args.semantic_root,
        replay_root=args.replay_root,
        output=args.output,
        viz_dir=args.viz_dir,
    )
    summary = {key: result[key] for key in (
        "integrity_passed", "episode_count", "frame_count", "pair_count",
        "valid_pair_count", "pair_valid_coverage", "valid_delta_abs_error_m",
        "nontrivial_delta_sign_agreement", "release_gate",
    )}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize a patched FreeOcc Habitat smoke run without changing online policy.

The patched FreeOcc process writes ``audit/freeocc_mapping_audit.json`` and
``audit/trajectories.npz``.  This script turns those raw diagnostics into a
small, reproducible report and optional plot.  It deliberately treats the run
as an offline shadow audit: no result here grants navigation safety authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _ply_header(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    result["bytes"] = path.stat().st_size
    vertex_count = None
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    result["vertices"] = vertex_count
    try:
        from plyfile import PlyData

        vertices = PlyData.read(str(path))["vertex"]
        if len(vertices):
            xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
            finite = np.isfinite(xyz).all(axis=1)
            result["finite_vertices"] = int(finite.sum())
            if finite.any():
                result["bbox_min"] = xyz[finite].min(axis=0).tolist()
                result["bbox_max"] = xyz[finite].max(axis=0).tolist()
                result["bbox_extent"] = np.ptp(xyz[finite], axis=0).tolist()
    except ImportError:
        result["ply_payload_note"] = "install plyfile to compute bbox"
    return result


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _umeyama(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"trajectory shape mismatch: {source.shape} vs {target.shape}")
    if len(source) < 3:
        raise ValueError("at least three trajectory pairs are required")
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src_centered, dst_centered = source - src_mean, target - dst_mean
    covariance = (dst_centered.T @ src_centered) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular) @ correction) / variance) if variance > 1e-12 else 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _trajectory_summary(npz_path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    estimated = np.asarray(data["estimated_c2w"], dtype=np.float64)[:, :3, 3]
    gt = np.asarray(data["gt_c2w"], dtype=np.float64)[:, :3, 3]
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    finite = np.isfinite(estimated).all(axis=1) & np.isfinite(gt).all(axis=1)
    estimated, gt, timestamps = estimated[finite], gt[finite], timestamps[finite]
    summary: Dict[str, Any] = {
        "pairs": int(len(estimated)),
        "estimated_path_length_raw": _path_length(estimated),
        "gt_path_length_m": _path_length(gt),
        "timestamp_min": float(timestamps.min()) if len(timestamps) else None,
        "timestamp_max": float(timestamps.max()) if len(timestamps) else None,
    }
    if len(estimated) >= 3 and np.ptp(estimated, axis=0).max() > 1e-8 and np.ptp(gt, axis=0).max() > 1e-8:
        scale, rotation, translation = _umeyama(estimated, gt)
        aligned = scale * (estimated @ rotation.T) + translation
        summary.update(
            sim3_scale=scale,
            pre_rmse_m=float(np.sqrt(np.mean(np.sum((estimated - gt) ** 2, axis=1)))),
            post_rmse_m=float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1)))),
            aligned_path_length_m=_path_length(aligned),
            sim3_rotation=rotation.tolist(),
            sim3_translation=translation.tolist(),
        )
    else:
        aligned = estimated.copy()
        summary["alignment_warning"] = "trajectory is too short or degenerate for Sim3"
    return summary, aligned, gt


def _latest_frame_rows(filter_calls: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for call in filter_calls:
        for row in call.get("frames", []):
            rows[int(row["frame_id"])] = row
    return rows


def _mapping_call(audit: Dict[str, Any]) -> Dict[str, Any]:
    calls = audit.get("mapping_calls", [])
    full = [call for call in calls if call.get("window_size") is None]
    return (full or calls or [{}])[-1]


def _diagnosis(filter_rows: Dict[int, Dict[str, Any]], mapping: Dict[str, Any]) -> str:
    raw = sum(int(row.get("raw_valid", 0)) for row in filter_rows.values())
    filtered = sum(int(row.get("final_valid", 0)) for row in filter_rows.values())
    gaussians = int(mapping.get("total_gaussians", 0))
    combined = sum(int(row.get("combined_valid", 0)) for row in mapping.get("frames", []))
    finite = sum(int(row.get("finite_valid", 0)) for row in mapping.get("frames", []))
    if raw and filtered / raw < 1e-3:
        return "filter_collapse"
    if filtered and combined / filtered < 0.1:
        return "mapper_frame_or_mask_collapse"
    if combined and finite / combined < 0.9:
        return "nonfinite_geometry_collapse"
    if gaussians < 1000:
        return "too_few_gaussians_for_occ"
    return "no_count_collapse_detected"


def _plot(out_path: Path, filter_rows: Dict[int, Dict[str, Any]], aligned: np.ndarray, gt: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_ids = sorted(filter_rows)
    raw = [filter_rows[idx].get("raw_valid", 0) for idx in frame_ids]
    mv = [filter_rows[idx].get("after_multiview", 0) for idx in frame_ids]
    final = [filter_rows[idx].get("final_valid", 0) for idx in frame_ids]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(frame_ids, raw, label="raw disparity")
    axes[0].plot(frame_ids, mv, label="after multiview")
    axes[0].plot(frame_ids, final, label="final filter")
    axes[0].set_xlabel("DROID keyframe id")
    axes[0].set_ylabel("valid pixels")
    axes[0].set_title("FreeOcc filtering audit")
    axes[0].legend()
    if len(gt):
        axes[1].plot(gt[:, 0], gt[:, 2], "o-", markersize=2, label="Habitat GT")
    if len(aligned):
        axes[1].plot(aligned[:, 0], aligned[:, 2], "o-", markersize=2, label="DROID Sim3 aligned")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title("Keyframe trajectory (top view)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze(
    run_dir: Path, expected_input_frames: int | None = None
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], np.ndarray, np.ndarray]:
    audit_path = run_dir / "audit" / "freeocc_mapping_audit.json"
    trajectory_path = run_dir / "audit" / "trajectories.npz"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    trajectory, aligned, gt = _trajectory_summary(trajectory_path)
    rows = _latest_frame_rows(audit.get("filter_calls", []))
    mapping = _mapping_call(audit)
    raw_total = sum(int(row.get("raw_valid", 0)) for row in rows.values())
    mv_total = sum(int(row.get("after_multiview", 0)) for row in rows.values())
    final_total = sum(int(row.get("final_valid", 0)) for row in rows.values())
    result: Dict[str, Any] = {
        "schema_version": "freeocc_habitat_audit_v1",
        "run_dir": str(run_dir),
        "offline_shadow_only": True,
        "video_keyframes": int(audit.get("video_frames", 0)),
        "mapper_camera_count": len(audit.get("camera_uids", [])),
        "gaussian_frame_count": len(audit.get("gaussian_frame_uids", [])),
        "filter_latest_by_frame": {
            "frames": len(rows),
            "raw_valid": raw_total,
            "after_multiview": mv_total,
            "final_valid": final_total,
            "multiview_retention": float(mv_total / raw_total) if raw_total else None,
            "final_retention": float(final_total / raw_total) if raw_total else None,
        },
        "final_mapping_call": mapping,
        "trajectory": trajectory,
        "ply_raw": _ply_header(run_dir / "mesh" / "final_mono_raw.ply"),
        "ply_aligned": _ply_header(run_dir / "mesh" / "final_mono.ply"),
    }
    result["diagnosis"] = _diagnosis(rows, mapping)
    if expected_input_frames is not None:
        result["expected_input_frames"] = int(expected_input_frames)
        # DROID applies a motion filter, so only the dataset loader—not the keyframe
        # buffer—is expected to preserve this count.  The authoritative loader count
        # is captured in the run log and checked separately by the launcher.
        result["note"] = "video_keyframes may be lower than input frames due to DROID motion filtering"
    return result, rows, aligned, gt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-input-frames", type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, rows, aligned, gt = analyze(args.run_dir, args.expected_input_frames)
    output_path = out_dir / "freeocc_mapping_summary.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        _plot(out_dir / "freeocc_filter_trajectory_audit.png", rows, aligned, gt)
    except ImportError:
        result["plot_warning"] = "matplotlib is unavailable"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

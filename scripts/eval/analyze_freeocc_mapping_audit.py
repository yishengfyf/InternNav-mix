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
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np


SEMANTIC_NAMES_11 = (
    "ceiling",
    "floor",
    "wall",
    "window",
    "chair",
    "bed",
    "sofa",
    "table",
    "television",
    "furniture",
    "objects",
)

SEMANTIC_PALETTE_11 = np.asarray(
    [
        [220, 45, 45],
        [40, 160, 40],
        [155, 210, 225],
        [115, 155, 210],
        [195, 200, 75],
        [255, 180, 110],
        [140, 105, 180],
        [25, 110, 180],
        [150, 180, 55],
        [255, 140, 0],
        [195, 180, 220],
    ],
    dtype=np.float32,
) / 255.0


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


def _property_index(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _semantic_gaussians(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read aligned Gaussian centers and open-vocabulary logits from a PLY."""
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"]
    names = tuple(vertices.data.dtype.names or ())
    feat_names = sorted((name for name in names if name.startswith("ov_feat_")), key=_property_index)
    if not feat_names:
        raise ValueError(f"{path} has no ov_feat_* properties")
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    logits = np.column_stack([vertices[name] for name in feat_names]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(logits).all(axis=1)
    xyz, logits = xyz[finite], logits[finite]
    labels = logits.argmax(axis=1).astype(np.int64) if len(logits) else np.empty(0, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(feat_names))
    metadata = {
        "semantic_features": len(feat_names),
        "finite_semantic_gaussians": int(len(xyz)),
        "class_counts": {
            (SEMANTIC_NAMES_11[idx] if idx < len(SEMANTIC_NAMES_11) else f"class_{idx}"): int(value)
            for idx, value in enumerate(counts)
        },
    }
    return xyz, labels, metadata


def _rgb_paths(rgb_dir: Path | None) -> Sequence[Path]:
    if rgb_dir is None or not rgb_dir.exists():
        return ()
    suffixes = {".jpg", ".jpeg", ".png"}
    preferred = [rgb_dir / "color", rgb_dir / "rgb", rgb_dir]
    for folder in preferred:
        if folder.is_dir():
            paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
            if paths:
                return paths
    return sorted(path for path in rgb_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _sample_indices(labels: np.ndarray, max_points: int = 120_000) -> np.ndarray:
    if len(labels) <= max_points:
        return np.arange(len(labels))
    # Deterministic stratified sampling keeps small semantic classes visible.
    rng = np.random.default_rng(0)
    selected = []
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        quota = max(1, int(round(max_points * count / len(labels))))
        candidates = np.flatnonzero(labels == label)
        selected.append(rng.choice(candidates, size=min(quota, len(candidates)), replace=False))
    merged = np.concatenate(selected)
    if len(merged) > max_points:
        merged = rng.choice(merged, size=max_points, replace=False)
    return np.sort(merged)


def _equal_3d_axes(axis: Any, xyz: np.ndarray) -> None:
    if not len(xyz):
        return
    low, high = np.quantile(xyz, [0.01, 0.99], axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[2] - radius, center[2] + radius)
    axis.set_zlim(center[1] - radius, center[1] + radius)


def _plot_rgb_semantic_gaussians(
    out_path: Path,
    ply_path: Path,
    rgb_dir: Path | None,
    aligned: np.ndarray,
    gt: np.ndarray,
    profile_label: str,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xyz, labels, metadata = _semantic_gaussians(ply_path)
    paths = _rgb_paths(rgb_dir)
    chosen = () if not paths else (paths[0], paths[len(paths) // 2], paths[-1])
    sample = _sample_indices(labels)
    points, point_labels = xyz[sample], labels[sample]
    palette = SEMANTIC_PALETTE_11
    colors = palette[np.minimum(point_labels, len(palette) - 1)] if len(points) else np.empty((0, 3))

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.78, 1.22))
    for col in range(3):
        axis = fig.add_subplot(grid[0, col])
        if chosen:
            image = plt.imread(chosen[col])
            axis.imshow(image)
            axis.set_title(f"RGB frame: {chosen[col].name}")
        else:
            axis.text(0.5, 0.5, "RGB input not found", ha="center", va="center")
            axis.set_title("RGB frame")
        axis.axis("off")

    axis_3d = fig.add_subplot(grid[1, 0], projection="3d")
    if len(points):
        size = max(0.15, min(3.0, 90_000 / len(points)))
        axis_3d.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors, s=size, alpha=0.72, linewidths=0)
        _equal_3d_axes(axis_3d, points)
    axis_3d.set_xlabel("x")
    axis_3d.set_ylabel("z")
    axis_3d.set_zlabel("y / height")
    axis_3d.view_init(elev=25, azim=-62)
    axis_3d.set_title(f"Aligned semantic Gaussians ({len(xyz):,})")

    axis_top = fig.add_subplot(grid[1, 1])
    if len(points):
        size = max(0.15, min(3.0, 90_000 / len(points)))
        axis_top.scatter(points[:, 0], points[:, 2], c=colors, s=size, alpha=0.68, linewidths=0)
    axis_top.set_aspect("equal", adjustable="datalim")
    axis_top.set_xlabel("x (m)")
    axis_top.set_ylabel("z (m)")
    axis_top.set_title("Semantic Gaussian top view")

    axis_traj = fig.add_subplot(grid[1, 2])
    if len(gt):
        axis_traj.plot(gt[:, 0], gt[:, 2], "o-", markersize=2.5, linewidth=1.4, label="Habitat GT pose")
    if len(aligned):
        axis_traj.plot(
            aligned[:, 0], aligned[:, 2], "o-", markersize=2.5, linewidth=1.4, label="DROID Sim3 aligned"
        )
    axis_traj.set_aspect("equal", adjustable="datalim")
    axis_traj.set_xlabel("x (m)")
    axis_traj.set_ylabel("z (m)")
    axis_traj.set_title("Trajectory audit (GT is offline only)")
    axis_traj.legend(loc="best")

    present = sorted(set(point_labels.tolist()))
    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=palette[min(idx, len(palette) - 1)],
            markeredgecolor="none", markersize=7,
            label=(SEMANTIC_NAMES_11[idx] if idx < len(SEMANTIC_NAMES_11) else f"class_{idx}"),
        )
        for idx in present
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False)
    fig.suptitle(
        f"FreeOcc Habitat RGB-only audit — {profile_label}\n"
        "Diagnostic visualization only; unknown space is not free and this map has no navigation authority",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    metadata.update(
        displayed_gaussians=int(len(sample)),
        rgb_frames_found=int(len(paths)),
        rgb_frames_shown=[path.name for path in chosen],
    )
    return metadata


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


def _runtime_summary(log_path: Path) -> Dict[str, Any]:
    if not log_path.is_file():
        return {"log_exists": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result: Dict[str, Any] = {"log_exists": True}
    patterns = {
        "total_fps": r"Total FPS:\s*([0-9]+(?:\.[0-9]+)?)",
        "input_images": r"INFO:\s*([0-9]+) images got!",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            result[key] = float(matches[-1]) if key == "total_fps" else int(matches[-1])
    return result


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
        "runtime": _runtime_summary(run_dir / "console.log"),
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
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument("--profile-label", default="unspecified")
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, rows, aligned, gt = analyze(args.run_dir, args.expected_input_frames)
    output_path = out_dir / "freeocc_mapping_summary.json"
    result["profile_label"] = args.profile_label
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        _plot(out_dir / "freeocc_filter_trajectory_audit.png", rows, aligned, gt)
    except ImportError:
        result["plot_warning"] = "matplotlib is unavailable"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        result["semantic_visualization"] = _plot_rgb_semantic_gaussians(
            out_dir / "freeocc_rgb_semantic_gaussians.png",
            args.run_dir / "mesh" / "final_mono.ply",
            args.rgb_dir,
            aligned,
            gt,
            args.profile_label,
        )
    except (ImportError, ValueError, OSError) as exc:
        result["semantic_visualization_warning"] = str(exc)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

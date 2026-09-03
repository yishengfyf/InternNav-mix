#!/usr/bin/env python3
"""Evaluate a FreeOcc Gaussian map against observed Habitat depth geometry.

This is an offline audit.  Habitat depth and pose are used only to construct a
surface-occupancy reference; they are never fed to the RGB-only reconstruction
or to DualVLN.  Unknown/unobserved voxels are deliberately not treated as free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from scripts.eval.analyze_freeocc_mapping_audit import (
    SEMANTIC_NAMES_11,
    SEMANTIC_PALETTE_11,
    _ground_axes,
    _rgb_paths,
    _sample_indices,
    _semantic_gaussians,
)


def _numeric_paths(folder: Path, suffixes: Iterable[str]) -> Sequence[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed]

    def key(path: Path) -> Tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 2**31 - 1, path.name

    return sorted(paths, key=key)


def _select_frame_range(paths: Sequence[Path], frame_start: int | None, frame_stop: int | None) -> Sequence[Path]:
    """Select the same numeric frame interval used by the FreeOcc run.

    ``frame_stop`` is exclusive, matching Python/Hydra ``t_stop`` semantics.
    Keeping this at the file-list level prevents an accidental comparison of a
    short prediction window against the entire RGB-D episode.
    """
    if frame_start is None and frame_stop is None:
        return paths
    selected = []
    for path in paths:
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        if frame_start is not None and frame_id < frame_start:
            continue
        if frame_stop is not None and frame_id >= frame_stop:
            continue
        selected.append(path)
    return selected


def _load_observed_surface(
    input_dir: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    pixel_stride: int,
    max_depth_m: float,
    frame_start: int | None = None,
    frame_stop: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    import cv2

    depth_paths = _select_frame_range(_numeric_paths(input_dir / "depth", (".png", ".npy")), frame_start, frame_stop)
    pose_paths = _select_frame_range(_numeric_paths(input_dir / "pose", (".txt",)), frame_start, frame_stop)
    color_paths = _select_frame_range(_numeric_paths(input_dir / "color", (".jpg", ".jpeg", ".png")), frame_start, frame_stop)
    by_id = lambda paths: {int(path.stem): path for path in paths if path.stem.isdigit()}
    depth_by_id, pose_by_id, color_by_id = by_id(depth_paths), by_id(pose_paths), by_id(color_paths)
    frame_ids = sorted(set(depth_by_id) & set(pose_by_id) & set(color_by_id))
    count = len(frame_ids)
    if count == 0:
        raise ValueError(f"missing color/depth/pose triplets under {input_dir}")

    points, colors = [], []
    per_frame = []
    for frame_id in frame_ids:
        depth_path = depth_by_id[frame_id]
        color_path = color_by_id[frame_id]
        pose_path = pose_by_id[frame_id]
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
        color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
        pose = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all():
            continue
        height, width = depth.shape
        vv, uu = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
        sampled_depth = depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & (sampled_depth <= max_depth_m)
        z = sampled_depth[valid].astype(np.float64)
        camera = np.column_stack(
            (
                (uu[valid] - cx) * z / fx,
                (vv[valid] - cy) * z / fy,
                z,
                np.ones_like(z),
            )
        )
        world = (pose @ camera.T).T[:, :3]
        finite = np.isfinite(world).all(axis=1)
        points.append(world[finite])
        colors.append(color[::pixel_stride, ::pixel_stride][valid][finite])
        per_frame.append({"frame": int(frame_id), "valid_points": int(finite.sum())})
    xyz = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float64)
    rgb = np.concatenate(colors, axis=0) if colors else np.empty((0, 3), dtype=np.uint8)
    return xyz, rgb, {
        "triplets": int(count),
        "frame_start": None if frame_start is None else int(frame_start),
        "frame_stop": None if frame_stop is None else int(frame_stop),
        "frame_ids": frame_ids,
        "pixel_stride": int(pixel_stride),
        "max_depth_m": float(max_depth_m),
        "surface_points": int(len(xyz)),
        "per_frame": per_frame,
    }


def _voxelize(points: np.ndarray, origin: np.ndarray, shape: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.int32)
    indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    valid = ((indices >= 0) & (indices < shape)).all(axis=1)
    return np.unique(indices[valid], axis=0).astype(np.int32)


def _voxel_centers(indices: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return origin + (indices.astype(np.float64) + 0.5) * voxel_size


def _linear_keys(indices: np.ndarray, shape: np.ndarray) -> np.ndarray:
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    return np.ravel_multi_index(indices.T, tuple(int(value) for value in shape)).astype(np.int64)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(source) or not len(target):
        return np.full(len(source), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(target).query(source, k=1, workers=-1)
        return np.asarray(distances, dtype=np.float64)
    except ImportError:
        result = np.empty(len(source), dtype=np.float64)
        for start in range(0, len(source), 2_000):
            chunk = source[start : start + 2_000]
            result[start : start + len(chunk)] = np.sqrt(
                np.min(np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2), axis=1)
            )
        return result


def _distance_stats(values: np.ndarray) -> Dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def _surface_metrics(
    gt_indices: np.ndarray,
    pred_indices: np.ndarray,
    origin: np.ndarray,
    shape: np.ndarray,
    voxel_size: float,
) -> Dict[str, Any]:
    gt_keys = _linear_keys(gt_indices, shape)
    pred_keys = _linear_keys(pred_indices, shape)
    intersection = np.intersect1d(gt_keys, pred_keys, assume_unique=True)
    union_count = len(gt_keys) + len(pred_keys) - len(intersection)
    gt_centers = _voxel_centers(gt_indices, origin, voxel_size)
    pred_centers = _voxel_centers(pred_indices, origin, voxel_size)
    pred_to_gt = _nearest_distances(pred_centers, gt_centers)
    gt_to_pred = _nearest_distances(gt_centers, pred_centers)
    result: Dict[str, Any] = {
        "gt_observed_voxels": int(len(gt_indices)),
        "pred_gaussian_center_voxels_in_gt_grid": int(len(pred_indices)),
        "exact_intersection": int(len(intersection)),
        "exact_iou": float(len(intersection) / union_count) if union_count else None,
    }
    for threshold in (0.10, 0.20, 0.50):
        precision = float(np.mean(pred_to_gt <= threshold)) if len(pred_to_gt) else 0.0
        recall = float(np.mean(gt_to_pred <= threshold)) if len(gt_to_pred) else 0.0
        result[f"precision_at_{threshold:.2f}m"] = precision
        result[f"recall_at_{threshold:.2f}m"] = recall
        result[f"f1_at_{threshold:.2f}m"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["pred_to_gt_distance_m"] = _distance_stats(pred_to_gt)
    result["gt_to_pred_distance_m"] = _distance_stats(gt_to_pred)
    return result


def _plot(
    out_path: Path,
    input_dir: Path,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    pred_labels: np.ndarray,
    gt_c2w: np.ndarray,
    profile_label: str,
    frame_start: int | None = None,
    frame_stop: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = _select_frame_range(_rgb_paths(input_dir), frame_start, frame_stop)
    chosen = () if not paths else (paths[0], paths[len(paths) // 2], paths[-1])
    ground_x, ground_y, up_axis = _ground_axes(gt_c2w[:, :3, 3])
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.78, 1.22))
    for column in range(3):
        axis = fig.add_subplot(grid[0, column])
        if chosen:
            axis.imshow(plt.imread(chosen[column]))
            axis.set_title(f"RGB frame: {chosen[column].name}")
        else:
            axis.text(0.5, 0.5, "RGB input not found", ha="center", va="center")
        axis.axis("off")

    gt_center = gt_c2w[:, :3, 3]
    gt_min, gt_max = gt_xyz.min(axis=0), gt_xyz.max(axis=0)
    inside = ((pred_xyz >= gt_min) & (pred_xyz <= gt_max)).all(axis=1)
    pred_local, label_local = pred_xyz[inside], pred_labels[inside]
    pred_sample = _sample_indices(label_local, max_points=100_000)
    gt_sample = np.linspace(0, len(gt_xyz) - 1, min(len(gt_xyz), 100_000), dtype=np.int64)

    pred_axis = fig.add_subplot(grid[1, 0])
    if len(pred_sample):
        colors = SEMANTIC_PALETTE_11[np.minimum(label_local[pred_sample], len(SEMANTIC_PALETTE_11) - 1)]
        pred_axis.scatter(
            pred_local[pred_sample, ground_x], pred_local[pred_sample, ground_y],
            c=colors, s=0.35, alpha=0.65, linewidths=0,
        )
    pred_axis.plot(gt_center[:, ground_x], gt_center[:, ground_y], "k.-", linewidth=1, markersize=2)
    pred_axis.set_title(f"Predicted semantic centers inside GT bounds ({int(inside.sum()):,})")

    gt_axis = fig.add_subplot(grid[1, 1])
    gt_axis.scatter(
        gt_xyz[gt_sample, ground_x], gt_xyz[gt_sample, ground_y],
        c=gt_xyz[gt_sample, up_axis], cmap="viridis", s=0.25, alpha=0.5, linewidths=0,
    )
    gt_axis.plot(gt_center[:, ground_x], gt_center[:, ground_y], "r.-", linewidth=1, markersize=2)
    gt_axis.set_title(f"Habitat observed depth surface ({len(gt_xyz):,} samples)")

    overlay_axis = fig.add_subplot(grid[1, 2])
    overlay_axis.scatter(
        gt_xyz[gt_sample, ground_x], gt_xyz[gt_sample, ground_y],
        c="0.72", s=0.25, alpha=0.35, linewidths=0, label="GT observed surface",
    )
    if len(pred_sample):
        overlay_axis.scatter(
            pred_local[pred_sample, ground_x], pred_local[pred_sample, ground_y],
            c="#d62728", s=0.4, alpha=0.45, linewidths=0, label="pred center",
        )
    overlay_axis.plot(gt_center[:, ground_x], gt_center[:, ground_y], "k-", linewidth=1, label="GT path")
    overlay_axis.set_title("Prediction / GT overlay")
    overlay_axis.legend(loc="best", markerscale=5)

    for axis in (pred_axis, gt_axis, overlay_axis):
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel(f"{'xyz'[ground_x]} (m)")
        axis.set_ylabel(f"{'xyz'[ground_y]} (m)")
    fig.suptitle(
        f"FreeOcc vs Habitat observed surface — {profile_label}\n"
        "Offline audit only; unobserved voxels remain unknown and are never counted as free",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--pred-ply", required=True, type=Path)
    parser.add_argument("--trajectory-npz", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--profile-label", default="unspecified")
    parser.add_argument("--fx", type=float, default=388.19104)
    parser.add_argument("--fy", type=float, default=388.19104)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--frame-start", type=int, default=None, help="inclusive numeric frame id")
    parser.add_argument("--frame-stop", type=int, default=None, help="exclusive numeric frame id")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_xyz, _gt_rgb, gt_meta = _load_observed_surface(
        args.input_dir, args.fx, args.fy, args.cx, args.cy,
        args.depth_scale, args.pixel_stride, args.max_depth_m,
        args.frame_start, args.frame_stop,
    )
    if not len(gt_xyz):
        raise ValueError("Habitat depth produced no valid observed surface points")
    pred_xyz, pred_labels, pred_meta = _semantic_gaussians(args.pred_ply)
    trajectory = np.load(args.trajectory_npz, allow_pickle=False)
    gt_c2w = np.asarray(trajectory["gt_c2w"], dtype=np.float64)

    margin = max(2 * args.voxel_size, 0.20)
    origin = np.floor((gt_xyz.min(axis=0) - margin) / args.voxel_size) * args.voxel_size
    maximum = np.ceil((gt_xyz.max(axis=0) + margin) / args.voxel_size) * args.voxel_size
    shape = np.maximum(np.ceil((maximum - origin) / args.voxel_size).astype(np.int64), 1)
    gt_indices = _voxelize(gt_xyz, origin, shape, args.voxel_size)
    pred_indices = _voxelize(pred_xyz, origin, shape, args.voxel_size)
    metrics = _surface_metrics(gt_indices, pred_indices, origin, shape, args.voxel_size)
    inside_pred = ((pred_xyz >= origin) & (pred_xyz < origin + shape * args.voxel_size)).all(axis=1)

    report: Dict[str, Any] = {
        "schema_version": "freeocc_habitat_observed_surface_v1",
        "offline_shadow_only": True,
        "profile_label": args.profile_label,
        "reference": "Habitat depth+pose observed surface; unobserved space is unknown, not free",
        "prediction": "Sim3-aligned FreeOcc Gaussian-center occupancy proxy",
        "semantic_gt_available": False,
        "semantic_miou": None,
        "semantic_metric_note": "No Habitat semantic frames/instance-category mapping were recorded in this sequence.",
        "intrinsics": {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy},
        "grid": {
            "origin": origin.tolist(),
            "shape": shape.tolist(),
            "voxel_size_m": args.voxel_size,
        },
        "gt": gt_meta,
        "prediction_metadata": pred_meta,
        "pred_gaussians_inside_gt_grid": int(inside_pred.sum()),
        "pred_gaussians_inside_gt_grid_fraction": float(inside_pred.mean()) if len(inside_pred) else 0.0,
        "metrics": metrics,
    }
    (args.out_dir / "habitat_gt_occ_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.out_dir / "habitat_observed_occ.npz",
        gt_occupied_indices=gt_indices,
        pred_occupied_indices=pred_indices,
        voxel_origin=origin,
        voxel_shape=shape,
        voxel_size=np.asarray(args.voxel_size, dtype=np.float32),
    )
    _plot(
        args.out_dir / "freeocc_rgb_pred_gt_occ.png",
        args.input_dir,
        gt_xyz,
        pred_xyz,
        pred_labels,
        gt_c2w,
        args.profile_label,
        args.frame_start,
        args.frame_stop,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

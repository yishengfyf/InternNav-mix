#!/usr/bin/env python3
"""Disambiguate Habitat-native and OpenCV camera pose conventions.

The check reprojects measured depth between different frames.  Unlike a
same-frame point cloud comparison, it cannot look correct merely because the
prediction and reference repeat the same camera-axis mistake.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


HABITAT_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _numeric_paths(folder: Path, suffixes: Iterable[str]) -> Sequence[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: (int(path.stem), path.name),
    )


def _load_sequence(input_dir: Path, depth_scale: float) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    import cv2

    color_paths = _numeric_paths(input_dir / "color", (".jpg", ".jpeg", ".png"))
    depth_paths = _numeric_paths(input_dir / "depth", (".png", ".npy"))
    pose_paths = _numeric_paths(input_dir / "pose", (".txt",))
    count = min(len(color_paths), len(depth_paths), len(pose_paths))
    if count < 2:
        raise ValueError(f"need at least two color/depth/pose triplets under {input_dir}")
    colors, depths, poses = [], [], []
    for color_path, depth_path, pose_path in zip(color_paths[:count], depth_paths[:count], pose_paths[:count]):
        colors.append(cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB))
        if depth_path.suffix.lower() == ".npy":
            depths.append(np.load(depth_path).astype(np.float64))
        else:
            depths.append(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float64) / depth_scale)
        poses.append(np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4))
    return colors, depths, poses


def _candidate_poses(poses: Sequence[np.ndarray], conversion: np.ndarray) -> list[np.ndarray]:
    return [pose @ conversion for pose in poses]


def _reproject(
    source_color: np.ndarray,
    source_depth: np.ndarray,
    source_c2w: np.ndarray,
    target_color: np.ndarray,
    target_depth: np.ndarray,
    target_c2w: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    pixel_stride: int,
    max_depth_m: float,
) -> Dict[str, np.ndarray | int]:
    fx, fy, cx, cy = intrinsics
    height, width = source_depth.shape
    vv, uu = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    depth = source_depth[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    z = depth[valid]
    source_pixels = np.column_stack((uu[valid], vv[valid]))
    camera = np.column_stack(
        ((source_pixels[:, 0] - cx) * z / fx, (source_pixels[:, 1] - cy) * z / fy, z, np.ones_like(z))
    )
    target_camera = (np.linalg.inv(target_c2w) @ source_c2w @ camera.T).T
    positive = np.isfinite(target_camera).all(axis=1) & (target_camera[:, 2] > 0.0)
    target_camera = target_camera[positive]
    source_pixels = source_pixels[positive]
    projected_u = np.rint(fx * target_camera[:, 0] / target_camera[:, 2] + cx).astype(np.int64)
    projected_v = np.rint(fy * target_camera[:, 1] / target_camera[:, 2] + cy).astype(np.int64)
    inside = (
        (projected_u >= 0) & (projected_u < width) & (projected_v >= 0) & (projected_v < height)
    )
    projected_u, projected_v = projected_u[inside], projected_v[inside]
    source_pixels = source_pixels[inside].astype(np.int64)
    projected_z = target_camera[inside, 2]
    measured_z = target_depth[projected_v, projected_u]
    observed = np.isfinite(measured_z) & (measured_z > 0.0) & (measured_z <= max_depth_m)
    projected_u, projected_v = projected_u[observed], projected_v[observed]
    source_pixels = source_pixels[observed]
    projected_z, measured_z = projected_z[observed], measured_z[observed]
    relative_error = np.abs(projected_z - measured_z) / np.maximum(measured_z, 1e-6)
    source_rgb = source_color[source_pixels[:, 1], source_pixels[:, 0]].astype(np.float64) / 255.0
    target_rgb = target_color[projected_v, projected_u].astype(np.float64) / 255.0
    photometric_l1 = np.mean(np.abs(source_rgb - target_rgb), axis=1)
    return {
        "source_samples": int(valid.sum()),
        "target_observed": int(len(relative_error)),
        "projected_u": projected_u,
        "projected_v": projected_v,
        "projected_z": projected_z,
        "measured_z": measured_z,
        "relative_depth_error": relative_error,
        "photometric_l1": photometric_l1,
        "source_rgb": source_rgb,
    }


def _summarize(results: Sequence[Dict[str, np.ndarray | int]]) -> Dict[str, float | int | None]:
    observed = sum(int(result["target_observed"]) for result in results)
    source = sum(int(result["source_samples"]) for result in results)
    if not observed:
        return {"source_samples": source, "target_observed": 0}
    depth_error = np.concatenate([np.asarray(result["relative_depth_error"]) for result in results])
    photo_error = np.concatenate([np.asarray(result["photometric_l1"]) for result in results])
    consistent_10 = depth_error < 0.10
    return {
        "source_samples": source,
        "target_observed": observed,
        "target_observed_fraction": float(observed / max(source, 1)),
        "depth_consistent_5pct": float(np.mean(depth_error < 0.05)),
        "depth_consistent_10pct": float(np.mean(consistent_10)),
        "relative_depth_error_median": float(np.median(depth_error)),
        "relative_depth_error_p90": float(np.quantile(depth_error, 0.9)),
        "photometric_l1_on_depth_consistent": (
            float(np.mean(photo_error[consistent_10])) if consistent_10.any() else None
        ),
    }


def _plot_example(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    direct: Dict[str, np.ndarray | int],
    converted: Dict[str, np.ndarray | int],
    pair: Tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].imshow(source)
    axes[0, 0].set_title(f"Source RGB frame {pair[0]}")
    axes[0, 1].imshow(target)
    axes[0, 1].set_title(f"Target RGB frame {pair[1]}")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.95,
        "Cross-frame measured-depth reprojection\n"
        "breaks the same-error-on-both-sides ambiguity.\n\n"
        "Direct: saved pose is already OpenCV c2w.\n"
        "Axis-flipped: saved pose is Habitat-native c2w\n"
        "and requires diag(1,-1,-1).",
        va="top",
        fontsize=12,
    )

    for row, (title, result) in enumerate((
        ("Direct saved c2w", direct),
        ("Apply Habitat→OpenCV axis flip", converted),
    )):
        axis = axes[1, row]
        axis.imshow(target)
        error = np.asarray(result["relative_depth_error"])
        u = np.asarray(result["projected_u"])
        v = np.asarray(result["projected_v"])
        sample = np.arange(len(error))[:: max(1, len(error) // 12_000)]
        scatter = axis.scatter(u[sample], v[sample], c=np.clip(error[sample], 0, 0.5), s=3, cmap="turbo", vmin=0, vmax=0.5)
        consistent = float(np.mean(error < 0.10)) if len(error) else 0.0
        axis.set_title(f"{title}\ndepth agreement <10%: {consistent:.1%}")
        fig.colorbar(scatter, ax=axis, fraction=0.046, label="relative depth error")

    axes[1, 2].axis("off")
    labels = ["direct", "axis-flipped"]
    values = []
    for result in (direct, converted):
        error = np.asarray(result["relative_depth_error"])
        values.append(float(np.mean(error < 0.10)) if len(error) else 0.0)
    inset = axes[1, 2].inset_axes([0.12, 0.18, 0.78, 0.68])
    inset.bar(labels, values, color=["#2ca02c", "#d62728"])
    inset.set_ylim(0, 1)
    inset.set_ylabel("depth agreement <10%")
    inset.set_title("Example-pair convention check")
    for axis in axes.flat[:2]:
        axis.axis("off")
    fig.suptitle("Habitat / FreeOcc camera-coordinate audit (offline only)", fontsize=16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(
    input_dir: Path,
    out_dir: Path,
    intrinsics: Tuple[float, float, float, float],
    depth_scale: float = 1000.0,
    pixel_stride: int = 8,
    max_depth_m: float = 10.0,
    max_pair_gap: int = 5,
) -> Dict[str, object]:
    colors, depths, poses = _load_sequence(input_dir, depth_scale)
    candidates = {
        "saved_pose_is_opencv_c2w": _candidate_poses(poses, np.eye(4)),
        "saved_pose_is_habitat_c2w_apply_axis_flip": _candidate_poses(poses, HABITAT_CAMERA_TO_OPENCV),
    }
    pairs = [(i, j) for i in range(len(poses)) for j in range(i + 1, min(len(poses), i + max_pair_gap + 1))]
    all_results: Dict[str, list[Dict[str, np.ndarray | int]]] = {key: [] for key in candidates}
    for key, candidate_poses in candidates.items():
        for i, j in pairs:
            all_results[key].append(
                _reproject(
                    colors[i], depths[i], candidate_poses[i], colors[j], depths[j], candidate_poses[j],
                    intrinsics, pixel_stride, max_depth_m,
                )
            )
    candidate_metrics = {key: _summarize(results) for key, results in all_results.items()}
    winner = max(candidate_metrics, key=lambda key: float(candidate_metrics[key].get("depth_consistent_10pct", 0.0)))
    direct_score = float(candidate_metrics["saved_pose_is_opencv_c2w"].get("depth_consistent_10pct", 0.0))
    flipped_score = float(candidate_metrics["saved_pose_is_habitat_c2w_apply_axis_flip"].get("depth_consistent_10pct", 0.0))
    summary: Dict[str, object] = {
        "schema_version": "habitat_camera_convention_audit_v1",
        "offline_shadow_only": True,
        "input_frames": len(poses),
        "cross_frame_pairs": len(pairs),
        "pixel_stride": pixel_stride,
        "intrinsics": {"fx": intrinsics[0], "fy": intrinsics[1], "cx": intrinsics[2], "cy": intrinsics[3]},
        "candidates": candidate_metrics,
        "winner": winner,
        "direct_to_flipped_depth_agreement_ratio": direct_score / max(flipped_score, 1e-12),
        "interpretation": (
            "saved pose matrices already use OpenCV camera axes (+z forward, +y image-down)"
            if winner == "saved_pose_is_opencv_c2w"
            else "saved pose matrices appear Habitat-native and need the camera-axis flip"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "habitat_camera_convention_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    example_index = max(range(len(pairs)), key=lambda index: int(all_results[winner][index]["target_observed"]))
    i, j = pairs[example_index]
    _plot_example(
        out_dir / "habitat_camera_convention_audit.png",
        colors[i], colors[j],
        all_results["saved_pose_is_opencv_c2w"][example_index],
        all_results["saved_pose_is_habitat_c2w_apply_axis_flip"][example_index],
        (i, j),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fx", type=float, default=388.19104)
    parser.add_argument("--fy", type=float, default=388.19104)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--max-pair-gap", type=int, default=5)
    args = parser.parse_args()
    result = analyze(
        args.input_dir, args.out_dir, (args.fx, args.fy, args.cx, args.cy),
        args.depth_scale, args.pixel_stride, args.max_depth_m, args.max_pair_gap,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

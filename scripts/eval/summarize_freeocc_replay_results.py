#!/usr/bin/env python3
"""Summarize completed FreeOcc/Habitat replay audits.

Each ``--run`` points at an analysis directory containing
``freeocc_mapping_summary.json`` and ``habitat_gt_occ_metrics.json``.  The
summary deliberately reports observed-surface metrics only; it is not a
navigation-safety score and semantic mIoU remains null without Habitat labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"expected NAME=ANALYSIS_DIR, got {spec!r}")
    name, raw = spec.split("=", 1)
    root = Path(raw)
    mapping = json.loads((root / "freeocc_mapping_summary.json").read_text())
    gt = json.loads((root / "habitat_gt_occ_metrics.json").read_text())
    metrics = gt.get("metrics", {})
    trajectory = mapping.get("trajectory", {})
    ply = mapping.get("ply_aligned", {})
    extent = ply.get("bbox_extent") or []
    return {
        "profile": name,
        "analysis_dir": str(root),
        "input_frames": mapping.get("expected_input_frames"),
        "mapper_cameras": mapping.get("mapper_camera_count"),
        "gaussians": mapping.get("final_mapping_call", {}).get("total_gaussians"),
        "ply_bytes": ply.get("bytes"),
        "bbox_max_extent_m": max(extent) if extent else None,
        "trajectory_post_rmse_m": trajectory.get("post_rmse_m"),
        "trajectory_path_length_m": trajectory.get("aligned_path_length_m"),
        "exact_surface_iou": metrics.get("exact_iou"),
        "precision_at_0.10m": metrics.get("precision_at_0.10m"),
        "recall_at_0.10m": metrics.get("recall_at_0.10m"),
        "f1_at_0.10m": metrics.get("f1_at_0.10m"),
        "f1_at_0.20m": metrics.get("f1_at_0.20m"),
        "f1_at_0.50m": metrics.get("f1_at_0.50m"),
        "pred_inside_gt_grid_fraction": gt.get("pred_gaussians_inside_gt_grid_fraction"),
        "semantic_miou": gt.get("semantic_miou"),
    }


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["profile"] for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(x, [max(1, row["gaussians"] or 0) for row in rows], color="#5470c6")
    axes[0].set_yscale("log")
    axes[0].set_title("Gaussian centers (log)")
    axes[1].bar(x, [row["exact_surface_iou"] or 0 for row in rows], label="exact IoU")
    axes[1].bar(x, [row["f1_at_0.20m"] or 0 for row in rows], alpha=0.75, label="F1 @ 0.20 m")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Observed-surface agreement")
    axes[1].legend()
    axes[2].bar(x, [row["recall_at_0.10m"] or 0 for row in rows], label="recall @ 0.10 m")
    axes[2].bar(x, [row["precision_at_0.10m"] or 0 for row in rows], alpha=0.75, label="precision @ 0.10 m")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Density versus accuracy")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("FreeOcc replay audit — GT pose/depth oracle, offline only")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=ANALYSIS_DIR")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_load(spec) for spec in args.run]
    report = {
        "schema_version": "freeocc_replay_audit_summary_v1",
        "offline_shadow_only": True,
        "semantic_gt_available": False,
        "rows": rows,
        "metric_definition": "Habitat observed surface from matching RGB-D frames; unknown voxels are excluded.",
    }
    (args.out_dir / "freeocc_replay_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(args.out_dir / "freeocc_replay_summary.png", rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

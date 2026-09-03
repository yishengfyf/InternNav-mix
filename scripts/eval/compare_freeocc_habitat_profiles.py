#!/usr/bin/env python3
"""Create a compact comparison of bounded FreeOcc Habitat audit profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load(spec: str) -> Dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"expected NAME=RUN_DIR, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    run_dir = Path(raw_path)
    summary = json.loads((run_dir / "analysis" / "freeocc_mapping_summary.json").read_text())
    gt_path = run_dir / "analysis" / "habitat_gt_occ_metrics.json"
    gt = json.loads(gt_path.read_text()) if gt_path.is_file() else {}
    return {"name": name, "run_dir": str(run_dir), "summary": summary, "gt": gt}


def _row(item: Dict[str, Any]) -> Dict[str, Any]:
    summary, gt = item["summary"], item["gt"]
    filters = summary.get("filter_latest_by_frame", {})
    trajectory = summary.get("trajectory", {})
    ply = summary.get("ply_aligned", {})
    metrics = gt.get("metrics", {})
    extent = ply.get("bbox_extent") or []
    return {
        "profile": item["name"],
        "diagnosis": summary.get("diagnosis"),
        "final_valid": filters.get("final_valid"),
        "final_retention": filters.get("final_retention"),
        "gaussians": summary.get("final_mapping_call", {}).get("total_gaussians"),
        "bbox_max_extent_m": max(extent) if extent else None,
        "trajectory_post_rmse_m": trajectory.get("post_rmse_m"),
        "trajectory_length_ratio": (
            trajectory.get("aligned_path_length_m") / trajectory.get("gt_path_length_m")
            if trajectory.get("aligned_path_length_m") is not None and trajectory.get("gt_path_length_m")
            else None
        ),
        "fps": summary.get("runtime", {}).get("total_fps"),
        "pred_inside_gt_grid_fraction": gt.get("pred_gaussians_inside_gt_grid_fraction"),
        "surface_exact_iou": metrics.get("exact_iou"),
        "surface_f1_at_0.20m": metrics.get("f1_at_0.20m"),
        "surface_f1_at_0.50m": metrics.get("f1_at_0.50m"),
        "semantic_miou": gt.get("semantic_miou"),
    }


def _values(rows, key):
    return np.asarray([np.nan if row.get(key) is None else row[key] for row in rows], dtype=np.float64)


def _plot(path: Path, rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["profile"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    width = 0.26
    for offset, key, label in (
        (-width, "final_valid", "final valid pixels"),
        (0.0, "gaussians", "Gaussian centers"),
    ):
        axes[0, 0].bar(x + offset, np.maximum(_values(rows, key), 1), width, label=label)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Geometry retained (log scale)")
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, _values(rows, "bbox_max_extent_m"), width, label="max bbox extent (m)")
    axes[0, 1].bar(
        x + width / 2, 100 * _values(rows, "pred_inside_gt_grid_fraction"), width,
        label="centers inside GT grid (%)",
    )
    axes[0, 1].set_title("Spatial plausibility")
    axes[0, 1].legend()

    axes[1, 0].bar(x - width / 2, _values(rows, "trajectory_post_rmse_m"), width, label="trajectory RMSE (m)")
    axes[1, 0].bar(x + width / 2, _values(rows, "trajectory_length_ratio"), width, label="path length ratio")
    axes[1, 0].set_title("DROID trajectory after offline Sim3")
    axes[1, 0].legend()

    axes[1, 1].bar(x - width, _values(rows, "surface_exact_iou"), width, label="exact surface IoU")
    axes[1, 1].bar(x, _values(rows, "surface_f1_at_0.20m"), width, label="surface F1 @ 0.20m")
    axes[1, 1].bar(x + width, _values(rows, "surface_f1_at_0.50m"), width, label="surface F1 @ 0.50m")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Observed-surface agreement")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "FreeOcc Habitat RGB-only profile audit\n"
        "Relaxed filters are diagnostics only; unknown space remains unknown",
        fontsize=15,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(_load(spec)) for spec in args.run]
    report = {
        "schema_version": "freeocc_habitat_profile_comparison_v1",
        "offline_shadow_only": True,
        "rows": rows,
        "conclusion_guardrail": "No relaxed profile grants navigation safety authority.",
    }
    (args.out_dir / "freeocc_profile_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(args.out_dir / "freeocc_profile_comparison.png", rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

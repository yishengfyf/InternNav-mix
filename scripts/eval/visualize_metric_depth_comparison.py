#!/usr/bin/env python3
"""Visualize RGB, Habitat GT, and one or more metric-depth providers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _paths(folder: Path, suffixes: set[str]) -> dict[int, Path]:
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes and path.stem.isdigit():
            result[int(path.stem)] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=OUTPUT_DIR")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    args = parser.parse_args()

    rgb = _paths(args.input_dir / "color", {".jpg", ".jpeg", ".png"})
    gt = _paths(args.input_dir / "depth", {".png"})
    providers = []
    for spec in args.prediction:
        if "=" not in spec:
            raise ValueError(f"expected NAME=OUTPUT_DIR, got {spec!r}")
        name, raw = spec.split("=", 1)
        root = Path(raw)
        manifest = json.loads((root / "metric_depth_manifest.json").read_text())
        providers.append((name, _paths(root / "depth", {".png"}), manifest))
    common = set(rgb) & set(gt)
    for _, paths, _ in providers:
        common &= set(paths)
    frame_ids = sorted(common)
    if not frame_ids:
        raise ValueError("no common RGB/GT/predicted frame ids")
    chosen = [frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 2 + 2 * len(providers)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.6 * rows), constrained_layout=True)
    for column, frame_id in enumerate(chosen):
        image = cv2.cvtColor(cv2.imread(str(rgb[frame_id])), cv2.COLOR_BGR2RGB)
        target = cv2.imread(str(gt[frame_id]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        axes[0, column].imshow(image)
        axes[0, column].set_title(f"RGB frame {frame_id}")
        im = axes[1, column].imshow(target, vmin=0, vmax=args.max_depth_m, cmap="turbo")
        axes[1, column].set_title("Habitat GT depth (m)")
        for provider_index, (name, paths, manifest) in enumerate(providers):
            prediction = cv2.imread(str(paths[frame_id]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            pred_row = 2 + 2 * provider_index
            err_row = pred_row + 1
            axes[pred_row, column].imshow(prediction, vmin=0, vmax=args.max_depth_m, cmap="turbo")
            audit = manifest.get("gt_depth_audit") or {}
            axes[pred_row, column].set_title(
                f"{name}: pred depth (m)\nAbsRel={audit.get('abs_rel', float('nan')):.3f}, "
                f"delta1={audit.get('delta1', float('nan')):.3f}"
            )
            valid = (target > 0) & np.isfinite(target)
            error = np.where(valid, np.abs(prediction - target), np.nan)
            axes[err_row, column].imshow(error, vmin=0, vmax=2.0, cmap="magma")
            axes[err_row, column].set_title(f"{name}: absolute error (m)")
        for row in range(rows):
            axes[row, column].axis("off")
    fig.colorbar(im, ax=axes[:, :], shrink=0.35, label="depth (m); error panels clipped at 2 m")
    fig.suptitle("RGB-only metric depth audit against Habitat GT (GT used only for evaluation)", fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

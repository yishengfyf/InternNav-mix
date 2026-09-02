#!/usr/bin/env python3
"""Evaluate/visualize a FreeOcc occupancy artifact against Habitat/MP3D GT.

This is intentionally offline.  It consumes ``occ.npz`` files and RGB frames,
so it cannot alter the DualVLN evaluator or online SparseOcc authority.
The expected NPZ fields are ``pred`` (or ``labels``), optional ``valid_mask``,
``voxel_origin`` and ``voxel_size``.  GT uses the same convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _load(path: str):
    d = np.load(path, allow_pickle=False)
    key = "pred" if "pred" in d else "labels"
    if key not in d:
        raise ValueError(f"{path}: missing pred/labels")
    labels = np.asarray(d[key])
    valid = np.asarray(d["valid_mask"], bool) if "valid_mask" in d else np.ones_like(labels, bool)
    if labels.shape != valid.shape:
        raise ValueError(f"{path}: labels/valid_mask shape mismatch")
    origin = np.asarray(d["voxel_origin"], np.float32).reshape(3) if "voxel_origin" in d else np.zeros(3, np.float32)
    size = float(np.asarray(d["voxel_size"]).reshape(-1)[0]) if "voxel_size" in d else 0.08
    return labels, valid, origin, size


def _metrics(pred, pvalid, gt, gvalid) -> Dict[str, object]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}; resample externally")
    mask = pvalid & gvalid
    po, go = pred > 0, gt > 0
    inter, union = np.count_nonzero(mask & po & go), np.count_nonzero(mask & (po | go))
    out: Dict[str, object] = {"valid_voxels": int(mask.sum()), "occupancy_iou": float(inter / union) if union else None}
    classes = sorted(set(np.unique(gt[mask]).tolist()) | set(np.unique(pred[mask]).tolist()))
    per = {}
    vals = []
    for c in classes:
        if int(c) <= 0:
            continue
        a, b = mask & (pred == c), mask & (gt == c)
        i, u = np.count_nonzero(a & b), np.count_nonzero(a | b)
        if u:
            per[str(int(c))] = float(i / u)
            vals.append(i / u)
    out["semantic_iou_by_label"] = per
    out["semantic_miou"] = float(np.mean(vals)) if vals else None
    return out


def _render(pred, valid, origin, size, rgb_dir: Optional[str], out_dir: Path, max_points: int = 50000) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("visualization requires matplotlib") from exc
    vox = np.argwhere(valid & (pred > 0))
    if len(vox) > max_points:
        rng = np.random.default_rng(0)
        vox = vox[rng.choice(len(vox), max_points, replace=False)]
    xyz = origin[None] + (vox.astype(np.float32) + 0.5) * size
    labels = pred[tuple(vox.T)] if len(vox) else np.empty((0,), np.int32)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")
    if len(xyz):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=labels, s=2, cmap="tab20", alpha=0.75)
    ax.set_title("FreeOcc semantic occupancy (occupied voxels)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax2 = fig.add_subplot(122)
    if rgb_dir:
        from PIL import Image
        files = sorted(Path(rgb_dir).glob("*.png")) + sorted(Path(rgb_dir).glob("*.jpg"))
        if files:
            ax2.imshow(Image.open(files[len(files) // 2]).convert("RGB"))
            ax2.set_title(f"RGB frame ({files[len(files)//2].name})")
        else:
            ax2.text(0.5, 0.5, "No RGB frames", ha="center", va="center")
    else:
        ax2.text(0.5, 0.5, "Pass --rgb-dir for RGB comparison", ha="center", va="center")
    ax2.axis("off")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "freeocc_rgb_occ_overview.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--gt-npz", default="")
    ap.add_argument("--rgb-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    pred, pv, origin, size = _load(args.pred_npz)
    out = {"pred_npz": str(args.pred_npz), "voxel_origin": origin.tolist(), "voxel_size": size,
           "shape": list(pred.shape), "occupied_voxels": int(np.count_nonzero(pv & (pred > 0)))}
    if args.gt_npz:
        gt, gv, go, gs = _load(args.gt_npz)
        if not np.allclose(origin, go) or abs(size - gs) > 1e-6:
            out["warning"] = "pred/gt origin or voxel_size differ; metrics still require matching grids"
        out["metrics"] = _metrics(pred, pv, gt, gv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "freeocc_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _render(pred, pv, origin, size, args.rgb_dir or None, out_dir)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

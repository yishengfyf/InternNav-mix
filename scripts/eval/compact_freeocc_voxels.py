#!/usr/bin/env python3
"""Convert a large FreeOcc semantic Gaussian PLY into a compact voxel map.

The compact map is intended for shadow safety queries and visualization.  It
does not fill unobserved space and therefore cannot replace SparseOcc without
further validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.eval.analyze_freeocc_mapping_audit import _semantic_gaussians


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-ply", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    args = parser.parse_args()
    xyz, labels, metadata = _semantic_gaussians(args.pred_ply)
    if not len(xyz):
        raise ValueError("PLY contains no finite semantic Gaussians")
    size = float(args.voxel_size)
    origin = np.floor(xyz.min(axis=0) / size) * size
    indices = np.floor((xyz - origin) / size).astype(np.int32)
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
    sorted_idx = indices[order]
    starts = np.r_[0, np.flatnonzero(np.any(sorted_idx[1:] != sorted_idx[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(sorted_idx)]
    unique_idx = sorted_idx[starts]
    voxel_labels = np.empty(len(starts), dtype=np.int16)
    voxel_counts = np.empty(len(starts), dtype=np.int32)
    for i, (start, end) in enumerate(zip(starts, ends)):
        values, counts = np.unique(labels[order[start:end]], return_counts=True)
        voxel_labels[i] = values[np.argmax(counts)]
        voxel_counts[i] = end - start
    centers = origin + (unique_idx.astype(np.float32) + 0.5) * size
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        voxel_indices=unique_idx,
        voxel_centers=centers,
        semantic_labels=voxel_labels,
        support_count=voxel_counts,
        voxel_origin=origin.astype(np.float32),
        voxel_size=np.asarray(size, dtype=np.float32),
    )
    report = {
        "schema_version": "freeocc_compact_semantic_voxels_v1",
        "source_ply": str(args.pred_ply),
        "source_gaussians": int(len(xyz)),
        "occupied_voxels": int(len(unique_idx)),
        "compression_ratio": float(len(xyz) / len(unique_idx)),
        "voxel_size_m": size,
        "origin": origin.tolist(),
        "semantic_metadata": metadata,
    }
    report_path = args.out_npz.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

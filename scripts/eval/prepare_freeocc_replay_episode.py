#!/usr/bin/env python3
"""Convert an audit replay-ledger episode to the FreeOcc folder format.

This is an offline utility.  It copies the RGB/depth files already recorded by
DualVLN and writes the saved Habitat camera pose as ``pose/*.txt``.  No online
evaluator state or safety decision is modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True, help="episode observations.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
    color_dir, depth_dir, pose_dir = (args.out_dir / name for name in ("color", "depth", "pose"))
    for folder in (color_dir, depth_dir, pose_dir):
        folder.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        pose = row.get("pose") or {}
        matrix = pose.get("stage23_gt_camera_pose_map")
        rgb_rel, depth_rel = row.get("rgb_path"), row.get("depth_path")
        if not matrix or not rgb_rel or not depth_rel:
            continue
        rgb_path, depth_path = args.ledger.parent / rgb_rel, args.ledger.parent / depth_rel
        if rgb_path.is_file() and depth_path.is_file():
            selected.append((row, rgb_path, depth_path, np.asarray(matrix, dtype=np.float64)))
    selected = selected[args.start_index : args.start_index + max(1, args.max_frames)]
    if len(selected) < 3:
        raise SystemExit(f"only {len(selected)} usable RGB-D-pose frames found")
    manifest = []
    for out_id, (row, rgb_path, depth_path, matrix) in enumerate(selected):
        stem = f"{out_id:06d}"
        shutil.copy2(rgb_path, color_dir / f"{stem}{rgb_path.suffix.lower()}")
        with np.load(depth_path, allow_pickle=False) as data:
            if "depth_m" not in data:
                raise ValueError(f"{depth_path} has no depth_m array")
            depth = np.asarray(data["depth_m"], dtype=np.float32)
        # FreeOcc's ScanNet loader interprets PNG depth in millimetres.
        depth_mm = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535).astype(np.uint16)
        import cv2

        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
        np.savetxt(pose_dir / f"{stem}.txt", matrix, fmt="%.10f")
        manifest.append({"output_frame": out_id, "source_step": row.get("step_id"), "source_record_index": row.get("record_index"), "source_rgb": str(rgb_path), "source_depth": str(depth_path)})
    (args.out_dir / "replay_manifest.json").write_text(json.dumps({"schema_version": "freeocc_replay_conversion_v1", "source_ledger": str(args.ledger), "frames": manifest}, indent=2, ensure_ascii=False))
    print(json.dumps({"output_dir": str(args.out_dir), "frames": len(manifest), "first_source_step": manifest[0]["source_step"], "last_source_step": manifest[-1]["source_step"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

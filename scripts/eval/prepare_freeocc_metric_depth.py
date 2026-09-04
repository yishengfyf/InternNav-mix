#!/usr/bin/env python3
"""Prepare RGB-only replay data with predicted metric depth for FreeOcc.

The source sensor remains RGB-only.  Depth Anything V2 Metric (Hypersim) runs
as a separate preprocessing/worker stage and writes millimetre PNG files that
FreeOcc can consume through its mature RGB-D path.  Source GT depth and pose
are copied only when explicitly requested for offline audit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


def _numeric_images(folder: Path) -> Sequence[Path]:
    paths = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(paths, key=lambda path: (int(path.stem) if path.stem.isdigit() else 2**31, path.name))


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _depth_metrics(pred: np.ndarray, target: np.ndarray, max_depth_m: float) -> dict[str, float | int | None]:
    valid = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0) & (target <= max_depth_m)
    if not np.any(valid):
        return {"valid_pixels": 0, "abs_rel": None, "rmse_m": None, "delta1": None}
    p, t = pred[valid].astype(np.float64), target[valid].astype(np.float64)
    ratio = np.maximum(p / t, t / p)
    return {
        "valid_pixels": int(len(p)),
        "abs_rel": float(np.mean(np.abs(p - t) / t)),
        "rmse_m": float(np.sqrt(np.mean((p - t) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "pred_mean_m": float(np.mean(p)),
        "gt_mean_m": float(np.mean(t)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--model-max-depth-m", type=float, default=20.0)
    parser.add_argument("--output-max-depth-m", type=float, default=10.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--audit-gt-depth", action="store_true")
    parser.add_argument("--copy-audit-poses", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    color_source = args.input_dir / "color"
    paths = _numeric_images(color_source)
    paths = [path for path in paths if int(path.stem) >= args.frame_start]
    if args.frame_stop is not None:
        paths = [path for path in paths if int(path.stem) < args.frame_stop]
    if not paths:
        raise ValueError(f"no numeric RGB images found under {color_source}")

    color_out, depth_out = args.output_dir / "color", args.output_dir / "depth"
    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)
    for extra in ("intrinsic",):
        source = args.input_dir / extra
        target = args.output_dir / extra
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(source, target)
    if args.copy_audit_poses and (args.input_dir / "pose").is_dir():
        pose_out = args.output_dir / "pose"
        pose_out.mkdir(parents=True, exist_ok=True)
        for path in paths:
            source = args.input_dir / "pose" / f"{int(path.stem):06d}.txt"
            if not source.is_file():
                source = args.input_dir / "pose" / f"{int(path.stem)}.txt"
            if source.is_file():
                _link_or_copy(source, pose_out / f"{int(path.stem):06d}.txt")

    model = DepthAnythingV2(**MODEL_CONFIGS["vits"], max_depth=args.model_max_depth_m)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()

    rows = []
    aggregate_pred, aggregate_gt = [], []
    gt_dir = args.input_dir / "depth"
    with torch.inference_mode():
        for ordinal, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read {path}")
            tensor, (height, width) = model.image2tensor(image, input_size=args.input_size)
            tensor = tensor.to(device)
            prediction = model.forward(tensor)
            prediction = F.interpolate(
                prediction[:, None], (height, width), mode="bilinear", align_corners=True
            )[0, 0].float().cpu().numpy()
            prediction = prediction.astype(np.float32)
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
            prediction = np.clip(prediction, 0.0, args.output_max_depth_m)
            millimetres = np.rint(prediction * 1000.0).astype(np.uint16)
            target_depth = depth_out / f"{int(path.stem):06d}.png"
            if not cv2.imwrite(str(target_depth), millimetres):
                raise OSError(f"failed to write {target_depth}")
            _link_or_copy(path, color_out / f"{int(path.stem):06d}.jpg")
            row: dict[str, object] = {
                "frame": int(path.stem),
                "pred_min_m": float(prediction[prediction > 0].min()) if np.any(prediction > 0) else None,
                "pred_mean_m": float(prediction.mean()),
                "pred_max_m": float(prediction.max()),
            }
            if args.audit_gt_depth:
                gt_path = gt_dir / f"{int(path.stem):06d}.png"
                if not gt_path.is_file():
                    gt_path = gt_dir / f"{int(path.stem)}.png"
                if gt_path.is_file():
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                    frame_metrics = _depth_metrics(prediction, gt, args.output_max_depth_m)
                    row["gt_metrics"] = frame_metrics
                    valid = np.isfinite(gt) & (gt > 0) & (gt <= args.output_max_depth_m)
                    aggregate_pred.append(prediction[valid])
                    aggregate_gt.append(gt[valid])
            rows.append(row)
            print(f"[{ordinal + 1}/{len(paths)}] frame={path.stem} mean_depth={prediction.mean():.3f}m", flush=True)

    overall = None
    if aggregate_gt:
        overall = _depth_metrics(
            np.concatenate(aggregate_pred), np.concatenate(aggregate_gt), args.output_max_depth_m
        )
    report = {
        "schema_version": "freeocc_rgb_metric_depth_v1",
        "source_modality": "rgb_only",
        "provider": "Depth Anything V2 Metric Hypersim Small",
        "checkpoint": str(args.checkpoint),
        "frames": len(rows),
        "frame_start": args.frame_start,
        "frame_stop": args.frame_stop,
        "input_size": args.input_size,
        "model_max_depth_m": args.model_max_depth_m,
        "output_max_depth_m": args.output_max_depth_m,
        "depth_png_scale": 1000.0,
        "gt_depth_used_for_inference": False,
        "gt_pose_used_for_inference": False,
        "gt_depth_audit": overall,
        "per_frame": rows,
    }
    (args.output_dir / "metric_depth_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

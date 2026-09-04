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
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}

Predictor = Callable[[np.ndarray], np.ndarray]


def _load_depth_anything(args: argparse.Namespace, device: torch.device) -> tuple[Predictor, dict[str, Any]]:
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2

    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(f"Depth Anything checkpoint not found: {args.checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS["vits"], max_depth=args.model_max_depth_m)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    def predict(image_bgr: np.ndarray) -> np.ndarray:
        tensor, (height, width) = model.image2tensor(image_bgr, input_size=args.input_size)
        prediction = model.forward(tensor.to(device))
        return F.interpolate(
            prediction[:, None], (height, width), mode="bilinear", align_corners=True
        )[0, 0].float().cpu().numpy()

    return predict, {
        "provider": "depth_anything_v2_metric_hypersim_small",
        "checkpoint": str(args.checkpoint),
        "input_size": args.input_size,
    }


def _load_metric3d(args: argparse.Namespace, device: torch.device) -> tuple[Predictor, dict[str, Any]]:
    if args.metric3d_root is None or not (args.metric3d_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Metric3D repository not found: {args.metric3d_root}")
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(f"Metric3D checkpoint not found: {args.checkpoint}")
    model = torch.hub.load(
        str(args.metric3d_root.resolve()), "metric3d_vit_giant2", source="local", pretrain=False
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    input_height, input_width = args.metric3d_input_height, args.metric3d_input_width
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device=device).float()[:, None, None]

    def predict(image_bgr: np.ndarray) -> np.ndarray:
        rgb_origin = image_bgr[:, :, ::-1]
        height, width = rgb_origin.shape[:2]
        scale = min(input_height / height, input_width / width)
        resized = cv2.resize(
            rgb_origin, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR
        )
        resized_height, resized_width = resized.shape[:2]
        pad_height, pad_width = input_height - resized_height, input_width - resized_width
        top, left = pad_height // 2, pad_width // 2
        bottom, right = pad_height - top, pad_width - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53]
        )
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().to(device)
        tensor = ((tensor - mean) / std)[None]
        prediction, _confidence, _output = model.inference({"input": tensor})
        prediction = prediction.squeeze()[top : input_height - bottom, left : input_width - right]
        prediction = F.interpolate(
            prediction[None, None], (height, width), mode="bilinear", align_corners=False
        )[0, 0]
        # Metric3D predicts in a canonical camera with focal length 1000 px.
        prediction = prediction * ((args.fx * scale) / 1000.0)
        return prediction.float().cpu().numpy()

    return predict, {
        "provider": "metric3d_v2_vit_giant2",
        "checkpoint": str(args.checkpoint),
        "metric3d_root": str(args.metric3d_root),
        "input_size": [input_height, input_width],
        "fx": args.fx,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
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
    parser.add_argument(
        "--provider", choices=("depth_anything_v2_hypersim_small", "metric3d_vit_giant2"),
        default="depth_anything_v2_hypersim_small",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metric3d-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--metric3d-input-height", type=int, default=616)
    parser.add_argument("--metric3d-input-width", type=int, default=1064)
    parser.add_argument("--fx", type=float, default=388.19104)
    parser.add_argument("--model-max-depth-m", type=float, default=20.0)
    parser.add_argument("--output-max-depth-m", type=float, default=10.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--audit-gt-depth", action="store_true")
    parser.add_argument("--copy-audit-poses", action="store_true")
    args = parser.parse_args()

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

    device = torch.device(args.device)
    if args.provider == "metric3d_vit_giant2":
        predict, provider_metadata = _load_metric3d(args, device)
    else:
        predict, provider_metadata = _load_depth_anything(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    aggregate_pred, aggregate_gt = [], []
    gt_dir = args.input_dir / "depth"
    inference_seconds = []
    with torch.inference_mode():
        for ordinal, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read {path}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = predict(image).astype(np.float32)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds.append(time.perf_counter() - started)
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
        **provider_metadata,
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
        "runtime": {
            "mean_inference_seconds": float(np.mean(inference_seconds)),
            "median_inference_seconds": float(np.median(inference_seconds)),
            "fps": float(len(inference_seconds) / sum(inference_seconds)),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "per_frame": rows,
    }
    (args.output_dir / "metric_depth_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

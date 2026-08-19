"""Offline Stage24B LSeg audit over a Stage24A Replay Ledger.

This worker never imports the InternNav evaluator and never writes navigation
state. It reads saved RGB-D frames, runs the VLMaps LSeg checkpoint, and emits
2-D mask/logit and sparse RGB-D surface statistics for independent review.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_LABELS = [
    "door", "chair", "table", "stairs", "sofa", "bed", "cabinet",
    "window", "wall", "floor", "shelving", "closet", "painting", "other",
]


def _load_lseg(repo: Path, checkpoint: Path, device: str):
    import sys

    sys.path.insert(0, str(repo))
    from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
    from torchvision import transforms

    crop_size = 480
    model = LSegEncNet(
        "", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size
    )
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state_dict", payload)
    # Match the VLMaps loader's checkpoint key convention.
    state = {key[4:] if key.startswith("net.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    return model, transform, crop_size


def _infer_logits(model, transform, image: np.ndarray, labels: List[str], crop_size: int):
    from vlmaps.lseg.additional_utils.models import pad_image, resize_image, crop_image

    source_h, source_w = image.shape[:2]
    tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    _, _, h, w = tensor.shape
    # Low-cost pilot: one aspect-preserving 480x480 padded forward pass.
    # The original VLMaps 520/480 sliding-window path is a later cost ablation.
    long_size = crop_size
    if h > w:
        height, width = long_size, int(w * long_size / h + 0.5)
    else:
        width, height = long_size, int(h * long_size / w + 0.5)
    resized = resize_image(tensor, height, width, mode="bilinear", align_corners=True)
    padded = pad_image(resized, [0.5] * 3, [0.5] * 3, crop_size)
    with torch.inference_mode():
        _, logits = model(padded, labels)
    logits = crop_image(logits, 0, height, 0, width)
    logits = torch.nn.functional.interpolate(
        logits, size=(source_h, source_w), mode="bilinear", align_corners=True
    )
    return logits[0].float().cpu().numpy()


def _overlay(image: np.ndarray, pred: np.ndarray, confidence: np.ndarray, labels: List[str]):
    palette = np.array([
        [220, 20, 60], [30, 144, 255], [50, 205, 50], [255, 165, 0],
        [138, 43, 226], [255, 105, 180], [0, 206, 209], [255, 215, 0],
        [128, 128, 128], [244, 164, 96], [46, 139, 87], [70, 130, 180],
        [255, 99, 71], [30, 30, 30],
    ], dtype=np.uint8)
    mask = palette[pred % len(palette)]
    visible = confidence >= 0.35
    blend = image.copy()
    blend[visible] = (0.55 * image[visible] + 0.45 * mask[visible]).astype(np.uint8)
    return blend


def _surface_stats(pred, confidence, depth, labels, sample_stride, fx, fy):
    h, w = depth.shape[:2]
    ys, xs = np.mgrid[0:h:sample_stride, 0:w:sample_stride]
    sampled_count = int(ys.size)
    d = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(d) & (d >= 0.15) & (d <= 5.0)
    ys, xs, d = ys[valid], xs[valid], d[valid]
    cls = pred[ys, xs]
    conf = confidence[ys, xs]
    result: Dict[str, Dict[str, float]] = {}
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x = (xs.astype(np.float32) - cx) * d / fx
    y = (ys.astype(np.float32) - cy) * d / fy
    for idx, label in enumerate(labels):
        keep = (cls == idx) & (conf >= 0.35)
        if not np.any(keep):
            continue
        result[label] = {
            "pixel_count": int(np.count_nonzero(keep)),
            "mean_confidence": float(np.mean(conf[keep])),
            "mean_depth_m": float(np.mean(d[keep])),
            "surface_centroid_camera_xyz": [
                float(np.mean(x[keep])), float(np.mean(y[keep])), float(np.mean(d[keep]))
            ],
        }
    return result, sampled_count, int(len(d))


def audit(ledger_dir: Path, output_dir: Path, repo: Path, checkpoint: Path, device: str, max_frames: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (ledger_dir / "observations.jsonl").read_text().splitlines() if line.strip()]
    if max_frames > 0:
        ids = np.linspace(0, len(rows) - 1, min(max_frames, len(rows)), dtype=int).tolist()
        rows = [rows[i] for i in sorted(set(ids))]
    labels = DEFAULT_LABELS
    model, transform, crop_size = _load_lseg(repo, checkpoint, device)
    records = []
    total_seconds = 0.0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        rgb = np.asarray(Image.open(ledger_dir / row["rgb_path"]).convert("RGB"))
        depth = np.load(ledger_dir / row["depth_path"])["depth_m"]
        start = time.perf_counter()
        logits = _infer_logits(model, transform, rgb, labels, crop_size)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        pred = np.argmax(probs, axis=0).astype(np.int32)
        confidence = np.max(probs, axis=0).astype(np.float32)
        stats, sampled, valid = _surface_stats(
            pred, confidence, depth, labels, sample_stride=8,
            fx=640.0 / (2.0 * np.tan(np.deg2rad(79.0) / 2.0)),
            fy=640.0 / (2.0 * np.tan(np.deg2rad(79.0) / 2.0)),
        )
        overlay = _overlay(rgb, pred, confidence, labels)
        frame_id = int(row.get("observation_index", len(records)))
        Image.fromarray(overlay).save(output_dir / f"obs_{frame_id:05d}_lseg_overlay.jpg", quality=90)
        records.append({
            "observation_index": frame_id,
            "step_id": row.get("step_id"),
            "inference_seconds": float(elapsed),
            "rgb_shape": list(rgb.shape),
            "depth_valid_count": int(np.isfinite(depth).sum()),
            "sampled_depth_count": sampled,
            "sampled_valid_count": valid,
            "mean_pixel_confidence": float(np.mean(confidence)),
            "high_confidence_pixel_fraction": float(np.mean(confidence >= 0.35)),
            "class_surface_stats": stats,
        })
    peak = None
    if device.startswith("cuda"):
        peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    report = {
        "audit_name": "stage24b_lseg_replay",
        "ledger_dir": str(ledger_dir),
        "checkpoint": str(checkpoint),
        "device": device,
        "labels": labels,
        "frame_count": len(records),
        "total_inference_seconds": float(total_seconds),
        "mean_inference_seconds": float(total_seconds / max(1, len(records))),
        "peak_cuda_allocated_mb": peak,
        "records": records,
        "gt_status": "not_available_in_replay_ledger; use Habitat semantic_scene/pixel audit separately",
    }
    (output_dir / "stage24b_lseg_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlmaps-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()
    audit(args.ledger_dir, args.output_dir, args.vlmaps_repo, args.checkpoint, args.device, args.max_frames)


if __name__ == "__main__":
    main()

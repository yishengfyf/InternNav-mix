"""Compare VLMaps-inspired multi-view fusion on frozen Stage24D ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.lseg_multiview_fusion import (
    FrameVoxelEvidence, aggregate_frame_voxels, fuse_voxel_evidence,
    isolated_voxel_rate,
)
from internnav.utils.lseg_online_shadow import (
    ALIASES, DEFAULT_LABELS, PALETTE, OnlineLSegSemanticShadow, _jsonable, _tokens,
)


VARIANTS = {
    "f0_q_hard": ("q", "hard"),
    "f1_q_probability": ("q", "prob"),
    "f2_all_probability": ("all", "prob"),
    "f3_q_embedding": ("q", "embedding"),
    "f4_q_robust_probability": ("q", "robust_prob"),
}
PERSISTENT_FEATURE_DIMS = {
    "f0_q_hard": 1,
    "f1_q_probability": len(DEFAULT_LABELS),
    "f2_all_probability": len(DEFAULT_LABELS),
    "f3_q_embedding": 512,
    "f4_q_robust_probability": len(DEFAULT_LABELS),
}


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _valid_sample_pixels(
    depth: np.ndarray, stride: int, min_depth: float, max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    values = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(values) & (values >= min_depth) & (values <= max_depth)
    return ys[valid], xs[valid], values[valid]


def _project_frame(
    *, logits: np.ndarray, embeddings: np.ndarray, pixel_y: np.ndarray,
    pixel_x: np.ndarray, depth_m: np.ndarray, camera_pose_map: np.ndarray,
    intrinsic: np.ndarray, confidence_threshold: float, observation_index: int,
    voxel_size_m: float, keep_embeddings: bool,
) -> FrameVoxelEvidence:
    selected_logits = logits[:, pixel_y, pixel_x].T.astype(np.float32)
    probabilities = torch.softmax(torch.from_numpy(selected_logits), dim=1).numpy()
    confidence = np.max(probabilities, axis=1)
    keep = confidence >= float(confidence_threshold)
    ys, xs, values = pixel_y[keep], pixel_x[keep], depth_m[keep]
    probabilities = probabilities[keep]
    selected_logits = selected_logits[keep]
    selected_embeddings = embeddings[keep] if keep_embeddings else None

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (xs.astype(np.float32) - cx) * values / fx
    y = (ys.astype(np.float32) - cy) * values / fy
    camera = np.stack([x, y, values, np.ones_like(values)], axis=1)
    points = (camera_pose_map @ camera.T).T[:, :3].astype(np.float32)
    return aggregate_frame_voxels(
        points, probabilities, selected_logits, selected_embeddings, values,
        np.asarray(camera_pose_map[:3, 3], dtype=np.float32), observation_index,
        voxel_size_m,
    )


def _audit_gt(
    points: np.ndarray, class_id: np.ndarray, labels: Sequence[str],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    gt = meta.get("semantic_scene_gt") or {}
    entries = list(gt.get("objects") or []) + list(gt.get("regions") or [])
    transform = np.asarray(
        (meta.get("coordinate_transforms") or {}).get("map_to_habitat_world"),
        dtype=np.float32,
    )
    if not entries or transform.shape != (4, 4):
        return {"available": False, "compatible_voxel_count": 0}
    world = (transform @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
    all_distances: List[np.ndarray] = []
    per_label: Dict[str, Any] = {}
    for label_id, label in enumerate(labels):
        mask = class_id == label_id
        if not np.any(mask):
            continue
        aliases = set(ALIASES.get(label, {label}))
        compatible = [
            item for item in entries
            if aliases.intersection(_tokens(item.get("category", "")))
        ]
        if not compatible:
            continue
        lower = np.asarray([item.get("lower", item["center"]) for item in compatible], dtype=np.float32)
        upper = np.asarray([item.get("upper", item["center"]) for item in compatible], dtype=np.float32)
        label_points = world[mask]
        nearest_parts = []
        for start in range(0, len(label_points), 4096):
            batch = label_points[start:start + 4096, None, :]
            delta = np.maximum(np.maximum(lower[None] - batch, 0.0), batch - upper[None])
            nearest_parts.append(np.min(np.linalg.norm(delta, axis=2), axis=1))
        distances = np.concatenate(nearest_parts).astype(np.float32)
        all_distances.append(distances)
        per_label[label] = {
            "compatible_voxel_count": int(len(distances)),
            "surface_distance_le_050m_count": int(np.count_nonzero(distances <= 0.50)),
            "surface_distance_le_050m_rate": float(np.mean(distances <= 0.50)),
            "surface_distance_m_median": float(np.median(distances)),
        }
    values = np.concatenate(all_distances) if all_distances else np.zeros(0, dtype=np.float32)
    return {
        "available": True, "compatible_voxel_count": int(len(values)),
        "surface_distance_le_050m_count": int(np.count_nonzero(values <= 0.50)),
        "surface_distance_le_050m_rate": float(np.mean(values <= 0.50)) if len(values) else None,
        "surface_distance_m_median": float(np.median(values)) if len(values) else None,
        "surface_distance_m_p95": float(np.percentile(values, 95)) if len(values) else None,
        "per_label": per_label,
    }


def _gt_hit_mask(
    points: np.ndarray, class_id: np.ndarray, labels: Sequence[str],
    meta: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return class-conditioned compatibility and <=0.5m hit per voxel."""
    compatible_mask = np.zeros(len(points), dtype=bool)
    hit_mask = np.zeros(len(points), dtype=bool)
    gt = meta.get("semantic_scene_gt") or {}
    entries = list(gt.get("objects") or []) + list(gt.get("regions") or [])
    transform = np.asarray(
        (meta.get("coordinate_transforms") or {}).get("map_to_habitat_world"),
        dtype=np.float32,
    )
    if not entries or transform.shape != (4, 4):
        return compatible_mask, hit_mask
    world = (transform @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
    for label_id, label in enumerate(labels):
        indices = np.flatnonzero(class_id == label_id)
        if not len(indices):
            continue
        aliases = set(ALIASES.get(label, {label}))
        matched = [
            item for item in entries
            if aliases.intersection(_tokens(item.get("category", "")))
        ]
        if not matched:
            continue
        compatible_mask[indices] = True
        lower = np.asarray([item.get("lower", item["center"]) for item in matched], dtype=np.float32)
        upper = np.asarray([item.get("upper", item["center"]) for item in matched], dtype=np.float32)
        for start in range(0, len(indices), 4096):
            batch_indices = indices[start:start + 4096]
            batch = world[batch_indices, None, :]
            delta = np.maximum(np.maximum(lower[None] - batch, 0.0), batch - upper[None])
            hit_mask[batch_indices] = np.min(np.linalg.norm(delta, axis=2), axis=1) <= 0.50
    return compatible_mask, hit_mask


def _compare_to_baseline(
    baseline: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray],
    labels: Sequence[str], meta: Mapping[str, Any],
) -> Dict[str, Any]:
    base_lookup = {tuple(key.tolist()): index for index, key in enumerate(baseline["keys"])}
    candidate_lookup = {tuple(key.tolist()): index for index, key in enumerate(candidate["keys"])}
    common = sorted(set(base_lookup).intersection(candidate_lookup))
    if not common:
        return {"common_voxel_count": 0}
    base_indices = np.asarray([base_lookup[key] for key in common], dtype=np.int64)
    candidate_indices = np.asarray([candidate_lookup[key] for key in common], dtype=np.int64)
    base_class = baseline["class_id"][base_indices]
    candidate_class = candidate["class_id"][candidate_indices]
    changed = base_class != candidate_class
    base_compatible, base_hit = _gt_hit_mask(
        baseline["map_xyz"][base_indices], base_class, labels, meta
    )
    candidate_compatible, candidate_hit = _gt_hit_mask(
        candidate["map_xyz"][candidate_indices], candidate_class, labels, meta
    )
    comparable = base_compatible & candidate_compatible
    corrected = comparable & changed & ~base_hit & candidate_hit
    harmed = comparable & changed & base_hit & ~candidate_hit
    return {
        "common_voxel_count": int(len(common)),
        "changed_voxel_count": int(np.count_nonzero(changed)),
        "changed_voxel_rate": float(np.mean(changed)),
        "gt_comparable_changed_count": int(np.count_nonzero(comparable & changed)),
        "gt_corrected_count": int(np.count_nonzero(corrected)),
        "gt_harmed_count": int(np.count_nonzero(harmed)),
        "gt_net_correction": int(np.count_nonzero(corrected) - np.count_nonzero(harmed)),
    }


def _bounds(points: np.ndarray, axes: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    projected = points[:, axes]
    lower = np.nanpercentile(projected, 1, axis=0)
    upper = np.nanpercentile(projected, 99, axis=0)
    center = (lower + upper) / 2.0
    extent = np.maximum(upper - lower, 0.5) * 1.08
    return center - extent / 2.0, center + extent / 2.0


def _render_panel(
    points: np.ndarray, class_id: np.ndarray, axes: Tuple[int, int],
    bounds: Tuple[np.ndarray, np.ndarray], title: str, size: int = 720,
) -> Image.Image:
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill=(0, 0, 0))
    lower, upper = bounds
    extent = np.maximum(upper - lower, 1e-6)
    projected = points[:, axes]
    margin = 35
    pixel = margin + (projected - lower) / extent * (size - 2 * margin)
    pixel[:, 1] = size - pixel[:, 1]
    valid = np.all((pixel >= margin) & (pixel <= size - margin), axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) > 50000:
        indices = indices[np.linspace(0, len(indices) - 1, 50000).astype(np.int64)]
    colors = PALETTE[class_id % len(PALETTE)]
    for index in indices:
        x, y = pixel[index]
        color = tuple(int(value) for value in colors[index])
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=color)
    return image


def _save_visualizations(
    output: Path, variants: Mapping[str, Dict[str, np.ndarray]], all_points: np.ndarray,
) -> Dict[str, Dict[str, str]]:
    output.mkdir(parents=True, exist_ok=True)
    angle = np.deg2rad(35.0)
    all_oblique = np.stack([
        all_points[:, 0] * np.cos(angle) - all_points[:, 1] * np.sin(angle),
        all_points[:, 2] + 0.35 * (all_points[:, 0] * np.sin(angle) + all_points[:, 1] * np.cos(angle)),
        all_points[:, 1],
    ], axis=1)
    common = {
        "bev_xy": ((0, 1), _bounds(all_points, (0, 1))),
        "side_xz": ((0, 2), _bounds(all_points, (0, 2))),
        "side_yz": ((1, 2), _bounds(all_points, (1, 2))),
        "oblique": ((0, 1), _bounds(all_oblique, (0, 1))),
    }
    result: Dict[str, Dict[str, str]] = {}
    contact_panels: Dict[str, List[Image.Image]] = {key: [] for key in common}
    for name, fused in variants.items():
        variant_dir = output / name
        variant_dir.mkdir(exist_ok=True)
        points = fused["map_xyz"]
        oblique = np.stack([
            points[:, 0] * np.cos(angle) - points[:, 1] * np.sin(angle),
            points[:, 2] + 0.35 * (points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)),
            points[:, 1],
        ], axis=1)
        paths = {}
        for view, (axes, view_bounds) in common.items():
            draw_points = oblique if view == "oblique" else points
            panel = _render_panel(draw_points, fused["class_id"], axes, view_bounds, f"{name} {view}")
            path = variant_dir / f"semantic_{view}.png"
            panel.save(path)
            paths[view] = str(path.relative_to(output.parent))
            contact_panels[view].append(panel.resize((360, 360)))
        result[name] = paths
    for view, panels in contact_panels.items():
        contact = Image.new("RGB", (720, 1080), (255, 255, 255))
        for index, panel in enumerate(panels):
            contact.paste(panel, ((index % 2) * 360, (index // 2) * 360))
        contact.save(output / f"comparison_{view}.png")
    legend = Image.new("RGB", (420, 40 + 25 * len(DEFAULT_LABELS)), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    draw.text((10, 10), "Stage24F LSeg classes", fill=(0, 0, 0))
    for index, label in enumerate(DEFAULT_LABELS):
        y = 36 + index * 25
        draw.rectangle((10, y, 30, y + 16), fill=tuple(PALETTE[index].tolist()))
        draw.text((40, y), label, fill=(0, 0, 0))
    legend.save(output / "semantic_legend.png")
    return result


def _variant_metrics(
    fused: Dict[str, np.ndarray], labels: Sequence[str], meta: Mapping[str, Any],
    frame_count: int, elapsed: float, variant_name: str,
) -> Dict[str, Any]:
    multi = fused["view_count"] >= 2
    class_counts = Counter(labels[int(value)] for value in fused["class_id"].tolist())
    gt = _audit_gt(fused["map_xyz"], fused["class_id"], labels, meta)
    per_label_rates = [
        float(item["surface_distance_le_050m_rate"])
        for item in (gt.get("per_label") or {}).values()
        if item.get("surface_distance_le_050m_rate") is not None
    ]
    feature_dim = PERSISTENT_FEATURE_DIMS[variant_name]
    return {
        "frame_count": int(frame_count), "fusion_seconds": float(elapsed),
        "voxel_count": int(len(fused["map_xyz"])),
        "class_count": len(class_counts), "class_voxel_counts": dict(sorted(class_counts.items())),
        "multi_view_voxel_count": int(np.count_nonzero(multi)),
        "multi_view_voxel_rate": float(np.mean(multi)),
        "multi_view_conflict_rate": float(np.mean(fused["conflict"][multi])) if np.any(multi) else 0.0,
        "multi_view_agreement_mean": float(np.mean(fused["cross_view_agreement"][multi])) if np.any(multi) else None,
        "confidence_mean": float(np.mean(fused["confidence"])),
        "margin_mean": float(np.mean(fused["margin"])),
        "normalized_entropy_mean": float(np.mean(fused["entropy"])),
        "isolated_voxel_rate": isolated_voxel_rate(fused["keys"], fused["class_id"]),
        "estimated_map_bytes": int(sum(value.nbytes for value in fused.values() if isinstance(value, np.ndarray))),
        "persistent_feature_dim": feature_dim,
        "persistent_semantic_payload_bytes_fp32": int(len(fused["map_xyz"]) * feature_dim * 4),
        "gt_episode_label_macro_hit_rate": float(np.mean(per_label_rates)) if per_label_rates else None,
        "gt_audit": gt,
    }


def replay_episode(ledger: Path, output: Path, args: argparse.Namespace) -> Dict[str, Any]:
    meta = json.loads((ledger / "episode_meta.json").read_text(encoding="utf-8"))
    observations = _jsonl(ledger / "observations.jsonl")
    queries = _jsonl(ledger / "queries.jsonl")
    query_keys = {str(item["observation_key"]) for item in queries}
    intrinsic = np.asarray(meta["camera_model"]["intrinsic"], dtype=np.float32)[:3, :3]
    cfg = {
        "lseg_online_shadow_enable": True, "lseg_online_shadow_repo": str(args.vlmaps_repo),
        "lseg_online_shadow_checkpoint": str(args.checkpoint), "lseg_online_shadow_device": args.device,
        "lseg_online_shadow_labels": DEFAULT_LABELS,
    }
    shadow = OnlineLSegSemanticShadow(cfg, intrinsic, args.device)
    shadow._load_model()
    text_features = shadow.lseg_text_features()
    frames: Dict[int, FrameVoxelEvidence] = {}
    inference_seconds = []
    query_indices = []
    online_class_counts: Dict[int, Dict[str, int]] = {}
    for ordinal, observation in enumerate(observations):
        index = int(observation["record_index"])
        is_query = str(observation["observation_key"]) in query_keys
        if is_query:
            query_indices.append(index)
        rgb = np.ascontiguousarray(np.asarray(
            Image.open(ledger / observation["rgb_path"]).convert("RGB")
        ).copy())
        if observation.get("rgb_storage_format") != "png" or hashlib.sha256(rgb.tobytes()).hexdigest() != observation.get("rgb_sha256"):
            raise RuntimeError(f"Lossless RGB contract failed: {ledger} frame {index}")
        with np.load(ledger / observation["depth_path"]) as payload:
            depth = np.ascontiguousarray(payload["depth_m"], dtype=np.float32)
        if hashlib.sha256(depth.tobytes()).hexdigest() != observation.get("depth_sha256"):
            raise RuntimeError(f"Depth hash contract failed: {ledger} frame {index}")
        ys, xs, values = _valid_sample_pixels(depth, args.stride, args.min_depth, args.max_depth)
        started = time.perf_counter()
        with shadow._deterministic_inference():
            logits, embeddings = shadow.infer_logits_and_sampled_embeddings(rgb, ys, xs)
        inference_seconds.append(time.perf_counter() - started)
        pose = np.asarray((observation.get("pose") or {})["stage23_gt_camera_pose_map"], dtype=np.float32)
        frames[index] = _project_frame(
            logits=logits, embeddings=embeddings, pixel_y=ys, pixel_x=xs, depth_m=values,
            camera_pose_map=pose, intrinsic=intrinsic, confidence_threshold=args.confidence,
            observation_index=int(observation["observation_index"]),
            voxel_size_m=args.voxel_size, keep_embeddings=is_query,
        )
        if is_query:
            sampled_logits = logits[:, ys, xs].T.astype(np.float32)
            sampled_probabilities = torch.softmax(
                torch.from_numpy(sampled_logits), dim=1
            ).numpy()
            sampled_keep = np.max(sampled_probabilities, axis=1) >= args.confidence
            counts = Counter(DEFAULT_LABELS[int(value)] for value in np.argmax(
                sampled_probabilities[sampled_keep], axis=1
            ).tolist())
            online_class_counts[int(observation["observation_index"])] = dict(counts)
        print(
            f"STAGE24F_FRAME scene={meta['scene_id']} episode={meta['episode_id']} "
            f"frame={ordinal + 1}/{len(observations)} query={int(is_query)}", flush=True,
        )
    all_indices = [int(item["record_index"]) for item in observations]
    fused_variants: Dict[str, Dict[str, np.ndarray]] = {}
    metrics = {}
    for name, (selection, mode) in VARIANTS.items():
        selected = query_indices if selection == "q" else all_indices
        started = time.perf_counter()
        fused = fuse_voxel_evidence(
            [frames[index] for index in selected], mode,
            text_features=text_features if mode == "embedding" else None,
        )
        elapsed = time.perf_counter() - started
        fused_variants[name] = fused
        metrics[name] = _variant_metrics(
            fused, DEFAULT_LABELS, meta, len(selected), elapsed, name
        )
        variant_dir = output / "variants" / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(variant_dir / "voxel_map.npz", **fused)
    baseline = fused_variants["f0_q_hard"]
    for name, fused in fused_variants.items():
        metrics[name]["comparison_to_f0"] = _compare_to_baseline(
            baseline, fused, DEFAULT_LABELS, meta
        )
    all_points = np.concatenate([frame.points for frame in frames.values() if len(frame.points)], axis=0)
    visualizations = _save_visualizations(output / "visualizations", fused_variants, all_points)

    online_dirs = [
        path.parent for path in args.ledger_root.glob("**/online_lseg_shadow/*/episode_meta.json")
        if (lambda item: str(item.get("scene_id")) == str(meta["scene_id"]) and
            str(item.get("episode_id")) == str(meta["episode_id"]))(
                json.loads(path.read_text(encoding="utf-8"))
            )
    ]
    consistency_errors = []
    if len(online_dirs) != 1:
        consistency_errors.append(f"expected_one_online_dir_got_{len(online_dirs)}")
    else:
        events = _jsonl(online_dirs[0] / "events.jsonl")
        for event in events:
            observation_index = int(event["observation_index"])
            expected = event.get("class_surface_counts") or {}
            actual = online_class_counts.get(observation_index) or {}
            if expected != actual:
                consistency_errors.append(f"observation_{observation_index}:class_counts")
    result = {
        "audit_name": "stage24f_multiview_fusion",
        "scene_id": str(meta["scene_id"]), "episode_id": str(meta["episode_id"]),
        "ledger": str(ledger), "observation_count": len(observations),
        "query_count": len(query_indices), "voxel_size_m": args.voxel_size,
        "confidence_threshold": args.confidence, "sample_stride": args.stride,
        "inference_seconds_mean": float(np.mean(inference_seconds)),
        "inference_seconds_total": float(np.sum(inference_seconds)),
        "variants": metrics, "visualizations": visualizations,
        "online_q_consistency": {"passed": not consistency_errors, "errors": consistency_errors},
        "decision_status": "audit_only_not_navigation_ready",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "episode_multiview_fusion.json").write_text(
        json.dumps(_jsonable(result), indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlmaps-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=5.0)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "distributed":
        args.device = f"cuda:{local_rank}"
    ledgers = sorted(path.parent for path in args.ledger_root.glob(
        "**/replay_ledger/*/episode_meta.json"
    ))
    if not ledgers:
        raise SystemExit(f"No replay ledgers under {args.ledger_root}")
    for index, ledger in enumerate(ledgers):
        if index % world != rank:
            continue
        meta = json.loads((ledger / "episode_meta.json").read_text(encoding="utf-8"))
        episode_output = args.output_root / f"rank{rank}" / f"{meta['scene_id']}_{meta['episode_id']}"
        result = replay_episode(ledger, episode_output, args)
        print("STAGE24F_EPISODE_COMPLETE " + json.dumps({
            "scene_id": result["scene_id"], "episode_id": result["episode_id"],
            "online_q_consistency": result["online_q_consistency"],
        }), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit HSGM-style YOLOE instance surfaces on a fixed InternNav replay.

This tool is deliberately offline and shadow-only.  It reads saved RGB-D-pose
observations, runs YOLOE on recorded S2 query frames, and writes paired raw and
HSGM edge-filtered instance surfaces.  It never imports the Habitat evaluator,
updates SparseOcc, or emits an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


ALIASES = {
    "door": {"door", "doorway", "entrance"},
    "stairs": {"stairs", "stair", "staircase", "steps"},
    "cabinet": {"cabinet", "chest", "drawer", "drawers"},
    "painting": {"painting", "picture", "artwork"},
    "sofa": {"sofa", "couch"},
    "tv": {"tv", "television", "monitor"},
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(kept)) if kept else None


def _percentile(values: Iterable[float | None], q: float) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(kept, q)) if kept else None


def _tokens(value: str) -> set[str]:
    normalized = str(value or "").lower().replace("-", "_").replace("/", "_")
    return {token for token in normalized.replace(" ", "_").split("_") if len(token) > 2}


def labels_compatible(left: str, right: str) -> bool:
    left_tokens = set(ALIASES.get(str(left).lower(), {str(left).lower()})) | _tokens(left)
    right_tokens = set(ALIASES.get(str(right).lower(), {str(right).lower()})) | _tokens(right)
    return bool(left_tokens.intersection(right_tokens))


def hsgm_center_filter(box_xyxy: Iterable[float], width: int, height: int, edge_fraction: float = 0.2) -> bool:
    """Return True when a detection survives HSGM's outer-center filter."""
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (
        width * edge_fraction <= center_x <= width * (1.0 - edge_fraction)
        and height * edge_fraction <= center_y <= height * (1.0 - edge_fraction)
    )


def project_mask_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    camera_pose_map: np.ndarray,
    *,
    min_depth_m: float = 0.15,
    max_depth_m: float = 5.0,
    sample_stride: int = 4,
    max_points: int = 4096,
) -> dict[str, Any]:
    """Project a binary mask into the SparseOcc map frame using metric z-depth."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    pose = np.asarray(camera_pose_map, dtype=np.float32)
    if mask.shape != depth.shape or intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("mask/depth/intrinsic/camera pose shape mismatch")
    ys, xs = np.nonzero(mask)
    mask_count = int(xs.size)
    if not mask_count:
        return {"mask_pixel_count": 0, "valid_depth_count": 0, "valid_depth_ratio": None,
                "map_xyz": np.zeros((0, 3), dtype=np.float32)}
    depth_values = depth[ys, xs]
    valid = np.isfinite(depth_values) & (depth_values >= min_depth_m) & (depth_values <= max_depth_m)
    valid_count = int(np.count_nonzero(valid))
    xs, ys, depth_values = xs[valid], ys[valid], depth_values[valid]
    if sample_stride > 1 and xs.size:
        keep = np.arange(xs.size) % int(sample_stride) == 0
        xs, ys, depth_values = xs[keep], ys[keep], depth_values[keep]
    if xs.size > max_points:
        keep = np.linspace(0, xs.size - 1, max_points).astype(np.int64)
        xs, ys, depth_values = xs[keep], ys[keep], depth_values[keep]
    if not xs.size:
        points = np.zeros((0, 3), dtype=np.float32)
    else:
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        optical = np.stack([
            (xs.astype(np.float32) - cx) * depth_values / fx,
            (ys.astype(np.float32) - cy) * depth_values / fy,
            depth_values,
            np.ones_like(depth_values),
        ], axis=1)
        points = (pose @ optical.T).T[:, :3].astype(np.float32)
    return {
        "mask_pixel_count": mask_count,
        "valid_depth_count": valid_count,
        "valid_depth_ratio": float(valid_count / mask_count),
        "sampled_point_count": int(points.shape[0]),
        "map_xyz": points,
    }


def hsgm_height_band_counts(points: np.ndarray, base_z: float, floor_tolerance_m: float = 0.2) -> dict[str, int]:
    """Summarize HSGM-like height bands without treating them as occupancy truth."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    relative = points[:, 2] - float(base_z) if len(points) else np.zeros(0, dtype=np.float32)
    return {
        "point_count": int(len(relative)),
        "floor_band_count": int(np.count_nonzero(np.abs(relative) <= floor_tolerance_m)),
        "stair_height_candidate_count": int(np.count_nonzero((np.abs(relative) > 0.2) & (np.abs(relative) < 0.7))),
        "obstacle_band_count": int(np.count_nonzero((relative > floor_tolerance_m) & (relative <= 1.5))),
        "above_agent_clearance_count": int(np.count_nonzero(relative > 1.5)),
    }


def associate_instances(instances: list[dict[str, Any]], merge_radius_m: float = 0.5) -> list[dict[str, Any]]:
    """Causally merge same-label instance centroids across replay frames."""
    nodes: list[dict[str, Any]] = []
    for item in sorted(instances, key=lambda row: (int(row["observation_index"]), int(row["detection_index"]))):
        centroid = np.asarray(item.get("centroid_map"), dtype=np.float32)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            continue
        candidates = []
        for index, node in enumerate(nodes):
            if node["label"] != item["label"]:
                continue
            distance = float(np.linalg.norm(np.asarray(node["centroid_map"]) - centroid))
            if distance <= merge_radius_m:
                candidates.append((distance, index))
        if candidates:
            _, index = min(candidates)
            node = nodes[index]
            count = int(node["instance_count"])
            node["centroid_map"] = (
                (np.asarray(node["centroid_map"]) * count + centroid) / (count + 1)
            ).tolist()
            node["instance_count"] = count + 1
            node["point_count"] += int(item.get("sampled_point_count", 0))
            node["source_observations"].add(int(item["observation_index"]))
            node["source_steps"].add(int(item["step_id"]))
            node["confidence_sum"] += float(item.get("confidence", 0.0))
        else:
            nodes.append({
                "node_id": f"YI{len(nodes):05d}",
                "label": str(item["label"]),
                "centroid_map": centroid.tolist(),
                "instance_count": 1,
                "point_count": int(item.get("sampled_point_count", 0)),
                "source_observations": {int(item["observation_index"])},
                "source_steps": {int(item["step_id"])},
                "confidence_sum": float(item.get("confidence", 0.0)),
            })
    for node in nodes:
        node["source_observations"] = sorted(node["source_observations"])
        node["source_steps"] = sorted(node["source_steps"])
        node["mean_confidence"] = float(node.pop("confidence_sum") / node["instance_count"])
        node["multi_view"] = len(node["source_observations"]) >= 2
    return nodes


def conflict_audit(nodes: list[dict[str, Any]], radius_m: float = 0.25) -> dict[str, Any]:
    pairs: Counter[str] = Counter()
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            if left["label"] == right["label"]:
                continue
            distance = np.linalg.norm(np.asarray(left["centroid_map"]) - np.asarray(right["centroid_map"]))
            if distance <= radius_m:
                pairs["|".join(sorted((left["label"], right["label"])))] += 1
    return {"count": int(sum(pairs.values())), "pairs": dict(sorted(pairs.items()))}


def gt_surface_audit(nodes: list[dict[str, Any]], episode_meta: dict[str, Any]) -> dict[str, Any]:
    gt = episode_meta.get("semantic_scene_gt") or {}
    entries = list(gt.get("objects") or []) + list(gt.get("regions") or [])
    map_to_world = np.asarray(
        (episode_meta.get("coordinate_transforms") or {}).get("map_to_habitat_world"), dtype=np.float32
    )
    if not entries or map_to_world.shape != (4, 4):
        return {"available": False, "compatible_node_count": 0}
    distances = []
    for node in nodes:
        compatible = [row for row in entries if labels_compatible(node["label"], row.get("category", ""))]
        if not compatible:
            continue
        point = map_to_world @ np.asarray([*node["centroid_map"], 1.0], dtype=np.float32)
        nearest_distance = None
        for row in compatible:
            lower = np.asarray(row.get("lower", row["center"]), dtype=np.float32)
            upper = np.asarray(row.get("upper", row["center"]), dtype=np.float32)
            delta = np.maximum(np.maximum(lower - point[:3], 0.0), point[:3] - upper)
            distance = float(np.linalg.norm(delta))
            nearest_distance = distance if nearest_distance is None else min(nearest_distance, distance)
        node["gt_surface_distance_m"] = nearest_distance
        distances.append(nearest_distance)
    return {
        "available": True,
        "compatible_node_count": len(distances),
        "surface_distance_le_050m_count": int(np.count_nonzero(np.asarray(distances) <= 0.5)),
        "surface_distance_le_050m_rate": float(np.mean(np.asarray(distances) <= 0.5)) if distances else None,
        "surface_distance_m_median": float(np.median(distances)) if distances else None,
        "surface_distance_m_p95": float(np.percentile(distances, 95)) if distances else None,
    }


def instruction_coverage(labels: Iterable[str], terms: Iterable[str]) -> dict[str, Any]:
    labels, terms = sorted(set(labels)), sorted(set(terms))
    matched = [term for term in terms if any(labels_compatible(label, term) for label in labels)]
    return {
        "landmark_terms": terms,
        "detected_labels": labels,
        "matched_terms": matched,
        "matched_count": len(matched),
        "term_count": len(terms),
        "coverage": float(len(matched) / len(terms)) if terms else None,
    }


def _load_depth(episode_dir: Path, row: dict[str, Any]) -> np.ndarray | None:
    relative = row.get("depth_path")
    if not relative:
        return None
    path = episode_dir / str(relative)
    if not path.exists():
        return None
    payload = np.load(path)
    return np.asarray(payload["depth_m"], dtype=np.float32)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.resize((width, height), resample=Image.Resampling.NEAREST)) > 0


def _query_observations(episode_dir: Path, max_frames: int) -> list[dict[str, Any]]:
    observations = {str(row.get("observation_key")): row for row in _jsonl(episode_dir / "observations.jsonl")}
    keys = []
    for query in _jsonl(episode_dir / "queries.jsonl"):
        key = str(query.get("observation_key"))
        if key in observations and key not in keys:
            keys.append(key)
    rows = [observations[key] for key in keys]
    if max_frames > 0 and len(rows) > max_frames:
        indices = np.linspace(0, len(rows) - 1, max_frames).round().astype(np.int64)
        rows = [rows[int(index)] for index in sorted(set(indices.tolist()))]
    return rows


def _landmark_terms(rows: list[dict[str, Any]]) -> list[str]:
    terms = set()
    for row in rows:
        semantic = row.get("semantic_state") or {}
        terms.update(str(term).lower() for term in semantic.get("landmark_terms") or [])
    return sorted(terms)


def _episode_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("**/replay_ledger/*/episode_meta.json"))


def _episode_annotations(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    annotations = {}
    for path in sorted(root.glob("**/episode_manifests/*.json")):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict) and row.get("scene_id") is not None and row.get("episode_id") is not None:
                annotations[(str(row["scene_id"]), str(row["episode_id"]))] = dict(row)
    return annotations


def _scene_split(scene_id: Any, holdout_fraction: float = 0.30) -> str:
    threshold = max(0, min(10, int(round(float(holdout_fraction) * 10))))
    bucket = int(hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "holdout" if bucket >= 10 - threshold else "dev"


def _base_z(row: dict[str, Any]) -> float:
    pose = np.asarray((row.get("pose") or {}).get("stage23_gt_base_pose_map"), dtype=np.float32)
    return float(pose[2, 3]) if pose.shape == (4, 4) else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _depth_hash_matches(depth: np.ndarray | None, row: dict[str, Any]) -> bool | None:
    expected = row.get("depth_sha256")
    if depth is None or not expected:
        return None
    actual = hashlib.sha256(np.ascontiguousarray(depth, dtype=np.float32).tobytes()).hexdigest()
    return actual == expected


def _lseg_summary(episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir.parent.parent / "online_lseg_shadow" / episode_dir.name / "summary.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    component = payload.get("component_filter") or {}
    raw_gt = payload.get("gt_audit") or {}
    filtered_gt = component.get("filtered_gt_audit") or {}
    filtered_conflict = component.get("filtered_cross_label_conflict_audit") or {}
    return {
        "available": True,
        "raw": {
            "node_count": payload.get("node_count"),
            "multi_view_node_count": payload.get("multi_view_node_count"),
            "severe_cross_label_conflict_count": payload.get("severe_cross_label_conflict_count"),
            "gt_compatible_node_count": raw_gt.get("compatible_node_count"),
            "gt_surface_hit_le_050m_count": raw_gt.get("surface_distance_le_050m_count"),
            "gt_surface_hit_le_050m_rate": raw_gt.get("surface_distance_le_050m_rate"),
        },
        "filtered": {
            "enabled": component.get("enabled"),
            "node_count": component.get("filtered_node_count"),
            "multi_view_node_count": component.get("filtered_multi_view_node_count"),
            "severe_cross_label_conflict_count": filtered_conflict.get("severe_count"),
            "gt_compatible_node_count": filtered_gt.get("compatible_node_count"),
            "gt_surface_hit_le_050m_count": filtered_gt.get("surface_distance_le_050m_count"),
            "gt_surface_hit_le_050m_rate": filtered_gt.get("surface_distance_le_050m_rate"),
        },
    }


def _bev(path: Path, instances: list[dict[str, Any]], route: list[list[float]]) -> None:
    points, colors = [], []
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14), (148, 103, 189)]
    for item in instances:
        xyz = np.asarray(item.get("points_map") or [], dtype=np.float32).reshape(-1, 3)
        if len(xyz):
            points.append(xyz[:, :2])
            color_index = int(hashlib.sha256(item["label"].encode("utf-8")).hexdigest()[:8], 16)
            colors.extend([palette[color_index % len(palette)]] * len(xyz))
    if not points:
        return
    xy = np.concatenate(points, axis=0)
    route_xy = np.asarray(route, dtype=np.float32).reshape(-1, 2) if route else np.zeros((0, 2))
    all_xy = np.concatenate([xy, route_xy], axis=0) if len(route_xy) else xy
    lower, upper = np.percentile(all_xy, 1, axis=0), np.percentile(all_xy, 99, axis=0)
    extent = np.maximum(upper - lower, 0.5)
    size, margin = 720, 30
    pix = margin + (xy - lower) / extent * (size - 2 * margin)
    pix[:, 1] = size - pix[:, 1]
    image = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    take = np.linspace(0, len(pix) - 1, min(40000, len(pix))).astype(np.int64)
    for index in take:
        x, y = pix[index]
        draw.point((float(x), float(y)), fill=colors[index])
    if len(route_xy):
        rp = margin + (route_xy - lower) / extent * (size - 2 * margin)
        rp[:, 1] = size - rp[:, 1]
        draw.line([tuple(row) for row in rp.tolist()], fill=(0, 0, 0), width=3)
    draw.text((12, 10), "YOLOE semantic surface BEV (audit only)", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


class YOLOERunner:
    def __init__(self, checkpoint: Path, hsgm_root: Path, device: str) -> None:
        import torch
        from ultralytics import YOLOE

        categories = runpy.run_path(str(hsgm_root / "src/segmentation/object_list.py"))["categories"]
        self.class_names = [str(row["name"]) for row in categories]
        self.torch, self.device = torch, device
        started = time.perf_counter()
        self.model = YOLOE(str(checkpoint))
        self.model.set_classes(self.class_names, self.model.get_text_pe(self.class_names))
        self.load_seconds = time.perf_counter() - started

    def predict(self, image: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        result = self.model.predict(
            image, conf=0.6, iou=0.3, retina_masks=True, verbose=False, device=self.device
        )[0]
        elapsed = time.perf_counter() - started
        output = []
        if result.masks is None:
            return output, elapsed
        masks = result.masks.data.detach().cpu().numpy()
        for index in range(len(masks)):
            output.append({
                "detection_index": index,
                "label": str(result.names.get(int(result.boxes.cls[index].item()))),
                "confidence": float(result.boxes.conf[index].item()),
                "box_xyxy": result.boxes.xyxy[index].detach().cpu().numpy().tolist(),
                "mask": masks[index] > 0.5,
            })
        return output, elapsed

    def cuda_stats(self) -> dict[str, Any]:
        if not self.torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_mb": self.torch.cuda.memory_allocated() / 1024 ** 2,
            "reserved_mb": self.torch.cuda.memory_reserved() / 1024 ** 2,
            "max_allocated_mb": self.torch.cuda.max_memory_allocated() / 1024 ** 2,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "stage31_frames.jsonl"
    if frame_path.exists():
        frame_path.unlink()
    episode_dirs = _episode_dirs(args.replay_root)
    annotations = _episode_annotations(args.replay_root)
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[:args.max_episodes]
    runner = YOLOERunner(args.checkpoint, args.hsgm_root, args.device)
    aggregate: dict[str, list[dict[str, Any]]] = {"raw": [], "hsgm_center_filtered": []}
    episode_reports = []
    manifest = []
    all_latencies = []
    for episode_dir in episode_dirs:
        meta = _load_json(episode_dir / "episode_meta.json")
        annotation = annotations.get((str(meta.get("scene_id")), str(meta.get("episode_id"))), {})
        rows = _query_observations(episode_dir, args.max_frames_per_episode)
        terms = _landmark_terms(rows)
        episode_key = f"{meta.get('scene_id')}|{meta.get('episode_id')}|r{meta.get('rank', 0)}"
        per_variant: dict[str, Any] = {}
        paired_lseg = _lseg_summary(episode_dir)
        route = []
        for row in rows:
            route_node = row.get("route_node") or {}
            gps = route_node.get("gps")
            if gps is not None:
                route.append([float(gps[0]), float(gps[1])])
            rgb_path = episode_dir / str(row.get("rgb_path", ""))
            if not rgb_path.exists():
                raise FileNotFoundError(rgb_path)
            image = np.asarray(Image.open(rgb_path).convert("RGB"))
            depth = _load_depth(episode_dir, row)
            depth_hash_matches = _depth_hash_matches(depth, row)
            if depth_hash_matches is False:
                raise ValueError(f"depth hash mismatch: {episode_key} {row.get('observation_key')}")
            detections, latency = runner.predict(image)
            all_latencies.append(latency)
            camera = meta.get("camera_model") or {}
            intrinsic = np.asarray(camera.get("intrinsic"), dtype=np.float32)
            pose = np.asarray((row.get("pose") or {}).get("stage23_gt_camera_pose_map"), dtype=np.float32)
            for variant in aggregate:
                kept = detections if variant == "raw" else [
                    det for det in detections if hsgm_center_filter(det["box_xyxy"], image.shape[1], image.shape[0])
                ]
                for det in kept:
                    mask = _resize_mask(det["mask"], image.shape[1], image.shape[0])
                    projection = None
                    if depth is not None and intrinsic.shape == (3, 3) and pose.shape == (4, 4):
                        projection = project_mask_depth(
                            mask, depth, intrinsic, pose, sample_stride=args.sample_stride,
                            max_points=args.max_points_per_instance,
                        )
                    points = projection.pop("map_xyz") if projection is not None else np.zeros((0, 3), dtype=np.float32)
                    centroid = np.median(points, axis=0).tolist() if len(points) else None
                    item = {
                        "episode_key": episode_key,
                        "variant": variant,
                        "step_id": int(row.get("step_id", -1)),
                        "observation_index": int(row.get("observation_index", -1)),
                        "observation_key": row.get("observation_key"),
                        "detection_index": int(det["detection_index"]),
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "box_xyxy": det["box_xyxy"],
                        "depth_available": depth is not None,
                        "centroid_map": centroid,
                        "points_map": points.tolist(),
                        **(projection or {"mask_pixel_count": int(np.count_nonzero(mask)),
                                          "valid_depth_count": 0, "valid_depth_ratio": None,
                                          "sampled_point_count": 0}),
                    }
                    aggregate[variant].append(item)
            frame_record = {
                "episode_key": episode_key,
                "step_id": row.get("step_id"),
                "observation_index": row.get("observation_index"),
                "observation_key": row.get("observation_key"),
                "rgb_sha256": row.get("rgb_sha256"),
                "rgb_storage_is_lossy": str(row.get("rgb_storage_format", "")).lower() in {"jpg", "jpeg"},
                "rgb_file_sha256": _sha256(rgb_path),
                "depth_sha256": row.get("depth_sha256"),
                "depth_available": depth is not None,
                "depth_hash_matches": depth_hash_matches,
                "raw_detection_count": len(detections),
                "hsgm_center_filtered_detection_count": sum(
                    hsgm_center_filter(det["box_xyxy"], image.shape[1], image.shape[0]) for det in detections
                ),
                "inference_seconds": latency,
                "shadow_only": True,
                "action_applied": False,
            }
            _append_jsonl(frame_path, frame_record)
            manifest.append({key: frame_record[key] for key in (
                "episode_key", "step_id", "observation_index", "observation_key", "rgb_sha256", "depth_sha256"
            )})
        for variant in aggregate:
            instances = [row for row in aggregate[variant] if row["episode_key"] == episode_key]
            nodes = associate_instances(instances, args.merge_radius_m)
            gt = gt_surface_audit(nodes, meta)
            coverage = instruction_coverage((row["label"] for row in instances), terms)
            bands = Counter()
            for row in instances:
                points = np.asarray(row["points_map"], dtype=np.float32).reshape(-1, 3)
                for key, value in hsgm_height_band_counts(points, _base_z(next(
                    source for source in rows if int(source.get("observation_index", -1)) == row["observation_index"]
                ))).items():
                    bands[key] += value
            per_variant[variant] = {
                "detection_count": len(instances),
                "depth_attached_detection_count": sum(row["sampled_point_count"] > 0 for row in instances),
                "valid_depth_ratio_mean": _mean(row["valid_depth_ratio"] for row in instances),
                "node_count": len(nodes),
                "multi_view_node_count": sum(bool(node["multi_view"]) for node in nodes),
                "multi_view_node_rate": sum(bool(node["multi_view"]) for node in nodes) / max(1, len(nodes)),
                "cross_label_conflict": conflict_audit(nodes),
                "gt_audit": gt,
                "instruction_coverage": coverage,
                "height_band_surface": dict(bands),
            }
            _write_json(output / "episodes" / episode_key / f"{variant}_nodes.json", nodes)
            if variant == "hsgm_center_filtered":
                _bev(output / "episodes" / episode_key / "semantic_bev.png", instances, route)
        episode_reports.append({
            "episode_key": episode_key,
            "scene_split": _scene_split(meta.get("scene_id")),
            "audit_role": annotation.get("audit_role"),
            "query_frame_count": len(rows),
            "readable_depth_frame_count": sum(_load_depth(episode_dir, row) is not None for row in rows),
            "paired_lseg": paired_lseg,
            "variants": per_variant,
        })
    variant_summary = {}
    for variant, instances in aggregate.items():
        episode_rows = [row["variants"][variant] for row in episode_reports]
        variant_summary[variant] = {
            "detection_count": len(instances),
            "depth_attached_detection_count": sum(row["sampled_point_count"] > 0 for row in instances),
            "valid_depth_ratio_mean": _mean(row["valid_depth_ratio"] for row in instances),
            "node_count": sum(row["node_count"] for row in episode_rows),
            "multi_view_node_count": sum(row["multi_view_node_count"] for row in episode_rows),
            "cross_label_conflict_count": sum(row["cross_label_conflict"]["count"] for row in episode_rows),
            "gt_compatible_node_count": sum(row["gt_audit"].get("compatible_node_count", 0) for row in episode_rows),
            "gt_surface_hit_le_050m_count": sum(row["gt_audit"].get("surface_distance_le_050m_count", 0) for row in episode_rows),
            "instruction_landmark_matched_count": sum(row["instruction_coverage"]["matched_count"] for row in episode_rows),
            "instruction_landmark_term_count": sum(row["instruction_coverage"]["term_count"] for row in episode_rows),
        }
        compatible = variant_summary[variant]["gt_compatible_node_count"]
        variant_summary[variant]["gt_surface_hit_le_050m_rate"] = (
            variant_summary[variant]["gt_surface_hit_le_050m_count"] / compatible if compatible else None
        )
        terms = variant_summary[variant]["instruction_landmark_term_count"]
        variant_summary[variant]["instruction_landmark_coverage"] = (
            variant_summary[variant]["instruction_landmark_matched_count"] / terms if terms else None
        )
    raw_count = variant_summary["raw"]["detection_count"]
    lseg_episodes = [row["paired_lseg"] for row in episode_reports if row.get("paired_lseg")]
    paired_lseg_summary = {"episode_count": len(lseg_episodes)}
    for variant in ("raw", "filtered"):
        rows = [row[variant] for row in lseg_episodes]
        compatible = sum(int(row.get("gt_compatible_node_count") or 0) for row in rows)
        hits = sum(int(row.get("gt_surface_hit_le_050m_count") or 0) for row in rows)
        paired_lseg_summary[variant] = {
            "node_count": sum(int(row.get("node_count") or 0) for row in rows),
            "multi_view_node_count": sum(int(row.get("multi_view_node_count") or 0) for row in rows),
            "severe_cross_label_conflict_count": sum(
                int(row.get("severe_cross_label_conflict_count") or 0) for row in rows
            ),
            "gt_compatible_node_count": compatible,
            "gt_surface_hit_le_050m_count": hits,
            "gt_surface_hit_le_050m_rate": hits / compatible if compatible else None,
        }
    strata = {}
    stratum_filters = {
        "dev": lambda row: row["scene_split"] == "dev",
        "holdout": lambda row: row["scene_split"] == "holdout",
        "flat": lambda row: "flat" in str(row.get("audit_role") or "").lower(),
        "vertical": lambda row: any(
            token in str(row.get("audit_role") or "").lower()
            for token in ("stairs", "mixed_height", "height_change")
        ),
    }
    for stratum, predicate in stratum_filters.items():
        selected = [row for row in episode_reports if predicate(row)]
        if not selected:
            continue
        strata[stratum] = {"episode_count": len(selected)}
        for variant in aggregate:
            rows = [row["variants"][variant] for row in selected]
            compatible = sum(row["gt_audit"].get("compatible_node_count", 0) for row in rows)
            hits = sum(row["gt_audit"].get("surface_distance_le_050m_count", 0) for row in rows)
            terms = sum(row["instruction_coverage"]["term_count"] for row in rows)
            matched = sum(row["instruction_coverage"]["matched_count"] for row in rows)
            strata[stratum][variant] = {
                "detection_count": sum(row["detection_count"] for row in rows),
                "node_count": sum(row["node_count"] for row in rows),
                "multi_view_node_count": sum(row["multi_view_node_count"] for row in rows),
                "cross_label_conflict_count": sum(row["cross_label_conflict"]["count"] for row in rows),
                "gt_surface_hit_le_050m_rate": hits / compatible if compatible else None,
                "instruction_landmark_coverage": matched / terms if terms else None,
            }
    report = {
        "audit_name": "stage31_hsgm_yoloe_fixed_replay",
        "schema_version": "stage31_hsgm_yoloe_replay_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "online_geometry_backend_changed": False,
        "online_semantic_backend_changed": False,
        "sparse_occ_is_only_safety_authority": True,
        "hsgm_surface_is_traversability_truth": False,
        "online_gt_fields_used": [],
        "audit_gt_fields_used": ["semantic_scene_gt", "coordinate_transforms.map_to_habitat_world"],
        "input": {
            "replay_root": str(args.replay_root),
            "checkpoint": str(args.checkpoint),
            "hsgm_root": str(args.hsgm_root),
            "model_confidence": 0.6,
            "model_iou": 0.3,
            "class_count": len(runner.class_names),
            "query_frames_only": True,
            "episode_count": len(episode_reports),
            "query_frame_count": len(manifest),
            "readable_depth_frame_count": sum(row["readable_depth_frame_count"] for row in episode_reports),
        },
        "model": {
            "load_seconds": runner.load_seconds,
            "inference_seconds_mean": _mean(all_latencies),
            "inference_seconds_p50": _percentile(all_latencies, 50),
            "inference_seconds_p95": _percentile(all_latencies, 95),
            "cuda": runner.cuda_stats(),
        },
        "variants": variant_summary,
        "paired_lseg": paired_lseg_summary,
        "strata": strata,
        "hsgm_center_filter_retention_rate": (
            variant_summary["hsgm_center_filtered"]["detection_count"] / raw_count if raw_count else None
        ),
        "episodes": episode_reports,
        "decision_scope": "offline_semantic_and_height_surface_audit_only",
        "detector_revalidation_required": False,
        "integrity": {
            "manifest_count_matches_query_frame_count": len(manifest) == len(all_latencies),
            "depth_hash_mismatch_count": sum(
                row.get("depth_hash_matches") is False for row in _jsonl(frame_path)
            ),
            "rgb_raw_hash_reverification_possible": False,
            "rgb_raw_hash_reverification_reason": "ledger RGB is JPEG and original-array hash predates lossy encoding",
        },
    }
    _write_json(output / "stage31_manifest.json", manifest)
    _write_json(output / "stage31_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hsgm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--max-points-per-instance", type=int, default=4096)
    parser.add_argument("--merge-radius-m", type=float, default=0.5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

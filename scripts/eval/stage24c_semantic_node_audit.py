"""Offline Stage24C semantic surface/node fusion audit.

This script consumes Stage24B sparse RGB-D surface samples only.  It never
imports the evaluator and cannot affect S2, SparseOcc safety, triage, or
actions.  Nodes are deliberately class-conditioned and merged by a fixed
metric radius so the result is an auditable memory baseline rather than a
navigation heuristic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _tokens(value: str) -> set:
    return {
        token
        for token in str(value or "").lower().replace("-", "_").split("_")
        if len(token) > 2
    }


ALIASES = {
    "shelving": {"shelving", "shelf", "cabinet"},
    "closet": {"closet", "wardrobe"},
    "floor": {"floor", "floors"},
    "wall": {"wall", "walls"},
    "door": {"door", "doorway", "entrance"},
    "painting": {"painting", "picture", "artwork"},
    "cabinet": {"cabinet", "chest", "chest_of_drawers", "drawer"},
    "stairs": {"stairs", "stair", "staircase"},
}


def _compatible(label: str, category: str) -> bool:
    return bool(set(ALIASES.get(label, {label})).intersection(_tokens(category)))


def _load_samples(report: dict, report_dir: Path, confidence_threshold: float):
    labels = list(report.get("labels") or [])
    frames = []
    total = 0
    kept = 0
    for record in report.get("records") or []:
        sample_path = report_dir / str(record.get("surface_samples_path") or "")
        if not sample_path.is_file():
            raise FileNotFoundError(f"missing Stage24B surface sample: {sample_path}")
        data = np.load(sample_path)
        xyz = np.asarray(data["map_xyz"], dtype=np.float32)
        class_id = np.asarray(data["class_id"], dtype=np.int32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
        valid = np.all(np.isfinite(xyz), axis=1) & np.isfinite(confidence)
        valid &= confidence >= float(confidence_threshold)
        valid &= class_id >= 0
        valid &= class_id < len(labels)
        total += int(valid.size)
        kept += int(np.count_nonzero(valid))
        frames.append({
            "observation_index": int(record.get("observation_index", -1)),
            "step_id": record.get("step_id"),
            "map_xyz": xyz[valid],
            "class_id": class_id[valid],
            "confidence": confidence[valid],
        })
    return labels, frames, total, kept


def _merge_nodes(labels: List[str], frames: Iterable[dict], radius_m: float):
    nodes: List[dict] = []
    buckets: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    radius = float(radius_m)

    def bucket_for(point):
        return tuple(np.floor(np.asarray(point, dtype=np.float32) / radius).astype(np.int64).tolist())

    for frame in frames:
        for point, class_id, confidence in zip(
            frame["map_xyz"], frame["class_id"], frame["confidence"]
        ):
            point = np.asarray(point, dtype=np.float32)
            class_id = int(class_id)
            base = bucket_for(point)
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        candidates.extend(buckets.get((class_id, base[0] + dx, base[1] + dy, base[2] + dz), []))
            best = None
            best_distance = None
            for index in candidates:
                distance = float(np.linalg.norm(nodes[index]["centroid"] - point))
                if distance <= radius and (best_distance is None or distance < best_distance):
                    best, best_distance = index, distance
            if best is None:
                best = len(nodes)
                nodes.append({
                    "node_id": f"SN{best:04d}",
                    "class_id": class_id,
                    "label": labels[class_id],
                    "centroid": point.copy(),
                    "point_count": 0,
                    "confidence_sum": 0.0,
                    "source_observations": set(),
                    "source_steps": set(),
                    "min_xyz": point.copy(),
                    "max_xyz": point.copy(),
                })
                buckets[(class_id, *base)].append(best)
            node = nodes[best]
            old_count = node["point_count"]
            node["centroid"] = (node["centroid"] * old_count + point) / float(old_count + 1)
            node["point_count"] = old_count + 1
            node["confidence_sum"] += float(confidence)
            node["source_observations"].add(int(frame["observation_index"]))
            if frame.get("step_id") is not None:
                node["source_steps"].add(int(frame["step_id"]))
            node["min_xyz"] = np.minimum(node["min_xyz"], point)
            node["max_xyz"] = np.maximum(node["max_xyz"], point)
    return nodes


def _finalize_nodes(nodes: List[dict], gt_entries: List[dict], radius_m: float, map_to_gt: np.ndarray):
    for node in nodes:
        node["centroid"] = [float(value) for value in node["centroid"]]
        node["min_xyz"] = [float(value) for value in node["min_xyz"]]
        node["max_xyz"] = [float(value) for value in node["max_xyz"]]
        node["mean_confidence"] = float(node.pop("confidence_sum") / max(1, node["point_count"]))
        node["source_observations"] = sorted(node["source_observations"])
        node["source_steps"] = sorted(node["source_steps"])
        compatible = [item for item in gt_entries if _compatible(node["label"], item.get("category", ""))]
        node["gt_compatible_count"] = int(len(compatible))
        if compatible:
            point_map = np.asarray(node["centroid"], dtype=np.float32)
            point = (map_to_gt @ np.array([*point_map, 1.0], dtype=np.float32))[:3]
            nearest = min(compatible, key=lambda item: float(np.linalg.norm(np.asarray(item["center"]) - point)))
            lower = np.asarray(nearest.get("lower", nearest["center"]), dtype=np.float32)
            upper = np.asarray(nearest.get("upper", nearest["center"]), dtype=np.float32)
            delta = np.maximum(np.maximum(lower - point, 0.0), point - upper)
            node["gt_nearest_category"] = nearest.get("category")
            node["gt_surface_distance_m"] = float(np.linalg.norm(delta))
            node["gt_category_agreement"] = True
        else:
            node["gt_nearest_category"] = None
            node["gt_surface_distance_m"] = None
            node["gt_category_agreement"] = None

    conflicts = []
    for i, left in enumerate(nodes):
        a = np.asarray(left["centroid"], dtype=np.float32)
        for right in nodes[i + 1:]:
            if left["class_id"] == right["class_id"]:
                continue
            b = np.asarray(right["centroid"], dtype=np.float32)
            distance = float(np.linalg.norm(a - b))
            if distance <= float(radius_m) / 2.0:
                conflicts.append({
                    "left_node_id": left["node_id"],
                    "left_label": left["label"],
                    "right_node_id": right["node_id"],
                    "right_label": right["label"],
                    "distance_m": distance,
                })
    return conflicts


def audit(report_path: Path, output_path: Path, radius_m: float, confidence_threshold: float):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels, frames, total, kept = _load_samples(report, report_path.parent, confidence_threshold)
    nodes = _merge_nodes(labels, frames, radius_m)
    gt_entries = list(report.get("semantic_gt_entries") or [])
    transforms = report.get("coordinate_transforms") or {}
    map_to_gt = np.asarray(transforms.get("map_to_habitat_world"), dtype=np.float32)
    if map_to_gt.shape != (4, 4):
        raise ValueError("Stage24B report is missing map_to_habitat_world transform")
    conflicts = _finalize_nodes(nodes, gt_entries, radius_m, map_to_gt)
    support_counts = Counter(len(node["source_observations"]) for node in nodes)
    by_label = Counter(node["label"] for node in nodes)
    multi_view = [node for node in nodes if len(node["source_observations"]) >= 2]
    gt_distances = [node["gt_surface_distance_m"] for node in nodes if node["gt_surface_distance_m"] is not None]
    result = {
        "audit_name": "stage24c_semantic_node",
        "source_report": str(report_path),
        "coordinate_transforms": transforms,
        "radius_m": float(radius_m),
        "confidence_threshold": float(confidence_threshold),
        "labels": labels,
        "frame_count": len(frames),
        "sample_count": int(total),
        "kept_sample_count": int(kept),
        "node_count": len(nodes),
        "nodes_by_label": dict(sorted(by_label.items())),
        "support_frame_histogram": {str(key): int(value) for key, value in sorted(support_counts.items())},
        "multi_view_node_count": len(multi_view),
        "multi_view_node_rate": float(len(multi_view) / max(1, len(nodes))),
        "cross_label_conflict_count": len(conflicts),
        "gt_compatible_node_count": len(gt_distances),
        "gt_surface_distance_m_median": float(np.median(gt_distances)) if gt_distances else None,
        "gt_surface_distance_le_050m_rate": float(np.mean(np.asarray(gt_distances) <= 0.50)) if gt_distances else None,
        "source_binding_valid": all(node["source_observations"] and node["source_steps"] for node in nodes),
        "nodes": nodes,
        "cross_label_conflicts": conflicts,
        "decision_status": "audit_only_not_navigation_ready",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=0.50)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    args = parser.parse_args()
    audit(args.report, args.output, args.radius_m, args.confidence_threshold)


if __name__ == "__main__":
    main()

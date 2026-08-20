"""Offline multi-view fusion utilities for Stage24F semantic audits.

The module never mutates SparseOcc or navigation state. It consumes frozen
RGB-D-pose replay evidence and produces comparable voxel maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FrameVoxelEvidence:
    keys: np.ndarray
    points: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    embeddings: Optional[np.ndarray]
    hard_votes: np.ndarray
    confidence: np.ndarray
    depth_m: np.ndarray
    view_direction: np.ndarray
    observation_index: int


def aggregate_frame_voxels(
    points: np.ndarray, probabilities: np.ndarray, logits: np.ndarray,
    embeddings: Optional[np.ndarray], depth_m: np.ndarray, camera_center: np.ndarray,
    observation_index: int, voxel_size_m: float,
) -> FrameVoxelEvidence:
    """Collapse repeated pixels from one frame before cross-frame fusion."""
    points = np.asarray(points, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    depth_m = np.asarray(depth_m, dtype=np.float32).reshape(-1)
    class_count = probabilities.shape[1]
    if not len(points):
        empty_vec = np.zeros((0, class_count), dtype=np.float32)
        return FrameVoxelEvidence(
            np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.float32),
            empty_vec, empty_vec.copy(),
            None if embeddings is None else np.zeros((0, embeddings.shape[1]), dtype=np.float32),
            empty_vec.copy(), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            int(observation_index),
        )
    keys = np.floor(points / float(voxel_size_m)).astype(np.int32)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)

    def mean_rows(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)[order]
        return np.add.reduceat(values, starts, axis=0) / counts.reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )

    frame_points = mean_rows(points)
    frame_probs = mean_rows(probabilities)
    frame_logits = mean_rows(logits)
    frame_embeddings = None if embeddings is None else mean_rows(embeddings)
    top1 = np.argmax(probabilities, axis=1)
    one_hot = np.eye(class_count, dtype=np.float32)[top1]
    hard_votes = mean_rows(one_hot)
    confidence = mean_rows(np.max(probabilities, axis=1)).reshape(-1)
    frame_depth = mean_rows(depth_m).reshape(-1)
    direction = frame_points - np.asarray(camera_center, dtype=np.float32).reshape(1, 3)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-6)
    return FrameVoxelEvidence(
        sorted_keys[starts], frame_points, frame_probs, frame_logits,
        frame_embeddings, hard_votes, confidence, frame_depth, direction,
        int(observation_index),
    )


def _concat_frames(
    frames: Sequence[FrameVoxelEvidence], field: str,
) -> np.ndarray:
    values = [getattr(frame, field) for frame in frames if len(frame.keys)]
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values, axis=0)


def _view_repeat_counts(keys: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Count near-identical azimuth/elevation observations per voxel."""
    azimuth = np.floor((np.arctan2(directions[:, 1], directions[:, 0]) + np.pi) /
                       np.deg2rad(30.0)).astype(np.int16)
    elevation = np.floor((np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)) + np.pi / 2) /
                         np.deg2rad(20.0)).astype(np.int16)
    view_keys = np.column_stack([keys, azimuth, elevation])
    _, inverse, counts = np.unique(view_keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.float32)


def fuse_voxel_evidence(
    frames: Sequence[FrameVoxelEvidence], mode: str,
    text_features: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Fuse per-frame voxel evidence with one of the Stage24F policies.

    Modes: ``hard`` (top-1 votes), ``prob`` (class probabilities),
    ``embedding`` (VLMaps-style LSeg embeddings), and ``robust_prob``
    (distance, confidence, and repeated-view weighted probabilities).
    """
    active = [frame for frame in frames if len(frame.keys)]
    if not active:
        raise ValueError("No frame evidence to fuse")
    keys = _concat_frames(active, "keys").astype(np.int32)
    points = _concat_frames(active, "points").astype(np.float32)
    probabilities = _concat_frames(active, "probabilities").astype(np.float32)
    hard_votes = _concat_frames(active, "hard_votes").astype(np.float32)
    confidence = _concat_frames(active, "confidence").astype(np.float32)
    depth_m = _concat_frames(active, "depth_m").astype(np.float32)
    directions = _concat_frames(active, "view_direction").astype(np.float32)
    observations = np.concatenate([
        np.full(len(frame.keys), frame.observation_index, dtype=np.int32)
        for frame in active
    ])

    distance_weight = np.exp(-(depth_m ** 2) / (2.0 * 0.6)).astype(np.float32)
    weights = (
        np.ones_like(distance_weight) if mode == "hard"
        else np.maximum(distance_weight, 1e-4)
    )
    if mode == "robust_prob":
        class_count = probabilities.shape[1]
        chance = 1.0 / float(class_count)
        reliability = np.clip((confidence - chance) / (1.0 - chance), 0.05, 1.0) ** 2
        repeat = _view_repeat_counts(keys, directions)
        weights *= reliability.astype(np.float32) / np.sqrt(np.maximum(repeat, 1.0))

    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1]
    view_count = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    sorted_weight = weights[order]
    weight_sum = np.add.reduceat(sorted_weight, starts)

    def weighted_mean(values: np.ndarray) -> np.ndarray:
        sorted_values = np.asarray(values)[order]
        weighted = sorted_values * sorted_weight.reshape(
            (-1,) + (1,) * (sorted_values.ndim - 1)
        )
        summed = np.add.reduceat(weighted, starts, axis=0)
        return summed / np.maximum(weight_sum, 1e-8).reshape(
            (-1,) + (1,) * (summed.ndim - 1)
        )

    fused_points = weighted_mean(points).astype(np.float32)
    vote_distribution = weighted_mean(hard_votes).astype(np.float32)
    if mode == "hard":
        evidence = vote_distribution
    elif mode in {"prob", "robust_prob"}:
        evidence = weighted_mean(probabilities).astype(np.float32)
    elif mode == "embedding":
        embeddings = _concat_frames(active, "embeddings")
        if embeddings.ndim != 2 or text_features is None:
            raise ValueError("Embedding fusion requires embeddings and text features")
        fused_embedding = weighted_mean(embeddings.astype(np.float32))
        evidence = fused_embedding @ np.asarray(text_features, dtype=np.float32).T
        evidence -= np.max(evidence, axis=1, keepdims=True)
        evidence = np.exp(evidence)
        evidence /= np.maximum(np.sum(evidence, axis=1, keepdims=True), 1e-8)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

    evidence = np.asarray(evidence, dtype=np.float32)
    evidence /= np.maximum(np.sum(evidence, axis=1, keepdims=True), 1e-8)
    class_id = np.argmax(evidence, axis=1).astype(np.int16)
    sorted_scores = np.sort(evidence, axis=1)
    margin = (sorted_scores[:, -1] - sorted_scores[:, -2]).astype(np.float32)
    entropy = (-np.sum(evidence * np.log(np.maximum(evidence, 1e-8)), axis=1) /
               np.log(evidence.shape[1])).astype(np.float32)
    agreement = np.max(vote_distribution, axis=1).astype(np.float32)
    conflict = (np.sum(vote_distribution > 1e-6, axis=1) > 1)
    return {
        "keys": sorted_keys[starts], "map_xyz": fused_points,
        "class_id": class_id, "confidence": np.max(evidence, axis=1),
        "margin": margin, "entropy": entropy, "view_count": view_count,
        "cross_view_agreement": agreement, "conflict": conflict,
        "weight_sum": weight_sum.astype(np.float32), "evidence": evidence,
        "source_record_count": np.asarray(len(observations), dtype=np.int64),
    }


def isolated_voxel_rate(keys: np.ndarray, class_id: np.ndarray) -> float:
    lookup = {tuple(key.tolist()): int(label) for key, label in zip(keys, class_id)}
    isolated = 0
    for key, label in lookup.items():
        if not any(
            lookup.get((key[0] + dx, key[1] + dy, key[2] + dz)) == label
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1))
        ):
            isolated += 1
    return float(isolated / max(1, len(lookup)))

"""Audit-only online LSeg surface memory for Frozen-S2 evaluation.

The shadow reads the same RGB-D observation and camera pose as S2, but none of
its outputs are exposed to prompts, safety checks, candidates, or actions.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_LABELS = [
    "door", "chair", "table", "stairs", "sofa", "bed", "cabinet",
    "window", "wall", "floor", "shelving", "closet", "painting", "other",
]

PALETTE = np.asarray([
    [220, 20, 60], [30, 144, 255], [50, 205, 50], [255, 165, 0],
    [138, 43, 226], [255, 105, 180], [0, 206, 209], [255, 215, 0],
    [128, 128, 128], [244, 164, 96], [46, 139, 87], [70, 130, 180],
    [255, 99, 71], [30, 30, 30],
], dtype=np.uint8)

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

BENIGN_NEARBY_LABEL_PAIRS = {
    tuple(sorted(pair)) for pair in (
        ("floor", "stairs"), ("cabinet", "shelving"),
        ("bed", "sofa"), ("chair", "sofa"),
        ("door", "floor"), ("floor", "wall"),
        ("stairs", "wall"), ("floor", "shelving"),
    )
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tokens(value: str) -> set:
    return {
        token for token in str(value or "").lower().replace("-", "_").split("_")
        if len(token) > 2
    }


class OnlineLSegSemanticShadow:
    """Run LSeg on S2 query frames and retain an independent 3-D audit map."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        camera_intrinsic: np.ndarray,
        device: Any,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("lseg_online_shadow_enable", False))
        self.repo = Path(str(cfg.get("lseg_online_shadow_repo", ""))).expanduser()
        self.checkpoint = Path(
            str(cfg.get("lseg_online_shadow_checkpoint", ""))
        ).expanduser()
        configured_device = str(cfg.get("lseg_online_shadow_device", "same"))
        self.device = str(device) if configured_device == "same" else configured_device
        self.labels = list(cfg.get("lseg_online_shadow_labels") or DEFAULT_LABELS)
        self.confidence_threshold = float(
            cfg.get("lseg_online_shadow_confidence_threshold", 0.35)
        )
        self.sample_stride = max(1, int(cfg.get("lseg_online_shadow_sample_stride", 8)))
        self.min_depth_m = float(cfg.get("lseg_online_shadow_min_depth_m", 0.15))
        self.max_depth_m = float(cfg.get("lseg_online_shadow_max_depth_m", 5.0))
        self.merge_radius_m = max(0.05, float(cfg.get("lseg_online_shadow_merge_radius_m", 0.50)))
        self.max_surface_samples = max(1000, int(
            cfg.get("lseg_online_shadow_max_surface_samples", 250000)
        ))
        self.component_filter_enable = bool(
            cfg.get("lseg_online_shadow_component_filter_enable", False)
        )
        self.component_filter_min_samples = max(1, int(
            cfg.get("lseg_online_shadow_component_filter_min_samples", 4)
        ))
        self.component_filter_radius_m = max(0.01, float(
            cfg.get("lseg_online_shadow_component_filter_radius_m", 0.20)
        ))
        self.component_filter_min_neighbors = max(1, int(
            cfg.get("lseg_online_shadow_component_filter_min_neighbors", 2)
        ))
        self.strong_min_views = max(2, int(
            cfg.get("lseg_online_shadow_strong_min_views", 2)
        ))
        self.strong_min_points = max(1, int(
            cfg.get("lseg_online_shadow_strong_min_points", 32)
        ))
        self.strong_min_confidence = float(
            cfg.get("lseg_online_shadow_strong_min_confidence", 0.40)
        )
        self.save_overlay = bool(cfg.get("lseg_online_shadow_save_overlay", True))
        self.save_surface = bool(cfg.get("lseg_online_shadow_save_surface", True))
        self.save_visualizations = bool(
            cfg.get("lseg_online_shadow_save_visualizations", True)
        )
        intrinsic = np.asarray(camera_intrinsic, dtype=np.float32)
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]
        if intrinsic.shape != (3, 3):
            raise ValueError(
                "Expected camera intrinsic shape (3, 3) or (4, 4), "
                f"got {intrinsic.shape}"
            )
        self.camera_intrinsic = intrinsic
        self.root: Optional[Path] = None
        self.episode_dir: Optional[Path] = None
        self.model = None
        self.transform = None
        self.crop_size = 480
        self.model_load_seconds: Optional[float] = None
        self.load_error: Optional[str] = None
        self.disabled_after_error = False
        self._baseline_cuda = self._cuda_stats()
        self._before_load_cuda: Dict[str, Any] = {}
        self._after_load_cuda: Dict[str, Any] = {}
        self.reset_episode()

    def set_root(self, root: Optional[str]) -> None:
        self.root = None if not root else Path(root)

    def _cuda_stats(self) -> Dict[str, Any]:
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return {"available": False}
        index = torch.device(self.device).index
        if index is None:
            index = torch.cuda.current_device()
        return {
            "available": True,
            "device": f"cuda:{index}",
            "allocated_mb": float(torch.cuda.memory_allocated(index) / 1048576.0),
            "reserved_mb": float(torch.cuda.memory_reserved(index) / 1048576.0),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated(index) / 1048576.0),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved(index) / 1048576.0),
        }

    @staticmethod
    def _capture_rng() -> Dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def reset_episode(self, **meta: Any) -> Dict[str, Any]:
        self.episode_meta = _jsonable(meta)
        self.records: List[Dict[str, Any]] = []
        self.surface_frames: List[Dict[str, np.ndarray]] = []
        self.filtered_surface_frames: List[Dict[str, np.ndarray]] = []
        self.inference_seconds: List[float] = []
        self.errors: List[str] = []
        self._stored_surface_count = 0
        self.episode_dir = None
        if self.enabled and self.root is not None:
            scene_id = str(meta.get("scene_id", "unknown_scene")).replace("/", "_")
            episode_id = str(meta.get("episode_id", "unknown_episode"))
            rank = str(meta.get("rank", "0"))
            self.episode_dir = (
                self.root / "online_lseg_shadow" / f"{scene_id}_{episode_id}_r{rank}"
            )
            (self.episode_dir / "overlays").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "surfaces").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "visualizations").mkdir(parents=True, exist_ok=True)
            self._write_json(self.episode_dir / "episode_meta.json", self.episode_meta)
        return {"enabled": self.enabled}

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_model(self) -> None:
        if self.model is not None or self.load_error is not None:
            return
        if not self.repo.is_dir():
            raise FileNotFoundError(f"VLMaps repository not found: {self.repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"LSeg checkpoint not found: {self.checkpoint}")
        rng = self._capture_rng()
        started = time.perf_counter()
        try:
            self._before_load_cuda = self._cuda_stats()
            repo_str = str(self.repo)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            from torchvision import transforms
            from vlmaps.lseg.modules.models.lseg_net import LSegEncNet

            model = LSegEncNet(
                "", arch_option=0, block_depth=0, activation="lrelu",
                crop_size=self.crop_size,
            )
            payload = self._load_checkpoint(self.checkpoint)
            state = payload.get("state_dict", payload)
            state = {
                key[4:] if key.startswith("net.") else key: value
                for key, value in state.items()
            }
            model.load_state_dict(state, strict=False)
            self.model = model.eval().to(self.device)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
            self.model_load_seconds = float(time.perf_counter() - started)
            self._after_load_cuda = self._cuda_stats()
        finally:
            self._restore_rng(rng)

    @staticmethod
    def _load_checkpoint(path: Path) -> Any:
        # Online/replay audits only accept the pre-extracted tensor state dict.
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    @contextmanager
    def _deterministic_inference():
        state = {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision("highest")
            torch.use_deterministic_algorithms(True, warn_only=False)
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
            torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
            torch.set_float32_matmul_precision(state["matmul_precision"])
            torch.use_deterministic_algorithms(
                state["deterministic_algorithms"],
                warn_only=state["deterministic_warn_only"],
            )

    def _infer_logits(self, image: np.ndarray) -> np.ndarray:
        from vlmaps.lseg.additional_utils.models import crop_image, pad_image, resize_image

        source_h, source_w = image.shape[:2]
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        _, _, height, width = tensor.shape
        if height > width:
            resized_h = self.crop_size
            resized_w = int(width * self.crop_size / height + 0.5)
        else:
            resized_w = self.crop_size
            resized_h = int(height * self.crop_size / width + 0.5)
        resized = resize_image(
            tensor, resized_h, resized_w, mode="bilinear", align_corners=True
        )
        padded = pad_image(
            resized, [0.5] * 3, [0.5] * 3, self.crop_size
        )
        with torch.inference_mode():
            _, logits = self.model(padded, self.labels)
        logits = crop_image(logits, 0, resized_h, 0, resized_w)
        logits = torch.nn.functional.interpolate(
            logits, size=(source_h, source_w), mode="bilinear", align_corners=True
        )
        return logits[0].float().cpu().numpy()

    def infer_logits_and_sampled_embeddings(
        self, image: np.ndarray, pixel_y: np.ndarray, pixel_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return dense logits and only the requested 512-D LSeg features.

        Stage24F uses this offline-only entry point to avoid materializing a
        source-resolution H x W x 512 tensor. The online shadow remains on the
        existing logits-only path.
        """
        from vlmaps.lseg.additional_utils.models import crop_image, pad_image, resize_image

        source_h, source_w = image.shape[:2]
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        _, _, height, width = tensor.shape
        if height > width:
            resized_h = self.crop_size
            resized_w = int(width * self.crop_size / height + 0.5)
        else:
            resized_w = self.crop_size
            resized_h = int(height * self.crop_size / width + 0.5)
        resized = resize_image(
            tensor, resized_h, resized_w, mode="bilinear", align_corners=True
        )
        padded = pad_image(resized, [0.5] * 3, [0.5] * 3, self.crop_size)
        with torch.inference_mode():
            embeddings, logits = self.model(padded, self.labels)
        embeddings = crop_image(embeddings, 0, resized_h, 0, resized_w)
        logits = crop_image(logits, 0, resized_h, 0, resized_w)
        logits = torch.nn.functional.interpolate(
            logits, size=(source_h, source_w), mode="bilinear", align_corners=True
        )

        sample_y = torch.as_tensor(pixel_y, device=self.device, dtype=torch.float32)
        sample_x = torch.as_tensor(pixel_x, device=self.device, dtype=torch.float32)
        if source_h > 1:
            sample_y = sample_y * float(embeddings.shape[2] - 1) / float(source_h - 1)
        else:
            sample_y.zero_()
        if source_w > 1:
            sample_x = sample_x * float(embeddings.shape[3] - 1) / float(source_w - 1)
        else:
            sample_x.zero_()
        grid_x = 2.0 * sample_x / max(1, embeddings.shape[3] - 1) - 1.0
        grid_y = 2.0 * sample_y / max(1, embeddings.shape[2] - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
        sampled = torch.nn.functional.grid_sample(
            embeddings, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        )[0, :, 0, :].T
        return (
            logits[0].float().cpu().numpy(),
            sampled.float().cpu().numpy(),
        )

    def lseg_text_features(self) -> np.ndarray:
        """Return normalized CLIP text features for the configured labels."""
        import clip

        tokens = clip.tokenize(self.labels).to(self.device)
        with torch.inference_mode():
            features = self.model.clip_pretrained.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return features.cpu().numpy()

    def _project(
        self, pred: np.ndarray, confidence: np.ndarray, depth: np.ndarray,
        camera_pose_map: np.ndarray, occ_memory: Any,
    ) -> Dict[str, np.ndarray]:
        height, width = depth.shape[:2]
        ys, xs = np.mgrid[0:height:self.sample_stride, 0:width:self.sample_stride]
        sampled_count = int(ys.size)
        values = depth[ys, xs].astype(np.float32)
        valid = np.isfinite(values) & (values >= self.min_depth_m) & (values <= self.max_depth_m)
        ys, xs, values = ys[valid], xs[valid], values[valid]
        class_id = pred[ys, xs].astype(np.int16)
        conf = confidence[ys, xs].astype(np.float32)
        fx, fy = float(self.camera_intrinsic[0, 0]), float(self.camera_intrinsic[1, 1])
        cx, cy = float(self.camera_intrinsic[0, 2]), float(self.camera_intrinsic[1, 2])
        x = (xs.astype(np.float32) - cx) * values / fx
        y = (ys.astype(np.float32) - cy) * values / fy
        camera = np.stack([x, y, values, np.ones_like(values)], axis=1)
        points = (camera_pose_map @ camera.T).T[:, :3].astype(np.float32)
        keep = conf >= self.confidence_threshold
        points, class_id, conf = points[keep], class_id[keep], conf[keep]
        ys, xs, values = ys[keep], xs[keep], values[keep]

        occ_state = np.full(len(points), 2, dtype=np.int8)  # 0 occupied, 1 free, 2 unknown
        if len(points) and occ_memory is not None:
            rows = (occ_memory.gs / 2 - (points[:, 0] / occ_memory.cs).astype(np.int64)).astype(np.int64)
            cols = (occ_memory.gs / 2 - (points[:, 1] / occ_memory.cs).astype(np.int64)).astype(np.int64)
            occupied = set(occ_memory.occ2d_counts.keys())
            free = set(occ_memory.free2d_counts.keys()) - occupied
            for index, cell in enumerate(zip(rows.tolist(), cols.tolist())):
                if cell in occupied:
                    occ_state[index] = 0
                elif cell in free:
                    occ_state[index] = 1
        return {
            "map_xyz": points, "class_id": class_id, "confidence": conf,
            "depth_m": values.astype(np.float32), "pixel_y": ys.astype(np.int16),
            "pixel_x": xs.astype(np.int16), "occ_state": occ_state,
            "sampled_count": np.asarray(sampled_count),
            "valid_depth_count": np.asarray(int(np.count_nonzero(valid))),
        }

    @staticmethod
    def _slice_samples(
        samples: Dict[str, np.ndarray], indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        count = len(samples["map_xyz"])
        return {
            key: value[indices]
            if isinstance(value, np.ndarray) and value.ndim and len(value) == count
            else value
            for key, value in samples.items()
        }

    def _filter_surface_samples(
        self, samples: Dict[str, np.ndarray], image_shape: Tuple[int, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray]:
        """Remove tiny 2-D islands and spatially isolated 3-D attachments.

        The result is always an indexed subset of ``samples``. Connectivity is
        evaluated on the sampled image grid and 3-D density only within each
        same-label component, so nearby surfaces with different labels cannot
        validate one another.
        """
        count = len(samples["map_xyz"])
        all_indices = np.arange(count, dtype=np.int64)
        if not self.component_filter_enable or not count:
            stats = {
                "enabled": self.component_filter_enable,
                "raw_sample_count": count,
                "retained_sample_count": count,
                "rejected_sample_count": 0,
                "retention_rate": 1.0 if count else None,
                "component_count": 0,
                "small_component_count": 0,
                "edge_touch_component_count": 0,
                "small_component_rejected_sample_count": 0,
                "density_rejected_sample_count": 0,
            }
            return self._slice_samples(samples, all_indices), stats, all_indices

        pixel_y = samples["pixel_y"].astype(np.int64)
        pixel_x = samples["pixel_x"].astype(np.int64)
        class_id = samples["class_id"].astype(np.int64)
        lookup = {
            (int(label), int(y), int(x)): index
            for index, (label, y, x) in enumerate(zip(class_id, pixel_y, pixel_x))
        }
        unseen = set(range(count))
        components: List[List[int]] = []
        stride = self.sample_stride
        while unseen:
            seed = unseen.pop()
            component = [seed]
            stack = [seed]
            while stack:
                index = stack.pop()
                label = int(class_id[index])
                y, x = int(pixel_y[index]), int(pixel_x[index])
                for dy in (-stride, 0, stride):
                    for dx in (-stride, 0, stride):
                        if dy == 0 and dx == 0:
                            continue
                        neighbor = lookup.get((label, y + dy, x + dx))
                        if neighbor is not None and neighbor in unseen:
                            unseen.remove(neighbor)
                            component.append(neighbor)
                            stack.append(neighbor)
            components.append(component)

        keep = np.zeros(count, dtype=bool)
        small_component_count = 0
        small_rejected = 0
        density_rejected = 0
        edge_touch_count = 0
        height, width = image_shape
        radius = self.component_filter_radius_m
        radius_sq = radius * radius
        for component in components:
            component_indices = np.asarray(component, dtype=np.int64)
            if np.any(
                (pixel_y[component_indices] == 0)
                | (pixel_x[component_indices] == 0)
                | (pixel_y[component_indices] + stride >= height)
                | (pixel_x[component_indices] + stride >= width)
            ):
                edge_touch_count += 1
            if len(component) < self.component_filter_min_samples:
                small_component_count += 1
                small_rejected += len(component)
                continue

            points = samples["map_xyz"][component_indices]
            cells = np.floor(points / radius).astype(np.int64)
            buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
            for local_index, cell in enumerate(cells):
                buckets[tuple(cell.tolist())].append(local_index)
            for local_index, (point, cell) in enumerate(zip(points, cells)):
                neighbor_count = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            candidates = buckets.get(
                                (int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)),
                                [],
                            )
                            if candidates:
                                delta = points[candidates] - point
                                neighbor_count += int(np.count_nonzero(
                                    np.einsum("ij,ij->i", delta, delta) <= radius_sq
                                ))
                if neighbor_count >= self.component_filter_min_neighbors:
                    keep[component_indices[local_index]] = True
                else:
                    density_rejected += 1

        kept_indices = np.flatnonzero(keep)
        retained = int(len(kept_indices))
        stats = {
            "enabled": True,
            "raw_sample_count": count,
            "retained_sample_count": retained,
            "rejected_sample_count": count - retained,
            "retention_rate": float(retained / count),
            "component_count": len(components),
            "small_component_count": small_component_count,
            "edge_touch_component_count": edge_touch_count,
            "small_component_rejected_sample_count": small_rejected,
            "density_rejected_sample_count": density_rejected,
        }
        return self._slice_samples(samples, kept_indices), stats, kept_indices

    def _save_overlay(
        self, rgb: np.ndarray, pred: np.ndarray, confidence: np.ndarray, frame_name: str
    ) -> Optional[str]:
        if not self.save_overlay or self.episode_dir is None:
            return None
        mask = PALETTE[pred % len(PALETTE)]
        visible = confidence >= self.confidence_threshold
        blended = np.asarray(rgb, dtype=np.uint8).copy()
        blended[visible] = (
            0.55 * blended[visible] + 0.45 * mask[visible]
        ).astype(np.uint8)
        path = self.episode_dir / "overlays" / f"{frame_name}_overlay.jpg"
        Image.fromarray(blended).save(path, quality=90)
        return str(path.relative_to(self.episode_dir))

    def process_query_frame(
        self, *, rgb: Any, depth_m: Any, camera_pose_map: Any, step_id: int,
        query_id: int, observation_index: int, occ_memory: Any = None,
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "event_type": "online_lseg_query_frame", "enabled": self.enabled,
            "shadow_only": True, "action_applied": False, "step_id": int(step_id),
            "query_id": int(query_id), "observation_index": int(observation_index),
            "valid": False,
        }
        if not self.enabled:
            event["reason"] = "disabled"
            return event
        if self.disabled_after_error:
            event["reason"] = "disabled_after_error"
            return event
        if camera_pose_map is None:
            event["reason"] = "missing_camera_pose_map"
            return event
        rng = self._capture_rng()
        try:
            self._load_model()
            image = np.asarray(rgb, dtype=np.uint8)
            image = np.ascontiguousarray(image)
            depth = np.ascontiguousarray(
                np.asarray(depth_m, dtype=np.float32).reshape(image.shape[:2])
            )
            pose = np.ascontiguousarray(
                np.asarray(camera_pose_map, dtype=np.float32).reshape(4, 4)
            )
            before_cuda = self._cuda_stats()
            if before_cuda.get("available"):
                torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            started = time.perf_counter()
            with self._deterministic_inference():
                logits = self._infer_logits(image)
            elapsed = float(time.perf_counter() - started)
            after_cuda = self._cuda_stats()
            probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            pred = np.argmax(probabilities, axis=0).astype(np.int16)
            confidence = np.max(probabilities, axis=0).astype(np.float32)
            samples = self._project(pred, confidence, depth, pose, occ_memory)
            filtered_samples, filter_stats, filtered_source_indices = (
                self._filter_surface_samples(samples, image.shape[:2])
            )
            frame_name = f"q{int(query_id):04d}_step{int(step_id):04d}_obs{int(observation_index):04d}"
            overlay_path = self._save_overlay(image, pred, confidence, frame_name)
            surface_path = None
            if self.save_surface and self.episode_dir is not None:
                path = self.episode_dir / "surfaces" / f"{frame_name}_surface.npz"
                np.savez_compressed(path, **samples)
                surface_path = str(path.relative_to(self.episode_dir))

            remaining = self.max_surface_samples - self._stored_surface_count
            stored = min(max(0, remaining), int(len(samples["map_xyz"])))
            if stored:
                ids = np.linspace(0, len(samples["map_xyz"]) - 1, stored).astype(np.int64)
                self.surface_frames.append({
                    "map_xyz": samples["map_xyz"][ids],
                    "class_id": samples["class_id"][ids],
                    "confidence": samples["confidence"][ids],
                    "occ_state": samples["occ_state"][ids],
                    "observation_index": np.full(stored, int(observation_index), dtype=np.int32),
                    "step_id": np.full(stored, int(step_id), dtype=np.int32),
                })
                filtered_ids = np.intersect1d(
                    ids, filtered_source_indices, assume_unique=True
                )
                if len(filtered_ids):
                    self.filtered_surface_frames.append({
                        "map_xyz": samples["map_xyz"][filtered_ids],
                        "class_id": samples["class_id"][filtered_ids],
                        "confidence": samples["confidence"][filtered_ids],
                        "occ_state": samples["occ_state"][filtered_ids],
                        "observation_index": np.full(
                            len(filtered_ids), int(observation_index), dtype=np.int32
                        ),
                        "step_id": np.full(
                            len(filtered_ids), int(step_id), dtype=np.int32
                        ),
                    })
                self._stored_surface_count += stored
            state_counts = Counter(samples["occ_state"].tolist())
            class_counts = Counter(
                self.labels[int(index)] for index in samples["class_id"].tolist()
            )
            self.inference_seconds.append(elapsed)
            event.update({
                "valid": True, "reason": "ok", "inference_seconds": elapsed,
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                "camera_pose_sha256": hashlib.sha256(pose.tobytes()).hexdigest(),
                "high_confidence_pixel_fraction": float(
                    np.mean(confidence >= self.confidence_threshold)
                ),
                "mean_pixel_confidence": float(np.mean(confidence)),
                "sampled_depth_count": int(samples["sampled_count"]),
                "valid_depth_count": int(samples["valid_depth_count"]),
                "surface_sample_count": int(len(samples["map_xyz"])),
                "stored_surface_sample_count": int(stored),
                "filtered_surface_sample_count": int(len(filtered_samples["map_xyz"])),
                "stored_filtered_surface_sample_count": int(
                    len(filtered_ids) if stored else 0
                ),
                "component_filter": filter_stats,
                "class_surface_counts": dict(sorted(class_counts.items())),
                "occ_state_counts": {
                    "occupied": int(state_counts.get(0, 0)),
                    "free": int(state_counts.get(1, 0)),
                    "unknown": int(state_counts.get(2, 0)),
                },
                "overlay_path": overlay_path, "surface_path": surface_path,
                "cuda_before": before_cuda, "cuda_after": after_cuda,
            })
        except Exception as exc:
            self.load_error = self.load_error or f"{type(exc).__name__}: {exc}"
            self.errors.append(f"step={step_id},query={query_id}: {type(exc).__name__}: {exc}")
            self.disabled_after_error = True
            event.update({"reason": "shadow_error", "error": self.errors[-1]})
        finally:
            self._restore_rng(rng)
        self.records.append(event)
        if self.episode_dir is not None:
            with open(self.episode_dir / "events.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")
        return event

    def _merge_nodes(self, frames: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        buckets: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
        radius = self.merge_radius_m
        for frame in frames:
            for point, class_id, confidence, observation, step in zip(
                frame["map_xyz"], frame["class_id"], frame["confidence"],
                frame["observation_index"], frame["step_id"],
            ):
                class_id = int(class_id)
                base = tuple(np.floor(point / radius).astype(np.int64).tolist())
                candidates = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            candidates.extend(buckets.get(
                                (class_id, base[0] + dx, base[1] + dy, base[2] + dz), []
                            ))
                best = None
                best_distance = None
                for index in candidates:
                    distance = float(np.linalg.norm(nodes[index]["centroid"] - point))
                    if distance <= radius and (best_distance is None or distance < best_distance):
                        best, best_distance = index, distance
                if best is None:
                    best = len(nodes)
                    nodes.append({
                        "node_id": f"SN{best:05d}", "class_id": class_id,
                        "label": self.labels[class_id], "centroid": point.copy(),
                        "point_count": 0, "confidence_sum": 0.0,
                        "source_observations": set(), "source_steps": set(),
                    })
                    buckets[(class_id, *base)].append(best)
                node = nodes[best]
                count = node["point_count"]
                node["centroid"] = (node["centroid"] * count + point) / float(count + 1)
                node["point_count"] = count + 1
                node["confidence_sum"] += float(confidence)
                node["source_observations"].add(int(observation))
                node["source_steps"].add(int(step))
        for node in nodes:
            node["centroid"] = [float(value) for value in node["centroid"]]
            node["mean_confidence"] = float(
                node.pop("confidence_sum") / max(1, node["point_count"])
            )
            node["source_observations"] = sorted(node["source_observations"])
            node["source_steps"] = sorted(node["source_steps"])
            strong = (
                len(node["source_observations"]) >= self.strong_min_views
                or (
                    node["point_count"] >= self.strong_min_points
                    and node["mean_confidence"] >= self.strong_min_confidence
                )
            )
            node["evidence_tier"] = "strong" if strong else "weak"
        return nodes

    def snapshot_nodes(self, *, filtered: bool = False) -> List[Dict[str, Any]]:
        """Return a causal node snapshot from frames observed so far."""
        frames = self.filtered_surface_frames if filtered else self.surface_frames
        return self._merge_nodes(frames)

    def _audit_conflicts(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_pairs = Counter()
        severe_pairs = Counter()
        strong_severe_pairs = Counter()
        for left_index, left in enumerate(nodes):
            left_point = np.asarray(left["centroid"], dtype=np.float32)
            for right in nodes[left_index + 1:]:
                if left["class_id"] == right["class_id"]:
                    continue
                if np.linalg.norm(
                    left_point - np.asarray(right["centroid"], dtype=np.float32)
                ) > self.merge_radius_m / 2:
                    continue
                pair = tuple(sorted((str(left["label"]), str(right["label"]))))
                key = "|".join(pair)
                raw_pairs[key] += 1
                if pair in BENIGN_NEARBY_LABEL_PAIRS:
                    continue
                severe_pairs[key] += 1
                if (
                    left.get("evidence_tier") == "strong"
                    and right.get("evidence_tier") == "strong"
                ):
                    strong_severe_pairs[key] += 1
        return {
            "raw_count": int(sum(raw_pairs.values())),
            "raw_pairs": dict(sorted(raw_pairs.items())),
            "severe_count": int(sum(severe_pairs.values())),
            "severe_pairs": dict(sorted(severe_pairs.items())),
            "strong_severe_count": int(sum(strong_severe_pairs.values())),
            "strong_severe_pairs": dict(sorted(strong_severe_pairs.items())),
        }

    def _audit_nodes_with_gt(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        gt = self.episode_meta.get("semantic_scene_gt") or {}
        entries = list(gt.get("objects") or []) + list(gt.get("regions") or [])
        transforms = self.episode_meta.get("coordinate_transforms") or {}
        map_to_gt = np.asarray(transforms.get("map_to_habitat_world"), dtype=np.float32)
        distances = []
        per_label_distances: Dict[str, List[float]] = defaultdict(list)
        compatible_nodes = 0
        if not entries or map_to_gt.shape != (4, 4):
            return {"available": False, "compatible_node_count": 0}
        for node in nodes:
            aliases = set(ALIASES.get(node["label"], {node["label"]}))
            compatible = [
                item for item in entries
                if aliases.intersection(_tokens(item.get("category", "")))
            ]
            node["gt_compatible_count"] = len(compatible)
            if not compatible:
                node["gt_surface_distance_m"] = None
                continue
            point = map_to_gt @ np.asarray([*node["centroid"], 1.0], dtype=np.float32)
            nearest = min(
                compatible,
                key=lambda item: float(np.linalg.norm(np.asarray(item["center"]) - point[:3])),
            )
            lower = np.asarray(nearest.get("lower", nearest["center"]), dtype=np.float32)
            upper = np.asarray(nearest.get("upper", nearest["center"]), dtype=np.float32)
            delta = np.maximum(np.maximum(lower - point[:3], 0.0), point[:3] - upper)
            distance = float(np.linalg.norm(delta))
            node["gt_nearest_category"] = nearest.get("category")
            node["gt_surface_distance_m"] = distance
            distances.append(distance)
            per_label_distances[str(node["label"])].append(distance)
            compatible_nodes += 1
        values = np.asarray(distances, dtype=np.float32)
        per_label = {}
        for label, label_distances in sorted(per_label_distances.items()):
            label_values = np.asarray(label_distances, dtype=np.float32)
            per_label[label] = {
                "compatible_node_count": int(label_values.size),
                "surface_distance_le_050m_count": int(
                    np.count_nonzero(label_values <= 0.50)
                ),
                "surface_distance_le_050m_rate": float(
                    np.mean(label_values <= 0.50)
                ),
                "surface_distance_m_median": float(np.median(label_values)),
            }
        return {
            "available": True, "compatible_node_count": int(compatible_nodes),
            "surface_distance_le_050m_count": int(np.count_nonzero(values <= 0.50)),
            "surface_distance_m_median": float(np.median(values)) if values.size else None,
            "surface_distance_m_p95": float(np.percentile(values, 95)) if values.size else None,
            "surface_distance_le_050m_rate": float(np.mean(values <= 0.50)) if values.size else None,
            "per_label": per_label,
        }

    @staticmethod
    def _project_panel(
        points: np.ndarray, colors: np.ndarray, axes: Tuple[int, int], title: str,
        route: Optional[np.ndarray] = None, size: int = 720,
    ) -> Image.Image:
        image = Image.new("RGB", (size, size), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        draw.text((12, 10), title, fill=(0, 0, 0))
        if not len(points):
            return image
        projected = points[:, axes].astype(np.float32)
        all_projected = projected
        route_projected = None
        if route is not None and len(route):
            route_projected = route[:, axes].astype(np.float32)
            all_projected = np.concatenate([projected, route_projected], axis=0)
        lower = np.nanpercentile(all_projected, 1, axis=0)
        upper = np.nanpercentile(all_projected, 99, axis=0)
        extent = np.maximum(upper - lower, 0.5)
        margin = 35
        pixel = margin + (projected - lower) / extent * (size - 2 * margin)
        pixel[:, 1] = size - pixel[:, 1]
        order = np.linspace(0, len(pixel) - 1, min(50000, len(pixel))).astype(np.int64)
        for index in order:
            x, y = pixel[index]
            color = tuple(int(value) for value in colors[index])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        if route_projected is not None and len(route_projected):
            route_pixel = margin + (route_projected - lower) / extent * (size - 2 * margin)
            route_pixel[:, 1] = size - route_pixel[:, 1]
            draw.line([tuple(value) for value in route_pixel.tolist()], fill=(0, 0, 0), width=3)
        return image

    def _save_semantic_visualizations(
        self, points: np.ndarray, class_id: np.ndarray, occ_memory: Any
    ) -> Dict[str, str]:
        if not self.save_visualizations or self.episode_dir is None or not len(points):
            return {}
        colors = PALETTE[class_id % len(PALETTE)]
        route = None
        if occ_memory is not None and getattr(occ_memory, "pose_trace", None):
            route = np.asarray([
                [item["x"], item["y"], float(item.get("z", 0.0))]
                for item in occ_memory.pose_trace
            ], dtype=np.float32)
        panels = {
            "semantic_bev_xy.png": self._project_panel(points, colors, (0, 1), "semantic BEV (XY)", route),
            "semantic_side_xz.png": self._project_panel(points, colors, (0, 2), "semantic side (XZ)", route),
            "semantic_side_yz.png": self._project_panel(points, colors, (1, 2), "semantic side (YZ)", route),
        }
        # Oblique view is a fixed metric rotation, not a perspective reconstruction.
        angle = np.deg2rad(35.0)
        oblique = np.stack([
            points[:, 0] * np.cos(angle) - points[:, 1] * np.sin(angle),
            points[:, 2] + 0.35 * (points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)),
            points[:, 1],
        ], axis=1)
        panels["semantic_3d_oblique.png"] = self._project_panel(
            oblique, colors, (0, 1), "semantic 3D oblique", None
        )
        result = {}
        for name, panel in panels.items():
            path = self.episode_dir / "visualizations" / name
            panel.save(path)
            result[name.removesuffix(".png")] = str(path.relative_to(self.episode_dir))
        legend = Image.new("RGB", (480, 40 + 28 * len(self.labels)), (255, 255, 255))
        draw = ImageDraw.Draw(legend)
        draw.text((12, 10), "LSeg semantic classes", fill=(0, 0, 0))
        for index, label in enumerate(self.labels):
            y = 38 + index * 28
            color = tuple(int(value) for value in PALETTE[index % len(PALETTE)])
            draw.rectangle((12, y, 34, y + 18), fill=color)
            draw.text((44, y), label, fill=(0, 0, 0))
        legend_path = self.episode_dir / "visualizations" / "semantic_legend.png"
        legend.save(legend_path)
        result["semantic_legend"] = str(legend_path.relative_to(self.episode_dir))
        return result

    def finish_episode(
        self, *, metrics: Optional[Dict[str, Any]] = None, steps: Any = None,
        occ_memory: Any = None, frequency: str = "s2_query",
    ) -> Dict[str, Any]:
        if self.surface_frames:
            points = np.concatenate([frame["map_xyz"] for frame in self.surface_frames])
            class_id = np.concatenate([frame["class_id"] for frame in self.surface_frames])
            confidence = np.concatenate([frame["confidence"] for frame in self.surface_frames])
            occ_state = np.concatenate([frame["occ_state"] for frame in self.surface_frames])
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            class_id = np.zeros(0, dtype=np.int16)
            confidence = np.zeros(0, dtype=np.float32)
            occ_state = np.zeros(0, dtype=np.int8)
        nodes = self._merge_nodes(self.surface_frames)
        gt_audit = self._audit_nodes_with_gt(nodes)
        conflict_audit = self._audit_conflicts(nodes)
        filtered_nodes = self._merge_nodes(self.filtered_surface_frames)
        filtered_gt_audit = self._audit_nodes_with_gt(filtered_nodes)
        filtered_conflict_audit = self._audit_conflicts(filtered_nodes)
        if self.filtered_surface_frames:
            filtered_points = np.concatenate([
                frame["map_xyz"] for frame in self.filtered_surface_frames
            ])
            filtered_class_id = np.concatenate([
                frame["class_id"] for frame in self.filtered_surface_frames
            ])
            filtered_confidence = np.concatenate([
                frame["confidence"] for frame in self.filtered_surface_frames
            ])
            filtered_occ_state = np.concatenate([
                frame["occ_state"] for frame in self.filtered_surface_frames
            ])
        else:
            filtered_points = np.zeros((0, 3), dtype=np.float32)
            filtered_class_id = np.zeros(0, dtype=np.int16)
            filtered_confidence = np.zeros(0, dtype=np.float32)
            filtered_occ_state = np.zeros(0, dtype=np.int8)
        visualizations = self._save_semantic_visualizations(points, class_id, occ_memory)
        latency = np.asarray(self.inference_seconds, dtype=np.float64)
        class_counts = Counter(self.labels[int(index)] for index in class_id.tolist())
        occ_counts = Counter(occ_state.tolist())
        summary = {
            "event_type": "online_lseg_episode_summary", **self.episode_meta,
            "enabled": self.enabled, "shadow_only": True, "action_applied_count": 0,
            "decision_status": "audit_only_not_navigation_ready",
            "frequency": str(frequency), "steps": steps,
            "success": (metrics or {}).get("success"),
            "frame_count": len(self.records),
            "valid_frame_count": sum(bool(record.get("valid")) for record in self.records),
            "error_count": len(self.errors), "errors": self.errors,
            "model_load_seconds": self.model_load_seconds, "load_error": self.load_error,
            "inference_seconds_mean": float(np.mean(latency)) if latency.size else None,
            "inference_seconds_p50": float(np.percentile(latency, 50)) if latency.size else None,
            "inference_seconds_p95": float(np.percentile(latency, 95)) if latency.size else None,
            "inference_seconds_max": float(np.max(latency)) if latency.size else None,
            "stored_surface_sample_count": int(len(points)),
            "class_surface_counts": dict(sorted(class_counts.items())),
            "occ_state_counts": {
                "occupied": int(occ_counts.get(0, 0)), "free": int(occ_counts.get(1, 0)),
                "unknown": int(occ_counts.get(2, 0)),
            },
            "node_count": len(nodes),
            "multi_view_node_count": sum(len(node["source_observations"]) >= 2 for node in nodes),
            "multi_view_node_rate": float(
                sum(len(node["source_observations"]) >= 2 for node in nodes) / max(1, len(nodes))
            ),
            "strong_node_count": sum(
                node.get("evidence_tier") == "strong" for node in nodes
            ),
            "weak_node_count": sum(
                node.get("evidence_tier") == "weak" for node in nodes
            ),
            "cross_label_conflict_count": conflict_audit["raw_count"],
            "severe_cross_label_conflict_count": conflict_audit["severe_count"],
            "strong_severe_cross_label_conflict_count": (
                conflict_audit["strong_severe_count"]
            ),
            "cross_label_conflict_audit": conflict_audit,
            "gt_audit": gt_audit, "visualizations": visualizations,
            "component_filter": {
                "enabled": self.component_filter_enable,
                "min_samples": self.component_filter_min_samples,
                "radius_m": self.component_filter_radius_m,
                "min_neighbors_including_self": self.component_filter_min_neighbors,
                "raw_stored_surface_sample_count": int(len(points)),
                "filtered_stored_surface_sample_count": int(len(filtered_points)),
                "stored_surface_retention_rate": float(
                    len(filtered_points) / len(points)
                ) if len(points) else None,
                "raw_node_count": len(nodes),
                "filtered_node_count": len(filtered_nodes),
                "node_retention_rate": float(
                    len(filtered_nodes) / len(nodes)
                ) if nodes else None,
                "filtered_multi_view_node_count": sum(
                    len(node["source_observations"]) >= 2 for node in filtered_nodes
                ),
                "filtered_strong_node_count": sum(
                    node.get("evidence_tier") == "strong" for node in filtered_nodes
                ),
                "filtered_weak_node_count": sum(
                    node.get("evidence_tier") == "weak" for node in filtered_nodes
                ),
                "filtered_cross_label_conflict_audit": filtered_conflict_audit,
                "filtered_gt_audit": filtered_gt_audit,
            },
            "cuda_s2_loaded_baseline": self._baseline_cuda,
            "cuda_immediately_before_lseg_load": self._before_load_cuda,
            "cuda_after_lseg_load": self._after_load_cuda,
            "cuda_final": self._cuda_stats(),
            "confidence_threshold": self.confidence_threshold,
            "sample_stride": self.sample_stride, "merge_radius_m": self.merge_radius_m,
            "strong_evidence_policy": {
                "min_views": self.strong_min_views,
                "single_view_min_points": self.strong_min_points,
                "single_view_min_confidence": self.strong_min_confidence,
            },
        }
        if self.episode_dir is not None:
            self._write_json(self.episode_dir / "nodes.json", nodes)
            self._write_json(self.episode_dir / "nodes_filtered.json", filtered_nodes)
            if len(points):
                np.savez_compressed(
                    self.episode_dir / "semantic_surface_memory.npz", map_xyz=points,
                    class_id=class_id, confidence=confidence, occ_state=occ_state,
                )
            if len(filtered_points):
                np.savez_compressed(
                    self.episode_dir / "semantic_surface_memory_filtered.npz",
                    map_xyz=filtered_points, class_id=filtered_class_id,
                    confidence=filtered_confidence, occ_state=filtered_occ_state,
                )
            self._write_json(self.episode_dir / "summary.json", summary)
        return summary

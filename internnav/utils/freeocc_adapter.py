"""Optional FreeOcc bridge for RGB-only, shadow-mode experiments.

FreeOcc remains an external dependency with its own CUDA/PyTorch stack.  This
module deliberately imports none of it at module load time and never changes
SparseOcc, action queues, or waypoint safety.  Heavy mapping is launched only
when ``run_external`` is explicitly called.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FreeOccConfig:
    output_dir: str = ""
    command: Optional[List[str]] = None
    semantic_names: List[str] = field(default_factory=list)
    save_frames: bool = True


@dataclass
class FreeOccOccupancy:
    labels: np.ndarray
    valid_mask: np.ndarray
    voxel_origin: np.ndarray
    voxel_size: float
    path: str = ""


class FreeOccWorldMemory:
    """Small reset/update/query boundary for an external FreeOcc worker."""

    def __init__(self, config: Optional[FreeOccConfig] = None):
        self.config = config or FreeOccConfig()
        self.scene_id: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.frame_dir: Optional[Path] = None
        self.frames: List[Dict[str, Any]] = []
        self.occupancy: Optional[FreeOccOccupancy] = None

    def reset(self, scene_id: str, episode_id: Optional[str] = None) -> None:
        self.scene_id, self.episode_id = str(scene_id), None if episode_id is None else str(episode_id)
        self.frames, self.occupancy = [], None
        self.frame_dir = None
        if self.config.output_dir and self.config.save_frames:
            self.frame_dir = Path(self.config.output_dir) / "rgb" / self.scene_id
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rgb_array(rgb: Any) -> np.ndarray:
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"RGB must have shape HxWx3, got {arr.shape}")
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    def update(self, rgb: Any, step_id: int, *, pose: Any = None, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Store an observation.  ``pose`` is diagnostic metadata only."""
        arr = self._rgb_array(rgb)
        frame_path = None
        if self.frame_dir is not None:
            from PIL import Image
            frame_path = self.frame_dir / f"{int(step_id):06d}.png"
            Image.fromarray(arr).save(frame_path)
        row = {"step_id": int(step_id), "path": str(frame_path) if frame_path else None,
               "height": int(arr.shape[0]), "width": int(arr.shape[1]),
               "pose_present": pose is not None,
               "timestamp": None if timestamp is None else float(timestamp)}
        self.frames.append(row)
        self._write_ledger()
        return dict(row)

    def _write_ledger(self) -> None:
        if not self.config.output_dir:
            return
        path = Path(self.config.output_dir) / "freeocc_frame_ledger.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": "freeocc_adapter_v1", "scene_id": self.scene_id,
                                    "episode_id": self.episode_id, "rgb_only_external_input": True,
                                    "frames": self.frames}, indent=2, ensure_ascii=False))

    def run_external(self, *, extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Run an explicit command; ``{frames}`` and ``{output}`` are expanded."""
        if not self.config.command:
            raise RuntimeError("FreeOcc command is not configured")
        if not self.frames:
            raise RuntimeError("No RGB frames recorded")
        argv = [str(x).format(frames=str(self.frame_dir or ""), output=str(self.config.output_dir or ""))
                for x in self.config.command]
        env = os.environ.copy()
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return subprocess.run(argv, env=env, check=True)

    def load_occupancy(self, npz_path: str | os.PathLike[str]) -> FreeOccOccupancy:
        data = np.load(str(npz_path), allow_pickle=False)
        key = "pred" if "pred" in data else "labels"
        if key not in data:
            raise ValueError(f"FreeOcc artifact has no pred/labels array: {npz_path}")
        labels = np.asarray(data[key])
        valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else np.ones_like(labels, dtype=bool)
        if labels.shape != valid.shape:
            raise ValueError(f"labels/valid_mask shape mismatch: {labels.shape} vs {valid.shape}")
        origin = np.asarray(data["voxel_origin"], dtype=np.float32).reshape(3) if "voxel_origin" in data else np.zeros(3, dtype=np.float32)
        size = float(np.asarray(data["voxel_size"]).reshape(-1)[0]) if "voxel_size" in data else 0.08
        self.occupancy = FreeOccOccupancy(labels, valid, origin, size, str(npz_path))
        return self.occupancy

    def query_semantics(self, name: str) -> Dict[str, Any]:
        if self.occupancy is None:
            return {"valid": False, "reason": "occupancy_not_loaded", "count": 0}
        query = str(name).strip().lower()
        names = [str(x).strip().lower() for x in self.config.semantic_names]
        ids = [i + 1 for i, n in enumerate(names) if n == query or query in n]
        if not ids:
            return {"valid": False, "reason": "semantic_name_not_found", "query": name, "count": 0}
        mask = self.occupancy.valid_mask & np.isin(self.occupancy.labels, ids)
        vox = np.argwhere(mask)
        xyz = self.occupancy.voxel_origin + (vox.astype(np.float32) + 0.5) * self.occupancy.voxel_size
        return {"valid": True, "query": name, "label_ids": ids, "count": int(len(xyz)), "xyz": xyz}


__all__ = ["FreeOccConfig", "FreeOccOccupancy", "FreeOccWorldMemory"]

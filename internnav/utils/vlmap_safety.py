from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _quat_to_rot_matrix(quat: np.ndarray, order: str = "wxyz") -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "xyzw":
        x, y, z, w = quat
    elif order == "wxyz":
        w, x, y, z = quat
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _yaw_to_tf(position: np.ndarray, yaw: float) -> np.ndarray:
    tf = np.eye(4, dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    tf[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    tf[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return tf


class VLMapActionSafety:
    """Optional action-level safety wrapper backed by VLMaps online obstacle map."""

    def __init__(self, config: Dict[str, Any], camera_intrinsic: np.ndarray):
        self.config = dict(config)
        self.enabled = bool(self.config.get("enable", False))
        self.verbose = bool(self.config.get("verbose", True))
        self.strict = bool(self.config.get("strict_import", False))
        self.forward_action = int(self.config.get("forward_action", 1))
        self.left_action = int(self.config.get("left_action", 2))
        self.right_action = int(self.config.get("right_action", 3))
        self.fallback_action = int(self.config.get("fallback_action", self.left_action))
        self.forward_distance = float(self.config.get("forward_distance", 0.25))
        self.turn_angle_deg = float(self.config.get("turn_angle_deg", 15.0))
        self.radius_cells = int(self.config.get("radius_cells", 2))
        self.depth_scale = float(self.config.get("depth_scale", 10.0))
        self.update_every_steps = max(1, int(self.config.get("update_every_steps", 1)))
        self.min_update_distance = float(self.config.get("min_update_distance", 0.0))
        self.quat_order = str(self.config.get("quat_order", "wxyz"))
        self._last_update_position: Optional[np.ndarray] = None
        self._step = 0
        self._disabled_reason: Optional[str] = None

        self.builder = None
        if self.enabled:
            self._init_builder(camera_intrinsic)

    def reset(self) -> None:
        self._last_update_position = None
        self._step = 0
        if self.builder is not None:
            self.builder.reset()

    def postprocess(self, obs: Dict[str, Any], action: int) -> Tuple[int, bool]:
        """Return possibly corrected action plus whether it changed."""
        if not self.enabled or self.builder is None:
            return action, False
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None or "depth" not in obs:
            return action, False

        self._maybe_update(obs["depth"], pose_tf)
        if int(action) != self.forward_action:
            return action, False

        if self.builder.is_forward_free(pose_tf, self.forward_distance, radius_cells=self.radius_cells):
            return action, False

        corrected = self._pick_turn_action(pose_tf)
        if self.verbose:
            print(
                "[VLMapSafety] forward blocked; "
                f"replace action {action} -> {corrected} at step {self._step}"
            )
        return corrected, corrected != int(action)

    def _init_builder(self, camera_intrinsic: np.ndarray) -> None:
        repo_path = self.config.get("vlmaps_repo") or self.config.get("vlmaps_root")
        if repo_path:
            repo_path = os.path.abspath(os.path.expanduser(str(repo_path)))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            from vlmaps.map.online_obstacle_builder import OnlineObstacleMapBuilder
        except Exception as exc:
            self.enabled = False
            self._disabled_reason = str(exc)
            if self.strict:
                raise
            print(f"[VLMapSafety] disabled because VLMaps import failed: {exc}")
            return

        obstacle_cfg = {
            "grid_size": int(self.config.get("grid_size", 1000)),
            "cell_size": float(self.config.get("cell_size", 0.05)),
            "camera_height": float(self.config.get("camera_height", 1.5)),
            "map_height": float(self.config.get("map_height", 2.5)),
            "depth_sample_rate": int(self.config.get("depth_sample_rate", 80)),
            "min_depth": float(self.config.get("min_depth", 0.1)),
            "max_depth": float(self.config.get("max_depth", 6.0)),
            "obstacle_height_min": float(self.config.get("obstacle_height_min", 0.05)),
            "obstacle_height_max": float(self.config.get("obstacle_height_max", 1.5)),
            "center_on_first_pose": bool(self.config.get("center_on_first_pose", True)),
        }
        self.builder = OnlineObstacleMapBuilder(
            obstacle_cfg,
            camera_intrinsic=np.asarray(camera_intrinsic)[:3, :3],
            cam_to_base_tf=self._cam_to_base_tf(),
        )
        if self.verbose:
            print(f"[VLMapSafety] enabled with config: {obstacle_cfg}")

    def _cam_to_base_tf(self) -> Optional[np.ndarray]:
        rot = self.config.get("cam_to_base_rot")
        trans = self.config.get("cam_to_base_trans")
        if rot is None and trans is None:
            return None
        tf = np.eye(4, dtype=np.float32)
        if rot is not None:
            tf[:3, :3] = np.asarray(rot, dtype=np.float32).reshape(3, 3)
        if trans is not None:
            tf[:3, 3] = np.asarray(trans, dtype=np.float32).reshape(3)
        return tf

    def _pose_from_obs(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        if "globalgps" in obs and "globalrotation" in obs:
            pos = np.asarray(obs["globalgps"], dtype=np.float32).reshape(3)
            rot = _quat_to_rot_matrix(np.asarray(obs["globalrotation"]), order=self.quat_order)
            tf = np.eye(4, dtype=np.float32)
            tf[:3, :3] = rot
            tf[:3, 3] = pos
            return tf

        if "gps" in obs and "compass" in obs:
            gps = np.asarray(obs["gps"], dtype=np.float32).reshape(-1)
            yaw = float(np.asarray(obs["compass"]).reshape(-1)[0])
            pos = np.array([gps[0], -gps[1], 0.0], dtype=np.float32)
            return _yaw_to_tf(pos, yaw)

        return None

    def _maybe_update(self, depth: np.ndarray, pose_tf: np.ndarray) -> None:
        self._step += 1
        if self._step % self.update_every_steps != 0:
            return
        position = pose_tf[:3, 3].copy()
        if self._last_update_position is not None and self.min_update_distance > 0:
            if np.linalg.norm(position[:2] - self._last_update_position[:2]) < self.min_update_distance:
                return

        depth_m = np.asarray(depth)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_m = depth_m.astype(np.float32, copy=False)
        if self.depth_scale != 1.0 and np.nanmax(depth_m) <= 1.5:
            depth_m = depth_m * self.depth_scale
        updated = self.builder.update(depth_m, pose_tf)
        if updated:
            self._last_update_position = position

    def _pick_turn_action(self, pose_tf: np.ndarray) -> int:
        row, col, yaw = self.builder.base_pose_to_grid(pose_tf)
        probe_distance = self.forward_distance
        left_yaw = yaw + math.radians(self.turn_angle_deg)
        right_yaw = yaw - math.radians(self.turn_angle_deg)
        left_probe = self._probe_grid(row, col, left_yaw, probe_distance)
        right_probe = self._probe_grid(row, col, right_yaw, probe_distance)
        left_free = self.builder.is_line_free((row, col), left_probe, radius_cells=self.radius_cells)
        right_free = self.builder.is_line_free((row, col), right_probe, radius_cells=self.radius_cells)
        if left_free and not right_free:
            return self.left_action
        if right_free and not left_free:
            return self.right_action
        if left_free and right_free:
            return self.left_action
        return self.fallback_action

    def _probe_grid(self, row: int, col: int, yaw: float, distance: float) -> Tuple[int, int]:
        cs = self.builder.cs
        gs = self.builder.gs
        x = (gs / 2 - row) * cs + distance * math.cos(yaw)
        y = (gs / 2 - col) * cs + distance * math.sin(yaw)
        probe_row = int(gs / 2 - int(x / cs))
        probe_col = int(gs / 2 - int(y / cs))
        return probe_row, probe_col

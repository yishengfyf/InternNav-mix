from __future__ import annotations

import json
import math
import os
import random
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
        self.line_skip_distance = float(self.config.get("line_skip_distance", 0.08))
        self.line_blocked_fraction = float(self.config.get("line_blocked_fraction", 0.67))
        self.line_blocked_min_cells = int(self.config.get("line_blocked_min_cells", 3))
        self.line_min_checked_cells = int(self.config.get("line_min_checked_cells", 3))
        self.line_cell_blocked_fraction = float(self.config.get("line_cell_blocked_fraction", 0.25))
        self.prefer_previous_turn = bool(self.config.get("prefer_previous_turn", True))
        self.repeat_block_enable = bool(self.config.get("repeat_block_enable", True))
        self.repeat_block_count = int(self.config.get("repeat_block_count", 3))
        self.repeat_block_window_steps = int(self.config.get("repeat_block_window_steps", 10))
        self.repeat_block_distance = float(self.config.get("repeat_block_distance", 0.60))
        self.repeat_turn_lock_steps = int(self.config.get("repeat_turn_lock_steps", 10))
        self.cluster_reset_distance = float(self.config.get("cluster_reset_distance", self.repeat_block_distance))
        self.cluster_reset_steps = int(
            self.config.get("cluster_reset_steps", max(20, self.repeat_block_window_steps * 3))
        )
        self.max_same_turn_in_cluster = max(1, int(self.config.get("max_same_turn_in_cluster", 2)))
        self.max_replans_per_cluster = int(self.config.get("max_replans_per_cluster", 2))
        self.replan_cooldown_steps = max(0, int(self.config.get("replan_cooldown_steps", 8)))
        self.recovery_turn_steps = max(1, int(self.config.get("recovery_turn_steps", 1)))
        self.stuck_block_count = max(1, int(self.config.get("stuck_block_count", 8)))
        self.stuck_distance = float(self.config.get("stuck_distance", 0.15))
        self.waypoint_repair_on_stuck = bool(self.config.get("waypoint_repair_on_stuck", True))
        self.max_waypoint_repairs_per_cluster = int(self.config.get("max_waypoint_repairs_per_cluster", 1))
        self.shadow_only = bool(self.config.get("shadow_only", False))
        self.debug = bool(self.config.get("debug", False))
        self.debug_root_dir = os.path.abspath(
            os.path.expanduser(str(self.config.get("debug_dir", "./logs/vlmap_safety_debug")))
        )
        self.debug_use_run_subdir = bool(self.config.get("debug_use_run_subdir", True))
        self.debug_run_prefix = str(self.config.get("debug_run_prefix", "run"))
        self.debug_dir: Optional[str] = None
        self.debug_log_all_events = bool(self.config.get("debug_log_all_events", True))
        self.debug_max_snapshots = int(self.config.get("debug_max_snapshots", 200))
        self.debug_save_on_change = bool(self.config.get("debug_save_on_change", True))
        self.debug_save_every_steps = int(self.config.get("debug_save_every_steps", 0))
        self.debug_sample_snapshots = bool(self.config.get("debug_sample_snapshots", False))
        self.debug_sample_total_snapshots = int(
            self.config.get("debug_sample_total_snapshots", self.debug_max_snapshots)
        )
        self.debug_sample_episode_count = int(self.config.get("debug_sample_episode_count", 0))
        self.debug_sample_images_per_episode = max(1, int(self.config.get("debug_sample_images_per_episode", 2)))
        self.debug_sample_seed = int(self.config.get("debug_sample_seed", 0))
        self.debug_crop_radius_cells = int(self.config.get("debug_crop_radius_cells", 80))
        self.debug_cell_scale = max(1, int(self.config.get("debug_cell_scale", 3)))
        self._last_update_position: Optional[np.ndarray] = None
        self._step = 0
        self._disabled_reason: Optional[str] = None
        self._debug_import_warned = False
        self._debug_saved_snapshots = 0
        self._debug_selected_episode_indices = None
        self._debug_episode_snapshot_counts = {}
        self._debug_episode_candidate_counts = {}
        self._recent_blocks = []
        self._last_safety_turn_action: Optional[int] = None
        self._last_safety_turn_step: Optional[int] = None
        self._last_safety_turn_position: Optional[np.ndarray] = None
        self._cluster_seq = 0
        self._active_cluster_id: Optional[int] = None
        self._cluster_start_position: Optional[np.ndarray] = None
        self._cluster_start_step: Optional[int] = None
        self._cluster_last_position: Optional[np.ndarray] = None
        self._cluster_last_step: Optional[int] = None
        self._cluster_block_count = 0
        self._cluster_replan_count = 0
        self._cluster_waypoint_repair_count = 0
        self._cluster_turn_counts = {self.left_action: 0, self.right_action: 0}
        self._same_turn_streak_action: Optional[int] = None
        self._same_turn_streak_count = 0
        self._replan_cooldown_until_step = -1
        self.last_decision: Dict[str, Any] = {}

        self.builder = None
        if self.enabled:
            self._init_builder(camera_intrinsic)

    def reset(self) -> None:
        self._last_update_position = None
        self._step = 0
        self._recent_blocks = []
        self._last_safety_turn_action = None
        self._last_safety_turn_step = None
        self._last_safety_turn_position = None
        self._clear_block_cluster()
        self._cluster_seq = 0
        self.last_decision = {}
        if self.builder is not None:
            self.builder.reset()

    def postprocess(self, obs: Dict[str, Any], action: int) -> Tuple[int, bool]:
        """Return possibly corrected action plus whether it changed."""
        if not self.enabled or self.builder is None:
            return action, False
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None or "depth" not in obs:
            return action, False

        original_action = int(action)
        safe_action = original_action
        changed = False
        front_free = None
        probe_info = None
        replan_required = False
        replan_suppressed_reason = None
        stuck_detected = False
        waypoint_repair_required = False
        cluster_displacement = 0.0
        cluster_duration_steps = 0
        recovery_actions = []
        repeat_block_count = 0
        position = pose_tf[:3, 3].copy()
        self.last_decision = {
            "input_action": original_action,
            "output_action": safe_action,
            "changed": False,
            "front_free": None,
            "replan_required": False,
            "replan_suppressed_reason": None,
            "repeat_block_count": 0,
            "cluster_id": self._active_cluster_id,
            "cluster_block_count": int(self._cluster_block_count),
            "cluster_replan_count": int(self._cluster_replan_count),
            "cluster_displacement": 0.0,
            "cluster_duration_steps": 0,
            "stuck": False,
            "waypoint_repair_required": False,
            "recovery_actions": [],
        }

        self._maybe_update(obs["depth"], pose_tf)
        if original_action == self.forward_action:
            front_free, front_stats = self._is_forward_free(pose_tf)
            probe_info = {"front": front_stats}
            if not front_free:
                self._ensure_block_cluster(position)
                obs_for_pick = dict(obs)
                locked_turn = self._locked_safety_turn(position)
                if locked_turn is not None:
                    obs_for_pick["preferred_turn_action"] = locked_turn
                corrected, probe_info = self._pick_turn_action(pose_tf, obs_for_pick)
                probe_info["front"] = front_stats
                repeat_block_count = self._record_block(position, corrected)
                cluster_displacement = self._cluster_displacement(position)
                cluster_duration_steps = self._cluster_duration_steps()
                stuck_detected = self._is_cluster_stuck(position)
                repeat_threshold_met = self.repeat_block_enable and repeat_block_count >= self.repeat_block_count
                if repeat_threshold_met and not self.shadow_only:
                    if self.recovery_turn_steps > 1:
                        recovery_actions = [int(corrected)] * (self.recovery_turn_steps - 1)
                    cooldown_remaining = self._replan_cooldown_remaining()
                    if cooldown_remaining > 0:
                        replan_suppressed_reason = "cooldown"
                    elif self.max_replans_per_cluster >= 0 and self._cluster_replan_count >= self.max_replans_per_cluster:
                        replan_suppressed_reason = "max_cluster_replans"
                    else:
                        replan_required = True
                        self._cluster_replan_count += 1
                        self._replan_cooldown_until_step = self._step + self.replan_cooldown_steps
                    self._clear_recent_blocks()
                if (
                    self.waypoint_repair_on_stuck
                    and not self.shadow_only
                    and stuck_detected
                    and self._cluster_replan_count >= self.max_replans_per_cluster
                    and (
                        self.max_waypoint_repairs_per_cluster < 0
                        or self._cluster_waypoint_repair_count < self.max_waypoint_repairs_per_cluster
                    )
                ):
                    waypoint_repair_required = True
                    self._cluster_waypoint_repair_count += 1
                probe_info["repeat_block_count"] = int(repeat_block_count)
                probe_info["replan_required"] = bool(replan_required)
                probe_info["replan_suppressed_reason"] = replan_suppressed_reason
                probe_info["cluster_id"] = self._active_cluster_id
                probe_info["cluster_block_count"] = int(self._cluster_block_count)
                probe_info["cluster_replan_count"] = int(self._cluster_replan_count)
                probe_info["cluster_waypoint_repair_count"] = int(self._cluster_waypoint_repair_count)
                probe_info["cluster_displacement"] = float(cluster_displacement)
                probe_info["cluster_duration_steps"] = int(cluster_duration_steps)
                probe_info["cluster_turn_counts"] = {
                    "left": int(self._cluster_turn_counts.get(self.left_action, 0)),
                    "right": int(self._cluster_turn_counts.get(self.right_action, 0)),
                }
                probe_info["same_turn_streak"] = int(self._same_turn_streak_count)
                probe_info["replan_cooldown_remaining"] = int(self._replan_cooldown_remaining())
                probe_info["stuck"] = bool(stuck_detected)
                probe_info["waypoint_repair_required"] = bool(waypoint_repair_required)
                probe_info["recovery_actions"] = recovery_actions
                if self.shadow_only:
                    safe_action = original_action
                else:
                    safe_action = corrected
                    changed = safe_action != original_action
                if self.verbose:
                    verb = "would replace" if self.shadow_only else "replace"
                    print(
                        "[VLMapSafety] forward blocked; "
                        f"{verb} action {original_action} -> {corrected} at step {self._step}"
                        f"{' (shadow only)' if self.shadow_only else ''}"
                    )
                    if replan_required:
                        print(
                            "[VLMapSafety] repeated blocked forward; "
                            f"request S2 replan after {repeat_block_count} local triggers "
                            f"in cluster {self._active_cluster_id}"
                        )
                    elif replan_suppressed_reason is not None:
                        print(
                            "[VLMapSafety] repeated blocked forward; "
                            f"suppress S2 replan by {replan_suppressed_reason} "
                            f"in cluster {self._active_cluster_id}"
                        )
                    if waypoint_repair_required:
                        print(
                            "[VLMapSafety] stuck cluster detected; "
                            f"request waypoint-level repair for cluster {self._active_cluster_id}"
                        )
            else:
                self._clear_recent_blocks()
                self._clear_block_cluster()
        self._maybe_save_debug_snapshot(
            obs=obs,
            pose_tf=pose_tf,
            input_action=original_action,
            output_action=safe_action,
            changed=changed,
            front_free=front_free,
            probe_info=probe_info,
        )
        self.last_decision = {
            "input_action": original_action,
            "output_action": int(safe_action),
            "changed": bool(changed),
            "front_free": None if front_free is None else bool(front_free),
            "replan_required": bool(replan_required),
            "replan_suppressed_reason": replan_suppressed_reason,
            "repeat_block_count": int(repeat_block_count),
            "cluster_id": self._active_cluster_id,
            "cluster_block_count": int(self._cluster_block_count),
            "cluster_replan_count": int(self._cluster_replan_count),
            "cluster_displacement": float(cluster_displacement),
            "cluster_duration_steps": int(cluster_duration_steps),
            "replan_cooldown_remaining": int(self._replan_cooldown_remaining()),
            "stuck": bool(stuck_detected),
            "waypoint_repair_required": bool(waypoint_repair_required),
            "recovery_actions": recovery_actions,
            "safety_step": int(self._step),
        }
        return safe_action, changed

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
            "obstacle_height_min": float(self.config.get("obstacle_height_min", 0.15)),
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

    def _pick_turn_action(self, pose_tf: np.ndarray, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        row, col, yaw = self.builder.base_pose_to_grid(pose_tf)
        probe_distance = self.forward_distance
        left_yaw = yaw + math.radians(self.turn_angle_deg)
        right_yaw = yaw - math.radians(self.turn_angle_deg)
        left_probe = self._probe_grid(row, col, left_yaw, probe_distance)
        right_probe = self._probe_grid(row, col, right_yaw, probe_distance)
        left_free, left_stats = self._is_line_free((row, col), left_probe)
        right_free, right_stats = self._is_line_free((row, col), right_probe)
        preferred_turn = self._preferred_turn_action(obs)
        corrected = self._choose_turn_action(left_free, right_free, preferred_turn)
        probe_info = {
            "row": int(row),
            "col": int(col),
            "yaw": float(yaw),
            "left_probe": [int(left_probe[0]), int(left_probe[1])],
            "right_probe": [int(right_probe[0]), int(right_probe[1])],
            "left_free": bool(left_free),
            "right_free": bool(right_free),
            "left_stats": left_stats,
            "right_stats": right_stats,
            "preferred_turn": int(preferred_turn) if preferred_turn is not None else None,
            "corrected": int(corrected),
        }
        return corrected, probe_info

    def _is_forward_free(self, pose_tf: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        start_row, start_col, _ = self.builder.base_pose_to_grid(pose_tf)
        goal_row, goal_col = self.builder.forward_target_grid(pose_tf, self.forward_distance)
        return self._is_line_free((start_row, start_col), (goal_row, goal_col))

    def _is_line_free(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[bool, Dict[str, Any]]:
        obstacle_map = self.builder.get_obstacle_map()
        line = list(self._grid_line(start, goal))
        skip_cells = max(0, int(math.floor(self.line_skip_distance / self.builder.cs)))
        if skip_cells > 0 and len(line) > skip_cells:
            line = line[skip_cells:]

        checked = 0
        blocked = 0
        radius = max(0, int(self.radius_cells))
        for row, col in line:
            if row < 0 or row >= self.builder.gs or col < 0 or col >= self.builder.gs:
                continue
            r0 = max(0, row - radius)
            r1 = min(self.builder.gs, row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(self.builder.gs, col + radius + 1)
            local = obstacle_map[r0:r1, c0:c1]
            checked += 1
            if local.size and float(np.mean(local == 0)) >= self.line_cell_blocked_fraction:
                blocked += 1

        blocked_fraction = blocked / checked if checked > 0 else 0.0
        is_blocked = (
            checked >= self.line_min_checked_cells
            and blocked >= self.line_blocked_min_cells
            and blocked_fraction >= self.line_blocked_fraction
        )
        stats = {
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "checked": int(checked),
            "blocked": int(blocked),
            "blocked_fraction": float(blocked_fraction),
            "skip_cells": int(skip_cells),
            "radius_cells": int(radius),
            "min_checked_cells": int(self.line_min_checked_cells),
            "blocked_min_cells": int(self.line_blocked_min_cells),
            "blocked_fraction_threshold": float(self.line_blocked_fraction),
            "cell_blocked_fraction_threshold": float(self.line_cell_blocked_fraction),
        }
        return not is_blocked, stats

    def _grid_line(self, start: Tuple[int, int], goal: Tuple[int, int]):
        r0, c0 = int(start[0]), int(start[1])
        r1, c1 = int(goal[0]), int(goal[1])
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        row, col = r0, c0
        while True:
            yield row, col
            if row == r1 and col == c1:
                break
            err2 = 2 * err
            if err2 > -dc:
                err -= dc
                row += sr
            if err2 < dr:
                err += dr
                col += sc

    def _preferred_turn_action(self, obs: Dict[str, Any]) -> Optional[int]:
        if not self.prefer_previous_turn:
            return None
        for key in ("preferred_turn_action", "last_nav_action", "last_action"):
            if key not in obs or obs[key] is None:
                continue
            action = int(obs[key])
            if action in (self.left_action, self.right_action):
                return action
        return None

    def _choose_turn_action(
        self,
        left_free: bool,
        right_free: bool,
        preferred_turn: Optional[int],
    ) -> int:
        options = []
        if left_free:
            options.append(self.left_action)
        if right_free:
            options.append(self.right_action)
        if not options:
            return int(self.fallback_action)
        if len(options) == 1:
            return int(options[0])

        opposite = {
            self.left_action: self.right_action,
            self.right_action: self.left_action,
        }
        if (
            self._same_turn_streak_action in options
            and self._same_turn_streak_count >= self.max_same_turn_in_cluster
            and opposite[self._same_turn_streak_action] in options
        ):
            return int(opposite[self._same_turn_streak_action])

        if preferred_turn in options:
            preferred_count = self._cluster_turn_counts.get(int(preferred_turn), 0)
            other = opposite[int(preferred_turn)]
            other_count = self._cluster_turn_counts.get(other, 0)
            if preferred_count <= other_count + self.max_same_turn_in_cluster:
                return int(preferred_turn)

        left_count = self._cluster_turn_counts.get(self.left_action, 0)
        right_count = self._cluster_turn_counts.get(self.right_action, 0)
        if right_count < left_count:
            return int(self.right_action)
        return int(self.left_action)

    def _locked_safety_turn(self, position: np.ndarray) -> Optional[int]:
        if self._last_safety_turn_action not in (self.left_action, self.right_action):
            return None
        if (
            self._same_turn_streak_action == self._last_safety_turn_action
            and self._same_turn_streak_count >= self.max_same_turn_in_cluster
        ):
            return None
        if self._last_safety_turn_step is None or self._last_safety_turn_position is None:
            return None
        if self._step - self._last_safety_turn_step > self.repeat_turn_lock_steps:
            return None
        if self.repeat_block_distance >= 0:
            dist = np.linalg.norm(position[:2] - self._last_safety_turn_position[:2])
            if dist > self.repeat_block_distance:
                return None
        return int(self._last_safety_turn_action)

    def _record_block(self, position: np.ndarray, turn_action: int) -> int:
        self._ensure_block_cluster(position)
        self._recent_blocks = self._filtered_recent_blocks(position)
        self._recent_blocks.append(
            {
                "step": int(self._step),
                "position": position.astype(np.float32, copy=True),
                "turn_action": int(turn_action),
            }
        )
        if int(turn_action) in (self.left_action, self.right_action):
            self._cluster_block_count += 1
            self._last_safety_turn_action = int(turn_action)
            self._last_safety_turn_step = int(self._step)
            self._last_safety_turn_position = position.astype(np.float32, copy=True)
            self._cluster_turn_counts[int(turn_action)] = self._cluster_turn_counts.get(int(turn_action), 0) + 1
            if self._same_turn_streak_action == int(turn_action):
                self._same_turn_streak_count += 1
            else:
                self._same_turn_streak_action = int(turn_action)
                self._same_turn_streak_count = 1
        return len(self._recent_blocks)

    def _filtered_recent_blocks(self, position: np.ndarray):
        filtered = []
        for item in self._recent_blocks:
            if self._step - int(item["step"]) > self.repeat_block_window_steps:
                continue
            if self.repeat_block_distance >= 0:
                dist = np.linalg.norm(position[:2] - item["position"][:2])
                if dist > self.repeat_block_distance:
                    continue
            filtered.append(item)
        return filtered

    def _clear_recent_blocks(self) -> None:
        self._recent_blocks = []

    def _ensure_block_cluster(self, position: np.ndarray) -> int:
        should_start = self._active_cluster_id is None
        if not should_start and self._cluster_last_step is not None:
            should_start = self._step - self._cluster_last_step > self.cluster_reset_steps
        if (
            not should_start
            and self.cluster_reset_distance >= 0
            and self._cluster_start_position is not None
        ):
            dist_from_start = np.linalg.norm(position[:2] - self._cluster_start_position[:2])
            should_start = dist_from_start > self.cluster_reset_distance
        if should_start:
            self._start_block_cluster(position)
        self._cluster_last_position = position.astype(np.float32, copy=True)
        self._cluster_last_step = int(self._step)
        return int(self._active_cluster_id)

    def _start_block_cluster(self, position: np.ndarray) -> None:
        self._cluster_seq += 1
        self._active_cluster_id = int(self._cluster_seq)
        self._cluster_start_position = position.astype(np.float32, copy=True)
        self._cluster_start_step = int(self._step)
        self._cluster_last_position = position.astype(np.float32, copy=True)
        self._cluster_last_step = int(self._step)
        self._cluster_block_count = 0
        self._cluster_replan_count = 0
        self._cluster_waypoint_repair_count = 0
        self._cluster_turn_counts = {self.left_action: 0, self.right_action: 0}
        self._same_turn_streak_action = None
        self._same_turn_streak_count = 0
        self._last_safety_turn_action = None
        self._last_safety_turn_step = None
        self._last_safety_turn_position = None
        self._replan_cooldown_until_step = -1

    def _clear_block_cluster(self) -> None:
        self._active_cluster_id = None
        self._cluster_start_position = None
        self._cluster_start_step = None
        self._cluster_last_position = None
        self._cluster_last_step = None
        self._cluster_block_count = 0
        self._cluster_replan_count = 0
        self._cluster_waypoint_repair_count = 0
        self._cluster_turn_counts = {self.left_action: 0, self.right_action: 0}
        self._same_turn_streak_action = None
        self._same_turn_streak_count = 0
        self._last_safety_turn_action = None
        self._last_safety_turn_step = None
        self._last_safety_turn_position = None
        self._replan_cooldown_until_step = -1

    def _replan_cooldown_remaining(self) -> int:
        return max(0, int(self._replan_cooldown_until_step - self._step))

    def _cluster_displacement(self, position: np.ndarray) -> float:
        if self._cluster_start_position is None:
            return 0.0
        return float(np.linalg.norm(position[:2] - self._cluster_start_position[:2]))

    def _cluster_duration_steps(self) -> int:
        if self._cluster_start_step is None:
            return 0
        return max(0, int(self._step - self._cluster_start_step))

    def _is_cluster_stuck(self, position: np.ndarray) -> bool:
        return (
            self._cluster_block_count >= self.stuck_block_count
            and self._cluster_displacement(position) <= self.stuck_distance
        )

    def _maybe_save_debug_snapshot(
        self,
        obs: Dict[str, Any],
        pose_tf: np.ndarray,
        input_action: int,
        output_action: int,
        changed: bool,
        front_free: Optional[bool],
        probe_info: Optional[Dict[str, Any]],
    ) -> None:
        if not self.debug or self.builder is None:
            return
        if self.debug_save_on_change and (changed or front_free is False):
            should_save = True
        elif self.debug_save_every_steps > 0 and self._step % self.debug_save_every_steps == 0:
            should_save = True
        else:
            should_save = False
        if self.debug_max_snapshots >= 0 and self._debug_saved_snapshots >= self.debug_max_snapshots:
            should_save = False

        debug_dir = self._get_debug_dir()
        os.makedirs(debug_dir, exist_ok=True)
        context = obs.get("debug_context", {}) or {}
        episode_id = context.get("episode_id", "unknown")
        scene_id = str(context.get("scene_id", "scene")).replace(os.sep, "_")
        eval_step = context.get("step_id", self._step)
        prefix = f"{scene_id}_ep{episode_id}_step{int(eval_step):05d}_safe{self._step:05d}"
        should_save = self._sample_debug_snapshot(should_save, context)
        if not should_save and not self.debug_log_all_events:
            return

        image_path = None
        if should_save:
            try:
                from PIL import Image, ImageDraw
            except Exception as exc:
                if not self._debug_import_warned:
                    print(f"[VLMapSafety] debug snapshot disabled because PIL import failed: {exc}")
                    self._debug_import_warned = True
                should_save = False

        if should_save:
            rgb_img = self._rgb_debug_image(obs.get("rgb"), Image, ImageDraw)
            depth_img = self._depth_debug_image(obs.get("depth"), Image, ImageDraw)
            map_img = self._map_debug_image(pose_tf, probe_info, front_free, Image, ImageDraw)
            panels = [img for img in (rgb_img, depth_img, map_img) if img is not None]
            if panels:
                height = max(img.height for img in panels)
                width = sum(img.width for img in panels)
                canvas = Image.new("RGB", (width, height), (20, 20, 20))
                offset = 0
                for img in panels:
                    canvas.paste(img, (offset, 0))
                    offset += img.width
                image_path = os.path.join(debug_dir, f"{prefix}.png")
                canvas.save(image_path)
                self._debug_saved_snapshots += 1

        event = {
            "scene_id": context.get("scene_id"),
            "episode_id": context.get("episode_id"),
            "eval_step": int(eval_step),
            "safety_step": int(self._step),
            "input_action": int(input_action),
            "output_action": int(output_action),
            "changed": bool(changed),
            "front_free": None if front_free is None else bool(front_free),
            "shadow_only": bool(self.shadow_only),
            "gps": self._jsonable(obs.get("gps")),
            "compass": self._jsonable(obs.get("compass")),
            "last_nav_action": self._jsonable(obs.get("last_nav_action")),
            "pixel_goal": self._jsonable(context.get("pixel_goal")),
            "repeat_block_count": int(probe_info.get("repeat_block_count", 0)) if probe_info else 0,
            "replan_required": bool(probe_info.get("replan_required", False)) if probe_info else False,
            "replan_suppressed_reason": probe_info.get("replan_suppressed_reason") if probe_info else None,
            "cluster_id": probe_info.get("cluster_id") if probe_info else None,
            "cluster_block_count": int(probe_info.get("cluster_block_count", 0)) if probe_info else 0,
            "cluster_replan_count": int(probe_info.get("cluster_replan_count", 0)) if probe_info else 0,
            "cluster_waypoint_repair_count": (
                int(probe_info.get("cluster_waypoint_repair_count", 0)) if probe_info else 0
            ),
            "cluster_displacement": float(probe_info.get("cluster_displacement", 0.0)) if probe_info else 0.0,
            "cluster_duration_steps": int(probe_info.get("cluster_duration_steps", 0)) if probe_info else 0,
            "cluster_turn_counts": probe_info.get("cluster_turn_counts") if probe_info else None,
            "same_turn_streak": int(probe_info.get("same_turn_streak", 0)) if probe_info else 0,
            "replan_cooldown_remaining": int(probe_info.get("replan_cooldown_remaining", 0)) if probe_info else 0,
            "stuck": bool(probe_info.get("stuck", False)) if probe_info else False,
            "waypoint_repair_required": (
                bool(probe_info.get("waypoint_repair_required", False)) if probe_info else False
            ),
            "recovery_actions": probe_info.get("recovery_actions", []) if probe_info else [],
            "probe": probe_info,
            "image_path": image_path,
        }
        with open(os.path.join(debug_dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _sample_debug_snapshot(self, should_save: bool, context: Dict[str, Any]) -> bool:
        if not should_save:
            return False
        if not self.debug_sample_snapshots:
            return True
        if self.debug_sample_total_snapshots >= 0 and self._debug_saved_snapshots >= self.debug_sample_total_snapshots:
            return False

        episode_index = context.get("episode_index")
        episode_count = context.get("episode_count")
        if episode_index is None or episode_count is None:
            return True

        try:
            episode_index = int(episode_index)
            episode_count = int(episode_count)
        except (TypeError, ValueError):
            return True
        if episode_count <= 0:
            return True

        selected = self._get_debug_selected_episode_indices(episode_count)
        if episode_index not in selected:
            return False

        current_count = self._debug_episode_snapshot_counts.get(episode_index, 0)
        if current_count >= self.debug_sample_images_per_episode:
            return False

        # Keep a deterministic, sparse sample within selected episodes instead of saving every early candidate.
        candidate_count = self._debug_episode_candidate_counts.get(episode_index, 0)
        self._debug_episode_candidate_counts[episode_index] = candidate_count + 1
        stride = max(1, int(self.config.get("debug_sample_candidate_stride", 2)))
        offset = self._debug_episode_sample_offset(episode_index, stride)
        if candidate_count % stride != offset and current_count > 0:
            return False

        self._debug_episode_snapshot_counts[episode_index] = current_count + 1
        return True

    def _get_debug_selected_episode_indices(self, episode_count: int):
        if self._debug_selected_episode_indices is not None:
            return self._debug_selected_episode_indices

        if self.debug_sample_episode_count > 0:
            sample_episode_count = self.debug_sample_episode_count
        else:
            total = self.debug_sample_total_snapshots if self.debug_sample_total_snapshots >= 0 else self.debug_max_snapshots
            total = max(1, int(total))
            sample_episode_count = int(math.ceil(total / self.debug_sample_images_per_episode))

        sample_episode_count = max(1, min(int(sample_episode_count), int(episode_count)))
        rng = random.Random(self.debug_sample_seed + episode_count)
        self._debug_selected_episode_indices = set(rng.sample(range(episode_count), sample_episode_count))
        if self.verbose:
            print(
                "[VLMapSafety] debug snapshot sampled episodes: "
                f"{sorted(self._debug_selected_episode_indices)} "
                f"({sample_episode_count}/{episode_count})"
            )
        return self._debug_selected_episode_indices

    def _debug_episode_sample_offset(self, episode_index: int, stride: int) -> int:
        if stride <= 1:
            return 0
        rng = random.Random(self.debug_sample_seed + 7919 * (episode_index + 1))
        return rng.randrange(stride)

    def _get_debug_dir(self) -> str:
        if self.debug_dir is not None:
            return self.debug_dir
        if not self.debug_use_run_subdir:
            self.debug_dir = self.debug_root_dir
            return self.debug_dir

        os.makedirs(self.debug_root_dir, exist_ok=True)
        for idx in range(1, 10000):
            candidate = os.path.join(self.debug_root_dir, f"{self.debug_run_prefix}_{idx:03d}")
            try:
                os.makedirs(candidate)
            except FileExistsError:
                continue
            self.debug_dir = candidate
            if self.verbose:
                print(f"[VLMapSafety] debug output dir: {self.debug_dir}")
            return self.debug_dir

        raise RuntimeError(f"Unable to allocate a debug run directory under {self.debug_root_dir}")

    def get_debug_dir(self) -> Optional[str]:
        if not self.debug:
            return None
        return self._get_debug_dir()

    def _rgb_debug_image(self, rgb: Any, Image: Any, ImageDraw: Any) -> Optional[Any]:
        if rgb is None:
            return None
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            return None
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB").resize((320, 320))
        return self._add_label(image, "RGB observation", Image, ImageDraw)

    def _depth_debug_image(self, depth: Any, Image: Any, ImageDraw: Any) -> Optional[Any]:
        if depth is None:
            return None
        arr = np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32, copy=False)
        max_depth = float(self.config.get("debug_depth_max", self.config.get("max_depth", 6.0)))
        arr = np.nan_to_num(arr, nan=max_depth, posinf=max_depth, neginf=0.0)
        arr = np.clip(arr / max(max_depth, 1e-6), 0.0, 1.0)
        gray = (255 * (1.0 - arr)).astype(np.uint8)
        image = Image.fromarray(gray, mode="L").convert("RGB").resize((320, 320))
        return self._add_label(image, "Depth, near=bright", Image, ImageDraw)

    def _map_debug_image(
        self,
        pose_tf: np.ndarray,
        probe_info: Optional[Dict[str, Any]],
        front_free: Optional[bool],
        Image: Any,
        ImageDraw: Any,
    ) -> Any:
        obstacle_map = self.builder.get_obstacle_map()
        seen = getattr(self.builder, "seen", None)
        row, col, yaw = self.builder.base_pose_to_grid(pose_tf)
        radius = max(10, int(self.debug_crop_radius_cells))
        r0 = max(0, row - radius)
        r1 = min(self.builder.gs, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(self.builder.gs, col + radius + 1)

        crop = obstacle_map[r0:r1, c0:c1]
        image_arr = np.full((crop.shape[0], crop.shape[1], 3), 45, dtype=np.uint8)
        if seen is not None:
            seen_crop = seen[r0:r1, c0:c1]
            image_arr[seen_crop & crop] = np.array([235, 235, 235], dtype=np.uint8)
            image_arr[seen_crop & ~crop] = np.array([225, 65, 65], dtype=np.uint8)
        else:
            image_arr[crop] = np.array([235, 235, 235], dtype=np.uint8)
            image_arr[~crop] = np.array([225, 65, 65], dtype=np.uint8)

        scale = self.debug_cell_scale
        resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
        image = Image.fromarray(image_arr, mode="RGB").resize(
            (image_arr.shape[1] * scale, image_arr.shape[0] * scale),
            resample=resample_nearest,
        )
        draw = ImageDraw.Draw(image)

        def to_xy(grid_row: int, grid_col: int) -> Tuple[int, int]:
            return (
                int((grid_col - c0) * scale + scale / 2),
                int((grid_row - r0) * scale + scale / 2),
            )

        agent_xy = to_xy(row, col)
        front_goal = self.builder.forward_target_grid(pose_tf, self.forward_distance)
        draw.line([agent_xy, to_xy(front_goal[0], front_goal[1])], fill=(255, 220, 0), width=max(1, scale))
        if probe_info is not None:
            left_probe = probe_info.get("left_probe")
            right_probe = probe_info.get("right_probe")
            if left_probe is not None:
                draw.line([agent_xy, to_xy(left_probe[0], left_probe[1])], fill=(50, 170, 255), width=max(1, scale))
            if right_probe is not None:
                draw.line([agent_xy, to_xy(right_probe[0], right_probe[1])], fill=(160, 120, 255), width=max(1, scale))

        heading_len = max(8, int(0.5 / self.builder.cs))
        heading_xy = to_xy(
            int(row - heading_len * math.cos(yaw)),
            int(col - heading_len * math.sin(yaw)),
        )
        draw.ellipse(
            [agent_xy[0] - 5, agent_xy[1] - 5, agent_xy[0] + 5, agent_xy[1] + 5],
            fill=(40, 220, 80),
        )
        draw.line([agent_xy, heading_xy], fill=(40, 220, 80), width=max(2, scale))

        front_text = "unknown" if front_free is None else ("free" if front_free else "blocked")
        front_stats = probe_info.get("front") if probe_info is not None else None
        if front_stats is not None:
            front_text += (
                f" F={front_stats.get('blocked', '?')}/{front_stats.get('checked', '?')}"
                f"@{front_stats.get('blocked_fraction', 0.0):.2f}"
            )
        if probe_info is not None:
            turn_text = (
                f"L={probe_info.get('left_free')} R={probe_info.get('right_free')} "
                f"pref={probe_info.get('preferred_turn')} -> {probe_info.get('corrected')} "
                f"c={probe_info.get('cluster_id')} "
                f"b={probe_info.get('cluster_block_count', 0)} "
                f"rp={probe_info.get('cluster_replan_count', 0)} "
                f"cd={probe_info.get('replan_cooldown_remaining', 0)} "
                f"stuck={probe_info.get('stuck', False)}"
            )
        else:
            turn_text = "no turn probe"
        return self._add_label(image, f"VLMap crop front={front_text} {turn_text}", Image, ImageDraw)

    def _add_label(self, image: Any, label: str, Image: Any, ImageDraw: Any) -> Any:
        label_h = 28
        canvas = Image.new("RGB", (image.width, image.height + label_h), (20, 20, 20))
        canvas.paste(image, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), label, fill=(235, 235, 235))
        return canvas

    def _jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        return value

    def _probe_grid(self, row: int, col: int, yaw: float, distance: float) -> Tuple[int, int]:
        cs = self.builder.cs
        gs = self.builder.gs
        x = (gs / 2 - row) * cs + distance * math.cos(yaw)
        y = (gs / 2 - col) * cs + distance * math.sin(yaw)
        probe_row = int(gs / 2 - int(x / cs))
        probe_col = int(gs / 2 - int(y / cs))
        return probe_row, probe_col

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _as_intrinsic3(intrinsic: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape == (4, 4):
        return intrinsic[:3, :3]
    if intrinsic.shape == (3, 3):
        return intrinsic
    raise ValueError(f"Expected camera intrinsic shape (3, 3) or (4, 4), got {intrinsic.shape}")


def _default_cam_to_base_tf(camera_height: float) -> np.ndarray:
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    tf[:3, 3] = np.array([0.0, 0.0, camera_height], dtype=np.float32)
    return tf


def _yaw_to_tf(pos: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tf[:3, 3] = np.asarray(pos, dtype=np.float32).reshape(3)
    return tf


def _depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False)
    h, w = depth.shape
    ids = np.arange(h * w, dtype=np.int64)[:: max(1, int(sample_rate))]
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    yy = ids // w
    xx = ids % w
    zz = depth.reshape(-1)[ids]
    valid = (zz > float(min_depth)) & (zz < float(max_depth)) & np.isfinite(zz)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    ids = ids[valid]
    xx = xx[valid].astype(np.float32) + 0.5
    yy = yy[valid].astype(np.float32) + 0.5
    zz = zz[valid].astype(np.float32)
    inv_intrinsic = np.linalg.inv(intrinsic)
    pix = np.stack([xx, yy, np.ones_like(xx)], axis=0)
    points = (inv_intrinsic @ pix) * zz.reshape(1, -1)
    return points.T.astype(np.float32, copy=False), ids


@dataclass
class SparseOccMemoryConfig:
    enable: bool = False
    shadow_only: bool = True
    grid_size: int = 1000
    cell_size: float = 0.05
    map_height: float = 2.5
    camera_height: float = 1.5
    min_depth: float = 0.15
    max_depth: float = 5.0
    depth_sample_rate: int = 240
    update_every_steps: int = 1
    center_on_first_pose: bool = True
    obstacle_height_min: float = 0.15
    obstacle_height_max: float = 1.2
    raycast_enable: bool = True
    raycast_stride_cells: int = 2
    raycast_max_points_per_update: int = 2500
    keyframe_every_steps: int = 10
    keyframe_min_distance: float = 0.50
    frontier_enable: bool = True
    frontier_connectivity: int = 8
    waypoint_probe_enable: bool = True
    waypoint_source_image_width: Optional[int] = None
    waypoint_source_image_height: Optional[int] = None
    waypoint_depth_patch_radius: int = 2
    waypoint_revisit_radius_cells: int = 8
    waypoint_frontier_sample_limit: int = 2000
    attribution_enable: bool = False
    attribution_frontier_sample_limit: int = 5000
    attribution_recent_semantic_window: int = 5
    attribution_high_conf_recent_window: int = 5
    attribution_stagnation_active_window_steps: int = 20
    attribution_dead_zone_min_step: int = 30
    attribution_dead_zone_unique_threshold: int = 2
    attribution_dead_zone_score_threshold: float = 0.65
    attribution_direction_match_degrees: float = 45.0
    save_bev: bool = True
    bev_every_updates: int = 20
    max_bev_snapshots: int = 12
    bev_crop_radius_cells: int = 140
    bev_cell_scale: int = 3
    validation_enable: bool = False
    validation_every_updates: int = 20
    validation_max_snapshots: int = 4
    validation_current_depth_sample_rate: int = 8
    validation_max_current_points: int = 120000
    validation_max_memory_points: int = 160000
    validation_max_occupied_points: int = 80000
    validation_max_free_points: int = 50000
    validation_max_frontier_points: int = 20000
    validation_save_rgb_depth: bool = True
    validation_save_current_rgb_ply: bool = True
    validation_save_memory_ply: bool = True
    validation_save_final_memory_ply: bool = True
    verbose: bool = False


class SparseOccSemanticMemory:
    """Shadow-only sparse 3D occupancy memory for VLN diagnostics.

    The first V8a implementation focuses on geometry and topology:
    endpoint occupied voxels, ray-cast free voxels, pose/keyframe nodes,
    frontier cells, waypoint memory probes, and BEV snapshots.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, camera_intrinsic: Optional[np.ndarray] = None):
        raw = dict(config or {})
        prefix = "occ_memory_"
        cfg_dict = {}
        for field_name in SparseOccMemoryConfig.__dataclass_fields__:
            if field_name in raw:
                cfg_dict[field_name] = raw[field_name]
            prefixed = prefix + field_name
            if prefixed in raw:
                cfg_dict[field_name] = raw[prefixed]
        for shared_key in (
            "grid_size",
            "cell_size",
            "map_height",
            "camera_height",
            "min_depth",
            "max_depth",
            "depth_sample_rate",
            "center_on_first_pose",
            "obstacle_height_min",
            "obstacle_height_max",
            "waypoint_source_image_width",
            "waypoint_source_image_height",
            "waypoint_depth_patch_radius",
            "verbose",
        ):
            if shared_key in raw and shared_key not in cfg_dict:
                cfg_dict[shared_key] = raw[shared_key]
        self.config = SparseOccMemoryConfig(**cfg_dict)
        self.gs = int(self.config.grid_size)
        self.cs = float(self.config.cell_size)
        self.vh = max(1, int(math.ceil(float(self.config.map_height) / self.cs)) + 1)
        self.camera_intrinsic = _as_intrinsic3(camera_intrinsic) if camera_intrinsic is not None else None
        self.cam_to_base_tf = _default_cam_to_base_tf(float(self.config.camera_height))
        self.debug_dir: Optional[str] = None
        self.reset_episode()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def set_debug_dir(self, debug_dir: Optional[str]) -> None:
        self.debug_dir = debug_dir

    def reset_episode(
        self,
        *,
        instruction: str = "",
        scene_id: str = "",
        episode_id: Optional[int] = None,
        episode_index: Optional[int] = None,
        episode_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.episode_meta = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "instruction": instruction,
        }
        self.occ_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.free_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.occ2d_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.free2d_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.visited2d_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pose_trace: List[Dict[str, Any]] = []
        self.keyframes: List[Dict[str, Any]] = []
        self.semantic_events: List[Dict[str, Any]] = []
        self.waypoint_events: List[Dict[str, Any]] = []
        self.init_base_tf: Optional[np.ndarray] = None
        self.inv_init_base_tf: Optional[np.ndarray] = None
        self.update_count = 0
        self.observation_count = 0
        self.free_update_count = 0
        self.occupied_update_count = 0
        self.frontier_cache: Optional[List[Tuple[int, int]]] = None
        self.frontier_cache_update = -1
        self.last_pose_grid: Optional[Tuple[int, int]] = None
        self.last_keyframe_xy: Optional[np.ndarray] = None
        self.saved_bev_count = 0
        self.saved_validation_count = 0
        self.saved_validation_final_count = 0
        self.last_semantic_decision: Dict[str, Any] = {}
        self.last_stagnation_step: Optional[int] = None
        event = {
            "event_type": "occ_memory_episode_start",
            **self.episode_meta,
            "enabled": bool(self.enabled),
            "shadow_only": bool(self.config.shadow_only),
        }
        self._write_event(event)
        return event

    def update_observation(
        self,
        obs: Dict[str, Any],
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        event = {
            "event_type": "occ_memory_update",
            **self.episode_meta,
            **context,
            "enabled": bool(self.enabled),
            "valid": False,
        }
        if not self.enabled:
            event["reason"] = "disabled"
            return event
        if self.camera_intrinsic is None:
            event["reason"] = "missing_intrinsic"
            self._write_event(event)
            return event
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None:
            event["reason"] = "missing_pose"
            self._write_event(event)
            return event
        self.observation_count += 1
        if self.observation_count % max(1, int(self.config.update_every_steps)) != 0:
            event["reason"] = "update_stride"
            self._write_event(event)
            return event

        if self.init_base_tf is None:
            self.init_base_tf = pose_tf.copy()
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        rel_base_tf = self._relative_base_tf(pose_tf)
        cam_pose_tf = rel_base_tf @ self.cam_to_base_tf
        pose_row, pose_col, pose_yaw = self._pose_to_grid(rel_base_tf)
        self.last_pose_grid = (int(pose_row), int(pose_col))
        self.visited2d_counts[(int(pose_row), int(pose_col))] += 1
        self.pose_trace.append(
            {
                "step_id": context.get("step_id"),
                "row": int(pose_row),
                "col": int(pose_col),
                "x": float(rel_base_tf[0, 3]),
                "y": float(rel_base_tf[1, 3]),
                "z": float(rel_base_tf[2, 3]),
                "yaw": float(pose_yaw),
            }
        )
        self._maybe_add_keyframe(context, rel_base_tf, pose_row, pose_col, pose_yaw)

        cam_points, point_ids = _depth_to_points(
            depth,
            self.camera_intrinsic,
            min_depth=float(self.config.min_depth),
            max_depth=float(self.config.max_depth),
            sample_rate=int(self.config.depth_sample_rate),
        )
        if cam_points.shape[0] == 0:
            event["reason"] = "no_valid_depth"
            self._write_event(event)
            return event

        if (
            int(self.config.raycast_max_points_per_update) > 0
            and cam_points.shape[0] > int(self.config.raycast_max_points_per_update)
        ):
            ids = np.linspace(
                0,
                cam_points.shape[0] - 1,
                int(self.config.raycast_max_points_per_update),
            ).astype(np.int64)
            cam_points = cam_points[ids]
            point_ids = point_ids[ids]

        cam_points_h = np.concatenate(
            [cam_points, np.ones((cam_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        world_points = (cam_pose_tf @ cam_points_h.T).T[:, :3]
        cam_origin = cam_pose_tf[:3, 3]
        occupied_added = 0
        free_added = 0
        for point in world_points:
            endpoint = self._xyz_to_grid(point)
            if endpoint is None:
                continue
            row, col, height = endpoint
            key3 = (row, col, height)
            before = self.occ_counts.get(key3, 0)
            self.occ_counts[key3] = before + 1
            if before == 0:
                occupied_added += 1
            if self._is_obstacle_height(height):
                self.occ2d_counts[(row, col)] += 1
            if self.config.raycast_enable:
                free_added += self._raycast_free(cam_origin, point, endpoint)

        self.update_count += 1
        self.occupied_update_count += occupied_added
        self.free_update_count += free_added
        self.frontier_cache = None
        frontier_count = None
        if self.config.save_bev and self.update_count % max(1, int(self.config.bev_every_updates)) == 0:
            frontier_count = len(self.get_frontier_cells(sample_limit=0))
        event.update(
            {
                "valid": True,
                "reason": "updated",
                "update_count": int(self.update_count),
                "sampled_point_count": int(cam_points.shape[0]),
                "occupied_added": int(occupied_added),
                "free_added": int(free_added),
                "occupied_voxel_count": int(len(self.occ_counts)),
                "free_voxel_count": int(len(self.free_counts)),
                "occupied_cell_count": int(len(self.occ2d_counts)),
                "free_cell_count": int(len(self.free2d_counts)),
                "frontier_count": None if frontier_count is None else int(frontier_count),
                "pose_grid": [int(pose_row), int(pose_col)],
            }
        )
        self._write_event(event)
        self._maybe_write_bev_snapshot(context)
        self._maybe_write_validation_snapshot(
            context,
            rgb=rgb,
            depth=depth,
            cam_pose_tf=cam_pose_tf,
            world_points=world_points,
            point_ids=point_ids,
        )
        return event

    def record_semantic(self, decision: Dict[str, Any]) -> None:
        if not self.enabled or not decision:
            return
        step_id = decision.get("step_id")
        top_margin = decision.get("top_margin")
        if top_margin is None:
            top_margin = decision.get("top_margin_to_second")
        if top_margin is None:
            top_margin = decision.get("rank1_margin_to_second")
        event = {
            "event_type": "occ_memory_semantic",
            **self.episode_meta,
            "step_id": step_id,
            "top_match": decision.get("top_match"),
            "top_score": decision.get("top_score"),
            "top_margin": top_margin,
            "threshold_hits": decision.get("threshold_hits"),
            "high_conf_semantic": decision.get("high_conf_semantic"),
            "stagnation_would_requery": decision.get("stagnation_would_requery"),
            "stagnation_recent_unique_count": decision.get("stagnation_recent_unique_count"),
            "stagnation_recent_terms": decision.get("stagnation_recent_terms"),
            "stagnation_would_requery_reason": decision.get("stagnation_would_requery_reason"),
            "status": decision.get("status"),
        }
        if self.last_pose_grid is not None:
            event["pose_grid"] = [int(self.last_pose_grid[0]), int(self.last_pose_grid[1])]
        if self.pose_trace:
            pose = self.pose_trace[-1]
            event["pose_xy"] = [float(pose["x"]), float(pose["y"])]
            event["pose_yaw"] = float(pose["yaw"])
        if decision.get("stagnation_would_requery"):
            stagnation_step = self._safe_int(step_id)
            if stagnation_step is not None:
                self.last_stagnation_step = stagnation_step
        event["last_stagnation_step"] = self.last_stagnation_step
        self.semantic_events.append(event)
        self.last_semantic_decision = dict(decision)
        if self.keyframes:
            self.keyframes[-1]["semantic_top_match"] = decision.get("top_match")
            self.keyframes[-1]["semantic_top_score"] = decision.get("top_score")
            self.keyframes[-1]["semantic_top_margin"] = top_margin
            self.keyframes[-1]["high_conf_semantic"] = bool(decision.get("high_conf_semantic"))
            self.keyframes[-1]["last_semantic_step_id"] = step_id
        self._write_event(event)

    def record_guidance_event(
        self,
        *,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        hint: str = "",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_type": "occ_memory_guidance",
            **self.episode_meta,
            **dict(context or {}),
            "enabled": bool(self.enabled),
            "action": action,
            "reason": reason,
            "hint": hint,
        }
        if decision:
            event.update(
                {
                    "semantic_dead_zone": decision.get("semantic_dead_zone"),
                    "semantic_dead_zone_score": decision.get("semantic_dead_zone_score"),
                    "semantic_recent_high_conf_count": decision.get("semantic_recent_high_conf_count"),
                    "semantic_recent_terms": decision.get("semantic_recent_terms"),
                    "frontier_dominant_direction": decision.get("frontier_dominant_direction"),
                    "frontier_dominant_angle_deg": decision.get("frontier_dominant_angle_deg"),
                    "frontier_dominant_count": decision.get("frontier_dominant_count"),
                    "frontier_direction_counts": decision.get("frontier_direction_counts"),
                    "waypoint_direction_bucket": decision.get("waypoint_direction_bucket"),
                    "waypoint_direction_angle_deg": decision.get("waypoint_direction_angle_deg"),
                    "waypoint_aligns_with_dominant_frontier": decision.get(
                        "waypoint_aligns_with_dominant_frontier"
                    ),
                    "goal_state": decision.get("goal_state"),
                    "goal_grid": decision.get("goal_grid"),
                    "start_grid": decision.get("start_grid"),
                }
            )
        self._write_event(event)
        return event

    def evaluate_waypoint(
        self,
        pixel_goal: Any,
        obs: Dict[str, Any],
        depth: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        event = {
            "event_type": "occ_memory_waypoint_probe",
            **self.episode_meta,
            **context,
            "enabled": bool(self.enabled and self.config.waypoint_probe_enable),
            "valid": False,
            "pixel_goal": self._jsonable(pixel_goal),
        }
        if not self.enabled or not self.config.waypoint_probe_enable:
            event["reason"] = "disabled"
            return event
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None or self.camera_intrinsic is None:
            event["reason"] = "missing_pose_or_intrinsic"
            self._write_event(event)
            return event
        if self.init_base_tf is None:
            event["reason"] = "memory_not_initialized"
            self._write_event(event)
            return event
        target = self._pixel_goal_to_grid(pixel_goal, depth, pose_tf, context)
        if target is None:
            event["reason"] = "invalid_pixel_goal"
            self._write_event(event)
            return event
        goal_grid = target["goal_grid"]
        state = self._cell_state(goal_grid[0], goal_grid[1])
        frontier_distance = self._nearest_frontier_distance(goal_grid)
        revisit_count = self._nearby_visit_count(goal_grid, int(self.config.waypoint_revisit_radius_cells))
        attribution: Dict[str, Any] = {}
        if self.config.attribution_enable:
            start_grid = target["start_grid"]
            start_yaw = float(target.get("start_yaw", 0.0))
            waypoint_direction = self._direction_to_cell(start_grid, goal_grid, start_yaw)
            frontier_summary = self._frontier_direction_summary(start_grid, start_yaw)
            semantic_state = self._semantic_dead_zone_state(context)
            nearest_high_conf = self._nearest_semantic_keyframe(
                start_grid,
                start_yaw,
                high_conf_only=True,
            )
            waypoint_dir = waypoint_direction.get("bucket")
            waypoint_angle = waypoint_direction.get("angle_deg")
            dominant_frontier_dir = frontier_summary.get("dominant_direction")
            dominant_frontier_angle = frontier_summary.get("dominant_angle_deg")
            attribution = {
                "waypoint_direction_bucket": waypoint_dir,
                "waypoint_direction_angle_deg": waypoint_angle,
                "waypoint_distance_m": waypoint_direction.get("distance_m"),
                "frontier_total_count_for_direction": frontier_summary.get("total_count"),
                "frontier_sampled_count_for_direction": frontier_summary.get("sampled_count"),
                "frontier_sample_fraction_for_direction": frontier_summary.get("sample_fraction"),
                "frontier_direction_counts": frontier_summary.get("direction_counts"),
                "frontier_direction_nearest_m": frontier_summary.get("nearest_m"),
                "frontier_direction_mass_ratio": frontier_summary.get("direction_mass_ratio"),
                "frontier_dominant_direction": dominant_frontier_dir,
                "frontier_dominant_angle_deg": dominant_frontier_angle,
                "frontier_dominant_count": frontier_summary.get("dominant_count"),
                "frontier_direction_entropy": frontier_summary.get("direction_entropy"),
                "waypoint_aligns_with_dominant_frontier": self._directions_aligned(
                    waypoint_angle,
                    dominant_frontier_angle,
                ),
                "waypoint_frontier_direction_count": (
                    frontier_summary.get("direction_counts") or {}
                ).get(str(waypoint_dir), 0),
                "semantic_dead_zone": semantic_state.get("dead_zone"),
                "semantic_dead_zone_score": semantic_state.get("dead_zone_score"),
                "semantic_recent_unique_count": semantic_state.get("recent_unique_count"),
                "semantic_recent_high_conf_count": semantic_state.get("recent_high_conf_count"),
                "semantic_recent_terms": semantic_state.get("recent_terms"),
                "semantic_last_top_match": semantic_state.get("last_top_match"),
                "semantic_last_top_score": semantic_state.get("last_top_score"),
                "semantic_last_stagnation": semantic_state.get("last_stagnation"),
                "semantic_stagnation_active": semantic_state.get("stagnation_active"),
                "semantic_last_stagnation_step": semantic_state.get("last_stagnation_step"),
                "semantic_stagnation_age_steps": semantic_state.get("stagnation_age_steps"),
                "nearest_high_conf_keyframe": nearest_high_conf,
                "waypoint_aligns_with_high_conf_keyframe": self._directions_aligned(
                    waypoint_angle,
                    nearest_high_conf.get("direction_angle_deg") if nearest_high_conf else None,
                ),
            }
        event.update(
            {
                "valid": True,
                "reason": "ok",
                "goal_grid": [int(goal_grid[0]), int(goal_grid[1])],
                "start_grid": target["start_grid"],
                "goal_state": state,
                "depth_m": float(target["depth_m"]),
                "frontier_distance_cells": frontier_distance,
                "frontier_distance_m": (
                    None if frontier_distance is None else float(frontier_distance * self.cs)
                ),
                "nearby_visit_count": int(revisit_count),
                "points_to_revisited_region": bool(revisit_count > 0),
                "occupied_cell_count": int(len(self.occ2d_counts)),
                "free_cell_count": int(len(self.free2d_counts)),
                "frontier_count": int(len(self.get_frontier_cells(sample_limit=0))),
                **attribution,
            }
        )
        self.waypoint_events.append(event)
        self._write_event(event)
        return event

    def finish_episode(self, *, metrics: Optional[Dict[str, Any]] = None, steps: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        frontier_count = len(self.get_frontier_cells(sample_limit=0))
        waypoint_state_counts: Dict[str, int] = defaultdict(int)
        waypoint_direction_counts: Dict[str, int] = defaultdict(int)
        waypoint_frontier_direction_counts: Dict[str, int] = defaultdict(int)
        dead_zone_frontier_direction_counts: Dict[str, int] = defaultdict(int)
        frontier_distances = []
        dead_zone_scores = []
        dead_zone_count = 0
        dead_zone_with_frontier_count = 0
        first_dead_zone_step = None
        waypoint_frontier_align_count = 0
        waypoint_high_conf_align_count = 0
        for event in self.waypoint_events:
            state = str(event.get("goal_state", "invalid"))
            waypoint_state_counts[state] += 1
            dist = event.get("frontier_distance_m")
            if dist is not None:
                frontier_distances.append(float(dist))
            direction = event.get("waypoint_direction_bucket")
            if direction:
                waypoint_direction_counts[str(direction)] += 1
            frontier_direction = event.get("frontier_dominant_direction")
            if frontier_direction:
                waypoint_frontier_direction_counts[str(frontier_direction)] += 1
            if event.get("waypoint_aligns_with_dominant_frontier"):
                waypoint_frontier_align_count += 1
            if event.get("waypoint_aligns_with_high_conf_keyframe"):
                waypoint_high_conf_align_count += 1
            if event.get("semantic_dead_zone"):
                dead_zone_count += 1
                step = event.get("step_id")
                if first_dead_zone_step is None and step is not None:
                    first_dead_zone_step = step
                score = event.get("semantic_dead_zone_score")
                if score is not None:
                    dead_zone_scores.append(float(score))
                if event.get("frontier_dominant_count", 0):
                    dead_zone_with_frontier_count += 1
                    if frontier_direction:
                        dead_zone_frontier_direction_counts[str(frontier_direction)] += 1
        if self.config.validation_enable and self.config.validation_save_final_memory_ply:
            self._write_final_validation_snapshot({"step_id": steps, "final": True})
        final_frontier_summary = {}
        if self.config.attribution_enable and self.pose_trace:
            last_pose = self.pose_trace[-1]
            final_frontier_summary = self._frontier_direction_summary(
                [int(last_pose["row"]), int(last_pose["col"])],
                float(last_pose["yaw"]),
            )
        high_conf_keyframes = [node for node in self.keyframes if node.get("high_conf_semantic")]
        high_conf_events = [event for event in self.semantic_events if event.get("high_conf_semantic")]
        summary = {
            "event_type": "occ_memory_episode_summary",
            **self.episode_meta,
            "steps": steps,
            "metrics": self._compact_metrics(metrics or {}),
            "enabled": bool(self.enabled),
            "update_count": int(self.update_count),
            "observation_count": int(self.observation_count),
            "occupied_voxel_count": int(len(self.occ_counts)),
            "free_voxel_count": int(len(self.free_counts)),
            "occupied_cell_count": int(len(self.occ2d_counts)),
            "free_cell_count": int(len(self.free2d_counts)),
            "frontier_count": int(frontier_count),
            "pose_count": int(len(self.pose_trace)),
            "keyframe_count": int(len(self.keyframes)),
            "semantic_event_count": int(len(self.semantic_events)),
            "semantic_high_conf_event_count": int(len(high_conf_events)),
            "semantic_high_conf_keyframe_count": int(len(high_conf_keyframes)),
            "waypoint_probe_count": int(len(self.waypoint_events)),
            "waypoint_goal_state_counts": dict(waypoint_state_counts),
            "waypoint_mean_frontier_distance_m": (
                float(np.mean(frontier_distances)) if frontier_distances else None
            ),
            "waypoint_direction_counts": dict(waypoint_direction_counts),
            "waypoint_frontier_dominant_direction_counts": dict(waypoint_frontier_direction_counts),
            "waypoint_frontier_alignment_count": int(waypoint_frontier_align_count),
            "waypoint_frontier_alignment_ratio": (
                float(waypoint_frontier_align_count / len(self.waypoint_events))
                if self.waypoint_events else None
            ),
            "waypoint_high_conf_alignment_count": int(waypoint_high_conf_align_count),
            "waypoint_high_conf_alignment_ratio": (
                float(waypoint_high_conf_align_count / len(self.waypoint_events))
                if self.waypoint_events else None
            ),
            "semantic_dead_zone_waypoint_count": int(dead_zone_count),
            "semantic_dead_zone_waypoint_ratio": (
                float(dead_zone_count / len(self.waypoint_events))
                if self.waypoint_events else None
            ),
            "semantic_first_dead_zone_waypoint_step": first_dead_zone_step,
            "semantic_dead_zone_mean_score": (
                float(np.mean(dead_zone_scores)) if dead_zone_scores else None
            ),
            "semantic_dead_zone_max_score": (
                float(np.max(dead_zone_scores)) if dead_zone_scores else None
            ),
            "semantic_dead_zone_with_frontier_count": int(dead_zone_with_frontier_count),
            "semantic_dead_zone_frontier_direction_counts": dict(dead_zone_frontier_direction_counts),
            "final_frontier_total_count_for_direction": final_frontier_summary.get("total_count"),
            "final_frontier_sampled_count_for_direction": final_frontier_summary.get("sampled_count"),
            "final_frontier_sample_fraction_for_direction": final_frontier_summary.get("sample_fraction"),
            "final_frontier_direction_counts": final_frontier_summary.get("direction_counts"),
            "final_frontier_direction_nearest_m": final_frontier_summary.get("nearest_m"),
            "final_frontier_dominant_direction": final_frontier_summary.get("dominant_direction"),
            "final_frontier_dominant_angle_deg": final_frontier_summary.get("dominant_angle_deg"),
            "final_frontier_direction_entropy": final_frontier_summary.get("direction_entropy"),
            "bev_snapshot_count": int(self.saved_bev_count),
            "validation_snapshot_count": int(self.saved_validation_count),
            "validation_final_snapshot_count": int(self.saved_validation_final_count),
        }
        self._write_event(summary)
        self._write_summary(summary)
        if self.config.save_bev:
            self._write_bev_snapshot({"step_id": steps, "final": True})
        return summary

    def get_frontier_cells(self, *, sample_limit: int = 0) -> List[Tuple[int, int]]:
        if not self.config.frontier_enable:
            return []
        if self.frontier_cache is not None and self.frontier_cache_update == self.update_count:
            cells = self.frontier_cache
        else:
            occupied = set(self.occ2d_counts.keys())
            observed = set(self.free2d_counts.keys()) | occupied
            cells = []
            for cell in self.free2d_counts.keys():
                if cell in occupied:
                    continue
                row, col = cell
                for nbr in self._neighbors2d(row, col):
                    if nbr not in observed:
                        cells.append((row, col))
                        break
            self.frontier_cache = cells
            self.frontier_cache_update = self.update_count
        if sample_limit and sample_limit > 0 and len(cells) > sample_limit:
            ids = np.linspace(0, len(cells) - 1, int(sample_limit)).astype(np.int64)
            return [cells[int(i)] for i in ids]
        return list(cells)

    def _pose_from_obs(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        if obs.get("gps") is None or obs.get("compass") is None:
            return None
        gps = np.asarray(obs["gps"], dtype=np.float32).reshape(-1)
        if gps.shape[0] < 2:
            return None
        yaw = float(np.asarray(obs["compass"], dtype=np.float32).reshape(-1)[0])
        pos = np.array([float(gps[0]), -float(gps[1]), 0.0], dtype=np.float32)
        return _yaw_to_tf(pos, yaw)

    def _relative_base_tf(self, base_pose_tf: np.ndarray) -> np.ndarray:
        base_pose_tf = np.asarray(base_pose_tf, dtype=np.float32)
        if self.config.center_on_first_pose:
            if self.inv_init_base_tf is None:
                return np.eye(4, dtype=np.float32)
            return self.inv_init_base_tf @ base_pose_tf
        return base_pose_tf

    def _pose_to_grid(self, rel_base_tf: np.ndarray) -> Tuple[int, int, float]:
        x, y, _ = rel_base_tf[:3, 3]
        row = int(self.gs / 2 - int(float(x) / self.cs))
        col = int(self.gs / 2 - int(float(y) / self.cs))
        yaw = float(math.atan2(float(rel_base_tf[1, 0]), float(rel_base_tf[0, 0])))
        return row, col, yaw

    def _xyz_to_grid(self, xyz: np.ndarray) -> Optional[Tuple[int, int, int]]:
        x, y, z = [float(v) for v in xyz[:3]]
        row = int(self.gs / 2 - int(x / self.cs))
        col = int(self.gs / 2 - int(y / self.cs))
        height = int(z / self.cs)
        if row < 0 or row >= self.gs or col < 0 or col >= self.gs or height < 0 or height >= self.vh:
            return None
        return int(row), int(col), int(height)

    def _is_obstacle_height(self, height: int) -> bool:
        z = float(height) * self.cs
        return float(self.config.obstacle_height_min) < z < float(self.config.obstacle_height_max)

    def _raycast_free(self, origin: np.ndarray, endpoint: np.ndarray, endpoint_grid: Tuple[int, int, int]) -> int:
        delta = np.asarray(endpoint, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
        distance = float(np.linalg.norm(delta))
        if distance <= self.cs:
            return 0
        stride = max(1, int(self.config.raycast_stride_cells)) * self.cs
        steps = max(1, int(distance / stride))
        added = 0
        for idx in range(1, steps):
            alpha = float(idx) / float(steps)
            point = np.asarray(origin, dtype=np.float32) + alpha * delta
            grid = self._xyz_to_grid(point)
            if grid is None or grid == endpoint_grid:
                continue
            before = self.free_counts.get(grid, 0)
            self.free_counts[grid] = before + 1
            if before == 0:
                added += 1
            self.free2d_counts[(grid[0], grid[1])] += 1
        return added

    def _maybe_add_keyframe(
        self,
        context: Dict[str, Any],
        rel_base_tf: np.ndarray,
        row: int,
        col: int,
        yaw: float,
    ) -> None:
        step_id = context.get("step_id")
        try:
            step_int = int(step_id) if step_id is not None else len(self.pose_trace)
        except (TypeError, ValueError):
            step_int = len(self.pose_trace)
        xy = rel_base_tf[:2, 3].astype(np.float32, copy=True)
        should_add = not self.keyframes
        if not should_add and int(self.config.keyframe_every_steps) > 0:
            should_add = step_int % int(self.config.keyframe_every_steps) == 0
        if not should_add and self.last_keyframe_xy is not None:
            should_add = bool(np.linalg.norm(xy - self.last_keyframe_xy) >= float(self.config.keyframe_min_distance))
        if not should_add:
            return
        self.keyframes.append(
            {
                "node_type": "pose",
                "step_id": step_int,
                "row": int(row),
                "col": int(col),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "yaw": float(yaw),
                "semantic_top_match": self.last_semantic_decision.get("top_match"),
                "semantic_top_score": self.last_semantic_decision.get("top_score"),
            }
        )
        self.last_keyframe_xy = xy

    def _pixel_goal_to_grid(
        self,
        pixel_goal: Any,
        depth: np.ndarray,
        pose_tf: np.ndarray,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if pixel_goal is None:
            return None
        try:
            px = float(pixel_goal[0])
            py = float(pixel_goal[1])
        except (TypeError, ValueError, IndexError):
            return None
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        h, w = depth_arr.shape[:2]
        image_w = int(
            context.get("image_width")
            or self.config.waypoint_source_image_width
            or w
        )
        image_h = int(
            context.get("image_height")
            or self.config.waypoint_source_image_height
            or h
        )
        if image_w <= 0 or image_h <= 0:
            return None
        sx = px * w / image_w
        sy = py * h / image_h
        ix = int(np.clip(round(sx), 0, w - 1))
        iy = int(np.clip(round(sy), 0, h - 1))
        radius = max(0, int(self.config.waypoint_depth_patch_radius))
        y0 = max(0, iy - radius)
        y1 = min(h, iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(w, ix + radius + 1)
        patch = depth_arr[y0:y1, x0:x1].astype(np.float32, copy=False)
        valid = patch[np.isfinite(patch) & (patch > self.config.min_depth) & (patch < self.config.max_depth)]
        if valid.size == 0:
            return None
        depth_m = float(np.median(valid))
        inv_intrinsic = np.linalg.inv(self.camera_intrinsic)
        cam = inv_intrinsic @ np.array([sx + 0.5, sy + 0.5, 1.0], dtype=np.float32)
        cam = cam * depth_m
        rel_base_tf = self._relative_base_tf(pose_tf)
        cam_pose_tf = rel_base_tf @ self.cam_to_base_tf
        point = cam_pose_tf @ np.array([cam[0], cam[1], cam[2], 1.0], dtype=np.float32)
        goal = self._xyz_to_grid(point[:3])
        if goal is None:
            return None
        start_row, start_col, start_yaw = self._pose_to_grid(rel_base_tf)
        return {
            "start_grid": [int(start_row), int(start_col)],
            "goal_grid": [int(goal[0]), int(goal[1])],
            "start_yaw": float(start_yaw),
            "depth_m": depth_m,
        }

    def _cell_state(self, row: int, col: int) -> str:
        key = (int(row), int(col))
        if key in self.occ2d_counts:
            return "occupied"
        if key in self.free2d_counts:
            return "free"
        return "unknown"

    def _nearby_visit_count(self, cell: Iterable[int], radius: int) -> int:
        row, col = [int(v) for v in list(cell)[:2]]
        total = 0
        for (r, c), count in self.visited2d_counts.items():
            if abs(int(r) - row) <= radius and abs(int(c) - col) <= radius:
                total += int(count)
        return total

    def _nearest_frontier_distance(self, cell: Iterable[int]) -> Optional[float]:
        row, col = [int(v) for v in list(cell)[:2]]
        frontiers = self.get_frontier_cells(sample_limit=int(self.config.waypoint_frontier_sample_limit))
        if not frontiers:
            return None
        return float(min(math.hypot(row - r, col - c) for r, c in frontiers))

    def _frontier_direction_summary(self, start_grid: Iterable[int], yaw: float) -> Dict[str, Any]:
        buckets = ("front", "left", "right", "back")
        frontiers = self.get_frontier_cells(sample_limit=0)
        total_count = len(frontiers)
        counts = {name: 0 for name in buckets}
        grouped = {name: [] for name in buckets}
        nearest = {name: None for name in buckets}
        if not frontiers:
            return {
                "total_count": 0,
                "sampled_count": 0,
                "sample_fraction": None,
                "direction_counts": counts,
                "nearest_m": nearest,
                "direction_mass_ratio": {name: 0.0 for name in buckets},
                "dominant_direction": None,
                "dominant_angle_deg": None,
                "dominant_count": 0,
                "direction_entropy": None,
            }
        for cell in frontiers:
            direction = self._direction_to_cell(start_grid, cell, yaw)
            bucket = direction.get("bucket")
            if bucket not in counts:
                continue
            counts[bucket] += 1
            grouped[bucket].append(direction)
        sampled = {name: list(items) for name, items in grouped.items()}
        sample_limit = int(self.config.attribution_frontier_sample_limit)
        if sample_limit > 0 and total_count > sample_limit:
            sampled = {}
            for name, items in grouped.items():
                if not items:
                    sampled[name] = []
                    continue
                keep = max(1, int(round(sample_limit * len(items) / float(total_count))))
                keep = min(keep, len(items))
                if keep >= len(items):
                    sampled[name] = list(items)
                else:
                    ids = np.linspace(0, len(items) - 1, keep).astype(np.int64)
                    sampled[name] = [items[int(idx)] for idx in ids]
        sampled_count = sum(len(items) for items in sampled.values())
        for bucket, items in sampled.items():
            if bucket not in counts:
                continue
            for direction in items:
                dist = direction.get("distance_m")
                if dist is not None and (nearest[bucket] is None or dist < nearest[bucket]):
                    nearest[bucket] = float(dist)
        total = max(1, sum(counts.values()))
        ratios = {name: float(counts[name] / total) for name in buckets}
        dominant_direction = max(counts, key=lambda key: counts[key]) if any(counts.values()) else None
        dominant_angle = None
        if dominant_direction:
            dominant_angle = self._mean_direction_angle(
                item.get("angle_deg") for item in sampled.get(dominant_direction, [])
            )
        entropy = 0.0
        for ratio in ratios.values():
            if ratio > 0.0:
                entropy -= ratio * math.log(ratio)
        return {
            "total_count": int(total_count),
            "sampled_count": int(sampled_count),
            "sample_fraction": float(sampled_count / total_count) if total_count else None,
            "direction_counts": counts,
            "nearest_m": nearest,
            "direction_mass_ratio": ratios,
            "dominant_direction": dominant_direction,
            "dominant_angle_deg": dominant_angle,
            "dominant_count": int(counts.get(dominant_direction, 0)) if dominant_direction else 0,
            "direction_entropy": float(entropy) if any(counts.values()) else None,
        }

    def _mean_direction_angle(self, angles_deg: Iterable[Any]) -> Optional[float]:
        angles = []
        for angle in angles_deg:
            if angle is None:
                continue
            try:
                angle_float = float(angle)
            except (TypeError, ValueError):
                continue
            angles.append(math.radians(angle_float))
        if not angles:
            return None
        mean_sin = float(np.mean(np.sin(angles)))
        mean_cos = float(np.mean(np.cos(angles)))
        if abs(mean_sin) < 1e-8 and abs(mean_cos) < 1e-8:
            return None
        return float(math.degrees(math.atan2(mean_sin, mean_cos)))

    def _directions_aligned(self, angle_a: Any, angle_b: Any) -> bool:
        try:
            a = float(angle_a)
            b = float(angle_b)
        except (TypeError, ValueError):
            return False
        threshold = max(1.0, min(180.0, float(self.config.attribution_direction_match_degrees)))
        return self._angle_distance_degrees(a, b) <= threshold

    def _angle_distance_degrees(self, angle_a: float, angle_b: float) -> float:
        return abs(math.degrees(self._wrap_angle(math.radians(float(angle_a) - float(angle_b)))))

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _direction_to_cell(self, start_grid: Iterable[int], target_grid: Iterable[int], yaw: float) -> Dict[str, Any]:
        start_xy = self._grid_to_xy(start_grid)
        target_xy = self._grid_to_xy(target_grid)
        dx = float(target_xy[0] - start_xy[0])
        dy = float(target_xy[1] - start_xy[1])
        distance = float(math.hypot(dx, dy))
        if distance <= 1e-6:
            return {"bucket": "same", "angle_deg": 0.0, "distance_m": 0.0}
        world_angle = math.atan2(dy, dx)
        rel_angle = self._wrap_angle(world_angle - float(yaw))
        rel_deg = math.degrees(rel_angle)
        bucket = self._angle_to_direction_bucket(rel_deg)
        return {
            "bucket": bucket,
            "angle_deg": float(rel_deg),
            "distance_m": distance,
        }

    def _nearest_semantic_keyframe(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        high_conf_only: bool = True,
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        for node in self.keyframes:
            if high_conf_only and not node.get("high_conf_semantic"):
                continue
            if not node.get("semantic_top_match"):
                continue
            candidates.append(node)
        if not candidates:
            return None
        start_xy = self._grid_to_xy(start_grid)
        best = None
        best_dist = None
        for node in candidates:
            xy = np.array([float(node.get("x", 0.0)), float(node.get("y", 0.0))], dtype=np.float32)
            dist = float(np.linalg.norm(xy - start_xy))
            if best_dist is None or dist < best_dist:
                best = node
                best_dist = dist
        if best is None:
            return None
        direction = self._direction_to_xy(start_xy, [float(best.get("x", 0.0)), float(best.get("y", 0.0))], yaw)
        return {
            "step_id": best.get("step_id"),
            "semantic_top_match": best.get("semantic_top_match"),
            "semantic_top_score": best.get("semantic_top_score"),
            "distance_m": best_dist,
            "direction_bucket": direction.get("bucket"),
            "direction_angle_deg": direction.get("angle_deg"),
        }

    def _semantic_dead_zone_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        window = max(1, int(self.config.attribution_recent_semantic_window))
        high_conf_window = max(1, int(self.config.attribution_high_conf_recent_window))
        recent_events = [
            event for event in self.semantic_events[-window:]
            if event.get("status") in (None, "ok") and event.get("top_match")
        ]
        recent_high_conf_events = [
            event for event in self.semantic_events[-high_conf_window:]
            if event.get("high_conf_semantic")
        ]
        recent_terms = [str(event.get("top_match")) for event in recent_events if event.get("top_match")]
        unique_terms = sorted(set(recent_terms))
        last = self.semantic_events[-1] if self.semantic_events else {}
        step = context.get("step_id", last.get("step_id"))
        step_int = self._safe_int(step)
        if step_int is None:
            step_int = 0
        last_stagnation = bool(last.get("stagnation_would_requery"))
        last_stagnation_step = self.last_stagnation_step
        stagnation_age = None
        if last_stagnation_step is not None:
            stagnation_age = int(step_int - last_stagnation_step)
        active_window = max(0, int(self.config.attribution_stagnation_active_window_steps))
        stagnation_active = bool(
            last_stagnation
            or (
                stagnation_age is not None
                and 0 <= stagnation_age <= active_window
            )
        )
        recent_unique_count = len(unique_terms)
        recent_high_conf_count = len(recent_high_conf_events)
        no_recent_high_conf = recent_high_conf_count == 0
        low_diversity = bool(
            recent_events
            and recent_unique_count <= int(self.config.attribution_dead_zone_unique_threshold)
        )
        late_enough = bool(step_int >= int(self.config.attribution_dead_zone_min_step))
        score = 0.0
        if stagnation_active:
            score += 0.45
        if no_recent_high_conf:
            score += 0.25
        if low_diversity:
            score += 0.15
        if late_enough:
            score += 0.15
        score = min(1.0, float(score))
        return {
            "dead_zone": bool(score >= float(self.config.attribution_dead_zone_score_threshold)),
            "dead_zone_score": score,
            "last_stagnation": last_stagnation,
            "stagnation_active": stagnation_active,
            "last_stagnation_step": last_stagnation_step,
            "stagnation_age_steps": stagnation_age,
            "stagnation_active_window_steps": int(active_window),
            "late_enough": late_enough,
            "no_recent_high_conf": no_recent_high_conf,
            "low_recent_semantic_diversity": low_diversity,
            "recent_unique_count": int(recent_unique_count),
            "recent_high_conf_count": int(recent_high_conf_count),
            "recent_terms": recent_terms,
            "last_top_match": last.get("top_match"),
            "last_top_score": last.get("top_score"),
        }

    def _grid_to_xy(self, grid: Iterable[int]) -> np.ndarray:
        row, col = [int(v) for v in list(grid)[:2]]
        return np.array(
            [
                (self.gs / 2.0 - float(row)) * self.cs,
                (self.gs / 2.0 - float(col)) * self.cs,
            ],
            dtype=np.float32,
        )

    def _direction_to_xy(self, start_xy: Iterable[float], target_xy: Iterable[float], yaw: float) -> Dict[str, Any]:
        start = np.asarray(list(start_xy)[:2], dtype=np.float32)
        target = np.asarray(list(target_xy)[:2], dtype=np.float32)
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return {"bucket": "same", "angle_deg": 0.0, "distance_m": 0.0}
        world_angle = math.atan2(float(delta[1]), float(delta[0]))
        rel_deg = math.degrees(self._wrap_angle(world_angle - float(yaw)))
        return {
            "bucket": self._angle_to_direction_bucket(rel_deg),
            "angle_deg": float(rel_deg),
            "distance_m": distance,
        }

    def _angle_to_direction_bucket(self, angle_deg: float) -> str:
        angle = float(angle_deg)
        if -45.0 <= angle <= 45.0:
            return "front"
        if 45.0 < angle < 135.0:
            return "left"
        if -135.0 < angle < -45.0:
            return "right"
        return "back"

    def _wrap_angle(self, angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _compact_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        allowed = (
            "distance_to_goal",
            "success",
            "spl",
            "oracle_success",
            "oracle_navigation_error",
            "ndtw",
            "sdtw",
        )
        compact = {}
        for key in allowed:
            value = metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
                compact[key] = self._jsonable(value)
        return compact

    def _neighbors2d(self, row: int, col: int):
        offsets4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        offsets8 = offsets4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        offsets = offsets8 if int(self.config.frontier_connectivity) == 8 else offsets4
        for dr, dc in offsets:
            rr = int(row) + dr
            cc = int(col) + dc
            if 0 <= rr < self.gs and 0 <= cc < self.gs:
                yield rr, cc

    def _maybe_write_bev_snapshot(self, context: Dict[str, Any]) -> None:
        if not self.config.save_bev:
            return
        if self.config.max_bev_snapshots >= 0 and self.saved_bev_count >= self.config.max_bev_snapshots:
            return
        every = max(1, int(self.config.bev_every_updates))
        if self.update_count <= 0 or self.update_count % every != 0:
            return
        self._write_bev_snapshot(context)

    def _write_bev_snapshot(self, context: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return
        out_dir = os.path.join(self.debug_dir, "occ_memory")
        os.makedirs(out_dir, exist_ok=True)
        center = self.last_pose_grid or (self.gs // 2, self.gs // 2)
        radius = max(10, int(self.config.bev_crop_radius_cells))
        scale = max(1, int(self.config.bev_cell_scale))
        r0 = max(0, int(center[0]) - radius)
        r1 = min(self.gs, int(center[0]) + radius + 1)
        c0 = max(0, int(center[1]) - radius)
        c1 = min(self.gs, int(center[1]) + radius + 1)
        h = max(1, r1 - r0)
        w = max(1, c1 - c0)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:, :] = np.array([28, 28, 30], dtype=np.uint8)
        for row, col in self.free2d_counts.keys():
            if r0 <= row < r1 and c0 <= col < c1:
                image[row - r0, col - c0] = np.array([42, 95, 66], dtype=np.uint8)
        for row, col in self.occ2d_counts.keys():
            if r0 <= row < r1 and c0 <= col < c1:
                image[row - r0, col - c0] = np.array([205, 72, 62], dtype=np.uint8)
        for row, col in self.get_frontier_cells(sample_limit=4000):
            if r0 <= row < r1 and c0 <= col < c1:
                image[row - r0, col - c0] = np.array([235, 195, 71], dtype=np.uint8)
        img = Image.fromarray(image, mode="RGB").resize((w * scale, h * scale), resample=Image.NEAREST)
        draw = ImageDraw.Draw(img)

        def to_xy(row: int, col: int) -> Tuple[int, int]:
            return int((col - c0) * scale + scale / 2), int((row - r0) * scale + scale / 2)

        if len(self.pose_trace) >= 2:
            pts = [to_xy(item["row"], item["col"]) for item in self.pose_trace if r0 <= item["row"] < r1 and c0 <= item["col"] < c1]
            if len(pts) >= 2:
                draw.line(pts, fill=(95, 160, 245), width=max(1, scale))
        for node in self.keyframes:
            row = int(node["row"])
            col = int(node["col"])
            if r0 <= row < r1 and c0 <= col < c1:
                x, y = to_xy(row, col)
                rad = max(2, scale)
                draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(110, 185, 255))
        if self.last_pose_grid is not None:
            x, y = to_xy(self.last_pose_grid[0], self.last_pose_grid[1])
            rad = max(3, scale + 1)
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(255, 255, 255))
        label = f"occ={len(self.occ2d_counts)} free={len(self.free2d_counts)} frontier={len(self.get_frontier_cells(sample_limit=0))}"
        draw.rectangle((0, 0, max(220, len(label) * 7), 18), fill=(18, 18, 18))
        draw.text((4, 3), label, fill=(240, 240, 240))
        suffix = "final" if context.get("final") else f"{self.saved_bev_count:03d}"
        step = context.get("step_id")
        step_text = "end" if step is None else str(step)
        path = os.path.join(out_dir, f"bev_ep{self.episode_meta.get('episode_id')}_{step_text}_{suffix}.png")
        img.save(path)
        self.saved_bev_count += 1

    def _maybe_write_validation_snapshot(
        self,
        context: Dict[str, Any],
        *,
        rgb: Optional[np.ndarray],
        depth: np.ndarray,
        cam_pose_tf: np.ndarray,
        world_points: np.ndarray,
        point_ids: np.ndarray,
    ) -> None:
        if not self.config.validation_enable or not self.debug_dir:
            return
        if self.config.validation_max_snapshots >= 0 and self.saved_validation_count >= self.config.validation_max_snapshots:
            return
        every = max(1, int(self.config.validation_every_updates))
        if self.update_count <= 0 or self.update_count % every != 0:
            return
        self._write_validation_snapshot(
            context,
            rgb=rgb,
            depth=depth,
            cam_pose_tf=cam_pose_tf,
            world_points=world_points,
            point_ids=point_ids,
        )

    def _write_validation_snapshot(
        self,
        context: Dict[str, Any],
        *,
        rgb: Optional[np.ndarray],
        depth: np.ndarray,
        cam_pose_tf: np.ndarray,
        world_points: np.ndarray,
        point_ids: np.ndarray,
    ) -> None:
        out_dir = self._validation_dir()
        if out_dir is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        suffix = self._validation_suffix(context, self.saved_validation_count)
        paths: Dict[str, str] = {}
        depth_shape = self._depth_shape(depth)

        if self.config.validation_save_rgb_depth:
            image_paths = self._write_validation_images(out_dir, suffix, rgb, depth)
            paths.update(image_paths)

        current_points = None
        current_colors = None
        if self.config.validation_save_current_rgb_ply:
            current_points, current_colors = self._current_rgb_point_cloud(
                rgb,
                depth,
                cam_pose_tf,
                fallback_world_points=world_points,
                fallback_point_ids=point_ids,
                fallback_depth_shape=depth_shape,
            )
            current_ply_path = os.path.join(out_dir, f"{suffix}_current_rgb_cloud.ply")
            self._write_point_cloud_ply(current_ply_path, current_points, current_colors)
            paths["current_rgb_cloud_ply"] = current_ply_path

        memory_stats: Dict[str, Any] = {}
        if self.config.validation_save_memory_ply:
            memory_ply_path = os.path.join(out_dir, f"{suffix}_memory_cloud.ply")
            memory_stats = self._write_memory_ply_snapshot(memory_ply_path)
            paths["memory_cloud_ply"] = memory_ply_path

        event = {
            "event_type": "occ_memory_validation_snapshot",
            **self.episode_meta,
            **context,
            "snapshot_index": int(self.saved_validation_count),
            "update_count": int(self.update_count),
            "paths": paths,
            "current_rgb_point_count": 0 if current_points is None else int(current_points.shape[0]),
            **memory_stats,
        }
        self.saved_validation_count += 1
        self._write_event(event)

    def _write_final_validation_snapshot(self, context: Dict[str, Any]) -> None:
        if not self.debug_dir or not self.config.validation_save_memory_ply:
            return
        out_dir = self._validation_dir()
        if out_dir is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        suffix = self._validation_suffix(context, self.saved_validation_final_count, final=True)
        memory_ply_path = os.path.join(out_dir, f"{suffix}_memory_cloud.ply")
        memory_stats = self._write_memory_ply_snapshot(memory_ply_path)
        event = {
            "event_type": "occ_memory_validation_final_snapshot",
            **self.episode_meta,
            **context,
            "snapshot_index": int(self.saved_validation_final_count),
            "update_count": int(self.update_count),
            "paths": {"memory_cloud_ply": memory_ply_path},
            **memory_stats,
        }
        self.saved_validation_final_count += 1
        self._write_event(event)

    def _validation_dir(self) -> Optional[str]:
        if not self.debug_dir:
            return None
        return os.path.join(self.debug_dir, "occ_memory", "validation")

    def _validation_suffix(self, context: Dict[str, Any], index: int, *, final: bool = False) -> str:
        episode = self.episode_meta.get("episode_id")
        if episode is None:
            episode = self.episode_meta.get("episode_index", "unknown")
        step = context.get("step_id")
        step_text = "end" if step is None else str(step)
        tag = "final" if final else f"{int(index):03d}"
        return f"ep{episode}_step{step_text}_{tag}"

    def _write_validation_images(
        self,
        out_dir: str,
        suffix: str,
        rgb: Optional[np.ndarray],
        depth: np.ndarray,
    ) -> Dict[str, str]:
        try:
            from PIL import Image
        except Exception:
            return {}
        paths: Dict[str, str] = {}
        if rgb is not None:
            rgb_arr = np.asarray(rgb)
            if rgb_arr.ndim == 3 and rgb_arr.shape[-1] >= 3:
                rgb_arr = rgb_arr[..., :3]
                if rgb_arr.dtype != np.uint8:
                    rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)
                rgb_path = os.path.join(out_dir, f"{suffix}_rgb.png")
                Image.fromarray(rgb_arr, mode="RGB").save(rgb_path)
                paths["rgb"] = rgb_path
        depth_arr = self._depth_2d(depth)
        if depth_arr.size:
            valid = np.isfinite(depth_arr) & (depth_arr > 0.0)
            if np.any(valid):
                clipped = np.clip(depth_arr, float(self.config.min_depth), float(self.config.max_depth))
                denom = max(1e-6, float(self.config.max_depth) - float(self.config.min_depth))
                vis = ((clipped - float(self.config.min_depth)) / denom * 255.0).astype(np.uint8)
                vis[~valid] = 0
                depth_vis_path = os.path.join(out_dir, f"{suffix}_depth_vis.png")
                Image.fromarray(vis, mode="L").save(depth_vis_path)
                paths["depth_vis"] = depth_vis_path
                depth_mm = np.zeros_like(depth_arr, dtype=np.uint16)
                depth_mm[valid] = np.clip(depth_arr[valid] * 1000.0, 0, 65535).astype(np.uint16)
                depth_mm_path = os.path.join(out_dir, f"{suffix}_depth_mm.png")
                Image.fromarray(depth_mm, mode="I;16").save(depth_mm_path)
                paths["depth_mm"] = depth_mm_path
        return paths

    def _current_rgb_point_cloud(
        self,
        rgb: Optional[np.ndarray],
        depth: np.ndarray,
        cam_pose_tf: np.ndarray,
        *,
        fallback_world_points: np.ndarray,
        fallback_point_ids: np.ndarray,
        fallback_depth_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.camera_intrinsic is not None:
            cam_points, point_ids = _depth_to_points(
                depth,
                self.camera_intrinsic,
                min_depth=float(self.config.min_depth),
                max_depth=float(self.config.max_depth),
                sample_rate=int(self.config.validation_current_depth_sample_rate),
            )
            if cam_points.shape[0] > 0:
                if (
                    int(self.config.validation_max_current_points) > 0
                    and cam_points.shape[0] > int(self.config.validation_max_current_points)
                ):
                    ids = np.linspace(
                        0,
                        cam_points.shape[0] - 1,
                        int(self.config.validation_max_current_points),
                    ).astype(np.int64)
                    cam_points = cam_points[ids]
                    point_ids = point_ids[ids]
                cam_points_h = np.concatenate(
                    [cam_points, np.ones((cam_points.shape[0], 1), dtype=np.float32)],
                    axis=1,
                )
                points = (cam_pose_tf @ cam_points_h.T).T[:, :3]
                colors = self._rgb_colors_for_point_ids(rgb, point_ids, self._depth_shape(depth))
                return points.astype(np.float32, copy=False), colors

        points = np.asarray(fallback_world_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            points = np.zeros((0, 3), dtype=np.float32)
        colors = self._rgb_colors_for_point_ids(rgb, np.asarray(fallback_point_ids), fallback_depth_shape)
        if colors is not None and colors.shape[0] != points.shape[0]:
            colors = None
        return points, colors

    def _write_memory_ply_snapshot(self, path: str) -> Dict[str, Any]:
        points, colors, stats = self._memory_point_cloud()
        self._write_point_cloud_ply(path, points, colors)
        stats["memory_cloud_point_count"] = int(points.shape[0])
        stats["memory_cloud_ply_exists"] = bool(os.path.exists(path))
        return stats

    def _memory_point_cloud(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        occ_keys = self._sample_items(list(self.occ_counts.keys()), int(self.config.validation_max_occupied_points))
        free_keys = self._sample_items(list(self.free_counts.keys()), int(self.config.validation_max_free_points))
        frontier_cells = self.get_frontier_cells(sample_limit=int(self.config.validation_max_frontier_points))

        point_parts = []
        color_parts = []
        occ_points = self._grid_keys_to_points(occ_keys)
        if occ_points.shape[0] > 0:
            point_parts.append(occ_points)
            color_parts.append(np.tile(np.array([[220, 72, 62]], dtype=np.uint8), (occ_points.shape[0], 1)))
        free_points = self._grid_keys_to_points(free_keys)
        if free_points.shape[0] > 0:
            point_parts.append(free_points)
            color_parts.append(np.tile(np.array([[45, 155, 120]], dtype=np.uint8), (free_points.shape[0], 1)))
        frontier_points = self._cell_keys_to_points(frontier_cells, z=0.05)
        if frontier_points.shape[0] > 0:
            point_parts.append(frontier_points)
            color_parts.append(np.tile(np.array([[240, 205, 65]], dtype=np.uint8), (frontier_points.shape[0], 1)))
        pose_points = self._pose_trace_points()
        if pose_points.shape[0] > 0:
            point_parts.append(pose_points)
            color_parts.append(np.tile(np.array([[90, 160, 245]], dtype=np.uint8), (pose_points.shape[0], 1)))
        keyframe_points = self._keyframe_points()
        if keyframe_points.shape[0] > 0:
            point_parts.append(keyframe_points)
            color_parts.append(np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (keyframe_points.shape[0], 1)))

        if point_parts:
            points = np.concatenate(point_parts, axis=0).astype(np.float32, copy=False)
            colors = np.concatenate(color_parts, axis=0).astype(np.uint8, copy=False)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

        max_points = int(self.config.validation_max_memory_points)
        if max_points > 0 and points.shape[0] > max_points:
            ids = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
            points = points[ids]
            colors = colors[ids]

        stats = {
            "memory_occ_point_count": int(occ_points.shape[0]),
            "memory_free_point_count": int(free_points.shape[0]),
            "memory_frontier_point_count": int(frontier_points.shape[0]),
            "memory_pose_point_count": int(pose_points.shape[0]),
            "memory_keyframe_point_count": int(keyframe_points.shape[0]),
        }
        return points, colors, stats

    def _rgb_colors_for_point_ids(
        self,
        rgb: Optional[np.ndarray],
        point_ids: np.ndarray,
        depth_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if rgb is None or point_ids is None:
            return None
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim != 3 or rgb_arr.shape[-1] < 3:
            return None
        depth_h, depth_w = depth_shape
        if depth_h <= 0 or depth_w <= 0:
            return None
        ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        rows = ids // int(depth_w)
        cols = ids % int(depth_w)
        rgb_h, rgb_w = rgb_arr.shape[:2]
        rr = np.clip((rows.astype(np.float32) * rgb_h / depth_h).astype(np.int64), 0, rgb_h - 1)
        cc = np.clip((cols.astype(np.float32) * rgb_w / depth_w).astype(np.int64), 0, rgb_w - 1)
        colors = rgb_arr[rr, cc, :3]
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        return colors

    def _grid_keys_to_points(self, keys: List[Tuple[int, int, int]]) -> np.ndarray:
        if not keys:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.asarray(keys, dtype=np.float32)
        x = (self.gs / 2.0 - arr[:, 0]) * self.cs
        y = (self.gs / 2.0 - arr[:, 1]) * self.cs
        z = (arr[:, 2] + 0.5) * self.cs
        return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    def _cell_keys_to_points(self, cells: List[Tuple[int, int]], *, z: float = 0.0) -> np.ndarray:
        if not cells:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.asarray(cells, dtype=np.float32)
        x = (self.gs / 2.0 - arr[:, 0]) * self.cs
        y = (self.gs / 2.0 - arr[:, 1]) * self.cs
        z_arr = np.full_like(x, float(z), dtype=np.float32)
        return np.stack([x, y, z_arr], axis=1).astype(np.float32, copy=False)

    def _pose_trace_points(self) -> np.ndarray:
        if not self.pose_trace:
            return np.zeros((0, 3), dtype=np.float32)
        points = [[item["x"], item["y"], 0.08] for item in self.pose_trace]
        return np.asarray(points, dtype=np.float32)

    def _keyframe_points(self) -> np.ndarray:
        if not self.keyframes:
            return np.zeros((0, 3), dtype=np.float32)
        points = [[item["x"], item["y"], 0.14] for item in self.keyframes]
        return np.asarray(points, dtype=np.float32)

    def _sample_items(self, items: List[Any], limit: int) -> List[Any]:
        if limit < 0 or len(items) <= limit:
            return items
        if limit == 0:
            return []
        ids = np.linspace(0, len(items) - 1, int(limit)).astype(np.int64)
        return [items[int(i)] for i in ids]

    def _write_point_cloud_ply(
        self,
        path: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            points = np.zeros((0, 3), dtype=np.float32)
        if colors is None:
            colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
        else:
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[1] < 3 or colors.shape[0] != points.shape[0]:
                colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
            else:
                colors = np.clip(colors[:, :3], 0, 255).astype(np.uint8)
        with open(path, "w", encoding="ascii") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {int(points.shape[0])}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors):
                f.write(
                    f"{float(point[0]):.5f} {float(point[1]):.5f} {float(point[2]):.5f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

    def _depth_2d(self, depth: np.ndarray) -> np.ndarray:
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim != 2:
            return np.zeros((0, 0), dtype=np.float32)
        return depth_arr.astype(np.float32, copy=False)

    def _depth_shape(self, depth: np.ndarray) -> Tuple[int, int]:
        depth_arr = self._depth_2d(depth)
        if depth_arr.ndim != 2:
            return 0, 0
        return int(depth_arr.shape[0]), int(depth_arr.shape[1])

    def _write_event(self, event: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        out_dir = os.path.join(self.debug_dir, "occ_memory")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "memory_events.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        out_dir = os.path.join(self.debug_dir, "occ_memory")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "memory_episode_summary.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(summary), ensure_ascii=False) + "\n")

    def _jsonable(self, value: Any):
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

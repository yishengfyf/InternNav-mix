from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from internnav.utils.progress_ranker_shadow import ProgressRankerShadowScorer
from internnav.utils.stage21_multitask_shadow import Stage21MultiTaskShadowScorer

_GOAL_PROGRESS_LANDMARK_ALIASES = {
    "appliance": "appliances",
    "appliances": "appliances",
    "armchair": "chair",
    "arch": "door",
    "arched doorway": "door",
    "arched entry": "entryway",
    "archway": "door",
    "balcony": "balcony",
    "bath": "bathtub",
    "bath room": "bathroom",
    "bathroom": "bathroom",
    "bathtub": "bathtub",
    "bed": "bed",
    "bed room": "bedroom",
    "bedroom": "bedroom",
    "blind": "blinds",
    "blinds": "blinds",
    "book shelf": "shelving",
    "bookshelf": "shelving",
    "cabinet": "cabinet",
    "chair": "chair",
    "chairs": "chair",
    "closet": "closet",
    "column": "column",
    "corridor": "corridor",
    "couch": "sofa",
    "couches": "sofa",
    "counter": "counter",
    "curtain": "curtain",
    "curtains": "curtain",
    "cushion": "cushion",
    "dinning room": "dining room",
    "dining area": "dining area",
    "dining room": "dining room",
    "door": "door",
    "doors": "door",
    "doorway": "door",
    "drawer": "chest_of_drawers",
    "drawers": "chest_of_drawers",
    "dresser": "chest_of_drawers",
    "entrance": "entryway",
    "entryway": "entryway",
    "fireplace": "fireplace",
    "foyer": "entryway",
    "hall": "hall",
    "hallway": "hallway",
    "island counter": "counter",
    "kitchen": "kitchen",
    "lamp": "lighting",
    "light": "lighting",
    "lighting": "lighting",
    "living area": "living area",
    "living room": "living room",
    "mirror": "mirror",
    "office": "office",
    "painting": "picture",
    "patio": "patio",
    "photo": "picture",
    "photograph": "picture",
    "picture": "picture",
    "plant": "plant",
    "plants": "plant",
    "railing": "railing",
    "room": "room",
    "seat": "seating",
    "seating": "seating",
    "shelf": "shelving",
    "shelves": "shelving",
    "shelving": "shelving",
    "shower": "shower",
    "sink": "sink",
    "sofa": "sofa",
    "stair": "stairs",
    "staircase": "stairs",
    "stairs": "stairs",
    "table": "table",
    "tables": "table",
    "television": "tv_monitor",
    "toilet": "toilet",
    "towel": "towel",
    "tv": "tv_monitor",
    "window": "window",
    "windows": "window",
}

_GOAL_PROGRESS_LANDMARK_TERMS = (
    "appliances",
    "bathroom",
    "bathtub",
    "bed",
    "bedroom",
    "blinds",
    "cabinet",
    "chair",
    "chest_of_drawers",
    "closet",
    "column",
    "corridor",
    "counter",
    "curtain",
    "cushion",
    "dining area",
    "dining room",
    "door",
    "entryway",
    "fireplace",
    "hall",
    "hallway",
    "kitchen",
    "lighting",
    "living area",
    "living room",
    "mirror",
    "office",
    "patio",
    "balcony",
    "picture",
    "plant",
    "railing",
    "room",
    "seating",
    "shelving",
    "shower",
    "sink",
    "sofa",
    "stairs",
    "table",
    "toilet",
    "towel",
    "tv_monitor",
    "window",
)

_GOAL_PROGRESS_SPECIFIC_ROOMS = {
    "bathroom",
    "bedroom",
    "closet",
    "corridor",
    "dining area",
    "dining room",
    "entryway",
    "hall",
    "hallway",
    "kitchen",
    "living area",
    "living room",
    "office",
    "patio",
    "balcony",
}

_TARGET_FRONTIER_TRANSITION_TERMS = {
    "door",
    "entryway",
    "hall",
    "hallway",
    "corridor",
    "stairs",
    "balcony",
    "patio",
}

_SEMANTIC_RESILIENCE_OBSTACLE_TERMS = {
    "barrier",
    "cabinet",
    "closet",
    "column",
    "counter",
    "curtain",
    "door",  # closed / blocked doors are useful local obstacle evidence.
    "drawer",
    "fireplace",
    "railing",
    "shelf",
    "shelving",
    "shower",
    "table",
    "wall",
    "window",
}

_SEMANTIC_RESILIENCE_PASSAGE_TERMS = {
    "archway",
    "corridor",
    "door",
    "entryway",
    "hall",
    "hallway",
    "stairs",
}


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


def _cam_to_base_for_pitch(camera_height: float, pitch_down_deg: float) -> np.ndarray:
    """Return the optical-camera transform for a known downward pitch."""
    pitch = math.radians(float(pitch_down_deg))
    c, s = math.cos(pitch), math.sin(pitch)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = np.array(
        [
            [0.0, -s, c],
            [-1.0, 0.0, 0.0],
            [0.0, -c, -s],
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
    camera_pitch_aware_update: bool = False
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
    stage15_repair_shadow_enable: bool = False
    stage15_repair_active: bool = False
    stage15_repair_backtrack_max_steps: int = 20
    stage15_repair_gate_mode: str = "consecutive"
    stage15_repair_gate_min_count: int = 3
    stage15_repair_active_max_per_episode: int = 5
    attribution_enable: bool = False
    attribution_frontier_sample_limit: int = 5000
    attribution_recent_semantic_window: int = 5
    attribution_high_conf_recent_window: int = 5
    attribution_stagnation_active_window_steps: int = 20
    attribution_dead_zone_min_step: int = 30
    attribution_dead_zone_unique_threshold: int = 2
    attribution_dead_zone_score_threshold: float = 0.65
    attribution_direction_match_degrees: float = 45.0
    candidate_probe_enable: bool = False
    candidate_probe_max_candidates: int = 4
    candidate_probe_max_events_per_episode: int = 12
    candidate_probe_frontier_sample_limit: int = 5000
    candidate_probe_free_sample_limit: int = 5000
    candidate_probe_min_distance_m: float = 0.75
    candidate_probe_max_distance_m: float = 4.0
    candidate_probe_min_separation_m: float = 0.50
    candidate_probe_exclude_back_frontier: bool = True
    candidate_probe_semantic_enable: bool = False
    candidate_probe_semantic_high_conf_only: bool = False
    candidate_probe_semantic_min_score: float = 0.20
    candidate_probe_semantic_max_candidates: int = 3
    candidate_probe_semantic_bind_radius_m: float = 2.50
    candidate_probe_semantic_direction_match_degrees: float = 75.0
    candidate_probe_semantic_frontier_min_relevance: float = 0.15
    candidate_probe_semantic_score_weight: float = 0.90
    candidate_probe_semantic_novelty_weight: float = 0.45
    candidate_probe_topology_novelty_weight: float = 0.35
    candidate_probe_goal_progress_enable: bool = False
    candidate_probe_goal_progress_next_weight: float = 1.20
    candidate_probe_goal_progress_completed_penalty: float = 0.80
    candidate_probe_goal_progress_repeated_penalty: float = 0.55
    candidate_probe_goal_progress_seen_score_threshold: float = 0.25
    candidate_probe_goal_progress_high_conf_bonus: float = 0.25
    candidate_probe_goal_progress_unknown_target_bonus: float = 0.20
    candidate_probe_target_frontier_enable: bool = False
    candidate_probe_target_frontier_score_weight: float = 1.10
    candidate_probe_target_frontier_cluster_radius_cells: int = 8
    candidate_probe_target_frontier_cluster_norm: int = 18
    candidate_probe_target_frontier_doorway_threshold: float = 0.35
    candidate_probe_target_frontier_candidate_threshold: float = 0.35
    candidate_probe_target_frontier_intent_max_deviation_deg: float = 75.0
    candidate_probe_target_frontier_intent_penalty_weight: float = 0.45
    semantic_resilience_shadow_enable: bool = False
    semantic_resilience_local_radius_cells: int = 18
    semantic_resilience_min_observed_cells: int = 24
    semantic_resilience_occupied_ratio_threshold: float = 0.28
    semantic_resilience_blocked_bucket_threshold: float = 0.45
    semantic_resilience_frontier_escape_threshold: int = 4
    semantic_resilience_min_backtrack_distance_m: float = 0.75
    semantic_resilience_max_backtrack_distance_m: float = 4.0
    semantic_resilience_backtrack_min_step_gap: int = 6
    semantic_resilience_backtrack_score_weight: float = 1.35
    semantic_resilience_candidate_source_score: float = 2.40
    semantic_resilience_anchor_feature_radius_cells: int = 8
    semantic_resilience_anchor_semantic_radius_m: float = 2.50
    semantic_resilience_cycle_window_steps: int = 48
    semantic_resilience_cycle_radius_cells: int = 8
    semantic_resilience_branch_min_run_cells: int = 2
    semantic_anchor_enable: bool = False
    semantic_anchor_min_score: float = 0.20
    semantic_anchor_max_terms_per_event: int = 3
    semantic_anchor_include_threshold_hits: bool = True
    semantic_anchor_include_pixel_goal: bool = True
    semantic_anchor_include_view_center: bool = True
    semantic_anchor_include_view_left: bool = False
    semantic_anchor_include_view_right: bool = False
    semantic_anchor_include_view_upper: bool = False
    semantic_anchor_include_view_lower: bool = False
    semantic_anchor_view_center_x: float = 0.50
    semantic_anchor_view_center_y: float = 0.56
    semantic_anchor_view_left_x: float = 0.32
    semantic_anchor_view_right_x: float = 0.68
    semantic_anchor_view_upper_y: float = 0.36
    semantic_anchor_view_lower_y: float = 0.74
    semantic_anchor_merge_radius_cells: int = 6
    semantic_anchor_local_radius_cells: int = 6
    semantic_anchor_max_anchors_per_episode: int = 256
    progress_ranker_shadow_enable: bool = False
    progress_ranker_shadow_checkpoint: str = ""
    progress_ranker_shadow_device: str = "cpu"
    progress_ranker_shadow_resilience_weight: float = 0.20
    stage21_multitask_shadow_enable: bool = False
    stage21_multitask_shadow_checkpoint: str = ""
    stage21_multitask_shadow_device: str = "cpu"
    candidate_probe_save_bev: bool = True
    candidate_probe_max_bev_snapshots: int = 12
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
    validation_accumulate_rgb_surface: bool = True
    validation_max_accumulated_surface_points: int = 200000
    validation_projection_size: int = 768
    validation_pose_from_context: bool = False
    validation_camera_pose_from_context: bool = False
    # Independent audit-only semantic scene comparison.  These fields never
    # affect online OCC state or navigation decisions.
    validation_semantic_scene_audit_enable: bool = False
    validation_semantic_scene_audit_max_objects: int = 4000
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
        self.progress_ranker_shadow_scorer: Optional[ProgressRankerShadowScorer] = None
        if bool(self.config.progress_ranker_shadow_enable):
            self.progress_ranker_shadow_scorer = ProgressRankerShadowScorer(
                checkpoint_path=str(self.config.progress_ranker_shadow_checkpoint),
                device=str(self.config.progress_ranker_shadow_device or "cpu"),
                resilience_weight=float(self.config.progress_ranker_shadow_resilience_weight),
            )
        self.stage21_multitask_shadow_scorer: Optional[Stage21MultiTaskShadowScorer] = None
        self.stage21_multitask_shadow_init_error: Optional[str] = None
        if bool(self.config.stage21_multitask_shadow_enable):
            try:
                self.stage21_multitask_shadow_scorer = Stage21MultiTaskShadowScorer(
                    checkpoint_path=str(self.config.stage21_multitask_shadow_checkpoint),
                    device=str(self.config.stage21_multitask_shadow_device or "cpu"),
                )
            except Exception as exc:
                # A shadow model must never make the frozen navigator fail.
                self.stage21_multitask_shadow_init_error = str(exc)
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
        self.semantic_anchors: List[Dict[str, Any]] = []
        self.semantic_anchor_added_count = 0
        self.semantic_anchor_merged_count = 0
        self.semantic_anchor_max_anchors_count = 0
        self.semantic_anchor_invalid_count = 0
        self.semantic_anchor_source_operation_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.semantic_anchor_source_offset_x_px_values: List[float] = []
        self.semantic_anchor_source_offset_y_px_values: List[float] = []
        self.semantic_anchor_source_center_distance_px_values: List[float] = []
        self.semantic_anchor_source_ray_yaw_deg_values: List[float] = []
        self.semantic_anchor_source_ray_pitch_deg_values: List[float] = []
        self.semantic_anchor_source_ray_norm_values: List[float] = []
        self.semantic_anchor_global_bearing_deg_values: List[float] = []
        self.semantic_anchor_relative_bearing_deg_values: List[float] = []
        self.semantic_anchor_pose_origin_distance_m_values: List[float] = []
        self.semantic_anchor_pose_step_distance_m_values: List[float] = []
        self.semantic_anchor_pose_step_dyaw_deg_values: List[float] = []
        self.waypoint_events: List[Dict[str, Any]] = []
        self.candidate_probe_events: List[Dict[str, Any]] = []
        self.candidate_selection_events: List[Dict[str, Any]] = []
        self.init_base_tf: Optional[np.ndarray] = None
        self.inv_init_base_tf: Optional[np.ndarray] = None
        self.update_count = 0
        self.observation_count = 0
        self.free_update_count = 0
        self.occupied_update_count = 0
        self.frontier_cache: Optional[List[Tuple[int, int]]] = None
        self.frontier_cache_update = -1
        self.frontier_set_cache: Optional[set] = None
        self.frontier_set_cache_update = -1
        self.last_pose_grid: Optional[Tuple[int, int]] = None
        self.last_keyframe_xy: Optional[np.ndarray] = None
        self.saved_bev_count = 0
        self.saved_candidate_bev_count = 0
        self.saved_validation_count = 0
        self.saved_validation_final_count = 0
        self.validation_surface_points: List[np.ndarray] = []
        self.validation_surface_colors: List[np.ndarray] = []
        self.validation_surface_point_count = 0
        self.validation_pose_height_values: List[float] = []
        self.validation_gt_height_values: List[float] = []
        self.validation_height_abs_errors: List[float] = []
        self.validation_endpoint_total_count = 0
        self.validation_endpoint_mapped_count = 0
        self.validation_endpoint_below_volume_count = 0
        self.validation_endpoint_above_volume_count = 0
        self.validation_endpoint_outside_xy_count = 0
        self.validation_endpoint_negative_z_count = 0
        self.validation_endpoint_negative_z_mapped_count = 0
        self.validation_pose_xy_errors: List[float] = []
        self.validation_pose_z_errors: List[float] = []
        self.validation_pose_yaw_errors_deg: List[float] = []
        self.validation_endpoint_gt_errors: List[float] = []
        self.validation_endpoint_gt_abs_x_errors: List[float] = []
        self.validation_endpoint_gt_abs_y_errors: List[float] = []
        self.validation_endpoint_gt_abs_z_errors: List[float] = []
        self.validation_endpoint_gt_error_groups: Dict[str, List[float]] = defaultdict(list)
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
        direct_base_tf = self._validation_context_matrix(
            context, "stage23_gt_base_pose_map"
        )
        direct_cam_tf = self._validation_context_matrix(
            context, "stage23_gt_camera_pose_map"
        )
        if bool(self.config.validation_camera_pose_from_context):
            if direct_base_tf is None or direct_cam_tf is None:
                event["reason"] = "missing_gt_camera_pose"
                self._write_event(event)
                return event
            pose_tf = direct_base_tf
        else:
            pose_tf = self._pose_from_obs(obs, context=context)
            if pose_tf is None:
                event["reason"] = "missing_pose"
                self._write_event(event)
                return event
        self.observation_count += 1
        if self.observation_count % max(1, int(self.config.update_every_steps)) != 0:
            event["reason"] = "update_stride"
            self._write_event(event)
            return event

        if self.init_base_tf is None and not bool(
            self.config.validation_camera_pose_from_context
        ):
            self.init_base_tf = pose_tf.copy()
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        rel_base_tf = (
            pose_tf.copy()
            if bool(self.config.validation_camera_pose_from_context)
            else self._relative_base_tf(pose_tf)
        )
        requested_camera_pitch_deg = float(context.get("camera_pitch_deg", 0.0) or 0.0)
        applied_camera_pitch_deg = (
            requested_camera_pitch_deg
            if bool(self.config.camera_pitch_aware_update)
            else 0.0
        )
        if bool(self.config.validation_camera_pose_from_context):
            cam_pose_tf = direct_cam_tf
        else:
            cam_to_base_tf = _cam_to_base_for_pitch(
                float(self.config.camera_height), applied_camera_pitch_deg
            )
            cam_pose_tf = rel_base_tf @ cam_to_base_tf
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
                "camera_pitch_deg": float(requested_camera_pitch_deg),
                "mapping_camera_pitch_deg": float(applied_camera_pitch_deg),
                "pose_height_source": (
                    "context_oracle_height"
                    if bool(self.config.validation_pose_from_context)
                    else "gps_compass_2d"
                ),
                "gt_relative_height_m": context.get("stage23a_gt_relative_height_m"),
                "sim_position": context.get("stage23a_sim_position"),
            }
        )
        pose_height = float(rel_base_tf[2, 3])
        gt_height = context.get("stage23a_gt_relative_height_m")
        if gt_height is not None:
            gt_height = float(gt_height)
            self.validation_pose_height_values.append(pose_height)
            self.validation_gt_height_values.append(gt_height)
            self.validation_height_abs_errors.append(abs(pose_height - gt_height))
        pose_gt_event = self._record_validation_pose_gt_error(
            rel_base_tf, direct_base_tf
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
        endpoint_gt_event = self._record_validation_endpoint_gt_error(
            world_points,
            cam_points,
            cam_points_h,
            direct_cam_tf,
            requested_camera_pitch_deg,
        )
        if self.config.validation_enable and self.config.validation_accumulate_rgb_surface:
            surface_colors = self._rgb_colors_for_point_ids(
                rgb, point_ids, self._depth_shape(depth)
            )
            self._append_validation_surface(world_points, surface_colors)
        cam_origin = cam_pose_tf[:3, 3]
        occupied_added = 0
        free_added = 0
        validation_endpoint_mapped = 0
        validation_endpoint_negative_z = 0
        validation_endpoint_negative_z_mapped = 0
        validation_endpoint_rejection_counts: Dict[str, int] = defaultdict(int)
        if self.config.validation_enable:
            self.validation_endpoint_total_count += int(world_points.shape[0])
        for point in world_points:
            if self.config.validation_enable and float(point[2]) < 0.0:
                validation_endpoint_negative_z += 1
            endpoint = self._xyz_to_grid(point)
            if endpoint is None:
                if self.config.validation_enable:
                    reason = self._validation_endpoint_rejection_reason(point)
                    validation_endpoint_rejection_counts[reason] += 1
                continue
            if self.config.validation_enable:
                validation_endpoint_mapped += 1
                if float(point[2]) < 0.0:
                    validation_endpoint_negative_z_mapped += 1
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

        if self.config.validation_enable:
            self.validation_endpoint_mapped_count += int(validation_endpoint_mapped)
            self.validation_endpoint_below_volume_count += int(
                validation_endpoint_rejection_counts.get("below_volume", 0)
            )
            self.validation_endpoint_above_volume_count += int(
                validation_endpoint_rejection_counts.get("above_volume", 0)
            )
            self.validation_endpoint_outside_xy_count += int(
                validation_endpoint_rejection_counts.get("outside_xy", 0)
            )
            self.validation_endpoint_negative_z_count += int(
                validation_endpoint_negative_z
            )
            self.validation_endpoint_negative_z_mapped_count += int(
                validation_endpoint_negative_z_mapped
            )

        self.update_count += 1
        self.occupied_update_count += occupied_added
        self.free_update_count += free_added
        self.frontier_cache = None
        self.frontier_set_cache = None
        self._refresh_latest_keyframe_information(context)
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
                "camera_pitch_aware_update": bool(
                    self.config.camera_pitch_aware_update
                ),
                "requested_camera_pitch_deg": float(requested_camera_pitch_deg),
                "applied_camera_pitch_deg": float(applied_camera_pitch_deg),
                "validation_endpoint_total_count": int(world_points.shape[0]),
                "validation_endpoint_mapped_count": int(validation_endpoint_mapped),
                "validation_endpoint_negative_z_count": int(
                    validation_endpoint_negative_z
                ),
                "validation_endpoint_negative_z_mapped_count": int(
                    validation_endpoint_negative_z_mapped
                ),
                "validation_endpoint_rejection_counts": dict(
                    validation_endpoint_rejection_counts
                ),
                **pose_gt_event,
                **endpoint_gt_event,
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

    def record_semantic(
        self,
        decision: Dict[str, Any],
        *,
        obs: Optional[Dict[str, Any]] = None,
        depth: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not decision:
            return
        context = dict(context or {})
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
        anchor_summary = self._record_semantic_anchors(
            decision,
            obs=obs,
            depth=depth,
            context=context,
        )
        if anchor_summary.get("enabled"):
            event["semantic_anchor_summary"] = anchor_summary
        self._write_event(event)

    def _semantic_anchor_terms(self, decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not bool(self.config.semantic_anchor_enable):
            return []
        min_score = float(self.config.semantic_anchor_min_score)
        max_terms = max(1, int(self.config.semantic_anchor_max_terms_per_event))
        by_term: Dict[str, Dict[str, Any]] = {}

        def add_term(term: Any, score: Any = None, *, source: str = "unknown", term_type: Any = None) -> None:
            canonical = self._canonical_semantic_term(term)
            if not canonical:
                return
            try:
                score_float = float(score)
            except (TypeError, ValueError):
                score_float = 0.0
            if score_float < min_score and source != "rank1":
                return
            previous = by_term.get(canonical)
            item = {
                "term": canonical,
                "raw_term": str(term),
                "score": float(score_float),
                "source": source,
                "term_type": term_type,
                "semantic_kind": self._semantic_anchor_kind(canonical, term_type=term_type),
            }
            if previous is None or float(item["score"]) > float(previous.get("score", 0.0) or 0.0):
                by_term[canonical] = item

        add_term(
            decision.get("top_match"),
            decision.get("top_score"),
            source="rank1",
        )
        for score_item in list(decision.get("scores") or []):
            add_term(
                score_item.get("term"),
                score_item.get("score"),
                source="score_item",
                term_type=score_item.get("term_type"),
            )
        if bool(self.config.semantic_anchor_include_threshold_hits):
            score_by_term = {
                self._canonical_semantic_term(item.get("term")): item
                for item in list(decision.get("scores") or [])
                if item.get("term")
            }
            for term in list(decision.get("threshold_hits") or []):
                canonical = self._canonical_semantic_term(term)
                score_item = score_by_term.get(canonical, {})
                add_term(
                    term,
                    score_item.get("score", decision.get("top_score")),
                    source="threshold_hit",
                    term_type=score_item.get("term_type"),
                )
        return sorted(
            by_term.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                item.get("source") == "rank1",
            ),
            reverse=True,
        )[:max_terms]

    def _semantic_anchor_kind(self, term: Any, *, term_type: Any = None) -> str:
        canonical = self._canonical_semantic_term(term)
        tokens = set(re.findall(r"[a-z0-9]+", canonical or ""))
        is_obstacle = bool(
            canonical in _SEMANTIC_RESILIENCE_OBSTACLE_TERMS
            or tokens.intersection(_SEMANTIC_RESILIENCE_OBSTACLE_TERMS)
        )
        is_passage = bool(
            canonical in _SEMANTIC_RESILIENCE_PASSAGE_TERMS
            or tokens.intersection(_SEMANTIC_RESILIENCE_PASSAGE_TERMS)
        )
        if is_obstacle and is_passage:
            return "obstacle_passage"
        if is_obstacle:
            return "obstacle"
        if is_passage:
            return "passage"
        if canonical in _GOAL_PROGRESS_SPECIFIC_ROOMS or str(term_type or "") == "room":
            return "room"
        if canonical in _GOAL_PROGRESS_LANDMARK_TERMS or str(term_type or "") == "object":
            return "landmark"
        return "semantic"

    def _semantic_anchor_pixel_sources(
        self,
        decision: Dict[str, Any],
        depth: np.ndarray,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        h, w = depth_arr.shape[:2]
        image_w = int(
            context.get("image_width")
            or decision.get("image_width")
            or self.config.waypoint_source_image_width
            or w
        )
        image_h = int(
            context.get("image_height")
            or decision.get("image_height")
            or self.config.waypoint_source_image_height
            or h
        )
        sources: List[Dict[str, Any]] = []
        if bool(self.config.semantic_anchor_include_pixel_goal):
            pixel_goal = decision.get("pixel_goal") or context.get("pixel_goal") or context.get("s2_pixel_goal")
            if pixel_goal is not None:
                try:
                    px, py = [float(v) for v in list(pixel_goal)[:2]]
                    sources.append(
                        {
                            "source": "pixel_goal",
                            "pixel": [float(px), float(py)],
                            "image_width": int(image_w),
                            "image_height": int(image_h),
                        }
                    )
                except (TypeError, ValueError):
                    pass
        if bool(self.config.semantic_anchor_include_view_center):
            px = float(image_w) * float(self.config.semantic_anchor_view_center_x)
            py = float(image_h) * float(self.config.semantic_anchor_view_center_y)
            sources.append(
                {
                    "source": "view_center",
                    "pixel": [float(px), float(py)],
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
        if bool(self.config.semantic_anchor_include_view_left):
            px = float(image_w) * float(self.config.semantic_anchor_view_left_x)
            py = float(image_h) * float(self.config.semantic_anchor_view_center_y)
            sources.append(
                {
                    "source": "view_left",
                    "pixel": [float(px), float(py)],
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
        if bool(self.config.semantic_anchor_include_view_right):
            px = float(image_w) * float(self.config.semantic_anchor_view_right_x)
            py = float(image_h) * float(self.config.semantic_anchor_view_center_y)
            sources.append(
                {
                    "source": "view_right",
                    "pixel": [float(px), float(py)],
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
        if bool(self.config.semantic_anchor_include_view_upper):
            px = float(image_w) * float(self.config.semantic_anchor_view_center_x)
            py = float(image_h) * float(self.config.semantic_anchor_view_upper_y)
            sources.append(
                {
                    "source": "view_upper",
                    "pixel": [float(px), float(py)],
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
        if bool(self.config.semantic_anchor_include_view_lower):
            px = float(image_w) * float(self.config.semantic_anchor_view_center_x)
            py = float(image_h) * float(self.config.semantic_anchor_view_lower_y)
            sources.append(
                {
                    "source": "view_lower",
                    "pixel": [float(px), float(py)],
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
        return sources

    def _semantic_anchor_local_context(self, cell: Iterable[int]) -> Dict[str, Any]:
        try:
            row0, col0 = [int(v) for v in list(cell)[:2]]
        except (TypeError, ValueError):
            return {}
        radius = max(1, int(self.config.semantic_anchor_local_radius_cells))
        counts = {"free": 0, "occupied": 0, "unknown": 0}
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                row = row0 + dr
                col = col0 + dc
                if row < 0 or row >= self.gs or col < 0 or col >= self.gs:
                    continue
                counts[self._cell_state(row, col)] += 1
        total = max(1, int(sum(counts.values())))
        return {
            "local_radius_cells": int(radius),
            "local_free_count": int(counts["free"]),
            "local_occupied_count": int(counts["occupied"]),
            "local_unknown_count": int(counts["unknown"]),
            "local_free_ratio": float(counts["free"] / total),
            "local_occupied_ratio": float(counts["occupied"] / total),
            "local_unknown_ratio": float(counts["unknown"] / total),
            "open_score": float(self._semantic_resilience_open_score([row0, col0])),
        }

    def _semantic_anchor_pose_context(self) -> Dict[str, Any]:
        if not self.pose_trace:
            return {}
        pose = self.pose_trace[-1]
        try:
            x = float(pose.get("x", 0.0) or 0.0)
            y = float(pose.get("y", 0.0) or 0.0)
            z = float(pose.get("z", 0.0) or 0.0)
            yaw = float(pose.get("yaw", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {}
        context = {
            "pose_local_xy": [float(x), float(y)],
            "pose_local_z_m": float(z),
            "pose_local_yaw_deg": float(math.degrees(yaw)),
            "pose_origin_distance_m": float(math.hypot(x, y)),
            "pose_step_id": pose.get("step_id"),
        }
        if len(self.pose_trace) >= 2:
            prev = self.pose_trace[-2]
            try:
                prev_x = float(prev.get("x", 0.0) or 0.0)
                prev_y = float(prev.get("y", 0.0) or 0.0)
                prev_yaw = float(prev.get("yaw", 0.0) or 0.0)
            except (TypeError, ValueError):
                prev_x = prev_y = prev_yaw = 0.0
            step_dx = float(x - prev_x)
            step_dy = float(y - prev_y)
            step_dyaw_deg = float(math.degrees(self._wrap_angle(yaw - prev_yaw)))
            context.update(
                {
                    "pose_prev_step_id": prev.get("step_id"),
                    "pose_step_dx_m": step_dx,
                    "pose_step_dy_m": step_dy,
                    "pose_step_distance_m": float(math.hypot(step_dx, step_dy)),
                    "pose_step_dyaw_deg": step_dyaw_deg,
                }
            )
        return context

    def _semantic_anchor_source_context(
        self,
        source: Dict[str, Any],
        depth: np.ndarray,
    ) -> Dict[str, Any]:
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        h, w = depth_arr.shape[:2]
        image_w = int(source.get("image_width") or self.config.waypoint_source_image_width or w or 1)
        image_h = int(source.get("image_height") or self.config.waypoint_source_image_height or h or 1)
        try:
            pixel = list(source.get("pixel") or [0.0, 0.0])
            px = float(pixel[0])
            py = float(pixel[1])
        except (TypeError, ValueError, IndexError):
            px = 0.0
            py = 0.0
        center_x = float(image_w) * 0.5
        center_y = float(image_h) * 0.5
        offset_x_px = float(px - center_x)
        offset_y_px = float(py - center_y)
        context = {
            "source_offset_x_px": offset_x_px,
            "source_offset_y_px": offset_y_px,
            "source_offset_x_norm": float(offset_x_px / max(1.0, float(image_w))),
            "source_offset_y_norm": float(offset_y_px / max(1.0, float(image_h))),
            "source_center_distance_px": float(math.hypot(offset_x_px, offset_y_px)),
        }
        if self.camera_intrinsic is not None:
            sx = float(px) * float(w) / max(1.0, float(image_w))
            sy = float(py) * float(h) / max(1.0, float(image_h))
            fx = float(self.camera_intrinsic[0, 0])
            fy = float(self.camera_intrinsic[1, 1])
            cx = float(self.camera_intrinsic[0, 2])
            cy = float(self.camera_intrinsic[1, 2])
            cam_x = (sx + 0.5 - cx) / max(1e-6, fx)
            cam_y = (sy + 0.5 - cy) / max(1e-6, fy)
            cam_z = 1.0
            context.update(
                {
                    "source_ray_x": float(cam_x),
                    "source_ray_y": float(cam_y),
                    "source_ray_z": float(cam_z),
                    "source_ray_norm": float(math.sqrt(cam_x * cam_x + cam_y * cam_y + cam_z * cam_z)),
                    "source_ray_yaw_deg": float(math.degrees(math.atan2(cam_x, cam_z))),
                    "source_ray_pitch_deg": float(math.degrees(math.atan2(-cam_y, cam_z))),
                }
            )
        return context

    def _basic_stats(self, values: Iterable[Any]) -> Dict[str, Any]:
        cleaned: List[float] = []
        for value in values:
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        if not cleaned:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": int(len(cleaned)),
            "mean": float(mean(cleaned)),
            "min": float(min(cleaned)),
            "max": float(max(cleaned)),
        }

    def _merge_or_add_semantic_anchor(self, anchor: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        row = int(anchor["grid"][0])
        col = int(anchor["grid"][1])
        term = str(anchor.get("semantic_top_match") or "")
        merge_radius = max(0, int(self.config.semantic_anchor_merge_radius_cells))
        best_index = None
        best_distance = None
        for idx, existing in enumerate(self.semantic_anchors):
            if str(existing.get("semantic_top_match") or "") != term:
                continue
            erow, ecol = [int(v) for v in list(existing.get("grid") or [0, 0])[:2]]
            distance = math.sqrt(float((row - erow) ** 2 + (col - ecol) ** 2))
            if distance <= merge_radius and (best_distance is None or distance < best_distance):
                best_index = idx
                best_distance = distance
        if best_index is None:
            max_anchors = int(self.config.semantic_anchor_max_anchors_per_episode)
            if max_anchors >= 0 and len(self.semantic_anchors) >= max_anchors:
                return anchor, "max_anchors"
            anchor = dict(anchor)
            anchor["anchor_id"] = f"SA{len(self.semantic_anchors):04d}"
            anchor["observation_count"] = 1
            anchor["score_sum"] = float(anchor.get("semantic_top_score", 0.0) or 0.0)
            anchor["score_max"] = float(anchor.get("semantic_top_score", 0.0) or 0.0)
            anchor["score_mean"] = float(anchor.get("semantic_top_score", 0.0) or 0.0)
            anchor["source_counts"] = {str(anchor.get("anchor_source") or "unknown"): 1}
            self.semantic_anchors.append(anchor)
            return anchor, "added"

        existing = self.semantic_anchors[best_index]
        score = float(anchor.get("semantic_top_score", 0.0) or 0.0)
        count = int(existing.get("observation_count", 1) or 1) + 1
        existing["observation_count"] = int(count)
        existing["last_step_id"] = anchor.get("last_step_id")
        existing["last_grid"] = anchor.get("grid")
        existing["last_xy"] = anchor.get("xy")
        existing["last_anchor_source"] = anchor.get("anchor_source")
        existing["score_sum"] = float(existing.get("score_sum", 0.0) or 0.0) + score
        existing["score_max"] = max(float(existing.get("score_max", 0.0) or 0.0), score)
        existing["score_mean"] = float(existing["score_sum"] / max(1, count))
        existing["high_conf_semantic"] = bool(
            existing.get("high_conf_semantic") or anchor.get("high_conf_semantic")
        )
        source_counts = dict(existing.get("source_counts") or {})
        source = str(anchor.get("anchor_source") or "unknown")
        source_counts[source] = int(source_counts.get(source, 0)) + 1
        existing["source_counts"] = source_counts
        for key in (
            "goal_state",
            "semantic_kind",
            "local_free_count",
            "local_occupied_count",
            "local_unknown_count",
            "local_free_ratio",
            "local_occupied_ratio",
            "local_unknown_ratio",
            "open_score",
            "pose_local_xy",
            "pose_local_z_m",
            "pose_local_yaw_deg",
            "pose_origin_distance_m",
            "pose_step_id",
            "pose_prev_step_id",
            "pose_step_dx_m",
            "pose_step_dy_m",
            "pose_step_distance_m",
            "pose_step_dyaw_deg",
            "source_offset_x_px",
            "source_offset_y_px",
            "source_offset_x_norm",
            "source_offset_y_norm",
            "source_center_distance_px",
            "source_ray_x",
            "source_ray_y",
            "source_ray_z",
            "source_ray_norm",
            "source_ray_yaw_deg",
            "source_ray_pitch_deg",
            "global_bearing_deg",
            "relative_bearing_deg",
            "relative_bearing_angle_deg",
        ):
            existing[key] = anchor.get(key, existing.get(key))
        return existing, "merged"

    def _record_semantic_anchors(
        self,
        decision: Dict[str, Any],
        *,
        obs: Optional[Dict[str, Any]],
        depth: Optional[np.ndarray],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = {
            "enabled": bool(self.config.semantic_anchor_enable),
            "valid": False,
            "reason": None,
            "added": 0,
            "merged": 0,
            "invalid": 0,
            "max_anchors": 0,
        }
        if not bool(self.config.semantic_anchor_enable):
            summary["reason"] = "disabled"
            return summary
        if obs is None or depth is None or self.camera_intrinsic is None:
            summary["reason"] = "missing_obs_depth_or_intrinsic"
            return summary
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None or self.init_base_tf is None:
            summary["reason"] = "missing_pose_or_memory"
            return summary
        terms = self._semantic_anchor_terms(decision)
        if not terms:
            summary["reason"] = "no_semantic_terms"
            return summary
        sources = self._semantic_anchor_pixel_sources(decision, depth, context)
        if not sources:
            summary["reason"] = "no_pixel_sources"
            return summary

        projected_sources: List[Dict[str, Any]] = []
        for source in sources:
            source_context = {
                **context,
                "image_width": source.get("image_width"),
                "image_height": source.get("image_height"),
            }
            target = self._pixel_goal_to_grid(source.get("pixel"), depth, pose_tf, source_context)
            if target is None:
                summary["invalid"] += 1
                continue
            projected_sources.append({**source, "target": target})
        if not projected_sources:
            summary["reason"] = "projection_failed"
            return summary

        pose_context = self._semantic_anchor_pose_context()
        for source in projected_sources:
            target = dict(source.get("target") or {})
            grid = list(target.get("goal_grid") or [])
            if len(grid) < 2:
                summary["invalid"] += 1
                self.semantic_anchor_invalid_count += 1
                continue
            row, col = int(grid[0]), int(grid[1])
            xy = self._grid_to_xy([row, col])
            direction = self._direction_to_cell(
                target.get("start_grid"),
                [row, col],
                float(target.get("start_yaw", 0.0) or 0.0),
            )
            local_context = self._semantic_anchor_local_context([row, col])
            source_context = self._semantic_anchor_source_context(source, depth)
            goal_state = self._cell_state(row, col)
            for term in terms:
                anchor = {
                    "event_type": "occ_memory_semantic_anchor",
                    **self.episode_meta,
                    "step_id": decision.get("step_id") or context.get("step_id"),
                    "last_step_id": decision.get("step_id") or context.get("step_id"),
                    "anchor_source": source.get("source"),
                    "source_pixel": source.get("pixel"),
                    "source_image_width": source.get("image_width"),
                    "source_image_height": source.get("image_height"),
                    **source_context,
                    **pose_context,
                    "semantic_top_match": term.get("term"),
                    "semantic_raw_term": term.get("raw_term"),
                    "semantic_top_score": float(term.get("score", 0.0) or 0.0),
                    "semantic_kind": term.get("semantic_kind"),
                    "semantic_term_source": term.get("source"),
                    "high_conf_semantic": bool(decision.get("high_conf_semantic")),
                    "grid": [int(row), int(col)],
                    "xy": [float(xy[0]), float(xy[1])],
                    "world_z": target.get("goal_world_z"),
                    "depth_m": target.get("depth_m"),
                    "goal_state": goal_state,
                    "direction_bucket": direction.get("bucket"),
                    "direction_angle_deg": direction.get("angle_deg"),
                    "global_bearing_deg": direction.get("world_bearing_deg"),
                    "relative_bearing_deg": direction.get("angle_deg"),
                    "distance_m": direction.get("distance_m"),
                    **local_context,
                }
                stored, operation = self._merge_or_add_semantic_anchor(anchor)
                source_name = str(source.get("source") or "unknown")
                if operation == "added":
                    summary["added"] += 1
                    self.semantic_anchor_added_count += 1
                elif operation == "merged":
                    summary["merged"] += 1
                    self.semantic_anchor_merged_count += 1
                elif operation == "max_anchors":
                    summary["max_anchors"] += 1
                    self.semantic_anchor_max_anchors_count += 1
                else:
                    self.semantic_anchor_invalid_count += 1
                self.semantic_anchor_source_operation_counts[source_name][str(operation)] += 1
                if source_context:
                    if source_context.get("source_offset_x_px") is not None:
                        self.semantic_anchor_source_offset_x_px_values.append(
                            float(source_context["source_offset_x_px"])
                        )
                    if source_context.get("source_offset_y_px") is not None:
                        self.semantic_anchor_source_offset_y_px_values.append(
                            float(source_context["source_offset_y_px"])
                        )
                if source_context.get("source_ray_yaw_deg") is not None:
                    self.semantic_anchor_source_ray_yaw_deg_values.append(
                        float(source_context["source_ray_yaw_deg"])
                    )
                if source_context.get("source_ray_pitch_deg") is not None:
                    self.semantic_anchor_source_ray_pitch_deg_values.append(
                        float(source_context["source_ray_pitch_deg"])
                    )
                if source_context.get("source_center_distance_px") is not None:
                    self.semantic_anchor_source_center_distance_px_values.append(
                        float(source_context["source_center_distance_px"])
                    )
                if source_context.get("source_ray_norm") is not None:
                    self.semantic_anchor_source_ray_norm_values.append(
                        float(source_context["source_ray_norm"])
                    )
                if direction.get("world_bearing_deg") is not None:
                    self.semantic_anchor_global_bearing_deg_values.append(
                        float(direction["world_bearing_deg"])
                    )
                if direction.get("angle_deg") is not None:
                    self.semantic_anchor_relative_bearing_deg_values.append(float(direction["angle_deg"]))
                if pose_context.get("pose_origin_distance_m") is not None:
                    self.semantic_anchor_pose_origin_distance_m_values.append(
                        float(pose_context["pose_origin_distance_m"])
                    )
                if pose_context.get("pose_step_distance_m") is not None:
                    self.semantic_anchor_pose_step_distance_m_values.append(
                        float(pose_context["pose_step_distance_m"])
                    )
                if pose_context.get("pose_step_dyaw_deg") is not None:
                    self.semantic_anchor_pose_step_dyaw_deg_values.append(
                        float(pose_context["pose_step_dyaw_deg"])
                    )
                event = dict(anchor)
                event["anchor_id"] = stored.get("anchor_id")
                event["anchor_operation"] = operation
                event["observation_count"] = stored.get("observation_count")
                event["score_mean"] = stored.get("score_mean")
                event["score_max"] = stored.get("score_max")
                event["source_counts"] = stored.get("source_counts")
                self._write_event(event)
        summary["valid"] = bool(summary["added"] or summary["merged"])
        summary["reason"] = "ok" if summary["valid"] else "no_anchors_recorded"
        summary["anchor_count"] = int(len(self.semantic_anchors))
        return summary

    def record_guidance_event(
        self,
        *,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        hint: str = "",
        reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
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
        if extra:
            event.update(extra)
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
        stage15_repair_info = self._stage15_repair_shadow_info(
            pixel_goal=pixel_goal,
            depth=depth,
            pose_tf=pose_tf,
            context=context,
            target=target,
            goal_state=state,
        )
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
                **stage15_repair_info,
                **attribution,
            }
        )
        self.waypoint_events.append(event)
        self._write_event(event)
        return event

    def generate_query_candidates(
        self,
        *,
        obs: Optional[Dict[str, Any]] = None,
        current_waypoint_decision: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate sparse 3D memory candidates for V10 shadow probes.

        This is deliberately a query interface, not an action interface: it
        records candidate points that a downstream model could choose from, but
        it never changes navigation state.
        """
        context = dict(context or {})
        decision = dict(current_waypoint_decision or {})
        event = {
            "event_type": "occ_memory_query_candidates",
            "event_schema_version": "stage21a_r3_v3",
            "recovery_feature_schema_version": "v3",
            **self.episode_meta,
            **context,
            "enabled": bool(self.enabled and self.config.candidate_probe_enable),
            "valid": False,
            "reason": None,
        }
        if not self.enabled or not self.config.candidate_probe_enable:
            event["reason"] = "disabled"
            return event
        max_events = int(self.config.candidate_probe_max_events_per_episode)
        if max_events >= 0 and len(self.candidate_probe_events) >= max_events:
            event["reason"] = "max_events_per_episode"
            self._write_event(event)
            return event
        pose_state = self._current_pose_state(obs or {})
        if pose_state is None:
            event["reason"] = "missing_pose_or_memory"
            self._write_event(event)
            return event
        start_grid = pose_state["grid"]
        yaw = float(pose_state["yaw"])
        current_angle = decision.get("waypoint_direction_angle_deg")
        current_goal_grid = decision.get("goal_grid")
        current_policy_candidate = self._build_current_policy_candidate(
            decision,
            start_grid=start_grid,
            yaw=yaw,
            context=context,
        )
        goal_progress_state = self._semantic_goal_progress_state()
        semantic_nodes = self._semantic_memory_nodes(
            start_grid,
            yaw,
            goal_progress_state=goal_progress_state,
        )
        semantic_resilience_state = self._semantic_resilience_local_state(
            start_grid,
            yaw,
            semantic_nodes=semantic_nodes,
            current_policy_candidate=current_policy_candidate,
        )
        if bool(context.get("s2_action_loop_detected")):
            trigger_reasons = list(semantic_resilience_state.get("trigger_reasons") or [])
            recovery_context_tags = list(
                semantic_resilience_state.get("recovery_context_tags") or []
            )
            for reason in ("s2_repeated_turn_generation", "s2_low_translation"):
                if reason not in trigger_reasons:
                    trigger_reasons.append(reason)
            for tag in ("s2_policy_loop", "decision_state_restoration"):
                if tag not in recovery_context_tags:
                    recovery_context_tags.append(tag)
            semantic_resilience_state.update(
                {
                    "recovery_trigger": True,
                    "current_policy_problem": True,
                    "trigger_reasons": trigger_reasons,
                    "recovery_context_tags": recovery_context_tags,
                    "s2_action_loop_detected": True,
                }
            )
        raw_candidates: List[Dict[str, Any]] = []
        raw_candidates.extend(
            self._frontier_query_candidates(
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                semantic_nodes=semantic_nodes,
                goal_progress_state=goal_progress_state,
            )
        )
        raw_candidates.extend(
            self._semantic_query_candidates(
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                semantic_nodes=semantic_nodes,
                goal_progress_state=goal_progress_state,
            )
        )
        raw_candidates.extend(
            self._open_floor_query_candidates(
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                semantic_nodes=semantic_nodes,
                goal_progress_state=goal_progress_state,
            )
        )
        raw_candidates.extend(
            self._semantic_resilience_backtrack_candidates(
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                semantic_nodes=semantic_nodes,
                goal_progress_state=goal_progress_state,
                resilience_state=semantic_resilience_state,
            )
        )
        candidates = self._select_query_candidates(raw_candidates)
        for idx, item in enumerate(candidates):
            item["candidate_index"] = int(idx)
            item["candidate_id"] = chr(ord("A") + idx)
        candidate_types: Dict[str, int] = defaultdict(int)
        direction_counts: Dict[str, int] = defaultdict(int)
        geometry_safe_count = 0
        active_gate_safe_count = 0
        current_aligned_count = 0
        semantic_evidence_count = 0
        instruction_relevant_count = 0
        semanticized_count = 0
        next_landmark_relevant_count = 0
        completed_landmark_count = 0
        repeated_semantic_count = 0
        unknown_target_frontier_bonus_count = 0
        target_frontier_count = 0
        target_frontier_escape_count = 0
        target_frontier_intent_safe_count = 0
        target_frontier_doorway_like_count = 0
        semantic_resilience_candidate_count = 0
        semantic_resilience_recommended_count = 0
        semantic_resilience_obstacle_count = 0
        semantic_resilience_passage_count = 0
        for item in candidates:
            candidate_types[str(item.get("candidate_type", "unknown"))] += 1
            direction_counts[str(item.get("direction_bucket", "unknown"))] += 1
            if item.get("geometry_safe"):
                geometry_safe_count += 1
            if item.get("active_gate_safe"):
                active_gate_safe_count += 1
            if item.get("aligned_with_current_waypoint"):
                current_aligned_count += 1
            if item.get("semantic_evidence"):
                semantic_evidence_count += 1
            if item.get("instruction_relevant"):
                instruction_relevant_count += 1
            if item.get("semanticized_candidate"):
                semanticized_count += 1
            if float(item.get("next_landmark_relevance", 0.0) or 0.0) > 0.0:
                next_landmark_relevant_count += 1
            if float(item.get("completed_landmark_penalty", 0.0) or 0.0) > 0.0:
                completed_landmark_count += 1
            if float(item.get("repeated_semantic_penalty", 0.0) or 0.0) > 0.0:
                repeated_semantic_count += 1
            if float(item.get("unknown_target_frontier_bonus", 0.0) or 0.0) > 0.0:
                unknown_target_frontier_bonus_count += 1
            if item.get("target_frontier_candidate"):
                target_frontier_count += 1
            if item.get("target_frontier_escape_candidate"):
                target_frontier_escape_count += 1
            if item.get("target_frontier_intent_safe"):
                target_frontier_intent_safe_count += 1
            if float(item.get("target_frontier_doorway_like_score", 0.0) or 0.0) >= float(
                self.config.candidate_probe_target_frontier_doorway_threshold
            ):
                target_frontier_doorway_like_count += 1
            if item.get("semantic_resilience_candidate"):
                semantic_resilience_candidate_count += 1
            if item.get("semantic_resilience_recommended"):
                semantic_resilience_recommended_count += 1
            if item.get("semantic_resilience_obstacle_term_count", 0):
                semantic_resilience_obstacle_count += 1
            if item.get("semantic_resilience_passage_term_count", 0):
                semantic_resilience_passage_count += 1
        bev_path = None
        if self.config.candidate_probe_save_bev and candidates:
            bev_path = self._write_candidate_bev_snapshot(candidates, context)
        event.update(
            {
                "valid": bool(candidates),
                "reason": "ok" if candidates else "no_candidates",
                "start_grid": [int(start_grid[0]), int(start_grid[1])],
                "start_xy": [
                    float(pose_state["xy"][0]),
                    float(pose_state["xy"][1]),
                ],
                "start_yaw": yaw,
                "current_waypoint_goal_grid": self._jsonable(current_goal_grid),
                "current_waypoint_direction_angle_deg": current_angle,
                "current_waypoint_direction_bucket": decision.get("waypoint_direction_bucket"),
                "current_waypoint_goal_state": decision.get("goal_state"),
                "current_waypoint_semantic_dead_zone": decision.get("semantic_dead_zone"),
                "current_waypoint_semantic_dead_zone_score": decision.get("semantic_dead_zone_score"),
                "current_policy_candidate": current_policy_candidate,
                "current_policy_candidate_valid": bool(
                    (current_policy_candidate or {}).get("valid")
                ),
                "current_policy_candidate_reason": (
                    current_policy_candidate or {}
                ).get("reason"),
                "raw_candidate_count": int(len(raw_candidates)),
                "candidate_count": int(len(candidates)),
                "candidate_type_counts": dict(candidate_types),
                "candidate_direction_counts": dict(direction_counts),
                "candidate_geometry_safe_count": int(geometry_safe_count),
                "candidate_active_gate_safe_count": int(active_gate_safe_count),
                "candidate_current_aligned_count": int(current_aligned_count),
                "candidate_semantic_evidence_count": int(semantic_evidence_count),
                "candidate_instruction_relevant_count": int(instruction_relevant_count),
                "candidate_semanticized_count": int(semanticized_count),
                "candidate_next_landmark_relevant_count": int(next_landmark_relevant_count),
                "candidate_completed_landmark_count": int(completed_landmark_count),
                "candidate_repeated_semantic_count": int(repeated_semantic_count),
                "candidate_unknown_target_frontier_bonus_count": int(
                    unknown_target_frontier_bonus_count
                ),
                "candidate_target_frontier_count": int(target_frontier_count),
                "candidate_target_frontier_escape_count": int(target_frontier_escape_count),
                "candidate_target_frontier_intent_safe_count": int(
                    target_frontier_intent_safe_count
                ),
                "candidate_target_frontier_doorway_like_count": int(
                    target_frontier_doorway_like_count
                ),
                "semantic_resilience_state": semantic_resilience_state,
                "semantic_resilience_enabled": bool(
                    semantic_resilience_state.get("enabled")
                ),
                "semantic_resilience_recovery_trigger": bool(
                    semantic_resilience_state.get("recovery_trigger")
                ),
                "semantic_resilience_local_trap": bool(
                    semantic_resilience_state.get("local_trap")
                ),
                "semantic_resilience_trigger_reasons": semantic_resilience_state.get(
                    "trigger_reasons"
                ),
                "semantic_resilience_recovery_context_tags": semantic_resilience_state.get(
                    "recovery_context_tags"
                ),
                "semantic_resilience_raw_candidate_count": int(
                    sum(
                        1
                        for item in raw_candidates
                        if item.get("semantic_resilience_candidate")
                    )
                ),
                "candidate_semantic_resilience_count": int(
                    semantic_resilience_candidate_count
                ),
                "candidate_semantic_resilience_recommended_count": int(
                    semantic_resilience_recommended_count
                ),
                "candidate_semantic_resilience_obstacle_count": int(
                    semantic_resilience_obstacle_count
                ),
                "candidate_semantic_resilience_passage_count": int(
                    semantic_resilience_passage_count
                ),
                "semantic_memory_node_count": int(len(semantic_nodes)),
                "instruction_terms": self._instruction_terms(),
                "goal_progress_enabled": bool(goal_progress_state.get("enabled")),
                "goal_progress_landmark_sequence": goal_progress_state.get("landmark_sequence"),
                "goal_progress_completed_landmarks": goal_progress_state.get("completed_landmarks"),
                "goal_progress_next_landmark": goal_progress_state.get("next_landmark"),
                "goal_progress_next_landmark_index": goal_progress_state.get("next_landmark_index"),
                "goal_progress_recent_repeated_terms": goal_progress_state.get("recent_repeated_terms"),
                "candidate_best_score": (
                    float(candidates[0].get("score", 0.0)) if candidates else None
                ),
                "candidate_bev_path": bev_path,
                "candidates": candidates,
            }
        )
        event["progress_ranker_shadow"] = self._score_progress_ranker_shadow(candidates)
        event["stage21_multitask_shadow"] = self._score_stage21_multitask_shadow(
            candidates, current_policy_candidate
        )
        self.candidate_probe_events.append(event)
        self._write_event(event)
        return event

    def _build_current_policy_candidate(
        self,
        decision: Dict[str, Any],
        *,
        start_grid: Iterable[int],
        yaw: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize the frozen policy's current waypoint as a Stage18 candidate.

        Stage17 ranks only OccMem-generated candidates.  Stage18 needs the
        current S2/NextDiT intent as a comparable reference so a later adapter
        can learn keep/intervene/abstain instead of blindly replacing S2.
        """
        candidate: Dict[str, Any] = {
            "candidate_id": "S2",
            "candidate_index": -1,
            "candidate_type": "current_policy",
            "source": "s2_current_waypoint",
            "valid": False,
            "reason": None,
            "pixel_goal": self._jsonable(context.get("s2_pixel_goal") or decision.get("pixel_goal")),
            "vlmap_waypoint_valid": context.get("vlmap_waypoint_valid"),
            "vlmap_waypoint_reason": context.get("vlmap_waypoint_reason"),
            "occ_waypoint_valid": decision.get("valid"),
            "occ_waypoint_reason": decision.get("reason"),
        }
        if not decision:
            candidate["reason"] = "missing_current_waypoint_decision"
            return candidate
        if not bool(decision.get("valid")):
            candidate["reason"] = str(decision.get("reason") or "invalid_current_waypoint")
            return candidate
        goal_grid = decision.get("goal_grid")
        try:
            row, col = [int(v) for v in list(goal_grid)[:2]]
        except (TypeError, ValueError):
            candidate["reason"] = "invalid_current_waypoint_grid"
            return candidate
        state = str(decision.get("goal_state") or self._cell_state(row, col))
        direction = self._direction_to_cell(start_grid, [row, col], yaw)
        bucket = str(decision.get("waypoint_direction_bucket") or direction.get("bucket") or "unknown")
        angle = decision.get("waypoint_direction_angle_deg")
        if angle is None:
            angle = direction.get("angle_deg")
        distance_m = decision.get("waypoint_distance_m")
        if distance_m is None:
            distance_m = direction.get("distance_m")
        visit_count = self._nearby_visit_count(
            [row, col],
            int(self.config.waypoint_revisit_radius_cells),
        )
        revisit_risk = min(1.0, float(visit_count) / 3.0)
        geometry_safe = state != "occupied"
        xy = self._grid_to_xy([row, col])
        candidate.update(
            {
                "valid": True,
                "reason": "ok",
                "grid": [int(row), int(col)],
                "xy": [float(xy[0]), float(xy[1])],
                "start_grid": self._jsonable(start_grid),
                "goal_state": state,
                "geometry_safe": bool(geometry_safe),
                "active_gate_safe": bool(geometry_safe and bucket != "back"),
                "direction_bucket": bucket,
                "direction_angle_deg": angle,
                "distance_m": None if distance_m is None else float(distance_m),
                "frontier_distance_m": decision.get("frontier_distance_m"),
                "nearby_visit_count": int(visit_count),
                "revisit_risk": float(revisit_risk),
                "points_to_revisited_region": bool(visit_count > 0),
                "semantic_dead_zone": decision.get("semantic_dead_zone"),
                "semantic_dead_zone_score": decision.get("semantic_dead_zone_score"),
                "semantic_recent_unique_count": decision.get("semantic_recent_unique_count"),
                "semantic_recent_high_conf_count": decision.get("semantic_recent_high_conf_count"),
                "semantic_recent_terms": decision.get("semantic_recent_terms"),
                "semantic_last_top_match": decision.get("semantic_last_top_match"),
                "semantic_last_top_score": decision.get("semantic_last_top_score"),
                "semantic_last_stagnation": decision.get("semantic_last_stagnation"),
                "semantic_stagnation_active": decision.get("semantic_stagnation_active"),
                "frontier_dominant_direction": decision.get("frontier_dominant_direction"),
                "frontier_dominant_angle_deg": decision.get("frontier_dominant_angle_deg"),
                "frontier_dominant_count": decision.get("frontier_dominant_count"),
                "frontier_direction_counts": decision.get("frontier_direction_counts"),
                "waypoint_aligns_with_dominant_frontier": decision.get(
                    "waypoint_aligns_with_dominant_frontier"
                ),
                "waypoint_aligns_with_high_conf_keyframe": decision.get(
                    "waypoint_aligns_with_high_conf_keyframe"
                ),
                "nearest_high_conf_keyframe": decision.get("nearest_high_conf_keyframe"),
            }
        )
        return candidate

    def _score_progress_ranker_shadow(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not bool(self.config.progress_ranker_shadow_enable):
            return {"enabled": False, "valid": False, "reason": "disabled"}
        if self.progress_ranker_shadow_scorer is None:
            return {"enabled": True, "valid": False, "reason": "not_initialized"}
        try:
            return self.progress_ranker_shadow_scorer.score_candidates(candidates)
        except Exception as exc:  # shadow diagnostics must not affect navigation
            return {
                "enabled": True,
                "valid": False,
                "reason": "error",
                "error": str(exc),
            }

    def _score_stage21_multitask_shadow(
        self,
        candidates: List[Dict[str, Any]],
        current_policy_candidate: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not bool(self.config.stage21_multitask_shadow_enable):
            return {"enabled": False, "valid": False, "reason": "disabled", "action_applied": False}
        if self.stage21_multitask_shadow_scorer is None:
            return {
                "enabled": True, "valid": False, "reason": "not_initialized",
                "error": self.stage21_multitask_shadow_init_error,
                "action_applied": False,
            }
        try:
            return self.stage21_multitask_shadow_scorer.score_candidates(
                candidates, current_policy_candidate
            )
        except Exception as exc:  # shadow diagnostics must not affect navigation
            return {
                "enabled": True, "valid": False, "reason": "error", "error": str(exc),
                "action_applied": False,
            }

    def _semantic_resilience_term_kind(self, term: Any) -> Optional[str]:
        canonical = self._canonical_semantic_term(term)
        if not canonical:
            canonical = str(term or "").lower().strip().replace("_", " ")
        if not canonical:
            return None
        tokens = set(re.findall(r"[a-z0-9]+", canonical))
        if canonical in _SEMANTIC_RESILIENCE_OBSTACLE_TERMS or tokens.intersection(
            _SEMANTIC_RESILIENCE_OBSTACLE_TERMS
        ):
            return "obstacle"
        if canonical in _SEMANTIC_RESILIENCE_PASSAGE_TERMS or tokens.intersection(
            _SEMANTIC_RESILIENCE_PASSAGE_TERMS
        ):
            return "passage"
        return None

    def _semantic_resilience_semantic_counts(
        self,
        cell: Iterable[int],
        semantic_nodes: List[Dict[str, Any]],
        *,
        radius_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not semantic_nodes:
            return {
                "obstacle_term_count": 0,
                "passage_term_count": 0,
                "nearest_obstacle_term": None,
                "nearest_obstacle_distance_m": None,
                "nearest_passage_term": None,
                "nearest_passage_distance_m": None,
            }
        if radius_m is None:
            radius_m = max(
                self.cs,
                float(self.config.semantic_resilience_local_radius_cells) * self.cs,
            )
        center_xy = self._grid_to_xy(cell)
        obstacle_count = 0
        passage_count = 0
        nearest_obstacle = (None, None)
        nearest_passage = (None, None)
        for node in semantic_nodes:
            node_xy = np.asarray(node.get("xy", [0.0, 0.0])[:2], dtype=np.float32)
            distance = float(np.linalg.norm(center_xy - node_xy))
            if distance > float(radius_m):
                continue
            term = node.get("semantic_top_match")
            kind = self._semantic_resilience_term_kind(term)
            if kind == "obstacle":
                obstacle_count += 1
                if nearest_obstacle[1] is None or distance < float(nearest_obstacle[1]):
                    nearest_obstacle = (term, distance)
            elif kind == "passage":
                passage_count += 1
                if nearest_passage[1] is None or distance < float(nearest_passage[1]):
                    nearest_passage = (term, distance)
        return {
            "obstacle_term_count": int(obstacle_count),
            "passage_term_count": int(passage_count),
            "nearest_obstacle_term": nearest_obstacle[0],
            "nearest_obstacle_distance_m": nearest_obstacle[1],
            "nearest_passage_term": nearest_passage[0],
            "nearest_passage_distance_m": nearest_passage[1],
        }

    def _semantic_resilience_local_state(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        semantic_nodes: List[Dict[str, Any]],
        current_policy_candidate: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        enabled = bool(self.config.semantic_resilience_shadow_enable)
        result: Dict[str, Any] = {
            "enabled": enabled,
            "valid": False,
            "reason": None,
            "recovery_trigger": False,
            "local_trap": False,
            "trigger_reasons": [],
            "recovery_context_tags": [],
        }
        if not enabled:
            result["reason"] = "disabled"
            return result
        try:
            row0, col0 = [int(v) for v in list(start_grid)[:2]]
        except (TypeError, ValueError):
            result["reason"] = "invalid_start_grid"
            return result

        radius = max(1, int(self.config.semantic_resilience_local_radius_cells))
        frontier_set = self._frontier_cell_set()
        buckets = ("front", "left", "right", "back")
        state_counts = {
            bucket: {"occupied": 0, "free": 0, "unknown": 0, "frontier": 0}
            for bucket in buckets
        }
        total_counts = {"occupied": 0, "free": 0, "unknown": 0, "frontier": 0}
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                row = row0 + dr
                col = col0 + dc
                if row < 0 or row >= self.gs or col < 0 or col >= self.gs:
                    continue
                direction = self._direction_to_cell([row0, col0], [row, col], yaw)
                bucket = direction.get("bucket")
                if bucket not in state_counts:
                    continue
                state = self._cell_state(row, col)
                state_counts[bucket][state] += 1
                total_counts[state] += 1
                if (row, col) in frontier_set:
                    state_counts[bucket]["frontier"] += 1
                    total_counts["frontier"] += 1

        observed_count = int(total_counts["occupied"] + total_counts["free"])
        total_count = int(total_counts["occupied"] + total_counts["free"] + total_counts["unknown"])
        local_occupied_ratio = (
            float(total_counts["occupied"] / observed_count) if observed_count else 0.0
        )
        local_unknown_ratio = (
            float(total_counts["unknown"] / total_count) if total_count else 0.0
        )
        blocked_bucket_threshold = float(self.config.semantic_resilience_blocked_bucket_threshold)
        bucket_pressure: Dict[str, float] = {}
        blocked_buckets = []
        for bucket in buckets:
            counts = state_counts[bucket]
            observed = int(counts["occupied"] + counts["free"])
            pressure = float(counts["occupied"] / observed) if observed else 0.0
            bucket_pressure[bucket] = pressure
            if observed > 0 and pressure >= blocked_bucket_threshold and counts["occupied"] >= 3:
                blocked_buckets.append(bucket)

        non_back_blocked = [bucket for bucket in blocked_buckets if bucket != "back"]
        frontier_escape_count = int(
            sum(state_counts[bucket]["frontier"] for bucket in ("front", "left", "right"))
        )
        current = current_policy_candidate or {}
        current_valid = bool(current.get("valid"))
        current_dead_zone = bool(current.get("semantic_dead_zone"))
        current_stagnation = bool(current.get("semantic_stagnation_active"))
        current_revisited = bool(current.get("points_to_revisited_region"))
        current_unsafe = bool(current_valid and not current.get("geometry_safe"))
        current_not_active_safe = bool(current_valid and not current.get("active_gate_safe"))
        current_policy_problem = bool(
            current_dead_zone
            or current_stagnation
            or current_revisited
            or current_unsafe
            or current_not_active_safe
        )
        min_observed = int(self.config.semantic_resilience_min_observed_cells)
        occupied_threshold = float(self.config.semantic_resilience_occupied_ratio_threshold)
        frontier_escape_threshold = int(self.config.semantic_resilience_frontier_escape_threshold)
        front_blocked = "front" in blocked_buckets
        side_blocked = "left" in blocked_buckets or "right" in blocked_buckets
        local_trap = bool(
            observed_count >= min_observed
            and (
                (front_blocked and side_blocked and frontier_escape_count <= frontier_escape_threshold)
                or len(non_back_blocked) >= 3
                or (
                    local_occupied_ratio >= occupied_threshold
                    and len(non_back_blocked) >= 2
                    and frontier_escape_count <= frontier_escape_threshold
                )
            )
        )

        semantic_counts = self._semantic_resilience_semantic_counts(
            [row0, col0],
            semantic_nodes,
        )
        trigger_reasons = []
        if local_trap:
            trigger_reasons.append("local_trap")
        if current_dead_zone:
            trigger_reasons.append("semantic_dead_zone")
        if current_stagnation:
            trigger_reasons.append("semantic_stagnation")
        if current_revisited:
            trigger_reasons.append("current_points_to_revisited_region")
        if current_unsafe:
            trigger_reasons.append("current_waypoint_occupied")
        if current_not_active_safe and not current_unsafe:
            trigger_reasons.append("current_waypoint_not_active_safe")
        if semantic_counts["obstacle_term_count"] > 0 and (local_trap or len(non_back_blocked) >= 2):
            trigger_reasons.append("semantic_obstacle_near_trap")

        recovery_trigger = bool(
            local_trap
            or (
                (current_dead_zone or current_stagnation)
                and (
                    frontier_escape_count <= frontier_escape_threshold
                    or len(non_back_blocked) >= 2
                    or current_policy_problem
                )
            )
            or (current_policy_problem and len(non_back_blocked) >= 2)
        )
        recovery_context_tags = []
        if local_trap:
            recovery_context_tags.append("spatial_constriction")
        if frontier_escape_count <= frontier_escape_threshold:
            recovery_context_tags.append("limited_frontier_escape")
        if current_dead_zone or current_stagnation:
            recovery_context_tags.append("semantic_uncertainty_or_stagnation")
        if current_revisited:
            recovery_context_tags.append("revisit_loop_risk")
        if current_unsafe or current_not_active_safe:
            recovery_context_tags.append("policy_memory_conflict")
        if semantic_counts["obstacle_term_count"] > 0:
            recovery_context_tags.append("semantic_obstacle_context")
        if semantic_counts["passage_term_count"] > 0:
            recovery_context_tags.append("semantic_passage_context")
        if current_policy_problem and not recovery_context_tags:
            recovery_context_tags.append("current_policy_risk")
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "radius_cells": int(radius),
                "radius_m": float(radius * self.cs),
                "observed_cell_count": int(observed_count),
                "total_checked_cell_count": int(total_count),
                "local_occupied_ratio_observed": float(local_occupied_ratio),
                "local_unknown_ratio": float(local_unknown_ratio),
                "local_frontier_count": int(total_counts["frontier"]),
                "frontier_escape_count": int(frontier_escape_count),
                "bucket_state_counts": state_counts,
                "bucket_occupied_pressure": bucket_pressure,
                "blocked_buckets": blocked_buckets,
                "non_back_blocked_bucket_count": int(len(non_back_blocked)),
                "current_policy_valid": bool(current_valid),
                "current_policy_problem": bool(current_policy_problem),
                "current_policy_dead_zone": bool(current_dead_zone),
                "current_policy_stagnation": bool(current_stagnation),
                "current_policy_revisited": bool(current_revisited),
                "current_policy_unsafe": bool(current_unsafe),
                "current_policy_not_active_safe": bool(current_not_active_safe),
                "semantic_obstacle_term_count": int(semantic_counts["obstacle_term_count"]),
                "semantic_passage_term_count": int(semantic_counts["passage_term_count"]),
                "nearest_semantic_obstacle_term": semantic_counts["nearest_obstacle_term"],
                "nearest_semantic_obstacle_distance_m": semantic_counts[
                    "nearest_obstacle_distance_m"
                ],
                "nearest_semantic_passage_term": semantic_counts["nearest_passage_term"],
                "nearest_semantic_passage_distance_m": semantic_counts[
                    "nearest_passage_distance_m"
                ],
                "local_trap": bool(local_trap),
                "recovery_trigger": bool(recovery_trigger),
                "trigger_reasons": trigger_reasons,
                "recovery_context_tags": recovery_context_tags,
            }
        )
        return result

    def _semantic_resilience_open_score(self, cell: Iterable[int], *, radius_cells: int = 6) -> float:
        try:
            row0, col0 = [int(v) for v in list(cell)[:2]]
        except (TypeError, ValueError):
            return 0.0
        radius = max(1, int(radius_cells))
        free_count = 0
        occupied_count = 0
        checked = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                row = row0 + dr
                col = col0 + dc
                if row < 0 or row >= self.gs or col < 0 or col >= self.gs:
                    continue
                checked += 1
                state = self._cell_state(row, col)
                if state == "free":
                    free_count += 1
                elif state == "occupied":
                    occupied_count += 1
        if checked <= 0:
            return 0.0
        free_ratio = float(free_count / checked)
        occupied_ratio = float(occupied_count / checked)
        return max(0.0, min(1.0, free_ratio * (1.0 - occupied_ratio)))

    @staticmethod
    def _normalized_entropy(counts: Iterable[int]) -> float:
        values = [max(0, int(value)) for value in counts]
        total = sum(values)
        active = sum(value > 0 for value in values)
        if total <= 0 or active <= 1:
            return 0.0
        entropy = 0.0
        for value in values:
            if value <= 0:
                continue
            probability = float(value / total)
            entropy -= probability * math.log(probability)
        return float(entropy / math.log(len(values)))

    @staticmethod
    def _relative_sector(dr: int, dc: int) -> str:
        if abs(dr) >= abs(dc):
            return "north" if dr < 0 else "south"
        return "west" if dc < 0 else "east"

    def _anchor_spatial_information(
        self,
        cell: Iterable[int],
        *,
        radius_cells: Optional[int] = None,
    ) -> Dict[str, Any]:
        row0, col0 = [int(v) for v in list(cell)[:2]]
        radius = max(
            1,
            int(
                self.config.semantic_resilience_anchor_feature_radius_cells
                if radius_cells is None
                else radius_cells
            ),
        )
        frontier_set = self._frontier_cell_set() if self.config.frontier_enable else set()
        state_counts = {"free": 0, "occupied": 0, "unknown": 0, "frontier": 0}
        sector_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
        sector_run_lengths = {key: 0 for key in sector_counts}
        checked = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                if dr * dr + dc * dc > radius * radius:
                    continue
                row = row0 + dr
                col = col0 + dc
                if row < 0 or row >= self.gs or col < 0 or col >= self.gs:
                    continue
                checked += 1
                state = self._cell_state(row, col)
                state_counts[state] += 1
                is_frontier = (row, col) in frontier_set
                if is_frontier:
                    state_counts["frontier"] += 1
                if state == "free" or is_frontier:
                    sector = self._relative_sector(dr, dc)
                    sector_counts[sector] += 1
                    # Count a direction as an executable exit only when the
                    # immediate ray has at least N consecutive traversable
                    # cells. This avoids every wide radius becoming 4 branches.
        # Measure consecutive traversable cells along the four cardinal rays.
        for sector, (dr_step, dc_step) in {
            "north": (-1, 0), "south": (1, 0),
            "west": (0, -1), "east": (0, 1),
        }.items():
            run = 0
            for distance in range(1, radius + 1):
                rr, cc = row0 + dr_step * distance, col0 + dc_step * distance
                if not (0 <= rr < self.gs and 0 <= cc < self.gs):
                    break
                state = self._cell_state(rr, cc)
                if state != "free" and (rr, cc) not in frontier_set:
                    break
                run += 1
            sector_run_lengths[sector] = run
        min_run = max(1, int(self.config.semantic_resilience_branch_min_run_cells))
        executable_exit_count = sum(
            run >= min_run for run in sector_run_lengths.values()
        )
        connected_component_count = 0
        traversable = {
            (row0 + dr, col0 + dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (dr or dc) and dr * dr + dc * dc <= radius * radius
            and 0 <= row0 + dr < self.gs and 0 <= col0 + dc < self.gs
            and (self._cell_state(row0 + dr, col0 + dc) == "free"
                 or (row0 + dr, col0 + dc) in frontier_set)
        }
        while traversable:
            connected_component_count += 1
            stack = [traversable.pop()]
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (nr, nc) in traversable:
                        traversable.remove((nr, nc))
                        stack.append((nr, nc))
        observed = state_counts["free"] + state_counts["occupied"]
        return {
            "radius_cells": int(radius),
            "checked_cell_count": int(checked),
            "free_count": int(state_counts["free"]),
            "occupied_count": int(state_counts["occupied"]),
            "unknown_count": int(state_counts["unknown"]),
            "frontier_count": int(state_counts["frontier"]),
            "visible_free_ratio": float(state_counts["free"] / max(1, checked)),
            "occupied_ratio_observed": float(
                state_counts["occupied"] / max(1, observed)
            ),
            "branch_count": int(executable_exit_count),
            "executable_exit_count": int(executable_exit_count),
            "connected_component_count": int(connected_component_count),
            "branch_depth_mean": float(
                sum(sector_run_lengths.values()) / max(1, len(sector_run_lengths))
            ),
            "direction_entropy": float(
                self._normalized_entropy(sector_counts.values())
            ),
            "sector_counts": dict(sector_counts),
        }

    def _anchor_trace_information(
        self,
        cell: Iterable[int],
        *,
        latest_step: Optional[int],
    ) -> Dict[str, Any]:
        row0, col0 = [int(v) for v in list(cell)[:2]]
        radius = max(1, int(self.config.semantic_resilience_cycle_radius_cells))
        window = max(2, int(self.config.semantic_resilience_cycle_window_steps))
        recent = self.pose_trace[-window:]
        near_flags = [
            bool(
                (int(item.get("row", row0)) - row0) ** 2
                + (int(item.get("col", col0)) - col0) ** 2
                <= radius * radius
            )
            for item in recent
        ]
        returns = 0
        was_near = False
        for is_near in near_flags:
            if is_near and not was_near:
                returns += 1
            was_near = is_near
        near_steps = []
        visit_start_steps = []
        visit_end_steps = []
        active_visit = False
        for item, is_near in zip(recent, near_flags):
            step = self._safe_int(item.get("step_id"))
            if is_near and step is not None:
                near_steps.append(step)
            if is_near and not active_visit:
                if step is not None:
                    visit_start_steps.append(step)
                active_visit = True
            elif not is_near and active_visit:
                if near_steps:
                    visit_end_steps.append(near_steps[-1])
                active_visit = False
        if active_visit and near_steps:
            visit_end_steps.append(near_steps[-1])
        last_visit_step = max(near_steps) if near_steps else None
        last_visit_age = None
        if latest_step is not None and last_visit_step is not None:
            last_visit_age = max(0, int(latest_step - last_visit_step))
        outgoing_sectors = set()
        for index, is_near in enumerate(near_flags[:-1]):
            if not is_near or near_flags[index + 1]:
                continue
            next_pose = recent[index + 1]
            dr = int(next_pose.get("row", row0)) - row0
            dc = int(next_pose.get("col", col0)) - col0
            if dr != 0 or dc != 0:
                outgoing_sectors.add(self._relative_sector(dr, dc))
        revisit_intervals = [
            int(b - a) for a, b in zip(visit_start_steps, visit_start_steps[1:]) if b > a
        ]
        cycle_count = max(0, returns - 1)
        return {
            "trace_window_steps": int(window),
            "near_pose_count": int(sum(near_flags)),
            "return_count": int(returns),
            "recent_cycle_count": int(cycle_count),
            "short_cycle_risk": float(min(1.0, cycle_count / 2.0)),
            "last_visit_step": last_visit_step,
            "last_visit_age_steps": last_visit_age,
            "outgoing_trace_direction_count": int(len(outgoing_sectors)),
            "outgoing_trace_directions": sorted(outgoing_sectors),
            "revisit_interval_steps": revisit_intervals,
        }

    def _anchor_semantic_information(
        self,
        cell: Iterable[int],
        semantic_nodes: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        anchor_xy = self._grid_to_xy(cell)
        radius_m = max(
            0.25, float(self.config.semantic_resilience_anchor_semantic_radius_m)
        )
        terms = set()
        instruction_terms = set()
        high_conf_terms = set()
        next_landmark_terms = set()
        passage_terms = set()
        for node in semantic_nodes:
            xy = node.get("xy")
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            node_xy = np.asarray([float(xy[0]), float(xy[1])], dtype=np.float32)
            if float(np.linalg.norm(node_xy - anchor_xy)) > radius_m:
                continue
            term = self._canonical_semantic_term(node.get("semantic_top_match"))
            if term:
                terms.add(term)
            if float(node.get("instruction_relevance", 0.0) or 0.0) > 0.0:
                instruction_terms.add(term or f"node:{len(instruction_terms)}")
            if bool(node.get("high_conf_semantic")):
                high_conf_terms.add(term or f"node:{len(high_conf_terms)}")
            if float(node.get("next_landmark_relevance", 0.0) or 0.0) > 0.0:
                next_landmark_terms.add(term or f"node:{len(next_landmark_terms)}")
            if self._semantic_resilience_term_kind(term) == "passage":
                passage_terms.add(term or f"node:{len(passage_terms)}")
        return {
            "semantic_radius_m": float(radius_m),
            "semantic_unique_count": int(len(terms)),
            "semantic_terms": sorted(terms),
            "instruction_relevant_count": int(len(instruction_terms)),
            "high_conf_landmark_count": int(len(high_conf_terms)),
            "next_landmark_count": int(len(next_landmark_terms)),
            "passage_semantic_count": int(len(passage_terms)),
        }

    def _recovery_anchor_features(
        self,
        cell: Iterable[int],
        start_grid: Iterable[int],
        *,
        source_type: str,
        source_node: Dict[str, Any],
        semantic_nodes: Iterable[Dict[str, Any]],
        latest_step: Optional[int],
    ) -> Dict[str, Any]:
        anchor_spatial = self._anchor_spatial_information(cell)
        current_spatial = self._anchor_spatial_information(start_grid)
        trace = self._anchor_trace_information(cell, latest_step=latest_step)
        semantic = self._anchor_semantic_information(cell, semantic_nodes)
        return {
            "recovery_feature_schema_version": "v3",
            "anchor_source_is_keyframe": bool(source_type == "keyframe"),
            "anchor_visible_free_ratio": anchor_spatial["visible_free_ratio"],
            "anchor_occupied_ratio_observed": anchor_spatial[
                "occupied_ratio_observed"
            ],
            "anchor_frontier_count": anchor_spatial["frontier_count"],
            "anchor_branch_count": anchor_spatial["branch_count"],
            "anchor_executable_exit_count": anchor_spatial["executable_exit_count"],
            "anchor_connected_component_count": anchor_spatial[
                "connected_component_count"
            ],
            "anchor_branch_depth_mean": anchor_spatial["branch_depth_mean"],
            "anchor_direction_entropy": anchor_spatial["direction_entropy"],
            "anchor_semantic_unique_count": semantic["semantic_unique_count"],
            "anchor_instruction_relevant_count": semantic[
                "instruction_relevant_count"
            ],
            "anchor_high_conf_landmark_count": semantic[
                "high_conf_landmark_count"
            ],
            "anchor_next_landmark_count": semantic["next_landmark_count"],
            "anchor_passage_semantic_count": semantic["passage_semantic_count"],
            "anchor_outgoing_trace_direction_count": trace[
                "outgoing_trace_direction_count"
            ],
            "anchor_last_visit_step": trace["last_visit_step"],
            "anchor_last_visit_age_steps": trace["last_visit_age_steps"],
            "anchor_recent_return_count": trace["return_count"],
            "anchor_recent_cycle_count": trace["recent_cycle_count"],
            "anchor_short_cycle_risk": trace["short_cycle_risk"],
            "anchor_revisit_interval_min_steps": (
                min(trace["revisit_interval_steps"])
                if trace["revisit_interval_steps"] else None
            ),
            "anchor_revisit_interval_mean_steps": (
                float(sum(trace["revisit_interval_steps"]) / len(trace["revisit_interval_steps"]))
                if trace["revisit_interval_steps"] else None
            ),
            "current_visible_free_ratio": current_spatial["visible_free_ratio"],
            "current_frontier_count": current_spatial["frontier_count"],
            "current_branch_count": current_spatial["branch_count"],
            "current_executable_exit_count": current_spatial[
                "executable_exit_count"
            ],
            "current_connected_component_count": current_spatial[
                "connected_component_count"
            ],
            "current_branch_depth_mean": current_spatial["branch_depth_mean"],
            "current_direction_entropy": current_spatial["direction_entropy"],
            "current_to_anchor_free_ratio_gain": float(
                anchor_spatial["visible_free_ratio"]
                - current_spatial["visible_free_ratio"]
            ),
            "current_to_anchor_frontier_gain": int(
                anchor_spatial["frontier_count"] - current_spatial["frontier_count"]
            ),
            "current_to_anchor_branch_gain": int(
                anchor_spatial["branch_count"] - current_spatial["branch_count"]
            ),
            "current_to_anchor_direction_entropy_gain": float(
                anchor_spatial["direction_entropy"]
                - current_spatial["direction_entropy"]
            ),
            "anchor_semantic_top_match": source_node.get("semantic_top_match"),
            "anchor_semantic_top_score": source_node.get("semantic_top_score"),
            "anchor_high_conf_semantic": bool(source_node.get("high_conf_semantic")),
        }

    def _semantic_resilience_backtrack_candidates(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        current_angle: Any,
        current_goal_grid: Any,
        semantic_nodes: List[Dict[str, Any]],
        goal_progress_state: Optional[Dict[str, Any]],
        resilience_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not bool(self.config.semantic_resilience_shadow_enable):
            return []
        if not bool((resilience_state or {}).get("recovery_trigger")):
            return []
        if len(self.pose_trace) < 2:
            return []
        min_distance = max(0.0, float(self.config.semantic_resilience_min_backtrack_distance_m))
        max_distance = max(min_distance, float(self.config.semantic_resilience_max_backtrack_distance_m))
        min_step_gap = max(0, int(self.config.semantic_resilience_backtrack_min_step_gap))
        source_score = float(self.config.semantic_resilience_candidate_source_score)
        weight = float(self.config.semantic_resilience_backtrack_score_weight)
        start_xy = self._grid_to_xy(start_grid)
        latest_pose = self.pose_trace[-1]
        latest_step = self._safe_int(latest_pose.get("step_id"))

        sources: List[Tuple[str, Dict[str, Any]]] = []
        for node in reversed(self.keyframes):
            sources.append(("keyframe", node))
        for node in reversed(self.pose_trace[:-1]):
            sources.append(("pose_trace", node))

        candidates: List[Dict[str, Any]] = []
        seen_cells = set()
        for source_type, node in sources:
            row = self._safe_int(node.get("row"))
            col = self._safe_int(node.get("col"))
            if row is None or col is None:
                continue
            cell = (int(row), int(col))
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            if self._cell_state(row, col) == "occupied":
                continue
            xy = self._grid_to_xy([row, col])
            distance = float(np.linalg.norm(xy - start_xy))
            if distance < min_distance or distance > max_distance:
                continue
            step = self._safe_int(node.get("step_id"))
            step_gap = None
            if latest_step is not None and step is not None:
                step_gap = int(latest_step - step)
                if step_gap < min_step_gap:
                    continue

            semantic = self._semantic_evidence_for_cell(
                [row, col],
                start_grid,
                yaw,
                semantic_nodes,
            )
            candidate = self._build_query_candidate(
                [row, col],
                "resilience_backtrack",
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                source_score=source_score,
                semantic=semantic,
                goal_progress_state=goal_progress_state,
            )
            if candidate is None:
                continue
            semantic_counts = self._semantic_resilience_semantic_counts(
                [row, col],
                semantic_nodes,
                radius_m=max(1.0, float(self.config.candidate_probe_semantic_bind_radius_m)),
            )
            open_score = self._semantic_resilience_open_score([row, col])
            distance_center = 0.5 * (min_distance + max_distance)
            distance_score = 1.0 - min(
                1.0,
                abs(distance - distance_center) / max(1e-6, distance_center - min_distance),
            )
            trap_score = 1.0 if resilience_state.get("local_trap") else 0.0
            dead_zone_score = 1.0 if (
                resilience_state.get("current_policy_dead_zone")
                or resilience_state.get("current_policy_stagnation")
            ) else 0.0
            passage_score = 1.0 if semantic_counts["passage_term_count"] > 0 else 0.0
            obstacle_context_score = 1.0 if resilience_state.get("semantic_obstacle_term_count", 0) else 0.0
            resilience_score = max(
                0.0,
                min(
                    1.0,
                    0.30 * trap_score
                    + 0.25 * dead_zone_score
                    + 0.20 * open_score
                    + 0.10 * passage_score
                    + 0.10 * obstacle_context_score
                    + 0.05 * distance_score,
                ),
            )
            candidate["score"] = float(candidate.get("score", 0.0) or 0.0) + weight * resilience_score
            recovery_features = self._recovery_anchor_features(
                [row, col],
                start_grid,
                source_type=source_type,
                source_node=node,
                semantic_nodes=semantic_nodes,
                latest_step=latest_step,
            )
            candidate.update(
                {
                    "semantic_resilience_candidate": True,
                    "semantic_resilience_recommended": True,
                    "semantic_resilience_reason": "backtrack_to_recent_safe_observation",
                    "semantic_resilience_source": source_type,
                    "semantic_resilience_source_step_id": step,
                    "semantic_resilience_source_world_z": float(
                        node.get("z", 0.0) or 0.0
                    ),
                    "semantic_resilience_step_gap": step_gap,
                    "semantic_resilience_backtrack_distance_m": float(distance),
                    "semantic_resilience_open_score": float(open_score),
                    "semantic_resilience_distance_score": float(distance_score),
                    "semantic_resilience_score": float(resilience_score),
                    "semantic_resilience_trigger_reasons": list(
                        (resilience_state or {}).get("trigger_reasons") or []
                    ),
                    "semantic_resilience_recovery_context_tags": list(
                        (resilience_state or {}).get("recovery_context_tags") or []
                    ),
                    "semantic_resilience_local_trap": bool(
                        (resilience_state or {}).get("local_trap")
                    ),
                    "semantic_resilience_recovery_trigger": bool(
                        (resilience_state or {}).get("recovery_trigger")
                    ),
                    "semantic_resilience_active_safe": bool(candidate.get("geometry_safe")),
                    "semantic_resilience_obstacle_term_count": int(
                        semantic_counts["obstacle_term_count"]
                    ),
                    "semantic_resilience_passage_term_count": int(
                        semantic_counts["passage_term_count"]
                    ),
                    "semantic_resilience_nearest_obstacle_term": semantic_counts[
                        "nearest_obstacle_term"
                    ],
                    "semantic_resilience_nearest_obstacle_distance_m": semantic_counts[
                        "nearest_obstacle_distance_m"
                    ],
                    "semantic_resilience_nearest_passage_term": semantic_counts[
                        "nearest_passage_term"
                    ],
                    "semantic_resilience_nearest_passage_distance_m": semantic_counts[
                        "nearest_passage_distance_m"
                    ],
                    **recovery_features,
                }
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                bool(item.get("geometry_safe")),
                float(item.get("semantic_resilience_score", 0.0) or 0.0),
                float(item.get("semantic_resilience_open_score", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return candidates[:1]

    def bfs_to_semantic_frontier(
        self,
        *,
        obs: Optional[Dict[str, Any]] = None,
        current_waypoint_decision: Optional[Dict[str, Any]] = None,
        max_action_steps: int = 8,
        forward_step_m: float = 0.25,
        frontier_sample_limit: int = 5000,
        require_instruction_relevant: bool = True,
        allow_fallback_target_frontier: bool = False,
    ) -> Dict[str, Any]:
        """Shadow query: BFS through known free space to a semantic frontier.

        `max_action_steps` is interpreted in Habitat forward-action units, not
        grid cells. With the default 0.25m action and 0.05m cells, 8 action
        steps allow a 40-cell BFS radius.
        """
        decision = dict(current_waypoint_decision or {})
        result: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "valid": False,
            "reason": None,
            "bfs_reachable": False,
            "bfs_path": [],
            "bfs_path_preview": [],
            "bfs_target_grid": None,
            "bfs_target_direction": None,
            "bfs_target_instruction_relevant": False,
        }
        if not self.enabled:
            result["reason"] = "disabled"
            return result
        pose_state = self._current_pose_state(obs or {})
        if pose_state is None:
            result["reason"] = "missing_pose_or_memory"
            return result
        start_grid = pose_state.get("grid")
        if not start_grid or len(start_grid) < 2:
            result["reason"] = "missing_start_grid"
            return result
        start = (int(start_grid[0]), int(start_grid[1]))
        yaw = float(pose_state.get("yaw", 0.0) or 0.0)
        forward_step_m = max(float(self.cs), float(forward_step_m or 0.25))
        max_action_steps = max(1, int(max_action_steps))
        max_grid_steps = max(1, int(math.ceil(max_action_steps * forward_step_m / max(1e-6, self.cs))))

        goal_progress_state = self._semantic_goal_progress_state()
        semantic_nodes = self._semantic_memory_nodes(
            start,
            yaw,
            goal_progress_state=goal_progress_state,
        )
        current_angle = decision.get("waypoint_direction_angle_deg")
        current_goal_grid = decision.get("goal_grid")

        raw_targets: List[Dict[str, Any]] = []
        frontier_cells = self.get_frontier_cells(sample_limit=max(0, int(frontier_sample_limit)))
        for cell in frontier_cells:
            semantic = self._semantic_evidence_for_cell(cell, start, yaw, semantic_nodes)
            candidate = self._build_query_candidate(
                cell,
                "frontier",
                start,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                source_score=1.0,
                semantic=semantic,
                goal_progress_state=goal_progress_state,
            )
            if candidate is None:
                continue
            if not candidate.get("geometry_safe"):
                continue
            raw_targets.append(candidate)

        def _is_instruction_target(item: Dict[str, Any]) -> bool:
            return bool(
                item.get("instruction_relevant")
                or float(item.get("next_landmark_relevance", 0.0) or 0.0) > 0.0
                or float(item.get("semantic_progress_score", 0.0) or 0.0) > 0.0
                or float(item.get("semantic_relevance_score", 0.0) or 0.0) > 0.0
            )

        instruction_targets = [item for item in raw_targets if _is_instruction_target(item)]
        fallback_targets = [
            item
            for item in raw_targets
            if item.get("target_frontier_candidate") or item.get("target_frontier_escape_candidate")
        ]
        if require_instruction_relevant:
            targets = instruction_targets
            if not targets and allow_fallback_target_frontier:
                targets = fallback_targets
        else:
            targets = instruction_targets or fallback_targets or raw_targets

        result.update(
            {
                "valid": True,
                "reason": "ok",
                "start_grid": [int(start[0]), int(start[1])],
                "start_yaw": float(yaw),
                "max_action_steps": int(max_action_steps),
                "forward_step_m": float(forward_step_m),
                "max_grid_steps": int(max_grid_steps),
                "frontier_sample_count": int(len(frontier_cells)),
                "raw_target_count": int(len(raw_targets)),
                "instruction_target_count": int(len(instruction_targets)),
                "fallback_target_count": int(len(fallback_targets)),
                "target_count": int(len(targets)),
                "require_instruction_relevant": bool(require_instruction_relevant),
                "allow_fallback_target_frontier": bool(allow_fallback_target_frontier),
                "semantic_memory_node_count": int(len(semantic_nodes)),
                "goal_progress_enabled": bool(goal_progress_state.get("enabled")),
                "goal_progress_next_landmark": goal_progress_state.get("next_landmark"),
                "goal_progress_completed_landmarks": goal_progress_state.get("completed_landmarks"),
            }
        )
        if not targets:
            result["reason"] = "no_instruction_relevant_frontier" if require_instruction_relevant else "no_frontier_targets"
            return result

        target_by_cell: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for item in sorted(targets, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True):
            grid = item.get("grid")
            if not grid or len(grid) < 2:
                continue
            target_by_cell.setdefault((int(grid[0]), int(grid[1])), item)
        if not target_by_cell:
            result["reason"] = "no_target_cells"
            return result

        free_cells = set(self.free2d_counts.keys()) - set(self.occ2d_counts.keys())
        if not free_cells:
            result["reason"] = "no_free_cells"
            return result
        queue = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        depth: Dict[Tuple[int, int], int] = {start: 0}
        reached: Optional[Tuple[int, int]] = None
        visited_count = 0
        while queue:
            cell = queue.popleft()
            visited_count += 1
            if cell in target_by_cell and cell != start:
                reached = cell
                break
            cur_depth = int(depth[cell])
            if cur_depth >= max_grid_steps:
                continue
            for nbr in self._neighbors2d(cell[0], cell[1]):
                nbr = (int(nbr[0]), int(nbr[1]))
                if nbr in parent:
                    continue
                if nbr not in free_cells and nbr not in target_by_cell:
                    continue
                if nbr in self.occ2d_counts:
                    continue
                parent[nbr] = cell
                depth[nbr] = cur_depth + 1
                queue.append(nbr)

        result["bfs_visited_cell_count"] = int(visited_count)
        if reached is None:
            nearest = None
            nearest_dist = None
            for cell in target_by_cell:
                dist = abs(cell[0] - start[0]) + abs(cell[1] - start[1])
                if nearest_dist is None or dist < nearest_dist:
                    nearest = cell
                    nearest_dist = dist
            result.update(
                {
                    "bfs_reachable": False,
                    "reason": "no_reachable_target_within_budget",
                    "nearest_target_grid": None if nearest is None else [int(nearest[0]), int(nearest[1])],
                    "nearest_target_manhattan_cells": nearest_dist,
                }
            )
            return result

        path = []
        cur: Optional[Tuple[int, int]] = reached
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        target = target_by_cell[reached]
        direction = self._direction_to_cell(start, reached, yaw)
        path_edges = max(0, len(path) - 1)
        path_m = float(path_edges * self.cs)
        action_steps_est = int(math.ceil(path_m / max(1e-6, forward_step_m)))
        preview = path[:8] + ([path[-1]] if len(path) > 8 else [])
        result.update(
            {
                "bfs_reachable": True,
                "reason": "ok",
                "bfs_path": [[int(r), int(c)] for r, c in path],
                "bfs_path_preview": [[int(r), int(c)] for r, c in preview],
                "bfs_path_cell_count": int(len(path)),
                "bfs_path_edge_count": int(path_edges),
                "bfs_path_m": float(path_m),
                "bfs_action_steps_estimate": int(action_steps_est),
                "bfs_target_grid": [int(reached[0]), int(reached[1])],
                "bfs_target_direction": direction.get("bucket"),
                "bfs_target_direction_angle_deg": direction.get("angle_deg"),
                "bfs_target_distance_m": direction.get("distance_m"),
                "bfs_target_instruction_relevant": bool(_is_instruction_target(target)),
                "bfs_target_candidate": self._jsonable(target),
            }
        )
        return result

    def record_candidate_selection_event(
        self,
        *,
        candidate_event: Dict[str, Any],
        selection: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_type": "occ_memory_candidate_selection",
            **self.episode_meta,
            **dict(context or {}),
            "enabled": bool(self.enabled and self.config.candidate_probe_enable),
            "candidate_event_valid": bool(candidate_event.get("valid")),
            "candidate_count": int(candidate_event.get("candidate_count", 0) or 0),
            "selection_status": selection.get("status"),
            "selection_output": selection.get("output"),
            "selection_valid": bool(selection.get("valid")),
            "selection_none": bool(selection.get("none")),
            "selection_choice": selection.get("choice"),
            "selection_reason": selection.get("reason"),
            "selection_coordinate_numbers": selection.get("coordinate_numbers"),
            "selection_coordinate_distance_px": selection.get("coordinate_distance_px"),
            "selection_coordinate_convention": selection.get("coordinate_convention"),
            "selection_direction_token": selection.get("direction_token"),
            "selected_candidate": selection.get("selected_candidate"),
        }
        self.candidate_selection_events.append(event)
        self._write_event(event)
        return event

    def _current_pose_state(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Counterfactual probes keep the official trace frozen. They must
        # still use the temporary observation yaw during a path re-audit;
        # normal callers retain the persistent pose contract by default.
        prefer_observation_pose = bool(obs.get("_prefer_observation_pose"))
        if self.pose_trace and not prefer_observation_pose:
            pose = self.pose_trace[-1]
            return {
                "grid": [int(pose["row"]), int(pose["col"])],
                "xy": [float(pose["x"]), float(pose["y"])],
                "yaw": float(pose["yaw"]),
            }
        pose_tf = self._pose_from_obs(obs)
        if pose_tf is None:
            return None
        if self.config.center_on_first_pose and self.init_base_tf is None:
            return None
        rel_base_tf = self._relative_base_tf(pose_tf)
        row, col, yaw = self._pose_to_grid(rel_base_tf)
        return {
            "grid": [int(row), int(col)],
            "xy": [float(rel_base_tf[0, 3]), float(rel_base_tf[1, 3])],
            "yaw": float(yaw),
        }

    def score_local_xy_trajectory(
        self,
        local_xy,
        obs: Optional[Dict[str, Any]] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score a local XY trajectory against persistent sparse occupancy.

        The local trajectory uses the same convention as NextDiT action samples:
        x is forward from the current pose and y is lateral in the current local
        frame. This probe is shadow-only and does not mutate memory.
        """
        context = dict(context or {})
        result = {
            "enabled": bool(self.enabled),
            "valid": False,
            "reason": None,
        }
        if not self.enabled:
            result["reason"] = "disabled"
            return result
        pose_state = self._current_pose_state(obs or {})
        if pose_state is None:
            result["reason"] = "missing_pose_or_memory"
            return result
        try:
            points = np.asarray(local_xy, dtype=np.float32)
        except (TypeError, ValueError):
            result["reason"] = "invalid_trajectory"
            return result
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            result["reason"] = "invalid_trajectory_shape"
            return result

        x0, y0 = [float(v) for v in pose_state["xy"][:2]]
        yaw = float(pose_state["yaw"])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        checked_cells = []
        prev_world = None
        stride = max(float(self.cs), 0.05)
        for point in points[:, :2]:
            lx = float(point[0])
            ly = float(point[1])
            world = np.array(
                [
                    x0 + cos_yaw * lx - sin_yaw * ly,
                    y0 + sin_yaw * lx + cos_yaw * ly,
                ],
                dtype=np.float32,
            )
            if prev_world is None:
                prev_world = world
                continue
            delta = world - prev_world
            distance = float(np.linalg.norm(delta))
            sample_count = max(1, int(math.ceil(distance / stride)))
            for idx in range(1, sample_count + 1):
                alpha = float(idx) / float(sample_count)
                sample_xy = prev_world + alpha * delta
                row, col = self._xy_to_grid_cell(sample_xy[0], sample_xy[1])
                if 0 <= row < self.gs and 0 <= col < self.gs:
                    checked_cells.append((int(row), int(col)))
            prev_world = world

        unique_cells = []
        seen = set()
        for cell in checked_cells:
            if cell not in seen:
                seen.add(cell)
                unique_cells.append(cell)

        occupied_cells = []
        free_count = 0
        unknown_count = 0
        for row, col in unique_cells:
            state = self._cell_state(row, col)
            if state == "occupied":
                occupied_cells.append((row, col))
            elif state == "free":
                free_count += 1
            else:
                unknown_count += 1

        checked_count = len(unique_cells)
        occupied_count = len(occupied_cells)
        unknown_ratio = float(unknown_count / checked_count) if checked_count else 1.0
        occupied_ratio = float(occupied_count / checked_count) if checked_count else 0.0
        end_row, end_col = unique_cells[-1] if unique_cells else pose_state["grid"][:2]
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "checked_cell_count": int(checked_count),
                "occupied_hit_count": int(occupied_count),
                "free_hit_count": int(free_count),
                "unknown_hit_count": int(unknown_count),
                "occupied_hit_ratio": float(occupied_ratio),
                "unknown_hit_ratio": float(unknown_ratio),
                "would_reject": bool(occupied_count > 0),
                "has_unknown": bool(unknown_count > 0),
                "start_grid": [int(v) for v in pose_state["grid"][:2]],
                "end_grid": [int(end_row), int(end_col)],
                "sampled_point_count": int(points.shape[0]),
                "local_endpoint_xy": [float(points[-1, 0]), float(points[-1, 1])],
                "occupied_cells_preview": [[int(r), int(c)] for r, c in occupied_cells[:8]],
                "context": self._jsonable(context),
            }
        )
        return result

    def _frontier_query_candidates(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        current_angle: Any,
        current_goal_grid: Any,
        semantic_nodes: List[Dict[str, Any]],
        goal_progress_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        best_by_bucket: Dict[str, Dict[str, Any]] = {}
        frontiers = self.get_frontier_cells(
            sample_limit=int(self.config.candidate_probe_frontier_sample_limit)
        )
        for cell in frontiers:
            semantic = self._semantic_evidence_for_cell(cell, start_grid, yaw, semantic_nodes)
            candidate = self._build_query_candidate(
                cell,
                "frontier",
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                source_score=1.0,
                semantic=semantic,
                goal_progress_state=goal_progress_state,
            )
            if candidate is None:
                continue
            bucket = str(candidate.get("direction_bucket"))
            if self.config.candidate_probe_exclude_back_frontier and bucket == "back":
                continue
            previous = best_by_bucket.get(bucket)
            if previous is None or float(candidate["score"]) > float(previous["score"]):
                best_by_bucket[bucket] = candidate
        return list(best_by_bucket.values())

    def _open_floor_query_candidates(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        current_angle: Any,
        current_goal_grid: Any,
        semantic_nodes: List[Dict[str, Any]],
        goal_progress_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cells = [cell for cell in self.free2d_counts.keys() if cell not in self.occ2d_counts]
        limit = int(self.config.candidate_probe_free_sample_limit)
        if limit > 0 and len(cells) > limit:
            ids = np.linspace(0, len(cells) - 1, limit).astype(np.int64)
            cells = [cells[int(idx)] for idx in ids]
        best_by_bucket: Dict[str, Dict[str, Any]] = {}
        for cell in cells:
            semantic = self._semantic_evidence_for_cell(cell, start_grid, yaw, semantic_nodes)
            candidate = self._build_query_candidate(
                cell,
                "open_floor",
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                source_score=0.25,
                semantic=semantic,
                goal_progress_state=goal_progress_state,
            )
            if candidate is None:
                continue
            bucket = str(candidate.get("direction_bucket"))
            if bucket == "back":
                continue
            previous = best_by_bucket.get(bucket)
            if previous is None or float(candidate["score"]) > float(previous["score"]):
                best_by_bucket[bucket] = candidate
        return list(best_by_bucket.values())

    def _semantic_query_candidates(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        current_angle: Any,
        current_goal_grid: Any,
        semantic_nodes: List[Dict[str, Any]],
        goal_progress_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.config.candidate_probe_semantic_enable:
            return []
        max_items = max(0, int(self.config.candidate_probe_semantic_max_candidates))
        if max_items <= 0:
            return []
        candidates = []
        for node in sorted(
            semantic_nodes,
            key=lambda item: float(item.get("semantic_candidate_score", 0.0) or 0.0),
            reverse=True,
        ):
            if not node.get("grid"):
                continue
            candidate = self._build_query_candidate(
                node["grid"],
                "semantic_keyframe",
                start_grid,
                yaw,
                current_angle=current_angle,
                current_goal_grid=current_goal_grid,
                source_score=0.75,
                semantic=node,
                goal_progress_state=goal_progress_state,
            )
            if candidate is not None:
                candidates.append(candidate)
            if len(candidates) >= max_items:
                break
        return candidates

    def _semantic_memory_nodes(
        self,
        start_grid: Iterable[int],
        yaw: float,
        *,
        goal_progress_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.config.candidate_probe_semantic_enable:
            return []
        min_score = float(self.config.candidate_probe_semantic_min_score)
        high_conf_only = bool(self.config.candidate_probe_semantic_high_conf_only)
        instruction_terms = set(self._instruction_terms())
        goal_progress_state = goal_progress_state or self._semantic_goal_progress_state()
        nodes: List[Dict[str, Any]] = []
        seen = set()
        sources = []
        for event in self.semantic_events:
            sources.append(("semantic_event", event))
        for node in self.keyframes:
            sources.append(("keyframe", node))
        for anchor in self.semantic_anchors:
            sources.append(("semantic_anchor", anchor))
        for source_type, item in sources:
            term = item.get("top_match") or item.get("semantic_top_match")
            if not term:
                continue
            score = item.get("top_score")
            if score is None:
                score = item.get("semantic_top_score")
            try:
                score_float = float(score)
            except (TypeError, ValueError):
                score_float = 0.0
            high_conf = bool(item.get("high_conf_semantic"))
            if high_conf_only and not high_conf:
                continue
            if score_float < min_score and not high_conf:
                continue
            grid = item.get("pose_grid") or item.get("grid")
            if grid is None and item.get("row") is not None and item.get("col") is not None:
                grid = [item.get("row"), item.get("col")]
            if not grid or len(grid) < 2:
                continue
            row, col = int(grid[0]), int(grid[1])
            key = (row, col, str(term), source_type)
            if key in seen:
                continue
            seen.add(key)
            direction = self._direction_to_cell(start_grid, [row, col], yaw)
            relevance = self._semantic_instruction_relevance(term, instruction_terms)
            recent_novelty = self._semantic_recent_novelty(term)
            goal_progress = self._semantic_goal_progress_for_term(term, goal_progress_state)
            confidence_score = min(1.0, max(0.0, (score_float - min_score) / max(1e-6, 0.35 - min_score)))
            semantic_candidate_score = (
                1.20 * relevance
                + 0.45 * confidence_score
                + (0.25 if high_conf else 0.0)
                + 0.25 * recent_novelty
                - 0.03 * float(direction.get("distance_m", 0.0) or 0.0)
            )
            if goal_progress_state.get("enabled"):
                semantic_candidate_score += (
                    float(self.config.candidate_probe_goal_progress_next_weight)
                    * float(goal_progress.get("next_landmark_relevance", 0.0) or 0.0)
                    + (
                        float(self.config.candidate_probe_goal_progress_high_conf_bonus)
                        if high_conf and float(goal_progress.get("next_landmark_relevance", 0.0) or 0.0) > 0.0
                        else 0.0
                    )
                    - float(self.config.candidate_probe_goal_progress_completed_penalty)
                    * float(goal_progress.get("completed_landmark_penalty", 0.0) or 0.0)
                    - float(self.config.candidate_probe_goal_progress_repeated_penalty)
                    * float(goal_progress.get("repeated_semantic_penalty", 0.0) or 0.0)
                )
            xy = self._grid_to_xy([row, col])
            nodes.append(
                {
                    "source_type": source_type,
                    "step_id": item.get("step_id"),
                    "grid": [int(row), int(col)],
                    "xy": [float(xy[0]), float(xy[1])],
                    "semantic_top_match": str(term),
                    "semantic_top_score": score_float,
                    "semantic_top_margin": item.get("top_margin") or item.get("semantic_top_margin"),
                    "high_conf_semantic": high_conf,
                    "instruction_relevance": float(relevance),
                    "semantic_recent_novelty": float(recent_novelty),
                    "matched_landmark": goal_progress.get("matched_landmark"),
                    "landmark_status": goal_progress.get("landmark_status"),
                    "next_landmark_relevance": float(
                        goal_progress.get("next_landmark_relevance", 0.0) or 0.0
                    ),
                    "completed_landmark_penalty": float(
                        goal_progress.get("completed_landmark_penalty", 0.0) or 0.0
                    ),
                    "repeated_semantic_penalty": float(
                        goal_progress.get("repeated_semantic_penalty", 0.0) or 0.0
                    ),
                    "semantic_progress_score": float(
                        goal_progress.get("semantic_progress_score", 0.0) or 0.0
                    ),
                    "semantic_candidate_score": float(semantic_candidate_score),
                    "distance_m": direction.get("distance_m"),
                    "direction_bucket": direction.get("bucket"),
                    "direction_angle_deg": direction.get("angle_deg"),
                }
            )
        return nodes

    def _semantic_evidence_for_cell(
        self,
        cell: Iterable[int],
        start_grid: Iterable[int],
        yaw: float,
        semantic_nodes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not semantic_nodes:
            return None
        candidate_direction = self._direction_to_cell(start_grid, cell, yaw)
        candidate_angle = candidate_direction.get("angle_deg")
        candidate_xy = self._grid_to_xy(cell)
        bind_radius = max(0.0, float(self.config.candidate_probe_semantic_bind_radius_m))
        direction_threshold = max(
            1.0,
            min(180.0, float(self.config.candidate_probe_semantic_direction_match_degrees)),
        )
        best = None
        best_score = None
        for node in semantic_nodes:
            node_xy = np.asarray(node.get("xy", [0.0, 0.0])[:2], dtype=np.float32)
            spatial_distance = float(np.linalg.norm(candidate_xy - node_xy))
            node_angle = node.get("direction_angle_deg")
            angle_distance = None
            direction_aligned = False
            try:
                angle_distance = self._angle_distance_degrees(float(candidate_angle), float(node_angle))
                direction_aligned = angle_distance <= direction_threshold
            except (TypeError, ValueError):
                pass
            nearby = spatial_distance <= bind_radius
            if not nearby and not direction_aligned:
                continue
            relevance = float(node.get("instruction_relevance", 0.0) or 0.0)
            novelty = float(node.get("semantic_recent_novelty", 0.0) or 0.0)
            semantic_score = float(node.get("semantic_candidate_score", 0.0) or 0.0)
            next_relevance = float(node.get("next_landmark_relevance", 0.0) or 0.0)
            completed_penalty = float(node.get("completed_landmark_penalty", 0.0) or 0.0)
            repeated_penalty = float(node.get("repeated_semantic_penalty", 0.0) or 0.0)
            bind_score = (
                semantic_score
                + 0.45 * relevance
                + 0.20 * novelty
                + (
                    0.65 * next_relevance
                    - 0.35 * completed_penalty
                    - 0.25 * repeated_penalty
                    if self.config.candidate_probe_goal_progress_enable
                    else 0.0
                )
                + (0.25 if nearby else 0.0)
                + (0.20 if direction_aligned else 0.0)
                - 0.04 * spatial_distance
                - 0.002 * float(angle_distance if angle_distance is not None else 90.0)
            )
            if best_score is None or bind_score > best_score:
                best_score = bind_score
                best = dict(node)
                best["bind_score"] = float(bind_score)
                best["bind_spatial_distance_m"] = float(spatial_distance)
                best["bind_angle_distance_deg"] = angle_distance
                best["bind_nearby"] = bool(nearby)
                best["bind_direction_aligned"] = bool(direction_aligned)
        return best

    def _instruction_terms(self) -> List[str]:
        instruction = str(self.episode_meta.get("instruction") or "").lower()
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "at",
            "by",
            "go",
            "in",
            "into",
            "is",
            "keep",
            "left",
            "right",
            "of",
            "on",
            "past",
            "room",
            "straight",
            "the",
            "then",
            "to",
            "toward",
            "under",
            "wait",
            "walk",
            "with",
        }
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", instruction)
            if len(token) >= 3 and token not in stopwords
        ]
        phrases = []
        common_phrases = (
            "living room",
            "dining room",
            "bedroom",
            "bathroom",
            "kitchen",
            "hallway",
            "entrance",
            "balcony",
            "stairs",
            "staircase",
            "doorway",
            "archway",
            "office",
            "closet",
            "corridor",
        )
        for phrase in common_phrases:
            if phrase in instruction:
                phrases.append(phrase)
        return sorted(set(tokens + phrases))

    def _instruction_landmark_sequence(self) -> List[str]:
        instruction = str(self.episode_meta.get("instruction") or "").lower()
        text = f" {instruction} "
        alias_items = dict(_GOAL_PROGRESS_LANDMARK_ALIASES)
        for term in _GOAL_PROGRESS_LANDMARK_TERMS:
            alias_items.setdefault(term.replace("_", " "), term)
        candidates: List[Tuple[int, str, str]] = []
        for phrase, canonical in alias_items.items():
            pattern = r"(?<![a-z0-9])" + re.escape(str(phrase).lower()) + r"(?![a-z0-9])"
            match = re.search(pattern, text)
            if match:
                candidates.append((match.start(), str(phrase), self._canonical_semantic_term(canonical)))
        candidates.sort(key=lambda item: (item[0], -len(item[1])))
        has_specific_room = any(
            canonical in _GOAL_PROGRESS_SPECIFIC_ROOMS for _, _, canonical in candidates
        )
        sequence = []
        seen = set()
        for _, _, canonical in candidates:
            if canonical == "room" and has_specific_room:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            sequence.append(canonical)
        return sequence

    def _canonical_semantic_term(self, term: Any) -> str:
        text = str(term or "").lower().strip().replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        if not text:
            return ""
        return _GOAL_PROGRESS_LANDMARK_ALIASES.get(text, text)

    def _semantic_goal_progress_state(self) -> Dict[str, Any]:
        enabled = bool(
            self.config.candidate_probe_semantic_enable
            and self.config.candidate_probe_goal_progress_enable
        )
        sequence = self._instruction_landmark_sequence() if enabled else []
        completed: List[str] = []
        completed_set = set()
        threshold = float(self.config.candidate_probe_goal_progress_seen_score_threshold)
        for event in self.semantic_events:
            term = event.get("top_match")
            canonical = self._canonical_semantic_term(term)
            if not canonical:
                continue
            try:
                score = float(event.get("top_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if not bool(event.get("high_conf_semantic")) and score < threshold:
                continue
            for landmark in sequence:
                if landmark in completed_set:
                    continue
                if self._semantic_landmark_match_score(canonical, landmark) > 0.0:
                    completed_set.add(landmark)
                    completed.append(landmark)
                    break
        next_landmark = None
        next_index = None
        for idx, landmark in enumerate(sequence):
            if landmark not in completed_set:
                next_landmark = landmark
                next_index = idx
                break
        window = max(1, int(self.config.attribution_recent_semantic_window))
        recent_counts: Dict[str, int] = defaultdict(int)
        for event in self.semantic_events[-window:]:
            canonical = self._canonical_semantic_term(event.get("top_match"))
            if canonical:
                recent_counts[canonical] += 1
        recent_repeated = sorted([term for term, count in recent_counts.items() if count >= 2])
        return {
            "enabled": bool(enabled),
            "landmark_sequence": sequence,
            "completed_landmarks": completed,
            "next_landmark": next_landmark,
            "next_landmark_index": next_index,
            "recent_repeated_terms": recent_repeated,
            "seen_score_threshold": threshold,
        }

    def _semantic_landmark_match_score(self, term: Any, landmark: Any) -> float:
        term_text = self._canonical_semantic_term(term)
        landmark_text = self._canonical_semantic_term(landmark)
        if not term_text or not landmark_text:
            return 0.0
        if term_text == landmark_text:
            return 1.0
        if term_text in landmark_text or landmark_text in term_text:
            return 0.85
        term_tokens = set(re.findall(r"[a-z0-9]+", term_text))
        landmark_tokens = set(re.findall(r"[a-z0-9]+", landmark_text))
        if not term_tokens or not landmark_tokens:
            return 0.0
        overlap = len(term_tokens.intersection(landmark_tokens))
        if overlap <= 0:
            return 0.0
        return min(0.75, float(overlap) / float(max(len(term_tokens), len(landmark_tokens))))

    def _semantic_goal_progress_for_term(
        self,
        term: Any,
        goal_progress_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state = goal_progress_state or {}
        if not state.get("enabled"):
            return {
                "matched_landmark": None,
                "landmark_status": "disabled",
                "next_landmark_relevance": 0.0,
                "completed_landmark_penalty": 0.0,
                "repeated_semantic_penalty": 0.0,
                "semantic_progress_score": 0.0,
            }
        sequence = list(state.get("landmark_sequence") or [])
        completed = set(state.get("completed_landmarks") or [])
        repeated = set(state.get("recent_repeated_terms") or [])
        next_landmark = state.get("next_landmark")
        best_landmark = None
        best_score = 0.0
        for landmark in sequence:
            score = self._semantic_landmark_match_score(term, landmark)
            if score > best_score:
                best_landmark = landmark
                best_score = score
        next_relevance = 0.0
        completed_penalty = 0.0
        repeated_penalty = 0.0
        status = "unknown"
        if best_landmark is not None and best_score > 0.0:
            if next_landmark and best_landmark == next_landmark:
                status = "next"
                next_relevance = best_score
            elif best_landmark in completed:
                status = "completed"
                completed_penalty = best_score
            else:
                status = "future"
                next_relevance = 0.25 * best_score
        canonical = self._canonical_semantic_term(term)
        for repeated_term in repeated:
            repeated_penalty = max(
                repeated_penalty,
                self._semantic_landmark_match_score(canonical, repeated_term),
            )
        progress_score = (
            float(self.config.candidate_probe_goal_progress_next_weight) * next_relevance
            - float(self.config.candidate_probe_goal_progress_completed_penalty) * completed_penalty
            - float(self.config.candidate_probe_goal_progress_repeated_penalty) * repeated_penalty
        )
        return {
            "matched_landmark": best_landmark,
            "landmark_status": status,
            "next_landmark_relevance": float(next_relevance),
            "completed_landmark_penalty": float(completed_penalty),
            "repeated_semantic_penalty": float(repeated_penalty),
            "semantic_progress_score": float(progress_score),
        }

    def _semantic_instruction_relevance(self, term: Any, instruction_terms: Iterable[str]) -> float:
        term_text = str(term or "").lower()
        if not term_text:
            return 0.0
        term_tokens = set(re.findall(r"[a-z0-9]+", term_text))
        terms = set(str(item).lower() for item in instruction_terms if item)
        if not terms:
            return 0.0
        if term_text in terms:
            return 1.0
        for phrase in terms:
            if " " in phrase and (phrase in term_text or term_text in phrase):
                return 1.0
        overlap = term_tokens.intersection(terms)
        if overlap:
            return min(1.0, float(len(overlap)) / max(1.0, float(len(term_tokens))))
        for token in term_tokens:
            if any(token in phrase or phrase in token for phrase in terms):
                return 0.5
        return 0.0

    def _semantic_recent_novelty(self, term: Any) -> float:
        term_text = str(term or "")
        if not term_text:
            return 0.0
        window = max(1, int(self.config.attribution_recent_semantic_window))
        recent_terms = [
            str(event.get("top_match"))
            for event in self.semantic_events[-window:]
            if event.get("top_match")
        ]
        if not recent_terms:
            return 0.5
        if term_text not in recent_terms:
            return 1.0
        unique_count = len(set(recent_terms))
        if unique_count <= int(self.config.attribution_dead_zone_unique_threshold):
            return 0.0
        return 0.25

    def _build_query_candidate(
        self,
        cell: Iterable[int],
        candidate_type: str,
        start_grid: Iterable[int],
        yaw: float,
        *,
        current_angle: Any,
        current_goal_grid: Any,
        source_score: float,
        semantic: Optional[Dict[str, Any]] = None,
        goal_progress_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        row, col = [int(v) for v in list(cell)[:2]]
        direction = self._direction_to_cell(start_grid, [row, col], yaw)
        distance_m = float(direction.get("distance_m", 0.0) or 0.0)
        if distance_m < float(self.config.candidate_probe_min_distance_m):
            return None
        if distance_m > float(self.config.candidate_probe_max_distance_m):
            return None
        state = self._cell_state(row, col)
        if state == "occupied":
            geometry_score = -2.0
        elif state == "free":
            geometry_score = 0.50
        else:
            geometry_score = 0.10
        visit_count = self._nearby_visit_count([row, col], int(self.config.waypoint_revisit_radius_cells))
        revisit_risk = min(1.0, float(visit_count) / 3.0)
        if candidate_type == "frontier":
            frontier_distance_cells = 0.0
        elif candidate_type == "open_floor":
            frontier_distance_cells = None
        else:
            frontier_distance_cells = self._nearest_frontier_distance([row, col])
        frontier_distance_m = (
            None if frontier_distance_cells is None else float(frontier_distance_cells * self.cs)
        )
        frontier_progress_score = 0.0
        if candidate_type == "frontier":
            frontier_progress_score = 1.0
        elif frontier_distance_m is not None and frontier_distance_m <= 0.50:
            frontier_progress_score = 0.35
        bucket = str(direction.get("bucket") or "unknown")
        direction_score = {
            "front": 0.35,
            "left": 0.20,
            "right": 0.20,
            "same": -0.50,
            "back": -0.35,
        }.get(bucket, 0.0)
        angle = direction.get("angle_deg")
        angle_to_current = None
        intent_alignment_score = None
        aligned_with_current = False
        try:
            angle_to_current = self._angle_distance_degrees(float(angle), float(current_angle))
            intent_alignment_score = max(0.0, 1.0 - float(angle_to_current) / 180.0)
            aligned_with_current = self._directions_aligned(angle, current_angle)
        except (TypeError, ValueError):
            pass
        distance_to_current_goal_m = None
        try:
            current_xy = self._grid_to_xy(current_goal_grid)
            candidate_xy = self._grid_to_xy([row, col])
            distance_to_current_goal_m = float(np.linalg.norm(candidate_xy - current_xy))
        except Exception:
            pass
        semantic_relevance_score = 0.0
        semantic_novelty_score = 0.0
        semantic_confidence_score = 0.0
        semantic_bind_score = 0.0
        matched_landmark = None
        landmark_status = "none"
        next_landmark_relevance = 0.0
        completed_landmark_penalty = 0.0
        repeated_semantic_penalty = 0.0
        semantic_progress_score = 0.0
        instruction_relevant = False
        if semantic is not None:
            semantic_relevance_score = float(semantic.get("instruction_relevance", 0.0) or 0.0)
            semantic_novelty_score = float(semantic.get("semantic_recent_novelty", 0.0) or 0.0)
            semantic_bind_score = float(semantic.get("bind_score", 0.0) or 0.0)
            matched_landmark = semantic.get("matched_landmark")
            landmark_status = str(semantic.get("landmark_status") or "unknown")
            next_landmark_relevance = float(semantic.get("next_landmark_relevance", 0.0) or 0.0)
            completed_landmark_penalty = float(
                semantic.get("completed_landmark_penalty", 0.0) or 0.0
            )
            repeated_semantic_penalty = float(
                semantic.get("repeated_semantic_penalty", 0.0) or 0.0
            )
            semantic_progress_score = float(semantic.get("semantic_progress_score", 0.0) or 0.0)
            try:
                semantic_score = float(semantic.get("semantic_top_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                semantic_score = 0.0
            min_score = float(self.config.candidate_probe_semantic_min_score)
            semantic_confidence_score = min(
                1.0,
                max(0.0, (semantic_score - min_score) / max(1e-6, 0.35 - min_score)),
            )
            instruction_relevant = bool(semantic_relevance_score > 0.0)
        semanticized_candidate = bool(
            semantic is not None
            and (
                semantic_relevance_score >= float(self.config.candidate_probe_semantic_frontier_min_relevance)
                or next_landmark_relevance > 0.0
                or bool(semantic.get("high_conf_semantic"))
                or candidate_type == "semantic_keyframe"
            )
        )
        if candidate_type == "frontier" and semanticized_candidate:
            candidate_type = "semantic_frontier"
        topology_novelty_score = float(frontier_progress_score * (1.0 - revisit_risk))
        preferred_distance_m = 2.0
        distance_score = -0.08 * abs(distance_m - preferred_distance_m)
        goal_progress_enabled = bool(
            (goal_progress_state or {}).get("enabled")
            or self.config.candidate_probe_goal_progress_enable
        )
        next_landmark = (goal_progress_state or {}).get("next_landmark")
        unknown_target_frontier_bonus = 0.0
        if (
            goal_progress_enabled
            and next_landmark
            and next_landmark_relevance <= 0.0
            and completed_landmark_penalty <= 0.0
            and frontier_progress_score > 0.0
        ):
            unknown_target_frontier_bonus = (
                float(self.config.candidate_probe_goal_progress_unknown_target_bonus)
                * float(frontier_progress_score)
            )
        goal_progress_score = 0.0
        if goal_progress_enabled:
            goal_progress_score = (
                float(self.config.candidate_probe_goal_progress_next_weight)
                * next_landmark_relevance
                + unknown_target_frontier_bonus
                - float(self.config.candidate_probe_goal_progress_completed_penalty)
                * completed_landmark_penalty
                - float(self.config.candidate_probe_goal_progress_repeated_penalty)
                * repeated_semantic_penalty
            )
        target_frontier = self._target_frontier_features(
            [row, col],
            start_grid,
            frontier_progress_score=frontier_progress_score,
            revisit_risk=revisit_risk,
            angle_to_current=angle_to_current,
            goal_progress_state=goal_progress_state,
            completed_landmark_penalty=completed_landmark_penalty,
            repeated_semantic_penalty=repeated_semantic_penalty,
            unknown_target_frontier_bonus=unknown_target_frontier_bonus,
        )
        target_frontier_score = float(target_frontier.get("score", 0.0) or 0.0)
        score = (
            float(source_score)
            + geometry_score
            + frontier_progress_score
            + direction_score
            + 0.30 * float(intent_alignment_score or 0.0)
            + float(self.config.candidate_probe_semantic_score_weight)
            * (
                0.70 * semantic_relevance_score
                + 0.20 * semantic_confidence_score
                + 0.10 * min(1.0, max(0.0, semantic_bind_score))
            )
            + float(self.config.candidate_probe_semantic_novelty_weight) * semantic_novelty_score
            + float(self.config.candidate_probe_topology_novelty_weight) * topology_novelty_score
            + goal_progress_score
            + float(self.config.candidate_probe_target_frontier_score_weight)
            * target_frontier_score
            + distance_score
            - 0.65 * revisit_risk
        )
        geometry_safe = state != "occupied"
        return {
            "candidate_type": str(candidate_type),
            "grid": [int(row), int(col)],
            "xy": [float(v) for v in self._grid_to_xy([row, col]).tolist()],
            "goal_state": state,
            "geometry_safe": bool(geometry_safe),
            "active_gate_safe": bool(geometry_safe and bucket != "back"),
            "direction_bucket": bucket,
            "direction_angle_deg": angle,
            "distance_m": distance_m,
            "frontier_distance_m": frontier_distance_m,
            "frontier_progress_score": float(frontier_progress_score),
            "topology_novelty_score": float(topology_novelty_score),
            "nearby_visit_count": int(visit_count),
            "revisit_risk": float(revisit_risk),
            "points_to_revisited_region": bool(visit_count > 0),
            "angle_to_current_waypoint_deg": angle_to_current,
            "intent_alignment_score": intent_alignment_score,
            "aligned_with_current_waypoint": bool(aligned_with_current),
            "distance_to_current_waypoint_m": distance_to_current_goal_m,
            "semantic_evidence": semantic,
            "semanticized_candidate": bool(semanticized_candidate),
            "instruction_relevant": bool(instruction_relevant),
            "semantic_relevance_score": float(semantic_relevance_score),
            "semantic_novelty_score": float(semantic_novelty_score),
            "semantic_confidence_score": float(semantic_confidence_score),
            "semantic_bind_score": float(semantic_bind_score),
            "goal_progress_enabled": bool(goal_progress_enabled),
            "goal_progress_next_landmark": next_landmark,
            "matched_landmark": matched_landmark,
            "landmark_status": landmark_status,
            "next_landmark_relevance": float(next_landmark_relevance),
            "completed_landmark_penalty": float(completed_landmark_penalty),
            "repeated_semantic_penalty": float(repeated_semantic_penalty),
            "semantic_progress_score": float(semantic_progress_score),
            "unknown_target_frontier_bonus": float(unknown_target_frontier_bonus),
            "goal_progress_score": float(goal_progress_score),
            "target_frontier_enabled": bool(target_frontier.get("enabled")),
            "target_frontier_score": float(target_frontier_score),
            "target_frontier_candidate": bool(target_frontier.get("candidate")),
            "target_frontier_escape_candidate": bool(target_frontier.get("escape_candidate")),
            "target_frontier_cluster_count": int(target_frontier.get("cluster_count", 0) or 0),
            "target_frontier_cluster_score": float(
                target_frontier.get("cluster_score", 0.0) or 0.0
            ),
            "target_frontier_doorway_like_score": float(
                target_frontier.get("doorway_like_score", 0.0) or 0.0
            ),
            "target_frontier_corridor_continuation_score": float(
                target_frontier.get("corridor_continuation_score", 0.0) or 0.0
            ),
            "target_frontier_transition_prior": float(
                target_frontier.get("transition_prior", 0.0) or 0.0
            ),
            "target_frontier_intent_deviation_penalty": float(
                target_frontier.get("intent_deviation_penalty", 0.0) or 0.0
            ),
            "target_frontier_intent_safe": bool(target_frontier.get("intent_safe")),
            "target_frontier_local_free_count": int(
                target_frontier.get("local_free_count", 0) or 0
            ),
            "target_frontier_local_occupied_count": int(
                target_frontier.get("local_occupied_count", 0) or 0
            ),
            "target_frontier_local_unknown_count": int(
                target_frontier.get("local_unknown_count", 0) or 0
            ),
            "score": float(score),
        }

    def _select_query_candidates(self, raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_candidates = max(0, int(self.config.candidate_probe_max_candidates))
        if max_candidates <= 0:
            return []
        ordered = sorted(
            raw_candidates,
            key=lambda item: (
                bool(item.get("active_gate_safe")),
                bool(item.get("geometry_safe")),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        min_sep = max(0.0, float(self.config.candidate_probe_min_separation_m))
        for candidate in ordered:
            if len(selected) >= max_candidates:
                break
            xy = np.asarray(candidate.get("xy", [0.0, 0.0])[:2], dtype=np.float32)
            too_close = False
            for existing in selected:
                other_xy = np.asarray(existing.get("xy", [0.0, 0.0])[:2], dtype=np.float32)
                if float(np.linalg.norm(xy - other_xy)) < min_sep:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(candidate)
        if bool(self.config.semantic_resilience_shadow_enable):
            resilience_candidates = [
                item
                for item in raw_candidates
                if item.get("semantic_resilience_candidate")
            ]
            if resilience_candidates and not any(
                item.get("semantic_resilience_candidate") for item in selected
            ):
                best_resilience = max(
                    resilience_candidates,
                    key=lambda item: (
                        bool(item.get("geometry_safe")),
                        float(item.get("semantic_resilience_score", 0.0) or 0.0),
                        float(item.get("score", 0.0) or 0.0),
                    ),
                )
                if len(selected) < max_candidates:
                    selected.append(best_resilience)
                elif selected:
                    selected[-1] = best_resilience
        return selected

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
        stage15_repair_event_count = 0
        stage15_roundtrip_valid_count = 0
        stage15_roundtrip_errors = []
        stage15_repair_candidate_count = 0
        stage15_repair_free_found_count = 0
        stage15_repair_valid_count = 0
        stage15_repair_no_free_count = 0
        stage15_repair_projection_failed_count = 0
        stage15_repair_reason_counts: Dict[str, int] = defaultdict(int)
        stage15_repair_shifts = []
        stage15_repair_backtrack_cells = []
        for event in self.waypoint_events:
            state = str(event.get("goal_state", "invalid"))
            waypoint_state_counts[state] += 1
            if event.get("stage15_repair_enabled"):
                stage15_repair_event_count += 1
                if event.get("roundtrip_valid"):
                    stage15_roundtrip_valid_count += 1
                roundtrip_error = event.get("roundtrip_error_px")
                if roundtrip_error is not None:
                    stage15_roundtrip_errors.append(float(roundtrip_error))
                repair_reason = event.get("repair_reason")
                if repair_reason:
                    stage15_repair_reason_counts[str(repair_reason)] += 1
                if event.get("repair_candidate"):
                    stage15_repair_candidate_count += 1
                if event.get("repair_free_grid") is not None:
                    stage15_repair_free_found_count += 1
                if event.get("repair_valid"):
                    stage15_repair_valid_count += 1
                if repair_reason == "no_free_along_bearing":
                    stage15_repair_no_free_count += 1
                if repair_reason == "repair_projection_failed":
                    stage15_repair_projection_failed_count += 1
                repair_shift = event.get("repair_pixel_shift")
                if repair_shift is not None:
                    stage15_repair_shifts.append(float(repair_shift))
                backtrack = event.get("repair_backtrack_cells")
                if backtrack is not None:
                    stage15_repair_backtrack_cells.append(float(backtrack))
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
        candidate_event_count = len(self.candidate_probe_events)
        candidate_valid_event_count = 0
        candidate_count_sum = 0
        candidate_geometry_safe_sum = 0
        candidate_active_gate_safe_sum = 0
        candidate_current_aligned_sum = 0
        candidate_semantic_evidence_sum = 0
        candidate_instruction_relevant_sum = 0
        candidate_semanticized_sum = 0
        candidate_next_landmark_relevant_sum = 0
        candidate_completed_landmark_sum = 0
        candidate_repeated_semantic_sum = 0
        candidate_unknown_target_frontier_bonus_sum = 0
        candidate_target_frontier_sum = 0
        candidate_target_frontier_escape_sum = 0
        candidate_target_frontier_intent_safe_sum = 0
        candidate_target_frontier_doorway_like_sum = 0
        semantic_resilience_enabled_event_count = 0
        semantic_resilience_recovery_trigger_count = 0
        semantic_resilience_local_trap_count = 0
        semantic_resilience_raw_candidate_sum = 0
        candidate_semantic_resilience_sum = 0
        candidate_semantic_resilience_recommended_sum = 0
        candidate_semantic_resilience_obstacle_sum = 0
        candidate_semantic_resilience_passage_sum = 0
        semantic_resilience_obstacle_event_count = 0
        semantic_resilience_passage_event_count = 0
        semantic_resilience_reason_counts: Dict[str, int] = defaultdict(int)
        semantic_resilience_context_tag_counts: Dict[str, int] = defaultdict(int)
        current_policy_candidate_valid_count = 0
        current_policy_candidate_geometry_safe_count = 0
        current_policy_candidate_active_gate_safe_count = 0
        current_policy_candidate_revisited_count = 0
        current_policy_candidate_dead_zone_count = 0
        progress_ranker_shadow_enabled_count = 0
        progress_ranker_shadow_valid_count = 0
        progress_ranker_shadow_error_count = 0
        progress_ranker_shadow_ranker_change_count = 0
        progress_ranker_shadow_resilience_change_count = 0
        progress_ranker_shadow_resilience_completed_count = 0
        progress_ranker_shadow_resilience_repeated_count = 0
        progress_ranker_shadow_resilience_unsafe_count = 0
        progress_ranker_shadow_resilience_future_sum = 0.0
        progress_ranker_shadow_resilience_recoverability_sum = 0.0
        stage21_shadow_enabled_count = 0
        stage21_shadow_valid_count = 0
        stage21_shadow_error_count = 0
        stage21_shadow_action_applied_count = 0
        stage21_shadow_progress_change_count = 0
        stage21_shadow_intent_change_count = 0
        stage21_shadow_recovery_candidate_count = 0
        stage21_shadow_missing_sum = 0.0
        stage21_shadow_latency_sum = 0.0
        stage21_shadow_latency_values: List[float] = []
        candidate_type_counts: Dict[str, int] = defaultdict(int)
        candidate_direction_counts: Dict[str, int] = defaultdict(int)
        for event in self.candidate_probe_events:
            if event.get("valid"):
                candidate_valid_event_count += 1
            candidate_count_sum += int(event.get("candidate_count", 0) or 0)
            candidate_geometry_safe_sum += int(event.get("candidate_geometry_safe_count", 0) or 0)
            candidate_active_gate_safe_sum += int(event.get("candidate_active_gate_safe_count", 0) or 0)
            candidate_current_aligned_sum += int(event.get("candidate_current_aligned_count", 0) or 0)
            candidate_semantic_evidence_sum += int(event.get("candidate_semantic_evidence_count", 0) or 0)
            candidate_instruction_relevant_sum += int(
                event.get("candidate_instruction_relevant_count", 0) or 0
            )
            candidate_semanticized_sum += int(event.get("candidate_semanticized_count", 0) or 0)
            candidate_next_landmark_relevant_sum += int(
                event.get("candidate_next_landmark_relevant_count", 0) or 0
            )
            candidate_completed_landmark_sum += int(
                event.get("candidate_completed_landmark_count", 0) or 0
            )
            candidate_repeated_semantic_sum += int(
                event.get("candidate_repeated_semantic_count", 0) or 0
            )
            candidate_unknown_target_frontier_bonus_sum += int(
                event.get("candidate_unknown_target_frontier_bonus_count", 0) or 0
            )
            candidate_target_frontier_sum += int(
                event.get("candidate_target_frontier_count", 0) or 0
            )
            candidate_target_frontier_escape_sum += int(
                event.get("candidate_target_frontier_escape_count", 0) or 0
            )
            candidate_target_frontier_intent_safe_sum += int(
                event.get("candidate_target_frontier_intent_safe_count", 0) or 0
            )
            candidate_target_frontier_doorway_like_sum += int(
                event.get("candidate_target_frontier_doorway_like_count", 0) or 0
            )
            if event.get("semantic_resilience_enabled"):
                semantic_resilience_enabled_event_count += 1
            if event.get("semantic_resilience_recovery_trigger"):
                semantic_resilience_recovery_trigger_count += 1
            if event.get("semantic_resilience_local_trap"):
                semantic_resilience_local_trap_count += 1
            semantic_resilience_raw_candidate_sum += int(
                event.get("semantic_resilience_raw_candidate_count", 0) or 0
            )
            candidate_semantic_resilience_sum += int(
                event.get("candidate_semantic_resilience_count", 0) or 0
            )
            candidate_semantic_resilience_recommended_sum += int(
                event.get("candidate_semantic_resilience_recommended_count", 0) or 0
            )
            candidate_semantic_resilience_obstacle_sum += int(
                event.get("candidate_semantic_resilience_obstacle_count", 0) or 0
            )
            candidate_semantic_resilience_passage_sum += int(
                event.get("candidate_semantic_resilience_passage_count", 0) or 0
            )
            state = event.get("semantic_resilience_state") or {}
            if int(state.get("semantic_obstacle_term_count", 0) or 0) > 0:
                semantic_resilience_obstacle_event_count += 1
            if int(state.get("semantic_passage_term_count", 0) or 0) > 0:
                semantic_resilience_passage_event_count += 1
            for reason in event.get("semantic_resilience_trigger_reasons") or []:
                semantic_resilience_reason_counts[str(reason)] += 1
            for tag in event.get("semantic_resilience_recovery_context_tags") or state.get(
                "recovery_context_tags"
            ) or []:
                semantic_resilience_context_tag_counts[str(tag)] += 1
            current_candidate = event.get("current_policy_candidate") or {}
            if current_candidate.get("valid"):
                current_policy_candidate_valid_count += 1
                if current_candidate.get("geometry_safe"):
                    current_policy_candidate_geometry_safe_count += 1
                if current_candidate.get("active_gate_safe"):
                    current_policy_candidate_active_gate_safe_count += 1
                if current_candidate.get("points_to_revisited_region"):
                    current_policy_candidate_revisited_count += 1
                if current_candidate.get("semantic_dead_zone"):
                    current_policy_candidate_dead_zone_count += 1
            shadow = event.get("progress_ranker_shadow") or {}
            if shadow.get("enabled"):
                progress_ranker_shadow_enabled_count += 1
            if shadow.get("reason") == "error":
                progress_ranker_shadow_error_count += 1
            if shadow.get("valid"):
                progress_ranker_shadow_valid_count += 1
                if shadow.get("ranker_changes_target_frontier"):
                    progress_ranker_shadow_ranker_change_count += 1
                if shadow.get("ranker_resilience_changes_target_frontier"):
                    progress_ranker_shadow_resilience_change_count += 1
                selected = shadow.get("ranker_resilience_selected") or {}
                if selected.get("completed_landmark"):
                    progress_ranker_shadow_resilience_completed_count += 1
                if selected.get("repeated_semantic"):
                    progress_ranker_shadow_resilience_repeated_count += 1
                if not selected.get("geometry_safe") or not selected.get("active_gate_safe"):
                    progress_ranker_shadow_resilience_unsafe_count += 1
                progress_ranker_shadow_resilience_future_sum += float(
                    selected.get("future_observability_proxy", 0.0) or 0.0
                )
                progress_ranker_shadow_resilience_recoverability_sum += float(
                    selected.get("recoverability_proxy", 0.0) or 0.0
                )
            stage21_shadow = event.get("stage21_multitask_shadow") or {}
            if stage21_shadow.get("enabled"):
                stage21_shadow_enabled_count += 1
            if stage21_shadow.get("reason") in {"error", "not_initialized"}:
                stage21_shadow_error_count += 1
            if stage21_shadow.get("valid"):
                stage21_shadow_valid_count += 1
                if stage21_shadow.get("action_applied"):
                    stage21_shadow_action_applied_count += 1
                if stage21_shadow.get("progress_changes_candidate_score"):
                    stage21_shadow_progress_change_count += 1
                if stage21_shadow.get("progress_changes_intent_alignment"):
                    stage21_shadow_intent_change_count += 1
                stage21_shadow_recovery_candidate_count += int(
                    stage21_shadow.get("recovery_eligible_count", 0) or 0
                )
                stage21_shadow_missing_sum += float(
                    stage21_shadow.get("missing_numeric_mean", 0.0) or 0.0
                )
                latency = float(stage21_shadow.get("inference_latency_ms", 0.0) or 0.0)
                stage21_shadow_latency_sum += latency
                stage21_shadow_latency_values.append(latency)
            for key, value in (event.get("candidate_type_counts") or {}).items():
                candidate_type_counts[str(key)] += int(value or 0)
            for key, value in (event.get("candidate_direction_counts") or {}).items():
                candidate_direction_counts[str(key)] += int(value or 0)
        selection_event_count = len(self.candidate_selection_events)
        selection_valid_count = 0
        selection_none_count = 0
        selection_active_gate_safe_count = 0
        selection_current_aligned_count = 0
        selection_next_landmark_relevant_count = 0
        selection_completed_landmark_count = 0
        selection_repeated_semantic_count = 0
        selection_reason_counts: Dict[str, int] = defaultdict(int)
        for event in self.candidate_selection_events:
            if event.get("selection_valid"):
                selection_valid_count += 1
            if event.get("selection_none"):
                selection_none_count += 1
            reason = event.get("selection_reason")
            if reason:
                selection_reason_counts[str(reason)] += 1
            selected = event.get("selected_candidate") or {}
            if selected.get("active_gate_safe"):
                selection_active_gate_safe_count += 1
            if selected.get("aligned_with_current_waypoint"):
                selection_current_aligned_count += 1
            if float(selected.get("next_landmark_relevance", 0.0) or 0.0) > 0.0:
                selection_next_landmark_relevant_count += 1
            if float(selected.get("completed_landmark_penalty", 0.0) or 0.0) > 0.0:
                selection_completed_landmark_count += 1
            if float(selected.get("repeated_semantic_penalty", 0.0) or 0.0) > 0.0:
                selection_repeated_semantic_count += 1
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
        semantic_anchor_kind_counts: Dict[str, int] = defaultdict(int)
        semantic_anchor_source_counts: Dict[str, int] = defaultdict(int)
        semantic_anchor_term_counts: Dict[str, int] = defaultdict(int)
        semantic_anchor_high_conf_count = 0
        for anchor in self.semantic_anchors:
            semantic_anchor_kind_counts[str(anchor.get("semantic_kind") or "unknown")] += 1
            semantic_anchor_source_counts[str(anchor.get("anchor_source") or "unknown")] += 1
            semantic_anchor_term_counts[str(anchor.get("semantic_top_match") or "unknown")] += 1
            if anchor.get("high_conf_semantic"):
                semantic_anchor_high_conf_count += 1
        semantic_anchor_source_operation_counts = {
            str(source): dict(ops)
            for source, ops in self.semantic_anchor_source_operation_counts.items()
        }
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
            "semantic_anchor_enabled": bool(self.config.semantic_anchor_enable),
            "semantic_anchor_count": int(len(self.semantic_anchors)),
            "semantic_anchor_added_count": int(self.semantic_anchor_added_count),
            "semantic_anchor_merged_count": int(self.semantic_anchor_merged_count),
            "semantic_anchor_max_anchors_count": int(self.semantic_anchor_max_anchors_count),
            "semantic_anchor_invalid_count": int(self.semantic_anchor_invalid_count),
            "semantic_anchor_merge_rate": (
                float(self.semantic_anchor_merged_count)
                / max(1, int(self.semantic_anchor_added_count + self.semantic_anchor_merged_count))
            ),
            "semantic_anchor_high_conf_count": int(semantic_anchor_high_conf_count),
            "semantic_anchor_kind_counts": dict(semantic_anchor_kind_counts),
            "semantic_anchor_source_counts": dict(semantic_anchor_source_counts),
            "semantic_anchor_source_operation_counts": semantic_anchor_source_operation_counts,
            "semantic_anchor_top_terms": dict(semantic_anchor_term_counts),
            "semantic_anchor_source_offset_x_px_stats": self._basic_stats(
                self.semantic_anchor_source_offset_x_px_values
            ),
            "semantic_anchor_source_offset_y_px_stats": self._basic_stats(
                self.semantic_anchor_source_offset_y_px_values
            ),
            "semantic_anchor_source_center_distance_px_stats": self._basic_stats(
                self.semantic_anchor_source_center_distance_px_values
            ),
            "semantic_anchor_source_ray_norm_stats": self._basic_stats(
                self.semantic_anchor_source_ray_norm_values
            ),
            "semantic_anchor_source_ray_yaw_deg_stats": self._basic_stats(
                self.semantic_anchor_source_ray_yaw_deg_values
            ),
            "semantic_anchor_source_ray_pitch_deg_stats": self._basic_stats(
                self.semantic_anchor_source_ray_pitch_deg_values
            ),
            "semantic_anchor_global_bearing_deg_stats": self._basic_stats(
                self.semantic_anchor_global_bearing_deg_values
            ),
            "semantic_anchor_relative_bearing_deg_stats": self._basic_stats(
                self.semantic_anchor_relative_bearing_deg_values
            ),
            "semantic_anchor_pose_origin_distance_m_stats": self._basic_stats(
                self.semantic_anchor_pose_origin_distance_m_values
            ),
            "semantic_anchor_pose_step_distance_m_stats": self._basic_stats(
                self.semantic_anchor_pose_step_distance_m_values
            ),
            "semantic_anchor_pose_step_dyaw_deg_stats": self._basic_stats(
                self.semantic_anchor_pose_step_dyaw_deg_values
            ),
            "waypoint_probe_count": int(len(self.waypoint_events)),
            "candidate_probe_event_count": int(candidate_event_count),
            "candidate_probe_valid_event_count": int(candidate_valid_event_count),
            "candidate_probe_mean_candidate_count": (
                float(candidate_count_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_geometry_safe_count": (
                float(candidate_geometry_safe_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_active_gate_safe_count": (
                float(candidate_active_gate_safe_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_current_aligned_count": (
                float(candidate_current_aligned_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_semantic_evidence_count": (
                float(candidate_semantic_evidence_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_instruction_relevant_count": (
                float(candidate_instruction_relevant_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_semanticized_count": (
                float(candidate_semanticized_sum / candidate_event_count) if candidate_event_count else None
            ),
            "candidate_probe_mean_next_landmark_relevant_count": (
                float(candidate_next_landmark_relevant_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_completed_landmark_count": (
                float(candidate_completed_landmark_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_repeated_semantic_count": (
                float(candidate_repeated_semantic_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_unknown_target_frontier_bonus_count": (
                float(candidate_unknown_target_frontier_bonus_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_target_frontier_count": (
                float(candidate_target_frontier_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_target_frontier_escape_count": (
                float(candidate_target_frontier_escape_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_target_frontier_intent_safe_count": (
                float(candidate_target_frontier_intent_safe_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "candidate_probe_mean_target_frontier_doorway_like_count": (
                float(candidate_target_frontier_doorway_like_sum / candidate_event_count)
                if candidate_event_count else None
            ),
            "semantic_resilience_enabled_event_count": int(
                semantic_resilience_enabled_event_count
            ),
            "semantic_resilience_recovery_trigger_count": int(
                semantic_resilience_recovery_trigger_count
            ),
            "semantic_resilience_recovery_trigger_rate": (
                float(semantic_resilience_recovery_trigger_count / candidate_event_count)
                if candidate_event_count else None
            ),
            "semantic_resilience_local_trap_count": int(
                semantic_resilience_local_trap_count
            ),
            "semantic_resilience_local_trap_rate": (
                float(semantic_resilience_local_trap_count / candidate_event_count)
                if candidate_event_count else None
            ),
            "semantic_resilience_raw_candidate_count": int(
                semantic_resilience_raw_candidate_sum
            ),
            "semantic_resilience_candidate_count": int(
                candidate_semantic_resilience_sum
            ),
            "semantic_resilience_candidate_rate": (
                float(candidate_semantic_resilience_sum / candidate_count_sum)
                if candidate_count_sum else None
            ),
            "semantic_resilience_recommended_candidate_count": int(
                candidate_semantic_resilience_recommended_sum
            ),
            "semantic_resilience_obstacle_event_count": int(
                semantic_resilience_obstacle_event_count
            ),
            "semantic_resilience_obstacle_event_rate": (
                float(semantic_resilience_obstacle_event_count / candidate_event_count)
                if candidate_event_count else None
            ),
            "semantic_resilience_passage_event_count": int(
                semantic_resilience_passage_event_count
            ),
            "semantic_resilience_passage_event_rate": (
                float(semantic_resilience_passage_event_count / candidate_event_count)
                if candidate_event_count else None
            ),
            "semantic_resilience_candidate_obstacle_count": int(
                candidate_semantic_resilience_obstacle_sum
            ),
            "semantic_resilience_candidate_passage_count": int(
                candidate_semantic_resilience_passage_sum
            ),
            "semantic_resilience_trigger_reason_counts": dict(
                semantic_resilience_reason_counts
            ),
            "semantic_resilience_context_tag_counts": dict(
                semantic_resilience_context_tag_counts
            ),
            "candidate_probe_type_counts": dict(candidate_type_counts),
            "candidate_probe_direction_counts": dict(candidate_direction_counts),
            "current_policy_candidate_valid_count": int(current_policy_candidate_valid_count),
            "current_policy_candidate_valid_rate": (
                float(current_policy_candidate_valid_count / candidate_event_count)
                if candidate_event_count else None
            ),
            "current_policy_candidate_geometry_safe_count": int(
                current_policy_candidate_geometry_safe_count
            ),
            "current_policy_candidate_geometry_safe_rate": (
                float(current_policy_candidate_geometry_safe_count / current_policy_candidate_valid_count)
                if current_policy_candidate_valid_count else None
            ),
            "current_policy_candidate_active_gate_safe_count": int(
                current_policy_candidate_active_gate_safe_count
            ),
            "current_policy_candidate_active_gate_safe_rate": (
                float(
                    current_policy_candidate_active_gate_safe_count
                    / current_policy_candidate_valid_count
                )
                if current_policy_candidate_valid_count else None
            ),
            "current_policy_candidate_revisited_count": int(current_policy_candidate_revisited_count),
            "current_policy_candidate_revisited_rate": (
                float(current_policy_candidate_revisited_count / current_policy_candidate_valid_count)
                if current_policy_candidate_valid_count else None
            ),
            "current_policy_candidate_dead_zone_count": int(current_policy_candidate_dead_zone_count),
            "current_policy_candidate_dead_zone_rate": (
                float(current_policy_candidate_dead_zone_count / current_policy_candidate_valid_count)
                if current_policy_candidate_valid_count else None
            ),
            "progress_ranker_shadow_enabled_count": int(progress_ranker_shadow_enabled_count),
            "progress_ranker_shadow_valid_count": int(progress_ranker_shadow_valid_count),
            "progress_ranker_shadow_error_count": int(progress_ranker_shadow_error_count),
            "progress_ranker_shadow_ranker_change_count": int(
                progress_ranker_shadow_ranker_change_count
            ),
            "progress_ranker_shadow_resilience_change_count": int(
                progress_ranker_shadow_resilience_change_count
            ),
            "progress_ranker_shadow_resilience_completed_count": int(
                progress_ranker_shadow_resilience_completed_count
            ),
            "progress_ranker_shadow_resilience_repeated_count": int(
                progress_ranker_shadow_resilience_repeated_count
            ),
            "progress_ranker_shadow_resilience_unsafe_count": int(
                progress_ranker_shadow_resilience_unsafe_count
            ),
            "progress_ranker_shadow_resilience_change_ratio": (
                float(progress_ranker_shadow_resilience_change_count / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "progress_ranker_shadow_resilience_completed_rate": (
                float(progress_ranker_shadow_resilience_completed_count / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "progress_ranker_shadow_resilience_repeated_rate": (
                float(progress_ranker_shadow_resilience_repeated_count / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "progress_ranker_shadow_resilience_unsafe_rate": (
                float(progress_ranker_shadow_resilience_unsafe_count / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "progress_ranker_shadow_resilience_future_observability_mean": (
                float(progress_ranker_shadow_resilience_future_sum / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "progress_ranker_shadow_resilience_recoverability_mean": (
                float(progress_ranker_shadow_resilience_recoverability_sum / progress_ranker_shadow_valid_count)
                if progress_ranker_shadow_valid_count else None
            ),
            "stage21_multitask_shadow_enabled_count": int(stage21_shadow_enabled_count),
            "stage21_multitask_shadow_valid_count": int(stage21_shadow_valid_count),
            "stage21_multitask_shadow_error_count": int(stage21_shadow_error_count),
            "stage21_multitask_shadow_action_applied_count": int(stage21_shadow_action_applied_count),
            "stage21_multitask_shadow_progress_change_count": int(stage21_shadow_progress_change_count),
            "stage21_multitask_shadow_intent_change_count": int(stage21_shadow_intent_change_count),
            "stage21_multitask_shadow_recovery_candidate_count": int(stage21_shadow_recovery_candidate_count),
            "stage21_multitask_shadow_progress_change_rate": (
                float(stage21_shadow_progress_change_count / stage21_shadow_valid_count)
                if stage21_shadow_valid_count else None
            ),
            "stage21_multitask_shadow_intent_change_rate": (
                float(stage21_shadow_intent_change_count / stage21_shadow_valid_count)
                if stage21_shadow_valid_count else None
            ),
            "stage21_multitask_shadow_missing_numeric_mean": (
                float(stage21_shadow_missing_sum / stage21_shadow_valid_count)
                if stage21_shadow_valid_count else None
            ),
            "stage21_multitask_shadow_latency_mean_ms": (
                float(stage21_shadow_latency_sum / stage21_shadow_valid_count)
                if stage21_shadow_valid_count else None
            ),
            "stage21_multitask_shadow_latency_p95_ms": (
                float(np.percentile(stage21_shadow_latency_values, 95))
                if stage21_shadow_latency_values else None
            ),
            "candidate_selection_event_count": int(selection_event_count),
            "candidate_selection_valid_count": int(selection_valid_count),
            "candidate_selection_none_count": int(selection_none_count),
            "candidate_selection_active_gate_safe_count": int(selection_active_gate_safe_count),
            "candidate_selection_current_aligned_count": int(selection_current_aligned_count),
            "candidate_selection_next_landmark_relevant_count": int(
                selection_next_landmark_relevant_count
            ),
            "candidate_selection_completed_landmark_count": int(selection_completed_landmark_count),
            "candidate_selection_repeated_semantic_count": int(selection_repeated_semantic_count),
            "candidate_selection_reason_counts": dict(selection_reason_counts),
            "waypoint_goal_state_counts": dict(waypoint_state_counts),
            "stage15_repair_event_count": int(stage15_repair_event_count),
            "stage15_roundtrip_valid_count": int(stage15_roundtrip_valid_count),
            "stage15_roundtrip_error_mean_px": (
                float(np.mean(stage15_roundtrip_errors)) if stage15_roundtrip_errors else None
            ),
            "stage15_roundtrip_error_median_px": (
                float(np.median(stage15_roundtrip_errors)) if stage15_roundtrip_errors else None
            ),
            "stage15_roundtrip_error_p90_px": (
                float(np.percentile(stage15_roundtrip_errors, 90))
                if stage15_roundtrip_errors else None
            ),
            "stage15_roundtrip_error_max_px": (
                float(np.max(stage15_roundtrip_errors)) if stage15_roundtrip_errors else None
            ),
            "stage15_repair_candidate_count": int(stage15_repair_candidate_count),
            "stage15_repair_free_found_count": int(stage15_repair_free_found_count),
            "stage15_repair_valid_count": int(stage15_repair_valid_count),
            "stage15_repair_no_free_count": int(stage15_repair_no_free_count),
            "stage15_repair_projection_failed_count": int(stage15_repair_projection_failed_count),
            "stage15_repair_reason_counts": dict(stage15_repair_reason_counts),
            "stage15_repair_pixel_shift_mean": (
                float(np.mean(stage15_repair_shifts)) if stage15_repair_shifts else None
            ),
            "stage15_repair_pixel_shift_median": (
                float(np.median(stage15_repair_shifts)) if stage15_repair_shifts else None
            ),
            "stage15_repair_backtrack_cells_mean": (
                float(np.mean(stage15_repair_backtrack_cells))
                if stage15_repair_backtrack_cells else None
            ),
            "stage15_repair_backtrack_cells_median": (
                float(np.median(stage15_repair_backtrack_cells))
                if stage15_repair_backtrack_cells else None
            ),
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
            "candidate_bev_snapshot_count": int(self.saved_candidate_bev_count),
            "validation_snapshot_count": int(self.saved_validation_count),
            "validation_final_snapshot_count": int(self.saved_validation_final_count),
            "validation_pose_source": (
                "context_oracle_sensor_6dof"
                if bool(self.config.validation_camera_pose_from_context)
                else (
                    "context_oracle_height"
                    if bool(self.config.validation_pose_from_context)
                    else "gps_compass_2d"
                )
            ),
            "validation_accumulated_rgb_surface_point_count": int(
                self.validation_surface_point_count
            ),
            **self._validation_pose_audit_summary(),
            **self._validation_endpoint_volume_summary(),
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

    def _frontier_cell_set(self) -> set:
        if (
            self.frontier_set_cache is None
            or self.frontier_set_cache_update != self.update_count
        ):
            self.frontier_set_cache = set(self.get_frontier_cells(sample_limit=0))
            self.frontier_set_cache_update = self.update_count
        return self.frontier_set_cache

    def _pose_from_obs(
        self, obs: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        if obs.get("gps") is None or obs.get("compass") is None:
            return None
        gps = np.asarray(obs["gps"], dtype=np.float32).reshape(-1)
        if gps.shape[0] < 2:
            return None
        yaw = float(np.asarray(obs["compass"], dtype=np.float32).reshape(-1)[0])
        context = context or {}
        if bool(self.config.validation_pose_from_context):
            height = context.get("stage23a_gt_relative_height_m")
            if height is None:
                height = obs.get("stage23a_gt_relative_height_m", 0.0)
            try:
                height = float(height)
            except (TypeError, ValueError):
                height = 0.0
        else:
            height = 0.0
        pos = np.array([float(gps[0]), -float(gps[1]), height], dtype=np.float32)
        return _yaw_to_tf(pos, yaw)

    @staticmethod
    def _validation_context_matrix(
        context: Dict[str, Any], key: str
    ) -> Optional[np.ndarray]:
        value = context.get(key)
        if value is None:
            return None
        try:
            matrix = np.asarray(value, dtype=np.float32).reshape(4, 4)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(matrix)):
            return None
        return matrix

    @staticmethod
    def _distribution_stats(values: Iterable[Any]) -> Dict[str, Any]:
        cleaned = np.asarray(list(values), dtype=np.float64).reshape(-1)
        cleaned = cleaned[np.isfinite(cleaned)]
        if cleaned.size == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }
        return {
            "count": int(cleaned.size),
            "mean": float(np.mean(cleaned)),
            "median": float(np.median(cleaned)),
            "p95": float(np.percentile(cleaned, 95)),
            "max": float(np.max(cleaned)),
        }

    def _record_validation_pose_gt_error(
        self,
        pose_tf: np.ndarray,
        gt_pose_tf: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        if not self.config.validation_enable or gt_pose_tf is None:
            return {}
        delta = np.asarray(pose_tf[:3, 3] - gt_pose_tf[:3, 3], dtype=np.float64)
        xy_error = float(np.linalg.norm(delta[:2]))
        z_error = float(abs(delta[2]))
        yaw = float(math.atan2(float(pose_tf[1, 0]), float(pose_tf[0, 0])))
        # The Habitat agent's local forward axis is -Z, whereas the
        # internal pose matrix uses +X as forward.  Taking the XY yaw from
        # matrix column 0 would report a spurious fixed 90-degree offset.
        gt_forward = -np.asarray(gt_pose_tf[:3, 2], dtype=np.float64)
        gt_yaw = float(math.atan2(float(gt_forward[1]), float(gt_forward[0])))
        yaw_error = abs((yaw - gt_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        yaw_error_deg = float(math.degrees(yaw_error))
        self.validation_pose_xy_errors.append(xy_error)
        self.validation_pose_z_errors.append(z_error)
        self.validation_pose_yaw_errors_deg.append(yaw_error_deg)
        return {
            "validation_pose_gt_xy_error_m": xy_error,
            "validation_pose_gt_z_error_m": z_error,
            "validation_pose_gt_yaw_error_deg": yaw_error_deg,
        }

    def _record_validation_endpoint_gt_error(
        self,
        world_points: np.ndarray,
        cam_points: np.ndarray,
        cam_points_h: np.ndarray,
        gt_cam_pose_tf: Optional[np.ndarray],
        camera_pitch_deg: float,
    ) -> Dict[str, Any]:
        if not self.config.validation_enable or gt_cam_pose_tf is None:
            return {}
        gt_points = (gt_cam_pose_tf @ cam_points_h.T).T[:, :3]
        if gt_points.shape != world_points.shape or gt_points.shape[0] == 0:
            return {"validation_endpoint_gt_error": "shape_mismatch"}
        delta = np.asarray(world_points - gt_points, dtype=np.float64)
        errors = np.linalg.norm(delta, axis=1)
        abs_delta = np.abs(delta)
        self.validation_endpoint_gt_errors.extend(errors.tolist())
        self.validation_endpoint_gt_abs_x_errors.extend(abs_delta[:, 0].tolist())
        self.validation_endpoint_gt_abs_y_errors.extend(abs_delta[:, 1].tolist())
        self.validation_endpoint_gt_abs_z_errors.extend(abs_delta[:, 2].tolist())
        pitch_key = "lookdown" if abs(float(camera_pitch_deg)) > 1e-4 else "horizon"
        self.validation_endpoint_gt_error_groups[pitch_key].extend(errors.tolist())
        distances = np.linalg.norm(np.asarray(cam_points, dtype=np.float64), axis=1)
        for name, lower, upper in (
            ("range_0_1m", 0.0, 1.0),
            ("range_1_3m", 1.0, 3.0),
            ("range_3_5m", 3.0, 5.0),
        ):
            mask = (distances >= lower) & (distances < upper)
            if np.any(mask):
                self.validation_endpoint_gt_error_groups[name].extend(
                    errors[mask].tolist()
                )
        stats = self._distribution_stats(errors)
        return {
            "validation_endpoint_gt_error_count": stats["count"],
            "validation_endpoint_gt_error_mean_m": stats["mean"],
            "validation_endpoint_gt_error_p95_m": stats["p95"],
            "validation_endpoint_gt_error_max_m": stats["max"],
        }

    def _validation_pose_audit_summary(self) -> Dict[str, Any]:
        gt = self.validation_gt_height_values
        pose = self.validation_pose_height_values
        errors = self.validation_height_abs_errors
        return {
            "validation_pose_audit_sample_count": int(len(gt)),
            "validation_gt_relative_height_min_m": float(min(gt)) if gt else None,
            "validation_gt_relative_height_max_m": float(max(gt)) if gt else None,
            "validation_gt_relative_height_range_m": (
                float(max(gt) - min(gt)) if gt else None
            ),
            "validation_pose_height_min_m": float(min(pose)) if pose else None,
            "validation_pose_height_max_m": float(max(pose)) if pose else None,
            "validation_pose_height_range_m": (
                float(max(pose) - min(pose)) if pose else None
            ),
            "validation_height_abs_error_mean_m": (
                float(np.mean(errors)) if errors else None
            ),
            "validation_height_abs_error_p95_m": (
                float(np.percentile(errors, 95)) if errors else None
            ),
            "validation_height_abs_error_max_m": (
                float(max(errors)) if errors else None
            ),
            "validation_pose_gt_xy_error_stats": self._distribution_stats(
                self.validation_pose_xy_errors
            ),
            "validation_pose_gt_z_error_stats": self._distribution_stats(
                self.validation_pose_z_errors
            ),
            "validation_pose_gt_yaw_error_deg_stats": self._distribution_stats(
                self.validation_pose_yaw_errors_deg
            ),
            "validation_endpoint_gt_error_stats": self._distribution_stats(
                self.validation_endpoint_gt_errors
            ),
            "validation_endpoint_gt_abs_x_error_stats": self._distribution_stats(
                self.validation_endpoint_gt_abs_x_errors
            ),
            "validation_endpoint_gt_abs_y_error_stats": self._distribution_stats(
                self.validation_endpoint_gt_abs_y_errors
            ),
            "validation_endpoint_gt_abs_z_error_stats": self._distribution_stats(
                self.validation_endpoint_gt_abs_z_errors
            ),
            "validation_endpoint_gt_error_groups": {
                key: self._distribution_stats(values)
                for key, values in sorted(
                    self.validation_endpoint_gt_error_groups.items()
                )
            },
        }

    def _validation_endpoint_rejection_reason(self, xyz: np.ndarray) -> str:
        x, y, z = [float(value) for value in np.asarray(xyz).reshape(-1)[:3]]
        row, col = self._xy_to_grid_cell(x, y)
        if row < 0 or row >= self.gs or col < 0 or col >= self.gs:
            return "outside_xy"
        height = int(z / self.cs)
        if height < 0:
            return "below_volume"
        if height >= self.vh:
            return "above_volume"
        return "other"

    def _validation_endpoint_volume_summary(self) -> Dict[str, Any]:
        total = int(self.validation_endpoint_total_count)
        mapped = int(self.validation_endpoint_mapped_count)
        below = int(self.validation_endpoint_below_volume_count)
        above = int(self.validation_endpoint_above_volume_count)
        outside_xy = int(self.validation_endpoint_outside_xy_count)
        negative_z = int(self.validation_endpoint_negative_z_count)
        negative_z_mapped = int(self.validation_endpoint_negative_z_mapped_count)
        return {
            "validation_endpoint_total_count": total,
            "validation_endpoint_mapped_count": mapped,
            "validation_endpoint_mapped_ratio": (
                float(mapped / total) if total else None
            ),
            "validation_endpoint_below_volume_count": below,
            "validation_endpoint_below_volume_ratio": (
                float(below / total) if total else None
            ),
            "validation_endpoint_above_volume_count": above,
            "validation_endpoint_above_volume_ratio": (
                float(above / total) if total else None
            ),
            "validation_endpoint_outside_xy_count": outside_xy,
            "validation_endpoint_outside_xy_ratio": (
                float(outside_xy / total) if total else None
            ),
            "validation_endpoint_negative_z_count": negative_z,
            "validation_endpoint_negative_z_ratio": (
                float(negative_z / total) if total else None
            ),
            "validation_endpoint_negative_z_mapped_count": negative_z_mapped,
            "validation_endpoint_negative_z_mapped_ratio": (
                float(negative_z_mapped / total) if total else None
            ),
            "validation_endpoint_unclassified_rejection_count": int(
                max(0, total - mapped - below - above - outside_xy)
            ),
        }

    def validation_compare_to_reference(
        self, reference: "SparseOccSemanticMemory", *, tolerance_cells: int = 1
    ) -> Dict[str, Any]:
        """Compare observed OCC labels with an independent reference branch.

        The reference branch is expected to be built from the same RGB-D frames
        using a complete simulator sensor pose. Cells outside the reference's
        observed volume remain unknown and are reported separately instead of
        being treated as free.
        """
        pred_occ = set(self.occ_counts.keys())
        pred_free = set(self.free_counts.keys()) - pred_occ
        ref_occ = set(reference.occ_counts.keys())
        ref_free = set(reference.free_counts.keys()) - ref_occ
        observed = ref_occ | ref_free
        pred_observed_occ = pred_occ & observed
        pred_observed_free = pred_free & observed

        def _binary_metrics(pred: set, positive: set) -> Dict[str, Any]:
            tp = len(pred & positive)
            fp = len(pred - positive)
            fn = len(positive - pred)
            precision = tp / float(tp + fp) if tp + fp else None
            recall = tp / float(tp + fn) if tp + fn else None
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            )
            iou = tp / float(tp + fp + fn) if tp + fp + fn else None
            return {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }

        ref_occ_by_xy: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for row, col, height in ref_occ:
            ref_occ_by_xy[(int(row), int(col))].append(int(height))
        tolerance = max(0, int(tolerance_cells))
        def _dilate_voxels(voxels: set) -> set:
            dilated = set()
            for row, col, height in voxels:
                for dr in range(-tolerance, tolerance + 1):
                    for dc in range(-tolerance, tolerance + 1):
                        for dh in range(-tolerance, tolerance + 1):
                            dilated.add((row + dr, col + dc, height + dh))
            return dilated

        ref_occ_tolerant = _dilate_voxels(ref_occ)
        pred_occ_tolerant = _dilate_voxels(pred_observed_occ)
        tol_precision_tp = len(pred_observed_occ & ref_occ_tolerant)
        tol_recall_tp = len(ref_occ & pred_occ_tolerant)
        tol_fp = len(pred_observed_occ) - tol_precision_tp
        tol_fn = len(ref_occ) - tol_recall_tp
        tol_precision = (
            tol_precision_tp / float(len(pred_observed_occ))
            if pred_observed_occ
            else None
        )
        tol_recall = (
            tol_recall_tp / float(len(ref_occ)) if ref_occ else None
        )

        height_errors = []
        for row, col, height in pred_observed_occ:
            candidates = ref_occ_by_xy.get((int(row), int(col)))
            if candidates:
                height_errors.append(
                    min(abs(int(height) - candidate) for candidate in candidates)
                    * float(self.cs)
                )

        return {
            "reference_source": "independent_sparse_occ_branch",
            "reference_observed_cell_count": int(len(observed)),
            "reference_occupied_cell_count": int(len(ref_occ)),
            "reference_free_cell_count": int(len(ref_free)),
            "prediction_occupied_cell_count": int(len(pred_occ)),
            "prediction_free_cell_count": int(len(pred_free)),
            "prediction_outside_reference_observed_occupied_count": int(
                len(pred_occ - observed)
            ),
            "prediction_outside_reference_observed_free_count": int(
                len(pred_free - observed)
            ),
            "occupied_exact": _binary_metrics(pred_observed_occ, ref_occ),
            "free_exact": _binary_metrics(pred_observed_free, ref_free),
            "occupied_tolerance": {
                "tolerance_cells": tolerance,
                "tolerance_m": float(tolerance * self.cs),
                "precision_tp": int(tol_precision_tp),
                "recall_tp": int(tol_recall_tp),
                "fp": int(tol_fp),
                "fn": int(tol_fn),
                "precision": tol_precision,
                "recall": tol_recall,
                "f1": (
                    2.0 * tol_precision * tol_recall
                    / (tol_precision + tol_recall)
                    if tol_precision is not None
                    and tol_recall is not None
                    and tol_precision + tol_recall
                    else None
                ),
            },
            "unknown_coverage": (
                float(
                    len(observed - pred_observed_occ - pred_observed_free)
                    / len(observed)
                )
                if observed
                else None
            ),
            "false_free_rate": (
                float(len(pred_observed_free & ref_occ) / len(pred_observed_free))
                if pred_observed_free
                else None
            ),
            "false_occupied_rate": (
                float(len(pred_observed_occ & ref_free) / len(pred_observed_occ))
                if pred_observed_occ
                else None
            ),
            "occupied_height_bin_error_m": self._distribution_stats(height_errors),
        }

    def _relative_base_tf(self, base_pose_tf: np.ndarray) -> np.ndarray:
        base_pose_tf = np.asarray(base_pose_tf, dtype=np.float32)
        if self.config.center_on_first_pose:
            if self.inv_init_base_tf is None:
                return np.eye(4, dtype=np.float32)
            return self.inv_init_base_tf @ base_pose_tf
        return base_pose_tf

    def _pose_to_grid(self, rel_base_tf: np.ndarray) -> Tuple[int, int, float]:
        x, y, _ = rel_base_tf[:3, 3]
        row, col = self._xy_to_grid_cell(float(x), float(y))
        yaw = float(math.atan2(float(rel_base_tf[1, 0]), float(rel_base_tf[0, 0])))
        return row, col, yaw

    def _xy_to_grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        row = int(self.gs / 2 - int(float(x) / self.cs))
        col = int(self.gs / 2 - int(float(y) / self.cs))
        return int(row), int(col)

    def _xyz_to_grid(self, xyz: np.ndarray) -> Optional[Tuple[int, int, int]]:
        x, y, z = [float(v) for v in xyz[:3]]
        row, col = self._xy_to_grid_cell(x, y)
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
                "z": float(rel_base_tf[2, 3]),
                "yaw": float(yaw),
                "semantic_top_match": self.last_semantic_decision.get("top_match"),
                "semantic_top_score": self.last_semantic_decision.get("top_score"),
                "keyframe_feature_schema_version": "v3",
            }
        )
        self.last_keyframe_xy = xy

    def _refresh_latest_keyframe_information(self, context: Dict[str, Any]) -> None:
        if not self.keyframes:
            return
        latest = self.keyframes[-1]
        step_id = self._safe_int(context.get("step_id"))
        keyframe_step = self._safe_int(latest.get("step_id"))
        if step_id is not None and keyframe_step is not None and step_id != keyframe_step:
            return
        spatial = self._anchor_spatial_information([latest["row"], latest["col"]])
        latest.update(
            {
                "keyframe_visible_free_ratio": spatial["visible_free_ratio"],
                "keyframe_occupied_ratio_observed": spatial["occupied_ratio_observed"],
                "keyframe_frontier_count": spatial["frontier_count"],
                "keyframe_branch_count": spatial["branch_count"],
                "keyframe_direction_entropy": spatial["direction_entropy"],
            }
        )

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
            "goal_world_z": float(point[2]),
        }

    def _project_to_nearest_free_along_bearing(
        self,
        goal_grid: Iterable[int],
        start_grid: Iterable[int],
        *,
        max_steps: int = 20,
    ) -> Optional[Tuple[int, int]]:
        """Move an occupied goal back toward the agent and return the first free cell."""
        try:
            goal = np.asarray(list(goal_grid)[:2], dtype=np.float32)
            start = np.asarray(list(start_grid)[:2], dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if goal.shape[0] < 2 or start.shape[0] < 2:
            return None
        direction = start - goal
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return None
        direction = direction / norm
        seen = set()
        for step in range(1, max(1, int(max_steps)) + 1):
            pos = goal + direction * float(step)
            cell = (int(round(float(pos[0]))), int(round(float(pos[1]))))
            if cell in seen:
                continue
            seen.add(cell)
            if self._cell_state(cell[0], cell[1]) == "free":
                return cell
        return None

    def _grid_to_pixel_goal(
        self,
        free_grid: Iterable[int],
        goal_world_z: float,
        pose_tf: np.ndarray,
        context: Dict[str, Any],
        depth: np.ndarray,
    ) -> Optional[List[int]]:
        """Project a BEV grid cell back into the S2 pixel-goal coordinate space."""
        if self.camera_intrinsic is None:
            return None
        try:
            world_xy = self._grid_to_xy(free_grid)
            world_z = float(goal_world_z)
        except (TypeError, ValueError, IndexError):
            return None
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim < 2:
            return None
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
        if image_w <= 0 or image_h <= 0 or w <= 0 or h <= 0:
            return None
        rel_base_tf = self._relative_base_tf(pose_tf)
        cam_pose_tf = rel_base_tf @ self.cam_to_base_tf
        world_point = np.array([world_xy[0], world_xy[1], world_z, 1.0], dtype=np.float32)
        try:
            cam_point = np.linalg.inv(cam_pose_tf) @ world_point
        except np.linalg.LinAlgError:
            return None
        cam_z = float(cam_point[2])
        if not np.isfinite(cam_z) or cam_z <= 1e-6:
            return None
        uvw = self.camera_intrinsic @ cam_point[:3]
        if not np.all(np.isfinite(uvw)) or abs(float(uvw[2])) <= 1e-6:
            return None
        sx = float(uvw[0] / uvw[2]) - 0.5
        sy = float(uvw[1] / uvw[2]) - 0.5
        px = sx * float(image_w) / float(w)
        py = sy * float(image_h) / float(h)
        if not np.isfinite(px) or not np.isfinite(py):
            return None
        if px < 0.0 or px > float(image_w - 1) or py < 0.0 or py > float(image_h - 1):
            return None
        return [int(round(px)), int(round(py))]

    def project_grid_to_pixel_goal(
        self,
        grid: Iterable[int],
        obs: Dict[str, Any],
        depth: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
        goal_world_z: float = 0.0,
    ) -> Dict[str, Any]:
        """Project a BEV grid cell into S2 pixel-goal coordinates."""
        context = dict(context or {})
        result: Dict[str, Any] = {
            "valid": False,
            "reason": None,
            "grid": None,
            "pixel_goal": None,
            "goal_world_z": float(goal_world_z),
        }
        if self.camera_intrinsic is None:
            result["reason"] = "missing_intrinsic"
            return result
        pose_tf = self._pose_from_obs(obs or {})
        if pose_tf is None:
            result["reason"] = "missing_pose"
            return result
        if self.init_base_tf is None:
            result["reason"] = "memory_not_initialized"
            return result
        try:
            cell = [int(v) for v in list(grid)[:2]]
        except (TypeError, ValueError):
            result["reason"] = "invalid_grid"
            return result
        if len(cell) < 2:
            result["reason"] = "invalid_grid"
            return result
        result["grid"] = [int(cell[0]), int(cell[1])]
        projected = self._grid_to_pixel_goal(
            cell,
            float(goal_world_z),
            pose_tf,
            context,
            depth,
        )
        if projected is None:
            result["reason"] = "projection_failed"
            return result
        result["valid"] = True
        result["reason"] = "ok"
        result["pixel_goal"] = [int(projected[0]), int(projected[1])]
        return result

    def plan_recovery_projection_bridge(
        self,
        candidate: Dict[str, Any],
        obs: Dict[str, Any],
        depth: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
        baseline_pixel_goal: Any = None,
        sample_x_ratios: Iterable[float] = (
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ),
        sample_y_ratios: Iterable[float] = (0.55, 0.65, 0.75, 0.85),
        max_angle_error_deg: float = 30.0,
    ) -> Dict[str, Any]:
        """Find a visible free local pixel that preserves a map recovery intent.

        This diagnostic reads the current sparse map and depth image but never
        changes memory, navigation actions, or the S2 output. The exact map
        anchor is projected first. A small image grid is then searched for
        free local proxy waypoints whose map bearing is close to the original
        candidate bearing.
        """
        context = dict(context or {})
        result: Dict[str, Any] = {
            "valid": False,
            "reason": None,
            "candidate_grid": None,
            "candidate_direction_bucket": candidate.get("direction_bucket"),
            "candidate_direction_angle_deg": candidate.get("direction_angle_deg"),
            "candidate_distance_m": candidate.get("distance_m"),
            "baseline_probe": None,
            "exact_projection": None,
            "sample_count": 0,
            "projected_sample_count": 0,
            "free_sample_count": 0,
            "angle_aligned_free_sample_count": 0,
            "eligible_probe_count": 0,
            "sample_goal_state_counts": {},
            "selected_pixel_goal": None,
            "selected_probe": None,
            "sample_records": [],
        }
        if not self.enabled or not self.config.waypoint_probe_enable:
            result["reason"] = "disabled"
            return result
        if self.camera_intrinsic is None:
            result["reason"] = "missing_intrinsic"
            return result
        pose_tf = self._pose_from_obs(obs or {})
        if pose_tf is None:
            result["reason"] = "missing_pose"
            return result
        if self.init_base_tf is None:
            result["reason"] = "memory_not_initialized"
            return result
        try:
            candidate_grid = [int(value) for value in list(candidate.get("grid"))[:2]]
            candidate_angle = float(candidate.get("direction_angle_deg"))
        except (TypeError, ValueError):
            result["reason"] = "invalid_candidate_geometry"
            return result
        if len(candidate_grid) != 2 or not np.isfinite(candidate_angle):
            result["reason"] = "invalid_candidate_geometry"
            return result
        result["candidate_grid"] = candidate_grid

        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim < 2:
            result["reason"] = "invalid_depth"
            return result
        depth_h, depth_w = depth_arr.shape[:2]
        image_w = int(
            context.get("image_width")
            or self.config.waypoint_source_image_width
            or depth_w
        )
        image_h = int(
            context.get("image_height")
            or self.config.waypoint_source_image_height
            or depth_h
        )
        if image_w <= 0 or image_h <= 0:
            result["reason"] = "invalid_image_shape"
            return result
        result["image_width"] = image_w
        result["image_height"] = image_h
        max_angle_error = max(0.0, float(max_angle_error_deg))
        candidate_distance = max(
            0.5,
            float(candidate.get("distance_m", 0.0) or 0.0),
        )

        def probe(pixel_goal: Any, source: str) -> Dict[str, Any]:
            record: Dict[str, Any] = {
                "source": source,
                "pixel_goal": None,
                "in_bounds": False,
                "projection_valid": False,
                "goal_state": None,
                "goal_grid": None,
                "depth_m": None,
                "direction_bucket": None,
                "direction_angle_deg": None,
                "angle_error_deg": None,
                "proxy_to_candidate_distance_m": None,
                "nearby_visit_count": None,
                "points_to_revisited_region": None,
                "eligible": False,
                "selection_score": None,
                "reason": None,
            }
            try:
                px = int(round(float(pixel_goal[0])))
                py = int(round(float(pixel_goal[1])))
            except (TypeError, ValueError, IndexError):
                record["reason"] = "invalid_pixel_goal"
                return record
            record["pixel_goal"] = [px, py]
            record["in_bounds"] = bool(0 <= px < image_w and 0 <= py < image_h)
            if not record["in_bounds"]:
                record["reason"] = "pixel_out_of_bounds"
                return record
            target = self._pixel_goal_to_grid([px, py], depth_arr, pose_tf, context)
            if target is None:
                record["reason"] = "invalid_pixel_projection"
                return record
            goal_grid = [int(value) for value in target["goal_grid"][:2]]
            state = self._cell_state(goal_grid[0], goal_grid[1])
            direction = self._direction_to_cell(
                target["start_grid"],
                goal_grid,
                float(target.get("start_yaw", 0.0) or 0.0),
            )
            proxy_angle = direction.get("angle_deg")
            angle_error = (
                None
                if proxy_angle is None
                else self._angle_distance_degrees(candidate_angle, float(proxy_angle))
            )
            proxy_to_candidate_m = float(
                np.linalg.norm(
                    np.asarray(goal_grid, dtype=np.float32)
                    - np.asarray(candidate_grid, dtype=np.float32)
                )
                * self.cs
            )
            revisit_count = self._nearby_visit_count(
                goal_grid, int(self.config.waypoint_revisit_radius_cells)
            )
            eligible = bool(
                state == "free"
                and angle_error is not None
                and angle_error <= max_angle_error
            )
            selection_score = None
            if eligible:
                selection_score = float(
                    angle_error / max(1.0, max_angle_error)
                    + proxy_to_candidate_m / candidate_distance
                )
            record.update(
                {
                    "projection_valid": True,
                    "goal_state": state,
                    "goal_grid": goal_grid,
                    "depth_m": float(target["depth_m"]),
                    "direction_bucket": direction.get("bucket"),
                    "direction_angle_deg": proxy_angle,
                    "angle_error_deg": angle_error,
                    "proxy_to_candidate_distance_m": proxy_to_candidate_m,
                    "nearby_visit_count": int(revisit_count),
                    "points_to_revisited_region": bool(revisit_count > 0),
                    "eligible": eligible,
                    "selection_score": selection_score,
                    "reason": "eligible" if eligible else (
                        "goal_not_free" if state != "free" else "angle_mismatch"
                    ),
                }
            )
            return record

        if baseline_pixel_goal is not None:
            result["baseline_probe"] = probe(
                baseline_pixel_goal, "fixed_directional_pixel"
            )

        exact = self.project_grid_to_pixel_goal(
            candidate_grid,
            obs,
            depth_arr,
            context=context,
            goal_world_z=float(
                candidate.get("semantic_resilience_source_world_z", 0.0) or 0.0
            ),
        )
        exact_record = {"projection": exact, "probe": None}
        if exact.get("valid"):
            exact_record["probe"] = probe(
                exact.get("pixel_goal"), "exact_candidate_projection"
            )
        result["exact_projection"] = exact_record

        sample_records: List[Dict[str, Any]] = []
        seen_pixels = set()
        for y_ratio in sample_y_ratios:
            for x_ratio in sample_x_ratios:
                try:
                    px = int(
                        round(min(1.0, max(0.0, float(x_ratio))) * (image_w - 1))
                    )
                    py = int(
                        round(min(1.0, max(0.0, float(y_ratio))) * (image_h - 1))
                    )
                except (TypeError, ValueError):
                    continue
                key = (px, py)
                if key in seen_pixels:
                    continue
                seen_pixels.add(key)
                sample_records.append(probe([px, py], "sample_grid"))
        result["sample_records"] = sample_records
        result["sample_count"] = len(sample_records)
        result["projected_sample_count"] = sum(
            bool(record.get("projection_valid")) for record in sample_records
        )
        state_counts: Dict[str, int] = defaultdict(int)
        for record in sample_records:
            state_counts[str(record.get("goal_state") or "invalid")] += 1
        result["sample_goal_state_counts"] = dict(state_counts)
        result["free_sample_count"] = sum(
            record.get("goal_state") == "free" for record in sample_records
        )
        eligible_samples = [
            record for record in sample_records if record.get("eligible")
        ]
        result["angle_aligned_free_sample_count"] = len(eligible_samples)
        exact_probe = dict(exact_record.get("probe") or {})
        eligible = (
            [exact_probe] if exact_probe.get("eligible") else []
        ) + eligible_samples
        result["eligible_probe_count"] = len(eligible)
        if not eligible:
            result["reason"] = "no_visible_aligned_free_proxy"
            return result
        selected = min(
            eligible,
            key=lambda record: (
                float(record.get("selection_score", float("inf"))),
                float(record.get("proxy_to_candidate_distance_m", float("inf"))),
                float(record.get("angle_error_deg", float("inf"))),
            ),
        )
        result.update(
            {
                "valid": True,
                "reason": (
                    "exact_candidate_visible_free"
                    if selected.get("source") == "exact_candidate_projection"
                    else "visible_aligned_free_proxy"
                ),
                "selected_pixel_goal": list(selected["pixel_goal"]),
                "selected_probe": selected,
            }
        )
        return result

    def plan_recovery_path_bridge(
        self,
        candidate: Dict[str, Any],
        obs: Dict[str, Any],
        depth: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
        sample_x_ratios: Iterable[float] = (
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ),
        sample_y_ratios: Iterable[float] = (0.55, 0.65, 0.75, 0.85),
        max_path_cells: int = 160,
        max_visited_cells: int = 20000,
        path_corridor_m: float = 0.35,
        min_path_progress_m: float = 0.25,
        max_local_subgoal_m: float = 3.0,
        max_initial_heading_error_deg: float = 40.0,
        reorient_lookahead_m: float = 0.75,
    ) -> Dict[str, Any]:
        """Bridge a recovery anchor through a known-free map path.

        Unlike :meth:`plan_recovery_projection_bridge`, a visible free pixel
        is accepted only when its depth endpoint lies close to the shortest
        known-free path to the selected recovery anchor and advances along
        that path.  This preserves the identity of a ``resilience_backtrack``
        candidate instead of replacing it with an arbitrary same-bearing
        free point.

        The method is read-only.  It is safe to call both before a bounded
        reorientation and on the first observation after that reorientation.
        """
        context = dict(context or {})
        result: Dict[str, Any] = {
            "valid": False,
            "reason": None,
            "path_reachable": False,
            "start_grid": None,
            "anchor_grid": None,
            "path": [],
            "path_preview": [],
            "path_cell_count": 0,
            "path_edge_count": 0,
            "path_m": None,
            "path_visited_cell_count": 0,
            "path_pose_source": None,
            "path_pose_yaw_rad": None,
            "initial_path_grid": None,
            "initial_direction_bucket": None,
            "initial_direction_angle_deg": None,
            "projection_candidate_geometry": None,
            "base_projection_bridge": None,
            "path_eligible_probe_count": 0,
            "selected_pixel_goal": None,
            "selected_probe": None,
            "probe_records": [],
        }
        if not self.enabled or not self.config.waypoint_probe_enable:
            result["reason"] = "disabled"
            return result
        pose_state = self._current_pose_state(obs or {})
        if pose_state is None:
            result["reason"] = "missing_pose_or_memory"
            return result
        result["path_pose_source"] = (
            "current_observation"
            if bool((obs or {}).get("_prefer_observation_pose"))
            else "persistent_pose_trace"
        )
        result["path_pose_yaw_rad"] = float(pose_state.get("yaw", 0.0) or 0.0)
        try:
            start_values = list(pose_state.get("grid") or [])
            anchor_values = list(candidate.get("grid") or [])
            start = (int(start_values[0]), int(start_values[1]))
            anchor = (int(anchor_values[0]), int(anchor_values[1]))
        except (TypeError, ValueError, IndexError):
            result["reason"] = "invalid_start_or_anchor"
            return result
        result["start_grid"] = [int(start[0]), int(start[1])]
        result["anchor_grid"] = [int(anchor[0]), int(anchor[1])]
        if self._cell_state(anchor[0], anchor[1]) != "free":
            result["reason"] = "anchor_not_free"
            return result

        free_cells = set(self.free2d_counts.keys()) - set(self.occ2d_counts.keys())
        if not free_cells:
            result["reason"] = "no_known_free_cells"
            return result
        max_depth = max(1, int(max_path_cells))
        max_visited = max(1, int(max_visited_cells))
        queue = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        path_depth: Dict[Tuple[int, int], int] = {start: 0}
        reached = False
        visited_count = 0
        while queue and visited_count < max_visited:
            cell = queue.popleft()
            visited_count += 1
            if cell == anchor:
                reached = True
                break
            if int(path_depth[cell]) >= max_depth:
                continue
            for nbr_raw in self._neighbors2d(cell[0], cell[1]):
                nbr = (int(nbr_raw[0]), int(nbr_raw[1]))
                if nbr in parent:
                    continue
                if nbr != anchor and nbr not in free_cells:
                    continue
                if nbr in self.occ2d_counts:
                    continue
                # Do not cut diagonally through the corner of an occupied or
                # unknown cell.  Cardinal moves need no additional check.
                dr = int(nbr[0] - cell[0])
                dc = int(nbr[1] - cell[1])
                if dr and dc:
                    side_a = (cell[0] + dr, cell[1])
                    side_b = (cell[0], cell[1] + dc)
                    if side_a not in free_cells or side_b not in free_cells:
                        continue
                parent[nbr] = cell
                path_depth[nbr] = int(path_depth[cell]) + 1
                queue.append(nbr)
        result["path_visited_cell_count"] = int(visited_count)
        if not reached:
            result["reason"] = (
                "path_search_budget_exhausted"
                if visited_count >= max_visited
                else "no_known_free_path_to_anchor"
            )
            return result

        path: List[Tuple[int, int]] = []
        cursor: Optional[Tuple[int, int]] = anchor
        while cursor is not None:
            path.append(cursor)
            cursor = parent.get(cursor)
        path.reverse()
        path_edges = max(0, len(path) - 1)
        path_m = float(path_edges * self.cs)
        result.update(
            {
                "path_reachable": True,
                "path": [[int(row), int(col)] for row, col in path],
                "path_preview": [
                    [int(row), int(col)]
                    for row, col in (path[:12] + ([path[-1]] if len(path) > 12 else []))
                ],
                "path_cell_count": int(len(path)),
                "path_edge_count": int(path_edges),
                "path_m": float(path_m),
            }
        )

        yaw = float(pose_state.get("yaw", 0.0) or 0.0)
        lookahead_cells = max(
            1, int(round(max(float(self.cs), float(reorient_lookahead_m)) / self.cs))
        )
        lookahead_index = min(path_edges, lookahead_cells)
        initial_grid = path[lookahead_index]
        initial_direction = self._direction_to_cell(start, initial_grid, yaw)
        result.update(
            {
                "initial_path_grid": [int(initial_grid[0]), int(initial_grid[1])],
                "initial_path_index": int(lookahead_index),
                "initial_direction_bucket": initial_direction.get("bucket"),
                "initial_direction_angle_deg": initial_direction.get("angle_deg"),
            }
        )

        # Stage27 candidates intentionally store map/path geometry rather than
        # a pose-relative direction. Recompute the projection-only fields from
        # the current re-audited path so a reorientation never reuses a stale
        # bearing and the frozen candidate record remains unchanged.
        projection_candidate = dict(candidate)
        projection_candidate.update(
            {
                "direction_bucket": initial_direction.get("bucket"),
                "direction_angle_deg": initial_direction.get("angle_deg"),
                "distance_m": float(path_m),
            }
        )
        result["projection_candidate_geometry"] = {
            "source": "current_sparseocc_path_reaudit",
            "direction_bucket": projection_candidate.get("direction_bucket"),
            "direction_angle_deg": projection_candidate.get("direction_angle_deg"),
            "distance_m": projection_candidate.get("distance_m"),
            "candidate_mutated": False,
        }
        base_bridge = self.plan_recovery_projection_bridge(
            projection_candidate,
            obs,
            depth,
            context=context,
            baseline_pixel_goal=None,
            sample_x_ratios=sample_x_ratios,
            sample_y_ratios=sample_y_ratios,
            # Bearing is retained in the audit, but route membership below is
            # the active eligibility criterion.
            max_angle_error_deg=180.0,
        )
        result["base_projection_bridge"] = base_bridge
        raw_probes: List[Dict[str, Any]] = []
        exact_probe = dict(
            ((base_bridge.get("exact_projection") or {}).get("probe") or {})
        )
        if exact_probe:
            raw_probes.append(exact_probe)
        raw_probes.extend(dict(item) for item in base_bridge.get("sample_records") or [])

        corridor_m = max(float(self.cs), float(path_corridor_m))
        min_progress_m = max(float(self.cs), float(min_path_progress_m))
        max_subgoal_m = max(min_progress_m, float(max_local_subgoal_m))
        max_heading_error = max(0.0, float(max_initial_heading_error_deg))
        path_arr = np.asarray(path, dtype=np.float32)
        eligible_records: List[Dict[str, Any]] = []
        probe_records: List[Dict[str, Any]] = []
        for raw in raw_probes:
            record = dict(raw)
            record.update(
                {
                    "path_nearest_index": None,
                    "path_deviation_m": None,
                    "path_progress_m": None,
                    "path_remaining_m": None,
                    "proxy_from_current_m": None,
                    "initial_heading_error_deg": None,
                    "path_eligible": False,
                    "path_reason": None,
                    "path_selection_score": None,
                }
            )
            goal_grid = record.get("goal_grid")
            if not record.get("projection_valid") or record.get("goal_state") != "free":
                record["path_reason"] = str(record.get("reason") or "probe_not_free")
                probe_records.append(record)
                continue
            try:
                goal = np.asarray(list(goal_grid)[:2], dtype=np.float32)
            except (TypeError, ValueError):
                record["path_reason"] = "invalid_goal_grid"
                probe_records.append(record)
                continue
            distances = np.linalg.norm(path_arr - goal[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            path_deviation_m = float(distances[nearest_index] * self.cs)
            progress_m = float(nearest_index * self.cs)
            remaining_m = float(max(0, path_edges - nearest_index) * self.cs)
            proxy_from_current_m = float(
                np.linalg.norm(goal - np.asarray(start, dtype=np.float32)) * self.cs
            )
            proxy_angle = record.get("direction_angle_deg")
            initial_angle = initial_direction.get("angle_deg")
            heading_error = (
                None
                if proxy_angle is None or initial_angle is None
                else float(
                    self._angle_distance_degrees(
                        float(initial_angle), float(proxy_angle)
                    )
                )
            )
            is_exact = record.get("source") == "exact_candidate_projection"
            eligible = bool(
                path_deviation_m <= corridor_m
                and progress_m >= min_progress_m
                and (is_exact or proxy_from_current_m <= max_subgoal_m)
                and heading_error is not None
                and heading_error <= max_heading_error
            )
            if path_deviation_m > corridor_m:
                path_reason = "outside_path_corridor"
            elif progress_m < min_progress_m:
                path_reason = "insufficient_path_progress"
            elif not is_exact and proxy_from_current_m > max_subgoal_m:
                path_reason = "local_subgoal_too_far"
            elif heading_error is None or heading_error > max_heading_error:
                path_reason = "initial_heading_mismatch"
            else:
                path_reason = "eligible"
            selection_score = None
            if eligible:
                selection_score = float(
                    path_deviation_m / corridor_m
                    + float(heading_error) / max(1.0, max_heading_error)
                    - min(1.0, progress_m / max(float(self.cs), max_subgoal_m))
                )
            record.update(
                {
                    "path_nearest_index": int(nearest_index),
                    "path_deviation_m": float(path_deviation_m),
                    "path_progress_m": float(progress_m),
                    "path_remaining_m": float(remaining_m),
                    "proxy_from_current_m": float(proxy_from_current_m),
                    "initial_heading_error_deg": heading_error,
                    "path_eligible": bool(eligible),
                    "path_reason": path_reason,
                    "path_selection_score": selection_score,
                }
            )
            probe_records.append(record)
            if eligible:
                eligible_records.append(record)
        result["probe_records"] = probe_records
        result["path_eligible_probe_count"] = int(len(eligible_records))
        result["path_constraints"] = {
            "path_corridor_m": float(corridor_m),
            "min_path_progress_m": float(min_progress_m),
            "max_local_subgoal_m": float(max_subgoal_m),
            "max_initial_heading_error_deg": float(max_heading_error),
        }
        if not eligible_records:
            result["reason"] = "no_visible_path_consistent_free_proxy"
            return result
        selected = min(
            eligible_records,
            key=lambda item: (
                float(item.get("path_selection_score", float("inf"))),
                float(item.get("path_deviation_m", float("inf"))),
                -float(item.get("path_progress_m", 0.0) or 0.0),
            ),
        )
        result.update(
            {
                "valid": True,
                "reason": (
                    "exact_anchor_visible_path_consistent"
                    if selected.get("source") == "exact_candidate_projection"
                    else "visible_path_consistent_free_proxy"
                ),
                "selected_pixel_goal": list(selected.get("pixel_goal") or []),
                "selected_probe": selected,
            }
        )
        return result

    def plan_depth_short_lookahead_shadow(
        self,
        obs: Dict[str, Any],
        depth: np.ndarray,
        *,
        context: Optional[Dict[str, Any]] = None,
        sample_x_ratios: Iterable[float] = (0.25, 0.40, 0.50, 0.60, 0.75),
        sample_y_ratios: Iterable[float] = (0.65, 0.75, 0.85, 0.92),
        lookahead_distances_m: Iterable[float] = (0.25, 0.50, 0.75, 1.00),
        footprint_radius_m: float = 0.18,
        floor_height_max_m: float = 1.5,
        local_search_enable: bool = False,
        local_search_lateral_m: float = 0.20,
        local_search_detour_m: float = 0.25,
        local_search_max_paths: int = 16,
    ) -> Dict[str, Any]:
        """Find short current-depth free prefixes without changing memory.

        A depth surface is only an endpoint observation.  For each projected
        ray we report that endpoint separately, then test prefixes before it
        against the current 2-D map and the unchanged floor/footprint gate.
        No prefix is accepted when any traversed cell is unknown or occupied.
        """
        result: Dict[str, Any] = {
            "schema_version": "stage50_depth_short_lookahead_shadow_v1",
            "valid": False,
            "reason": None,
            "start_grid": None,
            "surface_endpoints": [],
            "lookahead_records": [],
            "eligible_records": [],
            "eligible_count": 0,
            "unknown_is_free": False,
            "semantic_can_override_safety": False,
            "shadow_only": True,
            "action_applied": False,
            "gt_fields_used": [],
            "local_search_enabled": bool(local_search_enable),
        }
        if not self.enabled or not self.config.waypoint_probe_enable:
            result["reason"] = "disabled"
            return result
        pose_tf = self._pose_from_obs(obs or {})
        if pose_tf is None or self.init_base_tf is None or self.camera_intrinsic is None:
            result["reason"] = "missing_pose_intrinsic_or_memory"
            return result
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim != 2 or not depth_arr.size:
            result["reason"] = "invalid_depth"
            return result
        ctx = dict(context or {})
        h, w = depth_arr.shape
        image_w = int(ctx.get("image_width") or self.config.waypoint_source_image_width or w)
        image_h = int(ctx.get("image_height") or self.config.waypoint_source_image_height or h)
        try:
            start_state = self._pose_to_grid(self._relative_base_tf(pose_tf))
            start = (int(start_state[0]), int(start_state[1]))
        except (TypeError, ValueError, IndexError):
            result["reason"] = "missing_start_grid"
            return result
        result["start_grid"] = [int(start[0]), int(start[1])]
        footprint_radius_m = max(0.0, float(footprint_radius_m))
        radius_cells = max(0, int(math.ceil(footprint_radius_m / max(self.cs, 1e-6))))
        rel_base_tf = self._relative_base_tf(pose_tf)
        current_floor_z = float(rel_base_tf[2, 3])

        def line_cells(target: Tuple[int, int]) -> List[Tuple[int, int]]:
            dr, dc = target[0] - start[0], target[1] - start[1]
            count = max(abs(dr), abs(dc), 1)
            return [
                (int(round(start[0] + dr * i / count)), int(round(start[1] + dc * i / count)))
                for i in range(count + 1)
            ]

        evidence_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        def cell_evidence(row: int, col: int, floor_z: float) -> Dict[str, Any]:
            key = (int(row), int(col))
            if key not in evidence_cache:
                evidence_cache[key] = self.validation_floor_aligned_cell_evidence(
                    key[0], key[1], floor_z,
                    height_max_m=float(floor_height_max_m),
                )
            return evidence_cache[key]

        def first_occupied_height(evidence: Dict[str, Any], row: int, col: int) -> Optional[Dict[str, Any]]:
            for height in range(
                int(evidence.get("height_index_min", 0)),
                int(evidence.get("height_index_max", -1)) + 1,
            ):
                hits = int(self.occ_counts.get((int(row), int(col), int(height)), 0) or 0)
                if hits > 0:
                    return {
                        "height_index": int(height),
                        "height_m": float(height) * float(self.cs),
                        "occupied_hits": hits,
                    }
            return None

        def footprint_path_audit(path: Iterable[Tuple[int, int]], floor_z: float) -> Dict[str, Any]:
            unknown_count = 0
            blocked_count = 0
            occupied_hits = 0
            occupied_voxels = 0
            checked_count = 0
            initial_exempted_count = 0
            first_blocker = None
            for path_index, cell in enumerate(path):
                # The current pose is already occupied by the robot body and
                # may contain self/sensor returns.  It is an origin, not a
                # cell the proposed primitive must enter.  All future cells
                # retain the complete footprint/floor safety audit.
                if path_index == 0:
                    continue
                for dr in range(-radius_cells, radius_cells + 1):
                    for dc in range(-radius_cells, radius_cells + 1):
                        offset_m = math.hypot(dr, dc) * float(self.cs)
                        if offset_m > footprint_radius_m + 1e-6:
                            continue
                        row, col = int(cell[0] + dr), int(cell[1] + dc)
                        # The robot already occupies this footprint at the
                        # current pose. Self/sensor returns there are not
                        # evidence against entering a future cell, while any
                        # cell outside this initial footprint remains audited.
                        start_offset_m = math.hypot(
                            row - start[0], col - start[1]
                        ) * float(self.cs)
                        if start_offset_m <= footprint_radius_m + 1e-6:
                            initial_exempted_count += 1
                            continue
                        evidence = cell_evidence(row, col, floor_z)
                        checked_count += 1
                        occupied_hits += int(evidence.get("occupied_hits", 0) or 0)
                        occupied_voxels += int(evidence.get("occupied_voxel_count", 0) or 0)
                        if evidence.get("state") == "blocked":
                            blocked_count += 1
                            if first_blocker is None:
                                first_blocker = {
                                    "reason": "floor_footprint_occupied",
                                    "path_index": int(path_index),
                                    "path_cell": [int(cell[0]), int(cell[1])],
                                    "footprint_cell": [row, col],
                                    "footprint_offset": [int(dr), int(dc)],
                                    "evidence": dict(evidence),
                                    "first_occupied_height": first_occupied_height(evidence, row, col),
                                }
                        elif evidence.get("state") != "free":
                            unknown_count += 1
                            if first_blocker is None:
                                first_blocker = {
                                    "reason": "floor_footprint_unknown",
                                    "path_index": int(path_index),
                                    "path_cell": [int(cell[0]), int(cell[1])],
                                    "footprint_cell": [row, col],
                                    "footprint_offset": [int(dr), int(dc)],
                                    "evidence": dict(evidence),
                                }
            state = "occupied" if blocked_count else ("unknown" if unknown_count else "free")
            return {
                "state": state,
                "checked_cell_count": int(checked_count),
                "blocked_cell_count": int(blocked_count),
                "unknown_cell_count": int(unknown_count),
                "occupied_hit_count": int(occupied_hits),
                "occupied_voxel_count": int(occupied_voxels),
                "first_blocker": first_blocker,
                "start_cell_exempted": True,
                "initial_footprint_exempted_cell_count": int(initial_exempted_count),
            }

        def reconstruct_path(parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]], end: Tuple[int, int]) -> List[Tuple[int, int]]:
            path = []
            cursor = end
            while cursor is not None:
                path.append(cursor)
                cursor = parent.get(cursor)
            return list(reversed(path))

        def local_paths(target: Tuple[int, int], requested_cells: int) -> List[List[Tuple[int, int]]]:
            if not local_search_enable:
                return []
            lateral_cells = max(1, int(math.ceil(float(local_search_lateral_m) / max(self.cs, 1e-6))))
            detour_cells = max(0, int(math.ceil(float(local_search_detour_m) / max(self.cs, 1e-6))))
            max_steps = max(1, int(requested_cells) + detour_cells)
            target_vector = np.asarray(
                [target[0] - start[0], target[1] - start[1]], dtype=np.float32
            )
            target_norm = float(np.linalg.norm(target_vector))
            if target_norm <= 1e-6:
                return []
            forward = target_vector / target_norm
            queue = deque([start])
            parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            distance = {start: 0}
            endpoints = []
            neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1))
            while queue:
                cell = queue.popleft()
                steps = distance[cell]
                if steps > 0:
                    offset = np.asarray(
                        [cell[0] - start[0], cell[1] - start[1]], dtype=np.float32
                    )
                    longitudinal = float(np.dot(offset, forward))
                    lateral = float(np.linalg.norm(offset - longitudinal * forward))
                    if longitudinal >= float(requested_cells) and lateral <= float(lateral_cells):
                        endpoints.append(cell)
                if steps >= max_steps:
                    continue
                for dr, dc in neighbors:
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if nxt in parent or self._cell_state(*nxt) != "free":
                        continue
                    if dr and dc and (
                        self._cell_state(cell[0] + dr, cell[1]) != "free"
                        or self._cell_state(cell[0], cell[1] + dc) != "free"
                    ):
                        continue
                    parent[nxt] = cell
                    distance[nxt] = steps + 1
                    queue.append(nxt)
            endpoints.sort(key=lambda cell: (
                math.hypot(cell[0] - target[0], cell[1] - target[1]),
                distance[cell], cell[0], cell[1],
            ))
            return [reconstruct_path(parent, cell) for cell in endpoints[:max(1, int(local_search_max_paths))]]

        seen = set()
        for y_ratio in sample_y_ratios:
            for x_ratio in sample_x_ratios:
                px = int(round(min(1.0, max(0.0, float(x_ratio))) * (image_w - 1)))
                py = int(round(min(1.0, max(0.0, float(y_ratio))) * (image_h - 1)))
                key = (px, py)
                if key in seen:
                    continue
                seen.add(key)
                target = self._pixel_goal_to_grid([px, py], depth_arr, pose_tf, ctx)
                endpoint = {"pixel": [px, py], "projection_valid": bool(target is not None)}
                if target is None:
                    endpoint["reason"] = "invalid_projection"
                    result["surface_endpoints"].append(endpoint)
                    continue
                goal = (int(target["goal_grid"][0]), int(target["goal_grid"][1]))
                endpoint.update({"surface_grid": [goal[0], goal[1]], "surface_state": self._cell_state(*goal),
                                 "surface_depth_m": float(target["depth_m"]),
                                 "goal_world_z": float(target.get("goal_world_z", 0.0))})
                result["surface_endpoints"].append(endpoint)
                cells = line_cells(goal)
                for distance_m in lookahead_distances_m:
                    requested = max(float(self.cs), float(distance_m))
                    index = min(len(cells) - 1, max(1, int(round(requested / max(self.cs, 1e-6)))))
                    prefix = cells[: index + 1]
                    start_cell_state = self._cell_state(*prefix[0])
                    states = ["free"] + [self._cell_state(*cell) for cell in prefix[1:]]
                    path_state = "occupied" if "occupied" in states else ("unknown" if "unknown" in states else "free")
                    direct_audit = footprint_path_audit(prefix, current_floor_z) if path_state == "free" else {
                        "state": path_state,
                        "first_blocker": next(({
                            "reason": "path_2d_" + state,
                            "path_index": int(i),
                            "path_cell": [int(prefix[i][0]), int(prefix[i][1])],
                        } for i, state in enumerate(states) if state != "free"), None),
                        "checked_cell_count": 0, "blocked_cell_count": 0,
                        "unknown_cell_count": 0, "occupied_hit_count": 0,
                        "occupied_voxel_count": 0,
                    }
                    selected_path = prefix
                    selected_audit = direct_audit
                    path_source = "direct_depth_ray"
                    pixel = self._grid_to_pixel_goal(prefix[-1], current_floor_z, pose_tf, ctx, depth_arr)
                    local_attempt_count = 0
                    if path_state != "free" or direct_audit["state"] != "free" or pixel is None:
                        for alternative in local_paths(prefix[-1], index):
                            local_attempt_count += 1
                            audit = footprint_path_audit(alternative, current_floor_z)
                            alternative_pixel = self._grid_to_pixel_goal(
                                alternative[-1], current_floor_z, pose_tf, ctx, depth_arr
                            )
                            if audit["state"] == "free" and alternative_pixel is not None:
                                selected_path = alternative
                                selected_audit = audit
                                path_state = "free"
                                pixel = alternative_pixel
                                path_source = "local_8n_depth_supported"
                                break
                    floor_state = selected_audit["state"] if path_state == "free" else path_state
                    record = {
                        "pixel": [px, py],
                        "surface_grid": [goal[0], goal[1]],
                        "surface_state": endpoint["surface_state"],
                        "surface_depth_m": endpoint["surface_depth_m"],
                        "lookahead_grid": [selected_path[-1][0], selected_path[-1][1]],
                        "lookahead_pixel_goal": pixel,
                        "lookahead_pixel_in_bounds": pixel is not None,
                        "lookahead_distance_m": float(len(selected_path) - 1) * self.cs,
                        "path_cells": [[int(r), int(c)] for r, c in selected_path],
                        "direct_path_cells": [[int(r), int(c)] for r, c in prefix],
                        "path_source": path_source,
                        "start_cell_state": start_cell_state,
                        "start_cell_exempted": True,
                        "local_search_attempt_count": int(local_attempt_count),
                        "path_state": path_state,
                        "floor_footprint_state": floor_state,
                        "floor_footprint_audit": selected_audit,
                        "direct_floor_footprint_audit": direct_audit,
                        "eligible": bool(
                            path_state == "free" and floor_state == "free"
                            and len(selected_path) > 1 and pixel is not None
                        ),
                        "safety_authority": "current_sparseocc_reaudit",
                        "unknown_is_free": False,
                    }
                    result["lookahead_records"].append(record)
                    if record["eligible"]:
                        result["eligible_records"].append(record)
        result["eligible_count"] = len(result["eligible_records"])
        result["valid"] = bool(result["eligible_records"])
        result["reason"] = "eligible_current_depth_lookahead" if result["valid"] else "no_current_depth_short_safe_path"
        return result

    def _stage15_repair_shadow_info(
        self,
        *,
        pixel_goal: Any,
        depth: np.ndarray,
        pose_tf: np.ndarray,
        context: Dict[str, Any],
        target: Dict[str, Any],
        goal_state: str,
    ) -> Dict[str, Any]:
        enabled = bool(
            self.config.stage15_repair_shadow_enable
            or self.config.stage15_repair_active
        )
        if not enabled:
            return {}
        info: Dict[str, Any] = {
            "stage15_repair_enabled": True,
            "stage15_repair_active": bool(self.config.stage15_repair_active),
            "stage15_repair_prev_consecutive_count": 0,
            "stage15_repair_prev_cumulative_count": 0,
            "stage15_repair_consecutive_count": 0,
            "stage15_repair_cumulative_count": 0,
            "roundtrip_pixel_goal": None,
            "roundtrip_error_px": None,
            "roundtrip_valid": False,
            "repair_candidate": False,
            "repair_free_grid": None,
            "repair_pixel_goal": None,
            "repair_pixel_shift": None,
            "repair_backtrack_cells": None,
            "repair_free_state_check": None,
            "repair_valid": False,
            "repair_reason": "goal_state_not_occupied",
        }
        try:
            prev_consecutive = max(
                0,
                int(context.get("stage15_repair_prev_consecutive_count", 0) or 0),
            )
        except (TypeError, ValueError):
            prev_consecutive = 0
        try:
            prev_cumulative = max(
                0,
                int(context.get("stage15_repair_prev_cumulative_count", 0) or 0),
            )
        except (TypeError, ValueError):
            prev_cumulative = 0
        if str(goal_state) == "occupied":
            current_consecutive = prev_consecutive + 1
            current_cumulative = prev_cumulative + 1
        else:
            current_consecutive = 0
            current_cumulative = prev_cumulative
        info["stage15_repair_prev_consecutive_count"] = int(prev_consecutive)
        info["stage15_repair_prev_cumulative_count"] = int(prev_cumulative)
        info["stage15_repair_consecutive_count"] = int(current_consecutive)
        info["stage15_repair_cumulative_count"] = int(current_cumulative)
        goal_world_z = target.get("goal_world_z")
        if goal_world_z is None:
            info["repair_reason"] = "missing_goal_world_z"
            return info
        info["goal_world_z"] = float(goal_world_z)
        rt = self._grid_to_pixel_goal(
            target.get("goal_grid"),
            float(goal_world_z),
            pose_tf,
            context,
            depth,
        )
        info["roundtrip_pixel_goal"] = None if rt is None else [int(rt[0]), int(rt[1])]
        info["roundtrip_valid"] = rt is not None
        if rt is not None:
            try:
                dx = float(rt[0]) - float(pixel_goal[0])
                dy = float(rt[1]) - float(pixel_goal[1])
                info["roundtrip_error_px"] = float(math.hypot(dx, dy))
            except (TypeError, ValueError, IndexError):
                info["roundtrip_error_px"] = None
        if str(goal_state) != "occupied":
            return info
        info["repair_candidate"] = True
        free_grid = self._project_to_nearest_free_along_bearing(
            target.get("goal_grid"),
            target.get("start_grid"),
            max_steps=int(self.config.stage15_repair_backtrack_max_steps),
        )
        if free_grid is None:
            info["repair_reason"] = "no_free_along_bearing"
            return info
        info["repair_free_grid"] = [int(free_grid[0]), int(free_grid[1])]
        info["repair_free_state_check"] = self._cell_state(free_grid[0], free_grid[1])
        try:
            goal = np.asarray(list(target.get("goal_grid"))[:2], dtype=np.float32)
            free = np.asarray(list(free_grid)[:2], dtype=np.float32)
            info["repair_backtrack_cells"] = float(np.linalg.norm(free - goal))
        except (TypeError, ValueError):
            info["repair_backtrack_cells"] = None
        repaired = self._grid_to_pixel_goal(
            free_grid,
            float(goal_world_z),
            pose_tf,
            context,
            depth,
        )
        info["repair_pixel_goal"] = None if repaired is None else [int(repaired[0]), int(repaired[1])]
        if repaired is None:
            info["repair_reason"] = "repair_projection_failed"
            return info
        try:
            dx = float(repaired[0]) - float(pixel_goal[0])
            dy = float(repaired[1]) - float(pixel_goal[1])
            info["repair_pixel_shift"] = float(math.hypot(dx, dy))
        except (TypeError, ValueError, IndexError):
            info["repair_pixel_shift"] = None
        info["repair_valid"] = info["repair_free_state_check"] == "free"
        info["repair_reason"] = "ok" if info["repair_valid"] else "free_state_check_failed"
        return info

    def _cell_state(self, row: int, col: int) -> str:
        key = (int(row), int(col))
        if key in self.occ2d_counts:
            return "occupied"
        if key in self.free2d_counts:
            return "free"
        return "unknown"

    def validation_floor_aligned_cell_evidence(
        self,
        row: int,
        col: int,
        floor_z_m: float,
        *,
        height_max_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Read local-height OCC evidence without mutating navigation state."""
        min_height = int(
            math.floor(
                (float(floor_z_m) + float(self.config.obstacle_height_min))
                / self.cs
            )
        )
        max_height = int(
            math.ceil(
                (
                    float(floor_z_m)
                    + float(
                        self.config.obstacle_height_max
                        if height_max_m is None
                        else height_max_m
                    )
                )
                / self.cs
            )
        )
        occupied_hits = 0
        free_hits = 0
        occupied_voxels = 0
        free_voxels = 0
        for height in range(min_height, max_height + 1):
            key = (int(row), int(col), int(height))
            occ = int(self.occ_counts.get(key, 0) or 0)
            free = int(self.free_counts.get(key, 0) or 0)
            occupied_hits += occ
            free_hits += free
            occupied_voxels += int(occ > 0)
            free_voxels += int(free > 0)
        state = "unknown"
        if occupied_hits > 0:
            state = "blocked"
        elif free_hits > 0:
            state = "free"
        return {
            "state": state,
            "floor_z_m": float(floor_z_m),
            "height_index_min": int(min_height),
            "height_index_max": int(max_height),
            "height_max_m": float(
                self.config.obstacle_height_max
                if height_max_m is None
                else height_max_m
            ),
            "occupied_hits": int(occupied_hits),
            "free_hits": int(free_hits),
            "occupied_voxel_count": int(occupied_voxels),
            "free_voxel_count": int(free_voxels),
        }

    def audit_route_cell_evidence(
        self,
        route_audit: Dict[str, Any],
        *,
        include_height_aligned: bool = False,
    ) -> Dict[str, Any]:
        """Expose per-cell evidence without changing the OCC decision rule."""
        ordered_cells = list(route_audit.get("route_cells") or [])
        unique_cells = []
        seen = set()
        for values in ordered_cells:
            try:
                cell = (int(values[0]), int(values[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if cell not in seen:
                seen.add(cell)
                unique_cells.append(cell)
        route_cell_set = set(unique_cells)
        occupied_band_by_cell: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)
        free_band_by_cell: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)
        if include_height_aligned:
            for (row, col, height), count in self.occ_counts.items():
                cell = (int(row), int(col))
                if (
                    cell in route_cell_set
                    and self._is_obstacle_height(int(height))
                    and int(count) > 0
                ):
                    occupied_band_by_cell[cell][int(height)] = int(count)
            for (row, col, height), count in self.free_counts.items():
                cell = (int(row), int(col))
                if (
                    cell in route_cell_set
                    and self._is_obstacle_height(int(height))
                    and int(count) > 0
                ):
                    free_band_by_cell[cell][int(height)] = int(count)

        cells = []
        for row, col in unique_cells:
            occupied_hits = int(self.occ2d_counts.get((row, col), 0))
            free_hits = int(self.free2d_counts.get((row, col), 0))
            visited_hits = int(self.visited2d_counts.get((row, col), 0))
            total_hits = occupied_hits + free_hits
            cell_record = {
                "grid": [int(row), int(col)],
                "state": self._cell_state(row, col),
                "occupied_hits": occupied_hits,
                "free_hits": free_hits,
                "visited_hits": visited_hits,
                "evidence_hits": total_hits,
                "occupied_free_ratio": (
                    float(occupied_hits / total_hits) if total_hits else None
                ),
                "occupied_free_margin": (
                    float((occupied_hits - free_hits) / total_hits)
                    if total_hits
                    else None
                ),
            }
            if include_height_aligned:
                # Keep this audit tied to the same 3D voxel height band used
                # for obstacle reasoning. The existing flattened 2D counts
                # remain untouched and continue to drive all decisions.
                occupied_by_height = occupied_band_by_cell[(row, col)]
                free_by_height = free_band_by_cell[(row, col)]
                occupied_band_hits = int(sum(occupied_by_height.values()))
                free_band_hits = int(sum(free_by_height.values()))
                band_total_hits = occupied_band_hits + free_band_hits
                occupied_heights = sorted(occupied_by_height)
                free_heights = sorted(free_by_height)
                shared_heights = sorted(
                    set(occupied_heights).intersection(free_heights)
                )
                shared_occupied_hits = int(
                    sum(occupied_by_height[height] for height in shared_heights)
                )
                shared_free_hits = int(
                    sum(free_by_height[height] for height in shared_heights)
                )
                cell_record.update(
                    {
                        "obstacle_height_min_m": float(self.config.obstacle_height_min),
                        "obstacle_height_max_m": float(self.config.obstacle_height_max),
                        "occupied_band_hits": occupied_band_hits,
                        "free_band_hits": free_band_hits,
                        "band_evidence_hits": int(band_total_hits),
                        "occupied_band_ratio": (
                            float(occupied_band_hits / band_total_hits)
                            if band_total_hits
                            else None
                        ),
                        "occupied_band_margin": (
                            float((occupied_band_hits - free_band_hits) / band_total_hits)
                            if band_total_hits
                            else None
                        ),
                        "occupied_band_height_indices": occupied_heights,
                        "free_band_height_indices": free_heights,
                        "shared_band_height_indices": shared_heights,
                        "shared_band_occupied_hits": shared_occupied_hits,
                        "shared_band_free_hits": shared_free_hits,
                        "shared_band_evidence_hits": int(
                            shared_occupied_hits + shared_free_hits
                        ),
                    }
                )
            cells.append(cell_record)
        height_aligned_cells = [
            item for item in cells if "occupied_band_hits" in item
        ]
        occupied_band_cells = [
            item for item in height_aligned_cells if item["occupied_band_hits"] > 0
        ]
        band_free_dominant_cells = [
            item
            for item in occupied_band_cells
            if item["free_band_hits"] > item["occupied_band_hits"]
        ]
        shared_band_cells = [
            item for item in occupied_band_cells if item["shared_band_height_indices"]
        ]
        return {
            "schema_version": (
                "stage22e_height_aligned_route_cell_evidence_v1"
                if include_height_aligned
                else "stage22d_route_cell_evidence_v1"
            ),
            "cell_count": int(len(cells)),
            "occupied_cell_count": int(
                sum(item["state"] == "occupied" for item in cells)
            ),
            "free_cell_count": int(
                sum(item["state"] == "free" for item in cells)
            ),
            "unknown_cell_count": int(
                sum(item["state"] == "unknown" for item in cells)
            ),
            "cells": cells,
            "decision_rule": "occupied_key_precedes_free_key",
            "height_aligned": bool(include_height_aligned),
            "height_aligned_cell_count": int(len(height_aligned_cells)),
            "occupied_band_cell_count": int(len(occupied_band_cells)),
            "band_free_dominant_cell_count": int(len(band_free_dominant_cells)),
            "shared_band_height_cell_count": int(len(shared_band_cells)),
            "occupied_band_height_min_m": (
                float(self.config.obstacle_height_min)
                if include_height_aligned
                else None
            ),
            "occupied_band_height_max_m": (
                float(self.config.obstacle_height_max)
                if include_height_aligned
                else None
            ),
            "mutated_memory": False,
        }

    def _rasterize_executed_route_edge(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        *,
        edge_length_m: float,
        sample_spacing_m: float,
    ) -> List[Tuple[int, int]]:
        """Return ordered map cells intersected by one executed pose edge."""
        dr = int(end[0]) - int(start[0])
        dc = int(end[1]) - int(start[1])
        grid_steps = max(abs(dr), abs(dc), 1)
        metric_steps = int(
            math.ceil(max(0.0, float(edge_length_m)) / max(1e-3, sample_spacing_m))
        )
        steps = max(grid_steps, metric_steps, 1)
        cells: List[Tuple[int, int]] = []
        for index in range(steps + 1):
            alpha = float(index) / float(steps)
            cell = (
                int(round(float(start[0]) + alpha * float(dr))),
                int(round(float(start[1]) + alpha * float(dc))),
            )
            if not cells or cell != cells[-1]:
                cells.append(cell)
        return cells

    def _known_free_connectivity_audit(
        self,
        start: Tuple[int, int],
        anchor: Tuple[int, int],
        *,
        max_path_cells: int,
        max_visited_cells: int,
    ) -> Dict[str, Any]:
        """Mirror the path-reobserve ray-free BFS without projecting a pixel."""
        result: Dict[str, Any] = {
            "reachable": False,
            "reason": None,
            "visited_cell_count": 0,
            "path_cell_count": 0,
            "path_edge_count": 0,
            "path_m": None,
            "path_preview": [],
        }
        if self._cell_state(anchor[0], anchor[1]) != "free":
            result["reason"] = "anchor_not_free"
            return result
        free_cells = set(self.free2d_counts.keys()) - set(self.occ2d_counts.keys())
        if not free_cells:
            result["reason"] = "no_known_free_cells"
            return result

        max_depth = max(1, int(max_path_cells))
        max_visited = max(1, int(max_visited_cells))
        queue = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        depth: Dict[Tuple[int, int], int] = {start: 0}
        reached = False
        visited_count = 0
        while queue and visited_count < max_visited:
            cell = queue.popleft()
            visited_count += 1
            if cell == anchor:
                reached = True
                break
            if int(depth[cell]) >= max_depth:
                continue
            for nbr_raw in self._neighbors2d(cell[0], cell[1]):
                nbr = (int(nbr_raw[0]), int(nbr_raw[1]))
                if nbr in parent:
                    continue
                if nbr != anchor and nbr not in free_cells:
                    continue
                if nbr in self.occ2d_counts:
                    continue
                dr = int(nbr[0] - cell[0])
                dc = int(nbr[1] - cell[1])
                if dr and dc:
                    side_a = (cell[0] + dr, cell[1])
                    side_b = (cell[0], cell[1] + dc)
                    if side_a not in free_cells or side_b not in free_cells:
                        continue
                parent[nbr] = cell
                depth[nbr] = int(depth[cell]) + 1
                queue.append(nbr)
        result["visited_cell_count"] = int(visited_count)
        if not reached:
            result["reason"] = (
                "path_search_budget_exhausted"
                if visited_count >= max_visited
                else "no_known_free_path_to_anchor"
            )
            return result

        path: List[Tuple[int, int]] = []
        cursor: Optional[Tuple[int, int]] = anchor
        while cursor is not None:
            path.append(cursor)
            cursor = parent.get(cursor)
        path.reverse()
        result.update(
            {
                "reachable": True,
                "reason": "ok",
                "path_cell_count": int(len(path)),
                "path_edge_count": int(max(0, len(path) - 1)),
                "path_m": float(max(0, len(path) - 1) * self.cs),
                "path_preview": [
                    [int(row), int(col)]
                    for row, col in (
                        path[:12] + ([path[-1]] if len(path) > 12 else [])
                    )
                ],
            }
        )
        return result

    def audit_executed_route_to_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        current_step: Optional[int] = None,
        max_edge_m: float = 0.75,
        sample_spacing_m: float = 0.05,
        max_path_cells: int = 160,
        max_visited_cells: int = 20000,
    ) -> Dict[str, Any]:
        """Compare a real pose chain with the current ray-derived OCC map.

        The pose chain remains a separate source of evidence.  This method does
        not promote visited cells to free space and does not mutate any memory
        state, which keeps map quality and route-history quality independently
        auditable.
        """
        result: Dict[str, Any] = {
            "event_schema_version": "stage22a_executed_route_occ_audit_v1",
            "valid": False,
            "reason": None,
            "shadow_only": True,
            "action_applied": False,
            "output_rewritten": False,
            "gt_fields_used": [],
            "mapping_camera_pitch_aware": bool(
                self.config.camera_pitch_aware_update
            ),
            "source_step": None,
            "trigger_step": self._safe_int(current_step),
            "anchor_grid": None,
            "source_pose_grid": None,
            "trigger_pose_grid": None,
            "source_anchor_pose_match": False,
            "route_raw_pose_count": 0,
            "route_pitch_observation_count": 0,
            "route_max_camera_pitch_deg": 0.0,
            "route_translation_node_count": 0,
            "rotation_or_duplicate_pose_count": 0,
            "route_movement_edge_count": 0,
            "route_length_m": 0.0,
            "route_max_edge_m": None,
            "route_chain_continuous": False,
            "route_discontinuity_edge_count": 0,
            "route_cells": [],
            "route_cell_count": 0,
            "route_unique_cell_count": 0,
            "route_cell_state_counts": {"free": 0, "unknown": 0, "occupied": 0},
            "route_cell_state_ratios": {"free": None, "unknown": None, "occupied": None},
            "route_conflict_edge_count": 0,
            "route_unknown_edge_count": 0,
            "longest_unknown_gap_cells": 0,
            "longest_unknown_gap_m": 0.0,
            "first_occupied_conflict": None,
            "first_unknown_gap": None,
            "route_occupied_height_diagnostics": {
                "occupied_route_cell_count": 0,
                "low_or_ground_conflict_cell_count": 0,
                "lower_obstacle_conflict_cell_count": 0,
                "body_obstacle_conflict_cell_count": 0,
                "obstacle_band_conflict_cell_count": 0,
                "high_conflict_cell_count": 0,
                "no_voxel_conflict_cell_count": 0,
                "mean_low_or_ground_voxel_count": 0.0,
                "mean_lower_obstacle_voxel_count": 0.0,
                "mean_body_obstacle_voxel_count": 0.0,
                "mean_obstacle_band_voxel_count": 0.0,
                "mean_high_voxel_count": 0.0,
                "cells": [],
            },
            "edge_records": [],
            "known_free_connectivity": None,
            "continuous_but_ray_disconnected": False,
        }
        if not self.enabled:
            result["reason"] = "memory_disabled"
            return result
        source_step = self._safe_int(
            (candidate or {}).get("semantic_resilience_source_step_id")
        )
        result["source_step"] = source_step
        try:
            anchor_values = list((candidate or {}).get("grid") or [])
            anchor = (int(anchor_values[0]), int(anchor_values[1]))
        except (TypeError, ValueError, IndexError):
            result["reason"] = "invalid_candidate_anchor"
            return result
        result["anchor_grid"] = [int(anchor[0]), int(anchor[1])]
        if source_step is None:
            result["reason"] = "missing_candidate_source_step"
            return result
        if not self.pose_trace:
            result["reason"] = "missing_pose_trace"
            return result

        trigger_step = self._safe_int(current_step)
        exact_source_indices: List[int] = []
        for index, node in enumerate(self.pose_trace):
            step = self._safe_int(node.get("step_id"))
            if step == source_step:
                exact_source_indices.append(index)
        if not exact_source_indices:
            result["reason"] = "source_step_not_in_pose_trace"
            return result
        source_index = min(
            exact_source_indices,
            key=lambda index: abs(int(self.pose_trace[index].get("row", -10**9)) - anchor[0])
            + abs(int(self.pose_trace[index].get("col", -10**9)) - anchor[1]),
        )

        raw_nodes: List[Dict[str, Any]] = []
        for node in self.pose_trace[source_index:]:
            step = self._safe_int(node.get("step_id"))
            if trigger_step is not None and step is not None and step > trigger_step:
                break
            try:
                row = int(node["row"])
                col = int(node["col"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_nodes.append(
                {
                    "step_id": step,
                    "row": row,
                    "col": col,
                    "x": node.get("x"),
                    "y": node.get("y"),
                    "camera_pitch_deg": float(node.get("camera_pitch_deg", 0.0) or 0.0),
                }
            )
        if not raw_nodes:
            result["reason"] = "empty_source_to_trigger_trace"
            return result

        source_grid = (int(raw_nodes[0]["row"]), int(raw_nodes[0]["col"]))
        trigger_grid = (int(raw_nodes[-1]["row"]), int(raw_nodes[-1]["col"]))
        result.update(
            {
                "source_pose_grid": [int(source_grid[0]), int(source_grid[1])],
                "trigger_pose_grid": [int(trigger_grid[0]), int(trigger_grid[1])],
                "source_pose_step": raw_nodes[0].get("step_id"),
                "trigger_pose_step": raw_nodes[-1].get("step_id"),
                "source_anchor_pose_match": bool(source_grid == anchor),
                "source_anchor_grid_distance_cells": float(
                    math.hypot(source_grid[0] - anchor[0], source_grid[1] - anchor[1])
                ),
                "route_raw_pose_count": int(len(raw_nodes)),
                "route_pitch_observation_count": int(
                    sum(abs(float(node.get("camera_pitch_deg", 0.0))) > 1e-4 for node in raw_nodes)
                ),
                "route_max_camera_pitch_deg": float(
                    max((abs(float(node.get("camera_pitch_deg", 0.0))) for node in raw_nodes), default=0.0)
                ),
            }
        )

        def node_distance(first: Dict[str, Any], second: Dict[str, Any]) -> float:
            try:
                first_xy = (float(first["x"]), float(first["y"]))
                second_xy = (float(second["x"]), float(second["y"]))
                if all(math.isfinite(value) for value in first_xy + second_xy):
                    return float(math.hypot(second_xy[0] - first_xy[0], second_xy[1] - first_xy[1]))
            except (KeyError, TypeError, ValueError):
                pass
            return float(
                math.hypot(
                    int(second["row"]) - int(first["row"]),
                    int(second["col"]) - int(first["col"]),
                )
                * self.cs
            )

        translation_nodes = [raw_nodes[0]]
        for node in raw_nodes[1:]:
            if node_distance(translation_nodes[-1], node) > 1e-4:
                translation_nodes.append(node)
        result["route_translation_node_count"] = int(len(translation_nodes))
        result["rotation_or_duplicate_pose_count"] = int(
            len(raw_nodes) - len(translation_nodes)
        )
        result["route_nodes"] = [
            {
                "step_id": node.get("step_id"),
                "grid": [int(node["row"]), int(node["col"])],
                "xy": (
                    [float(node["x"]), float(node["y"])]
                    if node.get("x") is not None and node.get("y") is not None
                    else None
                ),
            }
            for node in translation_nodes
        ]

        ordered_cells: List[Tuple[int, int]] = []
        edge_lengths: List[float] = []
        discontinuity_count = 0
        conflict_edge_count = 0
        unknown_edge_count = 0
        edge_records: List[Dict[str, Any]] = []
        for edge_index, (first, second) in enumerate(
            zip(translation_nodes[:-1], translation_nodes[1:])
        ):
            edge_length = node_distance(first, second)
            edge_lengths.append(edge_length)
            start = (int(first["row"]), int(first["col"]))
            end = (int(second["row"]), int(second["col"]))
            edge_cells = self._rasterize_executed_route_edge(
                start,
                end,
                edge_length_m=edge_length,
                sample_spacing_m=max(1e-3, float(sample_spacing_m)),
            )
            if ordered_cells and edge_cells and edge_cells[0] == ordered_cells[-1]:
                ordered_cells.extend(edge_cells[1:])
            else:
                ordered_cells.extend(edge_cells)
            unique_edge_cells = list(dict.fromkeys(edge_cells))
            edge_counts = {"free": 0, "unknown": 0, "occupied": 0}
            for row, col in unique_edge_cells:
                edge_counts[self._cell_state(row, col)] += 1
            continuous = bool(edge_length <= max(1e-3, float(max_edge_m)) + 1e-9)
            discontinuity_count += int(not continuous)
            conflict_edge_count += int(edge_counts["occupied"] > 0)
            unknown_edge_count += int(edge_counts["unknown"] > 0)
            edge_records.append(
                {
                    "edge_index": int(edge_index),
                    "source_step": first.get("step_id"),
                    "target_step": second.get("step_id"),
                    "source_grid": [int(start[0]), int(start[1])],
                    "target_grid": [int(end[0]), int(end[1])],
                    "length_m": float(edge_length),
                    "continuous": continuous,
                    "cell_count": int(len(unique_edge_cells)),
                    "cell_state_counts": edge_counts,
                }
            )
        if not ordered_cells:
            ordered_cells = [source_grid]

        unique_cells = list(dict.fromkeys(ordered_cells))
        state_counts = {"free": 0, "unknown": 0, "occupied": 0}
        first_occupied = None
        first_unknown = None
        unknown_run = 0
        longest_unknown = 0
        for index, (row, col) in enumerate(ordered_cells):
            state = self._cell_state(row, col)
            if state == "unknown":
                unknown_run += 1
                longest_unknown = max(longest_unknown, unknown_run)
                if first_unknown is None:
                    first_unknown = {
                        "route_cell_index": int(index),
                        "grid": [int(row), int(col)],
                    }
            else:
                unknown_run = 0
            if state == "occupied" and first_occupied is None:
                first_occupied = {
                    "route_cell_index": int(index),
                    "grid": [int(row), int(col)],
                }
        for row, col in unique_cells:
            state_counts[self._cell_state(row, col)] += 1

        # Keep the flattened 2D decision unchanged, but expose the 3D voxel
        # heights behind each occupied route cell.  This diagnoses floor/height
        # projection errors without changing connectivity or action selection.
        height_cells = []
        low_conflict = lower_conflict = body_conflict = 0
        band_conflict = high_conflict = no_voxel = 0
        low_counts = []
        lower_counts = []
        body_counts = []
        band_counts = []
        high_counts = []
        route_occupied_cells = {
            (int(row), int(col))
            for row, col in unique_cells
            if self._cell_state(row, col) == "occupied"
        }
        voxel_heights_by_cell: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for (voxel_row, voxel_col, height), count in self.occ_counts.items():
            cell = (int(voxel_row), int(voxel_col))
            if cell in route_occupied_cells:
                voxel_heights_by_cell[cell].append((int(height), int(count)))
        for row, col in unique_cells:
            if (int(row), int(col)) not in route_occupied_cells:
                continue
            per_height = voxel_heights_by_cell.get((int(row), int(col)), [])
            low = sum(
                count
                for height, count in per_height
                if float(height) * self.cs
                <= float(self.config.obstacle_height_min)
            )
            lower = sum(
                count
                for height, count in per_height
                if self._is_obstacle_height(height)
                and float(height) * self.cs <= 0.35
            )
            body = sum(
                count
                for height, count in per_height
                if self._is_obstacle_height(height)
                and float(height) * self.cs > 0.35
            )
            band = lower + body
            high = sum(
                count
                for height, count in per_height
                if float(height) * self.cs
                >= float(self.config.obstacle_height_max)
            )
            low_counts.append(low)
            lower_counts.append(lower)
            body_counts.append(body)
            band_counts.append(band)
            high_counts.append(high)
            low_conflict += int(low > 0)
            lower_conflict += int(lower > 0)
            body_conflict += int(body > 0)
            band_conflict += int(band > 0)
            high_conflict += int(high > 0)
            if not per_height:
                no_voxel += 1
            height_cells.append(
                {
                    "grid": [int(row), int(col)],
                    "low_or_ground_voxel_count": int(low),
                    "lower_obstacle_voxel_count": int(lower),
                    "body_obstacle_voxel_count": int(body),
                    "obstacle_band_voxel_count": int(band),
                    "high_voxel_count": int(high),
                    "voxel_heights": [
                        {"height_index": int(height), "z_m": float(height * self.cs), "count": int(count)}
                        for height, count in sorted(per_height)
                    ],
                }
            )
        occupied_route_cell_count = len(height_cells)
        result_height_diagnostics = {
            "occupied_route_cell_count": int(occupied_route_cell_count),
            "low_or_ground_conflict_cell_count": int(low_conflict),
            "lower_obstacle_conflict_cell_count": int(lower_conflict),
            "body_obstacle_conflict_cell_count": int(body_conflict),
            "obstacle_band_conflict_cell_count": int(band_conflict),
            "high_conflict_cell_count": int(high_conflict),
            "no_voxel_conflict_cell_count": int(no_voxel),
            "mean_low_or_ground_voxel_count": float(sum(low_counts) / occupied_route_cell_count) if occupied_route_cell_count else 0.0,
            "mean_lower_obstacle_voxel_count": float(sum(lower_counts) / occupied_route_cell_count) if occupied_route_cell_count else 0.0,
            "mean_body_obstacle_voxel_count": float(sum(body_counts) / occupied_route_cell_count) if occupied_route_cell_count else 0.0,
            "mean_obstacle_band_voxel_count": float(sum(band_counts) / occupied_route_cell_count) if occupied_route_cell_count else 0.0,
            "mean_high_voxel_count": float(sum(high_counts) / occupied_route_cell_count) if occupied_route_cell_count else 0.0,
            "cells": height_cells,
        }
        denominator = max(1, len(unique_cells))
        connectivity = self._known_free_connectivity_audit(
            trigger_grid,
            anchor,
            max_path_cells=max_path_cells,
            max_visited_cells=max_visited_cells,
        )
        chain_continuous = bool(discontinuity_count == 0)
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "route_movement_edge_count": int(len(edge_records)),
                "route_length_m": float(sum(edge_lengths)),
                "route_max_edge_m": (
                    float(max(edge_lengths)) if edge_lengths else 0.0
                ),
                "route_chain_continuous": chain_continuous,
                "route_discontinuity_edge_count": int(discontinuity_count),
                "route_cells": [
                    [int(row), int(col)] for row, col in ordered_cells
                ],
                "route_cell_count": int(len(ordered_cells)),
                "route_unique_cell_count": int(len(unique_cells)),
                "route_cell_state_counts": state_counts,
                "route_cell_state_ratios": {
                    name: float(count / denominator)
                    for name, count in state_counts.items()
                },
                "route_conflict_edge_count": int(conflict_edge_count),
                "route_unknown_edge_count": int(unknown_edge_count),
                "longest_unknown_gap_cells": int(longest_unknown),
                "longest_unknown_gap_m": float(longest_unknown * self.cs),
                "first_occupied_conflict": first_occupied,
                "first_unknown_gap": first_unknown,
                "route_occupied_height_diagnostics": result_height_diagnostics,
                "edge_records": edge_records,
                "known_free_connectivity": connectivity,
                "continuous_but_ray_disconnected": bool(
                    chain_continuous and not connectivity.get("reachable")
                ),
            }
        )
        return result

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

    def _target_frontier_features(
        self,
        cell: Iterable[int],
        start_grid: Iterable[int],
        *,
        frontier_progress_score: float,
        revisit_risk: float,
        angle_to_current: Any,
        goal_progress_state: Optional[Dict[str, Any]],
        completed_landmark_penalty: float,
        repeated_semantic_penalty: float,
        unknown_target_frontier_bonus: float,
    ) -> Dict[str, Any]:
        if not self.config.candidate_probe_target_frontier_enable:
            return {
                "enabled": False,
                "score": 0.0,
                "cluster_count": 0,
                "cluster_score": 0.0,
                "doorway_like_score": 0.0,
                "corridor_continuation_score": 0.0,
                "transition_prior": 0.0,
                "intent_deviation_penalty": 0.0,
                "intent_safe": False,
                "candidate": False,
                "escape_candidate": False,
            }
        row, col = [int(v) for v in list(cell)[:2]]
        radius = max(1, int(self.config.candidate_probe_target_frontier_cluster_radius_cells))
        frontier_set = self._frontier_cell_set()
        free_count = 0
        occupied_count = 0
        unknown_count = 0
        frontier_count = 0
        total_count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                rr = row + dr
                cc = col + dc
                total_count += 1
                state = self._cell_state(rr, cc)
                if state == "occupied":
                    occupied_count += 1
                elif state == "free":
                    free_count += 1
                else:
                    unknown_count += 1
                if (rr, cc) in frontier_set:
                    frontier_count += 1
        cluster_norm = max(1.0, float(self.config.candidate_probe_target_frontier_cluster_norm))
        cluster_score = min(1.0, float(frontier_count) / cluster_norm)
        occupied_score = min(1.0, float(occupied_count) / max(1.0, 0.12 * float(total_count)))
        unknown_score = min(1.0, float(unknown_count) / max(1.0, 0.20 * float(total_count)))
        free_score = min(1.0, float(free_count) / max(1.0, 0.30 * float(total_count)))
        doorway_like_score = 0.0
        if frontier_progress_score > 0.0:
            doorway_like_score = min(
                1.0,
                0.40 * occupied_score
                + 0.35 * unknown_score
                + 0.25 * min(free_score, cluster_score),
            )
        corridor_score = self._corridor_continuation_score(start_grid, [row, col])
        transition_prior = self._target_frontier_transition_prior(goal_progress_state)
        try:
            intent_deviation = float(angle_to_current) / 180.0
        except (TypeError, ValueError):
            intent_deviation = 0.0
        intent_deviation = min(1.0, max(0.0, intent_deviation))
        try:
            intent_safe = float(angle_to_current) <= float(
                self.config.candidate_probe_target_frontier_intent_max_deviation_deg
            )
        except (TypeError, ValueError):
            intent_safe = True
        topology_score = (
            0.35 * cluster_score
            + 0.35 * doorway_like_score
            + 0.30 * corridor_score
        )
        target_score = (
            transition_prior * topology_score
            + 0.25 * float(frontier_progress_score)
            + float(unknown_target_frontier_bonus)
            - 0.30 * float(revisit_risk)
            - 0.25 * float(completed_landmark_penalty)
            - 0.20 * float(repeated_semantic_penalty)
            - float(self.config.candidate_probe_target_frontier_intent_penalty_weight)
            * intent_deviation
        )
        target_score = max(0.0, min(1.0, target_score))
        candidate_threshold = float(self.config.candidate_probe_target_frontier_candidate_threshold)
        is_candidate = bool(target_score >= candidate_threshold)
        escape_candidate = bool(
            is_candidate
            and intent_safe
            and completed_landmark_penalty <= 0.0
            and repeated_semantic_penalty <= 0.0
        )
        return {
            "enabled": True,
            "score": float(target_score),
            "cluster_count": int(frontier_count),
            "cluster_score": float(cluster_score),
            "doorway_like_score": float(doorway_like_score),
            "corridor_continuation_score": float(corridor_score),
            "transition_prior": float(transition_prior),
            "intent_deviation_penalty": float(intent_deviation),
            "intent_safe": bool(intent_safe),
            "candidate": bool(is_candidate),
            "escape_candidate": bool(escape_candidate),
            "local_free_count": int(free_count),
            "local_occupied_count": int(occupied_count),
            "local_unknown_count": int(unknown_count),
        }

    def _corridor_continuation_score(
        self,
        start_grid: Iterable[int],
        target_grid: Iterable[int],
    ) -> float:
        start_row, start_col = [int(v) for v in list(start_grid)[:2]]
        target_row, target_col = [int(v) for v in list(target_grid)[:2]]
        steps = max(abs(target_row - start_row), abs(target_col - start_col))
        if steps <= 1:
            return 0.0
        free_count = 0
        occupied_count = 0
        checked = 0
        for idx in range(1, steps + 1):
            t = float(idx) / float(steps)
            row = int(round(start_row + t * (target_row - start_row)))
            col = int(round(start_col + t * (target_col - start_col)))
            state = self._cell_state(row, col)
            checked += 1
            if state == "occupied":
                occupied_count += 1
            elif state == "free":
                free_count += 1
        if checked <= 0:
            return 0.0
        free_ratio = float(free_count) / float(checked)
        occupied_ratio = float(occupied_count) / float(checked)
        return max(0.0, min(1.0, free_ratio * (1.0 - occupied_ratio)))

    def _target_frontier_transition_prior(
        self,
        goal_progress_state: Optional[Dict[str, Any]],
    ) -> float:
        state = goal_progress_state or {}
        next_landmark = self._canonical_semantic_term(state.get("next_landmark"))
        if not next_landmark:
            return 0.35
        if next_landmark in _TARGET_FRONTIER_TRANSITION_TERMS:
            return 1.0
        if next_landmark in _GOAL_PROGRESS_SPECIFIC_ROOMS:
            return 0.75
        return 0.45

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
            return {
                "bucket": "same",
                "angle_deg": 0.0,
                "world_bearing_deg": 0.0,
                "distance_m": 0.0,
            }
        world_angle = math.atan2(dy, dx)
        rel_angle = self._wrap_angle(world_angle - float(yaw))
        rel_deg = math.degrees(rel_angle)
        bucket = self._angle_to_direction_bucket(rel_deg)
        return {
            "bucket": bucket,
            "angle_deg": float(rel_deg),
            "world_bearing_deg": float(math.degrees(world_angle)),
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
            "grid": [int(best.get("row", 0)), int(best.get("col", 0))],
            "xy": [float(best.get("x", 0.0)), float(best.get("y", 0.0))],
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
            return {
                "bucket": "same",
                "angle_deg": 0.0,
                "world_bearing_deg": 0.0,
                "distance_m": 0.0,
            }
        world_angle = math.atan2(float(delta[1]), float(delta[0]))
        rel_deg = math.degrees(self._wrap_angle(world_angle - float(yaw)))
        return {
            "bucket": self._angle_to_direction_bucket(rel_deg),
            "angle_deg": float(rel_deg),
            "world_bearing_deg": float(math.degrees(world_angle)),
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
        anchor_colors = {
            "obstacle": (255, 90, 90),
            "obstacle_passage": (255, 170, 70),
            "passage": (80, 230, 160),
            "room": (170, 135, 255),
            "landmark": (255, 120, 230),
            "semantic": (230, 230, 120),
        }
        for anchor in self.semantic_anchors:
            grid = anchor.get("grid") or []
            if len(grid) < 2:
                continue
            row, col = int(grid[0]), int(grid[1])
            if r0 <= row < r1 and c0 <= col < c1:
                x, y = to_xy(row, col)
                rad = max(3, scale + 1)
                color = anchor_colors.get(str(anchor.get("semantic_kind") or "semantic"), (230, 230, 120))
                draw.rectangle((x - rad, y - rad, x + rad, y + rad), fill=color)
        if self.last_pose_grid is not None:
            x, y = to_xy(self.last_pose_grid[0], self.last_pose_grid[1])
            rad = max(3, scale + 1)
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(255, 255, 255))
        label = f"occ={len(self.occ2d_counts)} free={len(self.free2d_counts)} frontier={len(self.get_frontier_cells(sample_limit=0))} anchors={len(self.semantic_anchors)}"
        draw.rectangle((0, 0, max(220, len(label) * 7), 18), fill=(18, 18, 18))
        draw.text((4, 3), label, fill=(240, 240, 240))
        suffix = "final" if context.get("final") else f"{self.saved_bev_count:03d}"
        step = context.get("step_id")
        step_text = "end" if step is None else str(step)
        path = os.path.join(out_dir, f"bev_ep{self.episode_meta.get('episode_id')}_{step_text}_{suffix}.png")
        img.save(path)
        self.saved_bev_count += 1

    def _write_candidate_bev_snapshot(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        if not self.debug_dir:
            return None
        if (
            self.config.candidate_probe_max_bev_snapshots >= 0
            and self.saved_candidate_bev_count >= self.config.candidate_probe_max_bev_snapshots
        ):
            return None
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None
        out_dir = os.path.join(self.debug_dir, "occ_memory", "candidates")
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
            pts = [
                to_xy(item["row"], item["col"])
                for item in self.pose_trace
                if r0 <= item["row"] < r1 and c0 <= item["col"] < c1
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=(95, 160, 245), width=max(1, scale))
        if self.last_pose_grid is not None:
            x, y = to_xy(self.last_pose_grid[0], self.last_pose_grid[1])
            rad = max(4, scale + 2)
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(255, 255, 255))
        anchor_colors = {
            "obstacle": (255, 90, 90),
            "obstacle_passage": (255, 170, 70),
            "passage": (80, 230, 160),
            "room": (170, 135, 255),
            "landmark": (255, 120, 230),
            "semantic": (230, 230, 120),
        }
        for anchor in self.semantic_anchors:
            grid = anchor.get("grid") or []
            if len(grid) < 2:
                continue
            row, col = int(grid[0]), int(grid[1])
            if not (r0 <= row < r1 and c0 <= col < c1):
                continue
            x, y = to_xy(row, col)
            rad = max(3, scale + 1)
            color = anchor_colors.get(str(anchor.get("semantic_kind") or "semantic"), (230, 230, 120))
            draw.rectangle((x - rad, y - rad, x + rad, y + rad), fill=color)
        color_by_type = {
            "frontier": (255, 225, 80),
            "semantic_frontier": (255, 160, 80),
            "semantic_keyframe": (240, 105, 255),
            "open_floor": (90, 220, 225),
            "resilience_backtrack": (115, 255, 125),
        }
        for candidate in candidates:
            candidate["bev_pixel"] = None
            candidate["bev_image_size"] = [int(w * scale), int(h * scale)]
            grid = candidate.get("grid") or []
            if len(grid) < 2:
                continue
            row, col = int(grid[0]), int(grid[1])
            if not (r0 <= row < r1 and c0 <= col < c1):
                continue
            x, y = to_xy(row, col)
            candidate["bev_pixel"] = [int(x), int(y)]
            label = str(candidate.get("candidate_id") or "?")
            color = color_by_type.get(str(candidate.get("candidate_type")), (255, 255, 255))
            rad = max(7, scale * 3)
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=color, outline=(0, 0, 0), width=2)
            draw.text((x - 3, y - 6), label, fill=(0, 0, 0))
        label = (
            f"V10 candidates={len(candidates)} "
            f"occ={len(self.occ2d_counts)} free={len(self.free2d_counts)} "
            f"frontier={len(self.get_frontier_cells(sample_limit=0))} "
            f"anchors={len(self.semantic_anchors)}"
        )
        draw.rectangle((0, 0, max(320, len(label) * 7), 18), fill=(18, 18, 18))
        draw.text((4, 3), label, fill=(240, 240, 240))
        step = context.get("step_id")
        step_text = "none" if step is None else str(step)
        path = os.path.join(
            out_dir,
            (
                f"candidates_ep{self.episode_meta.get('episode_id')}_"
                f"step{step_text}_{self.saved_candidate_bev_count:03d}.png"
            ),
        )
        img.save(path)
        self.saved_candidate_bev_count += 1
        return path

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
        paths: Dict[str, str] = {"memory_cloud_ply": memory_ply_path}

        surface_points, surface_colors = self._validation_surface_cloud()
        surface_path = os.path.join(out_dir, f"{suffix}_accumulated_rgb_surface.ply")
        self._write_point_cloud_ply(surface_path, surface_points, surface_colors)
        paths["accumulated_rgb_surface_ply"] = surface_path

        occ_keys = self._sample_items(
            list(self.occ_counts.keys()), int(self.config.validation_max_occupied_points)
        )
        free_keys = self._sample_items(
            list(self.free_counts.keys()), int(self.config.validation_max_free_points)
        )
        occ_points = self._grid_keys_to_points(occ_keys)
        free_points = self._grid_keys_to_points(free_keys)
        occ_path = os.path.join(out_dir, f"{suffix}_occupied_only.ply")
        free_path = os.path.join(out_dir, f"{suffix}_free_only.ply")
        self._write_point_cloud_ply(
            occ_path,
            occ_points,
            np.tile(np.array([[225, 70, 60]], dtype=np.uint8), (occ_points.shape[0], 1)),
        )
        self._write_point_cloud_ply(
            free_path,
            free_points,
            np.tile(np.array([[70, 155, 235]], dtype=np.uint8), (free_points.shape[0], 1)),
        )
        paths["occupied_only_ply"] = occ_path
        paths["free_only_ply"] = free_path

        trajectory_points = self._pose_trace_points()
        trajectory_path = os.path.join(out_dir, f"{suffix}_trajectory.ply")
        self._write_point_cloud_ply(
            trajectory_path,
            trajectory_points,
            np.tile(
                np.array([[255, 220, 65]], dtype=np.uint8),
                (trajectory_points.shape[0], 1),
            ),
        )
        paths["trajectory_ply"] = trajectory_path
        paths.update(
            self._write_validation_projection_views(
                out_dir,
                suffix,
                surface_points=surface_points,
                surface_colors=surface_colors,
                occupied_points=occ_points,
                free_points=free_points,
                trajectory_points=trajectory_points,
            )
        )

        pose_audit_path = os.path.join(out_dir, f"{suffix}_pose_height_audit.json")
        with open(pose_audit_path, "w", encoding="utf-8") as f:
            json.dump(
                self._jsonable(
                    {
                        **self.episode_meta,
                        "pose_source": (
                            "context_oracle_height"
                            if bool(self.config.validation_pose_from_context)
                            else "gps_compass_2d"
                        ),
                        "gt_fields_used": (
                            ["habitat_sim_agent_position_y"]
                            if bool(self.config.validation_pose_from_context)
                            else []
                        ),
                        **self._validation_pose_audit_summary(),
                        "pose_trace": self.pose_trace,
                    }
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        paths["pose_height_audit_json"] = pose_audit_path
        event = {
            "event_type": "occ_memory_validation_final_snapshot",
            **self.episode_meta,
            **context,
            "snapshot_index": int(self.saved_validation_final_count),
            "update_count": int(self.update_count),
            "paths": paths,
            "gt_fields_used": (
                ["habitat_sim_agent_position_y"]
                if bool(self.config.validation_pose_from_context)
                else []
            ),
            "accumulated_rgb_surface_point_count": int(surface_points.shape[0]),
            "occupied_only_point_count": int(occ_points.shape[0]),
            "free_only_point_count": int(free_points.shape[0]),
            "trajectory_point_count": int(trajectory_points.shape[0]),
            **self._validation_pose_audit_summary(),
            **self._validation_endpoint_volume_summary(),
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

    def _append_validation_surface(
        self, points: np.ndarray, colors: Optional[np.ndarray]
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return
        if colors is None:
            colors = np.full((points.shape[0], 3), 210, dtype=np.uint8)
        else:
            colors = np.asarray(colors)
            if colors.ndim != 2 or colors.shape[0] != points.shape[0] or colors.shape[1] < 3:
                colors = np.full((points.shape[0], 3), 210, dtype=np.uint8)
            else:
                colors = np.clip(colors[:, :3], 0, 255).astype(np.uint8)
        self.validation_surface_points.append(points.astype(np.float32, copy=True))
        self.validation_surface_colors.append(colors.astype(np.uint8, copy=True))
        self.validation_surface_point_count += int(points.shape[0])

    def _validation_surface_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.validation_surface_points:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
            )
        points = np.concatenate(self.validation_surface_points, axis=0)
        colors = np.concatenate(self.validation_surface_colors, axis=0)
        max_points = int(self.config.validation_max_accumulated_surface_points)
        if max_points > 0 and points.shape[0] > max_points:
            ids = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
            points = points[ids]
            colors = colors[ids]
        return (
            points.astype(np.float32, copy=False),
            colors.astype(np.uint8, copy=False),
        )

    def _write_validation_projection_views(
        self,
        out_dir: str,
        suffix: str,
        *,
        surface_points: np.ndarray,
        surface_colors: np.ndarray,
        occupied_points: np.ndarray,
        free_points: np.ndarray,
        trajectory_points: np.ndarray,
    ) -> Dict[str, str]:
        size = max(256, int(self.config.validation_projection_size))
        layers = [
            (free_points, np.array([55, 105, 150], dtype=np.uint8)),
            (surface_points, surface_colors),
            (occupied_points, np.array([240, 70, 55], dtype=np.uint8)),
            (trajectory_points, np.array([255, 230, 55], dtype=np.uint8)),
        ]
        specs = {
            "bev_xy": (0, 1),
            "side_xz": (0, 2),
            "side_yz": (1, 2),
        }
        paths: Dict[str, str] = {}
        for name, axes in specs.items():
            path = os.path.join(out_dir, f"{suffix}_{name}.png")
            if self._write_projection_png(path, layers, axes=axes, size=size):
                paths[f"{name}_png"] = path
            surface_path = os.path.join(out_dir, f"{suffix}_surface_{name}.png")
            if self._write_projection_png(
                surface_path,
                [(surface_points, surface_colors)],
                axes=axes,
                size=size,
            ):
                paths[f"surface_{name}_png"] = surface_path
        return paths

    def _write_projection_png(
        self,
        path: str,
        layers: List[Tuple[np.ndarray, np.ndarray]],
        *,
        axes: Tuple[int, int],
        size: int,
    ) -> bool:
        try:
            from PIL import Image
        except Exception:
            return False
        valid_parts = []
        for points, _ in layers:
            arr = np.asarray(points, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
                mask = np.isfinite(arr[:, axes[0]]) & np.isfinite(arr[:, axes[1]])
                if np.any(mask):
                    valid_parts.append(arr[mask][:, [axes[0], axes[1]]])
        if not valid_parts:
            return False
        all_xy = np.concatenate(valid_parts, axis=0)
        lo = np.percentile(all_xy, 0.5, axis=0)
        hi = np.percentile(all_xy, 99.5, axis=0)
        span = np.maximum(hi - lo, 0.5)
        margin = span * 0.05
        lo -= margin
        hi += margin
        span = np.maximum(hi - lo, 1e-6)
        canvas = np.full((size, size, 3), 22, dtype=np.uint8)
        for points, colors in layers:
            arr = np.asarray(points, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                continue
            xy = arr[:, [axes[0], axes[1]]]
            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]
            if xy.shape[0] == 0:
                continue
            uv = np.rint((xy - lo) / span * float(size - 1)).astype(np.int64)
            uv[:, 0] = np.clip(uv[:, 0], 0, size - 1)
            uv[:, 1] = np.clip(size - 1 - uv[:, 1], 0, size - 1)
            color_arr = np.asarray(colors)
            if color_arr.ndim == 1:
                color_arr = np.tile(color_arr[:3], (uv.shape[0], 1))
            else:
                color_arr = color_arr[valid, :3]
            canvas[uv[:, 1], uv[:, 0]] = np.clip(color_arr, 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(canvas, mode="RGB").save(path)
        return True

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
        anchor_points, anchor_colors = self._semantic_anchor_points()
        if anchor_points.shape[0] > 0:
            point_parts.append(anchor_points)
            color_parts.append(anchor_colors)

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
            "memory_semantic_anchor_point_count": int(anchor_points.shape[0]),
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
        points = [
            [item["x"], item["y"], float(item.get("z", 0.0)) + 0.08]
            for item in self.pose_trace
        ]
        return np.asarray(points, dtype=np.float32)

    def _keyframe_points(self) -> np.ndarray:
        if not self.keyframes:
            return np.zeros((0, 3), dtype=np.float32)
        points = [
            [item["x"], item["y"], float(item.get("z", 0.0)) + 0.14]
            for item in self.keyframes
        ]
        return np.asarray(points, dtype=np.float32)

    def _semantic_anchor_points(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.semantic_anchors:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        color_by_kind = {
            "obstacle": np.array([255, 90, 90], dtype=np.uint8),
            "obstacle_passage": np.array([255, 170, 70], dtype=np.uint8),
            "passage": np.array([80, 230, 160], dtype=np.uint8),
            "room": np.array([170, 135, 255], dtype=np.uint8),
            "landmark": np.array([255, 120, 230], dtype=np.uint8),
            "semantic": np.array([230, 230, 120], dtype=np.uint8),
        }
        points = []
        colors = []
        for anchor in self.semantic_anchors:
            xy = anchor.get("xy")
            if not xy or len(xy) < 2:
                grid = anchor.get("grid") or []
                if len(grid) < 2:
                    continue
                xy_arr = self._grid_to_xy(grid)
                xy = [float(xy_arr[0]), float(xy_arr[1])]
            points.append([float(xy[0]), float(xy[1]), 0.22])
            colors.append(color_by_kind.get(str(anchor.get("semantic_kind") or "semantic"), color_by_kind["semantic"]))
        if not points:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)

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

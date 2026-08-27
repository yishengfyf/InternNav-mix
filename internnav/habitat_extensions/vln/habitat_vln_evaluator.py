import argparse
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import math
import random
import re
import time
from collections import OrderedDict, deque
from datetime import datetime
from typing import Any, Optional, Tuple

import cv2
import habitat
import habitat_sim
import imageio
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image, ImageDraw
from magnum import Vector3
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.semantic_recovery_triage import classify_semantic_recovery_triage
from internnav.s2_action_loop import (
    init_s2_action_loop_state,
    normalize_direct_turn_output,
    observe_s2_action_query,
)
from internnav.utils.sparse_occ_memory import SparseOccSemanticMemory
from internnav.utils.lseg_online_shadow import OnlineLSegSemanticShadow
from internnav.utils.replay_ledger import ReplayLedger
from internnav.utils.vlmap_safety import VLMapActionSafety
from internnav.utils.vlmap_semantic import VLMapSemanticShadow
from internnav.utils.stage27_candidate_generation import generate_from_sparse_memory
from internnav.utils.stage38_recovery_context import build_recovery_bev_spatial_snapshot
from internnav.utils.stage41_executor_contract import validate_executor_contract
from internnav.utils.stage45_candidate_rejection_truth import (
    audit_candidate_rejection_truth,
    summarize_event_audits,
)
from internnav.utils.stage46_active_recovery import (
    active_path_within_bound,
    bind_candidate_to_loop_event,
    iterative_reorientation_decision,
)
from internnav.utils.stage43_counterfactual_reobserve import (
    SCHEMA_VERSION as STAGE43_SCHEMA_VERSION,
    normalize_angle_deg,
    plan_bounded_reorientation,
)

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class _JsonlTeeStream:
    def __init__(self, stream, log_path: str, stream_name: str, rank: int):
        self.stream = stream
        self.log_path = log_path
        self.stream_name = stream_name
        self.rank = rank
        self._buffer = ""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        if not isinstance(data, str) or not data:
            return
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._write_json_line(line)

    def flush(self):
        self.stream.flush()
        self._file.flush()

    def close(self):
        if self._buffer:
            self._write_json_line(self._buffer.rstrip("\r"))
            self._buffer = ""
        self._file.close()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def _write_json_line(self, message: str) -> None:
        event = {
            "time": datetime.now().isoformat(timespec="microseconds"),
            "rank": int(self.rank),
            "stream": self.stream_name,
            "message": message,
        }
        self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            mesh_audit_requested = bool(
                (cfg.agent.model_settings.get("vlmap_safety", {}) or {}).get(
                    "occ_memory_validation_mesh_raycast_enable", False
                )
            )
            if mesh_audit_requested:
                self.config.habitat.simulator.habitat_sim_v0.enable_physics = True
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history
        eval_random_seed = getattr(self.model_args, "eval_random_seed", None)
        if eval_random_seed is not None:
            eval_random_seed = int(eval_random_seed) + int(getattr(self, "rank", 0))
            self._seed_eval_rng(eval_random_seed, "init")

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth
        self._tilt_angle_deg = float(getattr(self.config.habitat.simulator, "tilt_angle", 15.0))

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        vlmap_safety_cfg.setdefault("camera_height", float(self._camera_height))
        vlmap_safety_cfg.setdefault("depth_scale", 1.0)
        habitat_forward_step = float(getattr(self.config.habitat.simulator, "forward_step_size", 0.25))
        habitat_turn_angle = float(getattr(self.config.habitat.simulator, "turn_angle", 15.0))
        vlmap_safety_cfg["habitat_forward_step_size"] = habitat_forward_step
        vlmap_safety_cfg["habitat_turn_angle_deg"] = habitat_turn_angle
        if bool(vlmap_safety_cfg.get("sync_habitat_action_scale", True)):
            vlmap_safety_cfg["forward_distance"] = habitat_forward_step
            vlmap_safety_cfg["turn_angle_deg"] = habitat_turn_angle
        self.vlmap_safety = VLMapActionSafety(
            vlmap_safety_cfg,
            get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
        )
        self._vlmap_last_nav_action = None
        self._vlmap_log_stdout = None
        self._vlmap_log_stderr = None
        self._vlmap_run_dir = None
        self._setup_vlmap_run_logging()
        self.vlmap_semantic = VLMapSemanticShadow(vlmap_safety_cfg)
        self.vlmap_semantic.set_debug_dir(self._get_vlmap_run_dir())
        self.occ_memory = SparseOccSemanticMemory(
            vlmap_safety_cfg,
            get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
        )
        self.occ_memory.set_debug_dir(self._get_vlmap_run_dir())
        self.replay_ledger = ReplayLedger(vlmap_safety_cfg)
        self.replay_ledger.set_root(self._get_vlmap_run_dir() or self.output_path)
        self.online_lseg_shadow = OnlineLSegSemanticShadow(
            vlmap_safety_cfg,
            get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
            self.device,
        )
        self.online_lseg_shadow.set_root(
            self._get_vlmap_run_dir() or self.output_path
        )
        self._stage23a_oracle_pose_enabled = bool(
            vlmap_safety_cfg.get("occ_memory_validation_oracle_pose_enable", False)
        )
        self._stage23a_oracle_sensor_pose_enabled = bool(
            vlmap_safety_cfg.get(
                "occ_memory_validation_oracle_sensor_pose_enable", False
            )
        )
        self._stage23a_initial_sim_position = None
        self._stage23a_initial_agent_matrix = None
        self._stage23a_mesh_raycast_enabled = bool(
            vlmap_safety_cfg.get("occ_memory_validation_mesh_raycast_enable", False)
        )
        self._stage23a_mesh_raycast_max_rays = max(
            0,
            int(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_mesh_raycast_max_rays", 128
                )
            ),
        )
        self._stage23a_mesh_raycast_errors = []
        self._stage23a_mesh_raycast_signed_errors = []
        self._stage23a_mesh_raycast_total = 0
        self._stage23a_mesh_raycast_hits = 0
        self._stage23a_mesh_raycast_misses = 0
        self._stage23a_mesh_gt_occ_voxels = set()
        self._stage23a_mesh_gt_free_voxels = set()
        self._stage23b_navmesh_audit_enabled = bool(
            vlmap_safety_cfg.get(
                "occ_memory_validation_navmesh_traversability_enable", False
            )
        )
        self._stage23b_navmesh_max_cells = max(
            32,
            int(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_navmesh_max_cells", 1200
                )
            ),
        )
        self._stage23b_navmesh_max_pairs = max(
            0,
            int(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_navmesh_max_pairs", 12
                )
            ),
        )
        self._stage23b_agent_radius_m = max(
            0.0,
            float(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_navmesh_agent_radius_m", 0.18
                )
            ),
        )
        self._stage23b_clearance_ablation_enabled = bool(
            vlmap_safety_cfg.get(
                "occ_memory_validation_navmesh_clearance_ablation_enable",
                False,
            )
        )
        self._stage23b_clearance_height_max_m = max(
            float(vlmap_safety_cfg.get("obstacle_height_max", 1.2)),
            float(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_navmesh_clearance_height_max_m",
                    self._camera_height,
                )
            ),
        )
        self._stage23b_route_support_audit_enabled = bool(
            vlmap_safety_cfg.get(
                "occ_memory_validation_route_support_audit_enable", False
            )
        )
        self._stage23c_semantic_scene_audit_enabled = bool(
            vlmap_safety_cfg.get(
                "occ_memory_validation_semantic_scene_audit_enable", False
            )
        )
        self._stage23c_semantic_scene_audit_max_objects = max(
            0,
            int(
                vlmap_safety_cfg.get(
                    "occ_memory_validation_semantic_scene_audit_max_objects",
                    4000,
                )
            ),
        )
        self._stage27_candidate_audit_enabled = bool(
            vlmap_safety_cfg.get("stage27_candidate_audit_enable", False)
        )
        self._stage27_candidate_audit_cfg = dict(
            vlmap_safety_cfg.get("stage27_candidate_audit_config", {}) or {}
        )
        self._stage45_candidate_rejection_truth_enable = bool(
            vlmap_safety_cfg.get(
                "stage45_candidate_rejection_truth_enable", False
            )
        )
        self._stage45_candidate_rejection_truth_cfg = dict(
            vlmap_safety_cfg.get(
                "stage45_candidate_rejection_truth_config", {}
            )
            or {}
        )
        self._stage27_candidate_audit_entries = {
            (
                str(item.get("scene_id")),
                int(item.get("episode_id", -1)),
                int(item.get("step_id", -1)),
            ): dict(item)
            for item in list(
                vlmap_safety_cfg.get("stage27_candidate_audit_entries", []) or []
            )
            if isinstance(item, dict)
        }
        self._stage27_candidate_audit_records = {}
        self.occ_memory_oracle_pose = None
        if self._stage23a_oracle_pose_enabled:
            oracle_cfg = copy.deepcopy(vlmap_safety_cfg)
            oracle_cfg["occ_memory_validation_pose_from_context"] = True
            oracle_cfg["occ_memory_stage21_multitask_shadow_enable"] = False
            oracle_cfg["occ_memory_progress_ranker_shadow_enable"] = False
            oracle_cfg["occ_memory_candidate_probe_enable"] = False
            oracle_cfg["occ_memory_save_bev"] = False
            oracle_cfg["occ_memory_candidate_probe_save_bev"] = False
            self.occ_memory_oracle_pose = SparseOccSemanticMemory(
                oracle_cfg,
                get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
            )
            oracle_debug_dir = self._get_vlmap_run_dir()
            if oracle_debug_dir:
                oracle_debug_dir = os.path.join(
                    oracle_debug_dir, "stage23a_oracle_pose"
                )
            self.occ_memory_oracle_pose.set_debug_dir(oracle_debug_dir)
        self.occ_memory_oracle_sensor_pose = None
        if self._stage23a_oracle_sensor_pose_enabled:
            oracle_cfg = copy.deepcopy(vlmap_safety_cfg)
            oracle_cfg["occ_memory_validation_camera_pose_from_context"] = True
            oracle_cfg["occ_memory_validation_pose_from_context"] = True
            oracle_cfg["occ_memory_stage21_multitask_shadow_enable"] = False
            oracle_cfg["occ_memory_progress_ranker_shadow_enable"] = False
            oracle_cfg["occ_memory_candidate_probe_enable"] = False
            oracle_cfg["occ_memory_save_bev"] = False
            oracle_cfg["occ_memory_candidate_probe_save_bev"] = False
            self.occ_memory_oracle_sensor_pose = SparseOccSemanticMemory(
                oracle_cfg,
                get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
            )
            oracle_debug_dir = self._get_vlmap_run_dir()
            if oracle_debug_dir:
                oracle_debug_dir = os.path.join(
                    oracle_debug_dir, "stage23a_oracle_sensor_pose"
                )
            self.occ_memory_oracle_sensor_pose.set_debug_dir(oracle_debug_dir)

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws, collisions, collision_free, cf_sucs, cf_spls = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws, collisions, collision_free, cf_sucs, cf_spls = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
            "collisions": collisions,  # shape [N_local]
            "collision_free": collision_free,  # shape [N_local]
            "cf_sucs": cf_sucs,  # shape [N_local]
            "cf_spls": cf_spls,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]

        # Distributed all_gather requires every rank to use the same dtype.
        # A one-episode rank can otherwise infer float32/float64 differently
        # from the Python/NumPy scalar type returned by Habitat metrics.
        return {
            name: tensor.to(device=self.device, dtype=torch.float32)
            for name, tensor in result.items()
        }

    def _get_vlmap_run_dir(self) -> Optional[str]:
        if self._vlmap_run_dir is not None:
            return self._vlmap_run_dir
        if not hasattr(self, "vlmap_safety"):
            return None
        get_debug_dir = getattr(self.vlmap_safety, "get_debug_dir", None)
        if get_debug_dir is None:
            return None
        self._vlmap_run_dir = get_debug_dir()
        return self._vlmap_run_dir

    def _stage23a_sim_pose_context(self, *, initialize: bool = False) -> dict:
        """Return GT-only simulator pose fields for the Stage23A shadow audit."""
        if not (
            self._stage23a_oracle_pose_enabled
            or self._stage23a_oracle_sensor_pose_enabled
        ):
            return {}
        try:
            state = self.env._env.sim.get_agent_state()
            position = np.asarray(state.position, dtype=np.float32).reshape(3)
            agent_rotation = quaternion.as_rotation_matrix(state.rotation).astype(
                np.float32
            )
            agent_matrix = np.eye(4, dtype=np.float32)
            agent_matrix[:3, :3] = agent_rotation
            agent_matrix[:3, 3] = position
            if initialize or self._stage23a_initial_sim_position is None:
                self._stage23a_initial_sim_position = position.copy()
            if initialize or self._stage23a_initial_agent_matrix is None:
                self._stage23a_initial_agent_matrix = agent_matrix.copy()
            initial = np.asarray(
                self._stage23a_initial_sim_position, dtype=np.float32
            ).reshape(3)
            rotation = quaternion.as_float_array(state.rotation).astype(np.float32)
            context = {
                "stage23a_gt_relative_height_m": float(position[1] - initial[1]),
                "stage23a_sim_position": position.tolist(),
                "stage23a_initial_sim_position": initial.tolist(),
                "stage23a_sim_rotation_wxyz": rotation.tolist(),
                "stage23a_gt_only": True,
            }
            if self._stage23a_oracle_sensor_pose_enabled:
                sensor_states = getattr(state, "sensor_states", {}) or {}
                sensor_key = None
                for candidate in ("depth", "depth_sensor", "rgb", "rgb_sensor"):
                    if candidate in sensor_states:
                        sensor_key = candidate
                        break
                if sensor_key is None and sensor_states:
                    sensor_key = next(iter(sensor_states))
                sensor_state = sensor_states.get(sensor_key) if sensor_key else None
                if sensor_state is None:
                    raise RuntimeError("Habitat agent state has no sensor state")
                sensor_position = np.asarray(
                    sensor_state.position, dtype=np.float32
                ).reshape(3)
                sensor_rotation = quaternion.as_rotation_matrix(
                    sensor_state.rotation
                ).astype(np.float32)
                sensor_matrix = np.eye(4, dtype=np.float32)
                sensor_matrix[:3, :3] = sensor_rotation
                sensor_matrix[:3, 3] = sensor_position
                initial_agent_inverse = np.linalg.inv(
                    np.asarray(self._stage23a_initial_agent_matrix, dtype=np.float32)
                )
                # Habitat local axes: +X right, +Y up, -Z forward.
                # Internal map axes: +X forward, +Y left, +Z up.
                habitat_to_map = np.array(
                    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                )
                optical_to_habitat_sensor = np.diag(
                    [1.0, -1.0, -1.0, 1.0]
                ).astype(np.float32)
                habitat_to_map_tf = np.eye(4, dtype=np.float32)
                habitat_to_map_tf[:3, :3] = habitat_to_map
                context.update(
                    {
                        "stage23_gt_base_pose_map": (
                            habitat_to_map_tf
                            @ initial_agent_inverse
                            @ agent_matrix
                        ).tolist(),
                        "stage23_gt_camera_pose_map": (
                            habitat_to_map_tf
                            @ initial_agent_inverse
                            @ sensor_matrix
                            @ optical_to_habitat_sensor
                        ).tolist(),
                        "stage23a_sensor_state_key": sensor_key,
                        "stage23a_sensor_position": sensor_position.tolist(),
                        "stage23a_sensor_rotation_wxyz": quaternion.as_float_array(
                            sensor_state.rotation
                        ).astype(np.float32).tolist(),
                        "stage23a_gt_sensor_pose_only": True,
                    }
                )
            return context
        except Exception as exc:
            return {
                "stage23a_gt_pose_error": f"{type(exc).__name__}: {exc}",
                "stage23a_gt_only": True,
            }

    def _setup_vlmap_run_logging(self) -> None:
        run_dir = self._get_vlmap_run_dir()
        if not run_dir:
            return
        log_path = os.path.join(run_dir, "log.jsonl")
        if getattr(sys.stdout, "log_path", None) != log_path:
            self._vlmap_log_stdout = _JsonlTeeStream(sys.stdout, log_path, "stdout", self.rank)
            sys.stdout = self._vlmap_log_stdout
        if getattr(sys.stderr, "log_path", None) != log_path:
            self._vlmap_log_stderr = _JsonlTeeStream(sys.stderr, log_path, "stderr", self.rank)
            sys.stderr = self._vlmap_log_stderr
        print(f"[VLMapSafety][Habitat] run log path: {log_path}")

    def _write_episode_progress(self, result: dict) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
            f.write(json.dumps(self._jsonable(result)) + "\n")

        run_dir = self._get_vlmap_run_dir()
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, 'progress.json'), 'a', encoding="utf-8") as f:
                f.write(json.dumps(self._jsonable(result), ensure_ascii=False) + "\n")

    def _get_s2_action_loop_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        raw_variants = vlmap_safety_cfg.get(
            "s2_recovery_context_shadow_variants", ["text_only", "text_images"]
        )
        if isinstance(raw_variants, str):
            raw_variants = raw_variants.split(",")
        shadow_variants = tuple(
            item
            for item in (str(value).strip().lower() for value in list(raw_variants or []))
            if item in {"text_only", "text_images"}
        )
        return {
            "enable": bool(vlmap_safety_cfg.get("s2_action_loop_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("s2_action_loop_shadow_only", True)),
            "min_same_turn_generations": int(
                vlmap_safety_cfg.get("s2_action_loop_min_same_turn_generations", 5)
            ),
            "min_cumulative_turn_actions": int(
                vlmap_safety_cfg.get("s2_action_loop_min_cumulative_turn_actions", 12)
            ),
            "min_step_span": int(vlmap_safety_cfg.get("s2_action_loop_min_step_span", 6)),
            "min_episode_step": int(
                vlmap_safety_cfg.get("s2_action_loop_min_episode_step", 30)
            ),
            "max_translation_m": float(
                vlmap_safety_cfg.get("s2_action_loop_max_translation_m", 0.35)
            ),
            "max_snapshots_per_episode": int(
                vlmap_safety_cfg.get("s2_action_loop_max_snapshots_per_episode", 2)
            ),
            "executed_route_occ_audit_enable": bool(
                vlmap_safety_cfg.get("s2_loop_executed_route_occ_audit_enable", False)
            ),
            "fixed_route_occ_audit_enable": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_fixed_route_occ_audit_enable", False
                )
            ),
            "fixed_route_occ_evidence_audit_enable": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_fixed_route_occ_evidence_audit_enable", False
                )
            ),
            "fixed_route_height_evidence_audit_enable": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_fixed_route_height_evidence_audit_enable", False
                )
            ),
            "fixed_route_occ_audit_entries": tuple(
                dict(entry)
                for entry in (
                    vlmap_safety_cfg.get(
                        "s2_loop_fixed_route_occ_audit_entries", []
                    )
                    or []
                )
                if isinstance(entry, dict)
            ),
            "executed_route_occ_audit_max_edge_m": max(
                0.05,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_executed_route_occ_audit_max_edge_m", 0.75
                    )
                ),
            ),
            "executed_route_occ_audit_sample_spacing_m": max(
                0.01,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_executed_route_occ_audit_sample_spacing_m", 0.05
                    )
                ),
            ),
            "executed_route_occ_audit_max_path_cells": max(
                1,
                int(
                    vlmap_safety_cfg.get(
                        "s2_loop_executed_route_occ_audit_max_path_cells", 160
                    )
                ),
            ),
            "executed_route_occ_audit_max_visited_cells": max(
                1,
                int(
                    vlmap_safety_cfg.get(
                        "s2_loop_executed_route_occ_audit_max_visited_cells", 20000
                    )
                ),
            ),
            "recovery_context_enable": bool(
                vlmap_safety_cfg.get("s2_recovery_context_enable", False)
            ),
            "recovery_context_shadow_only": bool(
                vlmap_safety_cfg.get("s2_recovery_context_shadow_only", True)
            ),
            "recovery_context_max_images": int(
                vlmap_safety_cfg.get("s2_recovery_context_max_images", 3)
            ),
            "recovery_context_ttl_queries": int(
                vlmap_safety_cfg.get("s2_recovery_context_ttl_queries", 2)
            ),
            "recovery_context_save_images": bool(
                vlmap_safety_cfg.get("s2_recovery_context_save_images", False)
            ),
            "recovery_context_shadow_variants": shadow_variants,
            "strict_active_enable": bool(
                vlmap_safety_cfg.get("s2_loop_strict_active_enable", False)
            ),
            "strict_active_max_interventions_per_episode": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "s2_loop_strict_active_max_interventions_per_episode", 1
                    )
                ),
            ),
            "strict_active_require_active_gate_safe": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_strict_active_require_active_gate_safe", True
                )
            ),
            "strict_active_allowed_directions": tuple(
                str(value).strip().lower()
                for value in list(
                    vlmap_safety_cfg.get(
                        "s2_loop_strict_active_allowed_directions",
                        ["front", "left", "right"],
                    )
                    or []
                )
                if str(value).strip()
            ),
            "projection_bridge_enable": bool(
                vlmap_safety_cfg.get("s2_loop_projection_bridge_enable", False)
            ),
            "projection_bridge_shadow_only": bool(
                vlmap_safety_cfg.get("s2_loop_projection_bridge_shadow_only", True)
            ),
            "projection_bridge_sample_x_ratios": tuple(
                float(value)
                for value in list(
                    vlmap_safety_cfg.get(
                        "s2_loop_projection_bridge_sample_x_ratios",
                        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
                    )
                    or []
                )
            ),
            "projection_bridge_sample_y_ratios": tuple(
                float(value)
                for value in list(
                    vlmap_safety_cfg.get(
                        "s2_loop_projection_bridge_sample_y_ratios",
                        [0.55, 0.65, 0.75, 0.85],
                    )
                    or []
                )
            ),
            "projection_bridge_max_angle_error_deg": float(
                vlmap_safety_cfg.get(
                    "s2_loop_projection_bridge_max_angle_error_deg", 30.0
                )
            ),
            "path_reobserve_active_enable": bool(
                vlmap_safety_cfg.get("s2_loop_path_reobserve_active_enable", False)
            ),
            "path_reobserve_candidate_source": str(
                vlmap_safety_cfg.get(
                    "s2_loop_path_reobserve_candidate_source", "legacy_semantic"
                )
            ),
            "path_reobserve_one_primitive_per_reaudit": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_path_reobserve_one_primitive_per_reaudit", False
                )
            ),
            "path_reobserve_iterative_reorient_enable": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_path_reobserve_iterative_reorient_enable", False
                )
            ),
            "path_reobserve_max_interventions_per_episode": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_max_interventions_per_episode", 1
                    )
                ),
            ),
            "path_reobserve_max_turn_steps": max(
                0,
                int(vlmap_safety_cfg.get("s2_loop_path_reobserve_max_turn_steps", 4)),
            ),
            "path_reobserve_turn_deadband_deg": max(
                0.0,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_turn_deadband_deg", 7.5
                    )
                ),
            ),
            "path_reobserve_scan_when_aligned": bool(
                vlmap_safety_cfg.get(
                    "s2_loop_path_reobserve_scan_when_aligned", True
                )
            ),
            "path_reobserve_max_path_cells": max(
                1,
                int(vlmap_safety_cfg.get("s2_loop_path_reobserve_max_path_cells", 160)),
            ),
            "path_reobserve_max_active_path_m": max(
                0.0,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_max_active_path_m", 0.0
                    )
                ),
            ),
            "path_reobserve_path_corridor_m": max(
                0.05,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_path_corridor_m", 0.35
                    )
                ),
            ),
            "path_reobserve_min_path_progress_m": max(
                0.05,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_min_path_progress_m", 0.25
                    )
                ),
            ),
            "path_reobserve_max_local_subgoal_m": max(
                0.25,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_max_local_subgoal_m", 3.0
                    )
                ),
            ),
            "path_reobserve_max_heading_error_deg": max(
                0.0,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_max_heading_error_deg", 40.0
                    )
                ),
            ),
            "path_reobserve_lookahead_m": max(
                0.05,
                float(
                    vlmap_safety_cfg.get(
                        "s2_loop_path_reobserve_lookahead_m", 0.75
                    )
                ),
            ),
        }

    def _format_s2_recovery_context(self, context: dict) -> str:
        """Format a compact, temporary recovery card for an S2 re-query.

        This is deliberately textual and evidence-scoped.  It does not claim
        that the rejected direction is globally wrong, and it does not expose
        GT/reference-path fields.  The associated images are appended by the
        caller in the order listed in ``context['image_roles']``.
        """
        event = dict(context or {})
        failure_type = str(event.get("failure_type") or "unknown")
        direction = str(event.get("turn_direction") or "unknown")
        streak = event.get("same_turn_generation_streak")
        turn_actions = event.get("cumulative_turn_actions")
        translation = event.get("translation_m")
        candidate = dict(event.get("candidate") or {})
        candidate_direction = str(candidate.get("direction_bucket") or "unknown")
        candidate_distance = candidate.get("distance_m")
        candidate_open = candidate.get("semantic_resilience_open_score")
        candidate_safe = candidate.get("geometry_safe")
        semantic_term = (
            (candidate.get("semantic_evidence") or {}).get("semantic_top_match")
            or (candidate.get("semantic_evidence") or {}).get("matched_landmark")
            or candidate.get("semantic_top_match")
            or candidate.get("matched_landmark")
            or candidate.get("anchor_semantic_top_match")
            or "unknown"
        )
        return (
            "Temporary recovery context (observed local evidence, not ground truth): "
            f"failure={failure_type}; repeated_turn={direction}; "
            f"query_streak={streak}; cumulative_turn_actions={turn_actions}; "
            f"translation_m={translation}; "
            f"recovery_anchor_direction={candidate_direction}; "
            f"anchor_distance_m={candidate_distance}; "
            f"anchor_geometry_safe={candidate_safe}; "
            f"anchor_open_score={candidate_open}; "
            f"anchor_semantic={semantic_term}. "
            "The repeated direction has failed to produce new local progress in "
            "this recent state. Re-observe the current view, respect the instruction, "
            "and choose a visible, executable, non-redundant waypoint; do not treat "
            "this card as a forced left/right command."
        )

    @staticmethod
    def _nearest_recovery_frame_record(frame_records: list, step_id) -> Optional[dict]:
        if step_id is None or not frame_records:
            return None
        try:
            target = int(step_id)
        except (TypeError, ValueError):
            return None
        best = None
        best_gap = None
        for item in frame_records:
            try:
                gap = abs(int(item.get("step_id")) - target)
            except (TypeError, ValueError):
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best = item
        return best if isinstance((best or {}).get("image"), Image.Image) else None

    def _build_s2_recovery_context(
        self,
        event: Optional[dict],
        *,
        frame_records: list,
        current_image: Image.Image,
    ) -> Optional[dict]:
        """Build a short-lived event packet from already executed observations."""
        cfg = self._get_s2_action_loop_cfg()
        if not cfg.get("recovery_context_enable") or not event:
            return None
        if not isinstance(current_image, Image.Image):
            return None

        candidate = dict(event.get("candidate") or {})
        anchor_step = candidate.get("semantic_resilience_source_step_id")
        roles = []
        images = []
        selected_steps = []

        def add(role: str, record: Optional[dict]) -> None:
            if not record or not isinstance(record.get("image"), Image.Image):
                return
            record_step = int(record.get("step_id"))
            if record_step in selected_steps:
                return
            roles.append(str(role))
            selected_steps.append(record_step)
            # The frame history shares rgb_list references.  Only the sparse
            # event packet owns copies, avoiding an episode-long second buffer.
            images.append(record["image"].copy())

        start_step = event.get("start_step")
        current_step = event.get("step_id")
        if start_step != current_step:
            add(
                "frame near the first repeated S2 decision",
                self._nearest_recovery_frame_record(frame_records, start_step),
            )
        if anchor_step not in {start_step, current_step}:
            add(
                "recent safe recovery anchor frame",
                self._nearest_recovery_frame_record(frame_records, anchor_step),
            )
        # The current loop frame is already the current image in the base S2
        # query.  Do not append it a second time and accidentally reweight it.
        max_images = max(0, int(cfg.get("recovery_context_max_images", 2)))
        images = images[:max_images]
        roles = roles[: len(images)]
        selected_steps = selected_steps[: len(images)]
        context = {
            "event_type": "s2_recovery_context",
            "event_schema_version": "stage21d_recovery_context_v1",
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "episode_index": event.get("episode_index"),
            "episode_count": event.get("episode_count"),
            "episode_eval_seed": event.get("episode_eval_seed"),
            "failure_type": event.get("failure_type"),
            "turn_direction": event.get("turn_direction"),
            "triage_tier": event.get("triage_tier"),
            "triage_reason": event.get("triage_reason"),
            "same_turn_generation_streak": event.get("same_turn_generation_streak"),
            "cumulative_turn_actions": event.get("cumulative_turn_actions"),
            "translation_m": event.get("translation_m"),
            "trigger_step": event.get("step_id"),
            "current_query_step": event.get("step_id"),
            "first_repeated_decision_step": event.get("start_step"),
            "safe_anchor_step": anchor_step,
            "candidate": candidate,
            "base_current_frame_present": True,
            "image_roles": roles,
            "image_steps": selected_steps,
            "images": images,
            "remaining_queries": max(1, int(cfg.get("recovery_context_ttl_queries", 2))),
            "prompt": self._format_s2_recovery_context(event),
            "action_applied": False,
            "gt_fields_used": [],
        }
        if cfg.get("recovery_context_save_images"):
            run_dir = self._get_vlmap_run_dir()
            if run_dir:
                try:
                    snapshot_dir = os.path.join(run_dir, "s2_recovery_context_snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    stem = (
                        f"{context.get('scene_id')}_{int(context.get('episode_id'))}_"
                        f"trigger{int(context.get('trigger_step'))}"
                    )
                    snapshot_records = []
                    snapshot_items = [
                        (
                            "base current frame",
                            context.get("current_query_step"),
                            current_image,
                        )
                    ] + list(zip(roles, selected_steps, images))
                    for index, (role, frame_step, frame_image) in enumerate(snapshot_items):
                        path = os.path.join(snapshot_dir, f"{stem}_image{index}.jpg")
                        frame_image.save(path, quality=90)
                        snapshot_records.append(
                            {
                                "prompt_role": str(role),
                                "step_id": frame_step,
                                "path": path,
                            }
                        )
                    context["snapshot_records"] = snapshot_records
                except Exception as exc:
                    context["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        return context

    def _recovery_context_prompt(self, context: dict, variant: str) -> str:
        if variant == "text_only":
            return str(context.get("prompt") or "")
        roles = list(context.get("image_roles") or [])
        image_tokens = " ".join(
            f"{DEFAULT_IMAGE_TOKEN} ({role})" for role in roles
        )
        suffix = f" Recovery context images: {image_tokens}." if image_tokens else ""
        return f"{context.get('prompt', '')}{suffix}"

    def _s2_direct_action_codes(self, text: Any) -> list[int]:
        value = str(text or "").strip()
        if not value or re.search(r"\d", value) or "STOP" in value.upper():
            return []
        residue = re.sub(r"[↑←→↓\s,.;:!?，。；：！？]", "", value)
        return [] if residue else self.parse_actions(value)

    def _s2_recovery_change_metrics(self, base_parse: dict, hinted_parse: dict, context: dict) -> dict:
        base_valid = bool(base_parse.get("valid"))
        hinted_valid = bool(hinted_parse.get("valid"))
        base_stop = bool(base_parse.get("is_stop"))
        hinted_stop = bool(hinted_parse.get("is_stop"))
        base_action_codes = self._s2_direct_action_codes(base_parse.get("text"))
        hinted_action_codes = self._s2_direct_action_codes(hinted_parse.get("text"))
        hinted_reobserve = bool(
            hinted_action_codes
            and set(hinted_action_codes) == {int(action_code.LOOKDOWN)}
        )
        base_goal = base_parse.get("pixel_goal")
        hinted_goal = hinted_parse.get("pixel_goal")
        shift = None
        if base_valid and hinted_valid and base_goal is not None and hinted_goal is not None:
            shift = float(
                np.hypot(
                    float(hinted_goal[0]) - float(base_goal[0]),
                    float(hinted_goal[1]) - float(base_goal[1]),
                )
            )
        if not base_valid and not base_stop and hinted_valid:
            transition = "invalid_or_turn_to_valid_pixel"
        elif hinted_reobserve:
            transition = "invalid_or_turn_to_reobserve"
        elif base_valid and hinted_stop:
            transition = "valid_to_stop"
        elif base_valid and hinted_valid:
            transition = "valid_to_valid"
        elif base_stop and hinted_valid:
            transition = "stop_to_valid_pixel"
        elif hinted_stop:
            transition = "to_stop"
        elif not hinted_valid:
            transition = "remains_invalid_or_turn"
        else:
            transition = "other"
        hinted_turn = normalize_direct_turn_output(hinted_parse.get("text"))
        repeated_direction = str(context.get("turn_direction") or "unknown")
        return {
            "base_valid": base_valid,
            "hinted_valid": hinted_valid,
            "base_is_stop": base_stop,
            "hinted_is_stop": hinted_stop,
            "base_direct_action_codes": base_action_codes,
            "hinted_direct_action_codes": hinted_action_codes,
            "hinted_protocol_valid": bool(
                hinted_valid or hinted_stop or hinted_action_codes
            ),
            "hinted_reobserve": hinted_reobserve,
            "base_pixel_goal": base_goal,
            "hinted_pixel_goal": hinted_goal,
            "base_direction_bucket": base_parse.get("direction_bucket"),
            "hinted_direction_bucket": hinted_parse.get("direction_bucket"),
            "change_type": transition,
            "changed_pixel": bool(base_goal != hinted_goal),
            "valid_pixel_shift_px": shift,
            "large_valid_pixel_shift_40px": bool(shift is not None and shift > 40.0),
            "direction_bucket_changed": bool(
                base_parse.get("direction_bucket") != hinted_parse.get("direction_bucket")
            ),
            "hinted_direct_turn_direction": (
                None if hinted_turn is None else hinted_turn.get("direction")
            ),
            "continues_repeated_error_direction": bool(
                hinted_turn is not None
                and str(hinted_turn.get("direction")) == repeated_direction
            ),
        }

    def _run_s2_recovery_context_counterfactual(
        self,
        *,
        base_prompt_body: str,
        final_prompt: str,
        input_images: list,
        messages_prefix: Optional[list],
        base_output: str,
        context: dict,
        image_width: int,
        variant: str,
        current_query_step: int,
    ) -> dict:
        """Shadow-only extra S2 query; the resulting action is never applied."""
        variant = str(variant).strip().lower()
        extra_images = list(context.get("images") or []) if variant == "text_images" else []
        base_images = list(input_images)
        # The base prompt places historical image tokens before its final
        # current-view token. Recovery image tokens are inserted between those
        # two regions, so the processor image list must follow that same order.
        # Appending recovery images after the current frame silently binds every
        # prompt role to the wrong image.
        if extra_images and base_images:
            images = base_images[:-1] + extra_images + base_images[-1:]
        else:
            images = base_images
        prompt_image_roles = (
            [f"base historical frame {index}" for index in range(max(0, len(base_images) - 1))]
            + (list(context.get("image_roles") or []) if extra_images else [])
            + (["base current frame"] if base_images else [])
        )
        prompt = (
            f"{base_prompt_body} {self._recovery_context_prompt(context, variant)} "
            f"{final_prompt}."
        )
        event = {
            "event_type": "s2_recovery_context_counterfactual",
            "event_schema_version": "stage21d_recovery_context_v1",
            "variant": variant,
            "shadow_only": True,
            "action_applied": False,
            "gt_fields_used": [],
            "base_output": base_output,
            "scene_id": context.get("scene_id"),
            "episode_id": context.get("episode_id"),
            "episode_index": context.get("episode_index"),
            "episode_count": context.get("episode_count"),
            "episode_eval_seed": context.get("episode_eval_seed"),
            "failure_type": context.get("failure_type"),
            "triage_tier": context.get("triage_tier"),
            "triage_reason": context.get("triage_reason"),
            "turn_direction": context.get("turn_direction"),
            "trigger_step": context.get("trigger_step"),
            "current_query_step": int(current_query_step),
            "first_repeated_decision_step": context.get("first_repeated_decision_step"),
            "safe_anchor_step": context.get("safe_anchor_step"),
            "base_current_frame_present": bool(context.get("base_current_frame_present")),
            "image_roles": list(context.get("image_roles") or []) if extra_images else [],
            "image_steps": list(context.get("image_steps") or []) if extra_images else [],
            "extra_image_count": len(extra_images),
            "base_image_count": len(base_images),
            "prompt_image_roles": prompt_image_roles,
            "prompt_image_binding_valid": bool(len(prompt_image_roles) == len(images)),
            "snapshot_records": list(context.get("snapshot_records") or []),
        }
        rng_state = self._capture_torch_rng_state()
        started = time.perf_counter()
        try:
            hinted_output = self._generate_s2_text_from_prompt_instruction(
                prompt,
                images,
                messages_prefix=messages_prefix,
                max_new_tokens=128,
            )
            base_parse = self._parse_s2_candidate_output(
                base_output, image_width=image_width
            )
            hinted_parse = self._parse_s2_candidate_output(
                hinted_output, image_width=image_width
            )
            event.update({"status": "ok", "hinted_output": hinted_output})
            event.update(self._s2_recovery_change_metrics(base_parse, hinted_parse, context))
        except Exception as exc:
            event.update({"status": "error", "error": str(exc)})
        finally:
            event["latency_ms"] = float((time.perf_counter() - started) * 1000.0)
            self._restore_torch_rng_state(rng_state)
        return event

    def _plan_s2_loop_projection_bridge_shadow(
        self,
        event: Optional[dict],
        *,
        observations: Optional[dict] = None,
        depth_m: Optional[np.ndarray] = None,
    ) -> dict:
        """Audit map-candidate to visible-free-pixel bridging without acting."""
        cfg = self._get_s2_action_loop_cfg()
        candidate = dict((event or {}).get("candidate") or {})
        result = {
            "event_type": "s2_loop_projection_bridge_shadow",
            "event_schema_version": "stage21c_projection_bridge_shadow_v1",
            "scene_id": (event or {}).get("scene_id"),
            "episode_id": (event or {}).get("episode_id"),
            "episode_index": (event or {}).get("episode_index"),
            "episode_count": (event or {}).get("episode_count"),
            "episode_eval_seed": (event or {}).get("episode_eval_seed"),
            "step_id": (event or {}).get("step_id"),
            "failure_type": (event or {}).get("failure_type"),
            "triage_tier": (event or {}).get("triage_tier"),
            "triage_reason": (event or {}).get("triage_reason"),
            "turn_direction": (event or {}).get("turn_direction"),
            "candidate": candidate,
            "enabled": bool(cfg.get("projection_bridge_enable")),
            "shadow_only": True,
            "considered": bool(event),
            "proposal_valid": False,
            "selected_pixel_goal": None,
            "reason": None,
            "action_applied": False,
            "output_rewritten": False,
            "gt_fields_used": [],
            "bridge": None,
        }
        if not result["enabled"]:
            result["reason"] = "disabled"
            return result
        if not cfg.get("projection_bridge_shadow_only"):
            result["reason"] = "shadow_only_required"
            return result
        if not event:
            result["reason"] = "missing_loop_event"
            return result
        if str(event.get("triage_tier") or "") != "strict_intervention":
            result["reason"] = "non_strict_hold"
            return result
        if not bool(candidate.get("geometry_safe")):
            result["reason"] = "candidate_not_geometry_safe"
            return result
        if not bool(candidate.get("active_gate_safe")):
            result["reason"] = "candidate_not_active_gate_safe"
            return result
        if str(candidate.get("direction_bucket") or "").lower() not in set(
            cfg.get("strict_active_allowed_directions") or ()
        ):
            result["reason"] = "candidate_direction_not_allowed"
            return result
        if observations is None or depth_m is None:
            result["reason"] = "missing_observation_or_depth"
            return result

        baseline_plan = self._semantic_resilience_active_lite_directional_pixel_goal(
            candidate, self._get_semantic_resilience_active_lite_cfg()
        )
        result["fixed_directional_pixel_plan"] = baseline_plan
        bridge = self.occ_memory.plan_recovery_projection_bridge(
            candidate,
            {
                "gps": observations.get("gps"),
                "compass": observations.get("compass"),
            },
            depth_m,
            context={
                "step_id": event.get("step_id"),
                "scene_id": event.get("scene_id"),
                "episode_id": event.get("episode_id"),
                "image_width": int(getattr(self.model_args, "resize_w", 384)),
                "image_height": int(getattr(self.model_args, "resize_h", 384)),
                "probe_source": "s2_loop_projection_bridge_shadow",
            },
            baseline_pixel_goal=baseline_plan.get("pixel_goal"),
            sample_x_ratios=cfg.get("projection_bridge_sample_x_ratios") or (),
            sample_y_ratios=cfg.get("projection_bridge_sample_y_ratios") or (),
            max_angle_error_deg=float(
                cfg.get("projection_bridge_max_angle_error_deg", 30.0)
            ),
        )
        result["bridge"] = bridge
        result["proposal_valid"] = bool(bridge.get("valid"))
        result["selected_pixel_goal"] = bridge.get("selected_pixel_goal")
        result["reason"] = str(bridge.get("reason") or "bridge_failed")
        return result

    def _plan_s2_loop_strict_active(
        self,
        event: Optional[dict],
        active_count: int,
        *,
        observations: Optional[dict] = None,
        depth_m: Optional[np.ndarray] = None,
    ) -> dict:
        cfg = self._get_s2_action_loop_cfg()
        candidate = dict((event or {}).get("candidate") or {})
        result = {
            "event_type": "s2_loop_strict_active",
            "event_schema_version": "stage21c_strict_active_v1",
            "scene_id": (event or {}).get("scene_id"),
            "episode_id": (event or {}).get("episode_id"),
            "episode_index": (event or {}).get("episode_index"),
            "episode_count": (event or {}).get("episode_count"),
            "episode_eval_seed": (event or {}).get("episode_eval_seed"),
            "step_id": (event or {}).get("step_id"),
            "failure_type": (event or {}).get("failure_type"),
            "triage_tier": (event or {}).get("triage_tier"),
            "triage_reason": (event or {}).get("triage_reason"),
            "turn_direction": (event or {}).get("turn_direction"),
            "candidate": candidate,
            "enabled": bool(cfg.get("strict_active_enable")),
            "considered": bool(event),
            "action_applied": False,
            "output_rewritten": False,
            "execution_pending": False,
            "reason": None,
            "intervention_index": int(active_count + 1),
            "intervention_budget": int(
                cfg.get("strict_active_max_interventions_per_episode", 1)
            ),
            "geometry_preflight": {
                "geometry_safe": bool(candidate.get("geometry_safe")),
                "active_gate_safe": bool(candidate.get("active_gate_safe")),
                "direction_bucket": candidate.get("direction_bucket"),
            },
            "trajectory_preflight": "delegated_to_existing_nextdit_occ_safety",
            "gt_fields_used": [],
        }
        if not result["enabled"]:
            result["reason"] = "disabled"
        elif not event:
            result["reason"] = "missing_loop_event"
        elif str(event.get("triage_tier") or "") != "strict_intervention":
            result["reason"] = "non_strict_hold"
        elif active_count >= int(cfg.get("strict_active_max_interventions_per_episode", 1)):
            result["reason"] = "budget_exhausted"
        elif not bool(candidate.get("geometry_safe")):
            result["reason"] = "candidate_not_geometry_safe"
        elif bool(cfg.get("strict_active_require_active_gate_safe")) and not bool(
            candidate.get("active_gate_safe")
        ):
            result["reason"] = "candidate_not_active_gate_safe"
        elif str(candidate.get("direction_bucket") or "").lower() not in set(
            cfg.get("strict_active_allowed_directions") or ()
        ):
            result["reason"] = "candidate_direction_not_allowed"
        else:
            plan = self._semantic_resilience_active_lite_directional_pixel_goal(
                candidate, self._get_semantic_resilience_active_lite_cfg()
            )
            result["pixel_goal_plan"] = plan
            if not plan.get("valid"):
                result["reason"] = str(plan.get("reason") or "invalid_pixel_goal")
            else:
                waypoint_preflight = self.occ_memory.evaluate_waypoint(
                    plan.get("pixel_goal"),
                    {
                        "gps": (observations or {}).get("gps"),
                        "compass": (observations or {}).get("compass"),
                    },
                    depth_m,
                    context={
                        "step_id": (event or {}).get("step_id"),
                        "scene_id": (event or {}).get("scene_id"),
                        "episode_id": (event or {}).get("episode_id"),
                        "image_width": int(getattr(self.model_args, "resize_w", 384)),
                        "image_height": int(getattr(self.model_args, "resize_h", 384)),
                        "probe_source": "s2_loop_strict_active_preflight",
                    },
                )
                result["waypoint_preflight"] = {
                    "valid": bool(waypoint_preflight.get("valid")),
                    "reason": waypoint_preflight.get("reason"),
                    "goal_state": waypoint_preflight.get("goal_state"),
                    "goal_grid": waypoint_preflight.get("goal_grid"),
                    "depth_m": waypoint_preflight.get("depth_m"),
                    "points_to_revisited_region": waypoint_preflight.get(
                        "points_to_revisited_region"
                    ),
                }
                if not waypoint_preflight.get("valid"):
                    result["reason"] = "waypoint_preflight_invalid"
                elif str(waypoint_preflight.get("goal_state") or "") != "free":
                    result["reason"] = "waypoint_preflight_not_free"
                else:
                    result["reason"] = "preflight_pass"
                    result["execution_pending"] = True
        return result

    def _recovery_path_bridge(
        self,
        event: dict,
        *,
        observations: Optional[dict],
        depth_m: Optional[np.ndarray],
        probe_source: str,
    ) -> dict:
        cfg = self._get_s2_action_loop_cfg()
        if observations is None or depth_m is None:
            return {
                "valid": False,
                "reason": "missing_observation_or_depth",
                "path_reachable": False,
            }

        return self.occ_memory.plan_recovery_path_bridge(
            dict(event.get("candidate") or {}),
            {
                "gps": observations.get("gps"),
                "compass": observations.get("compass"),
                "_prefer_observation_pose": str(probe_source).startswith(
                    "stage43_counterfactual"
                ),
            },
            depth_m,
            context={
                "step_id": event.get("step_id"),
                "scene_id": event.get("scene_id"),
                "episode_id": event.get("episode_id"),
                "image_width": int(getattr(self.model_args, "resize_w", 384)),
                "image_height": int(getattr(self.model_args, "resize_h", 384)),
                "probe_source": str(probe_source),
            },
            sample_x_ratios=cfg.get("projection_bridge_sample_x_ratios") or (),
            sample_y_ratios=cfg.get("projection_bridge_sample_y_ratios") or (),
            max_path_cells=int(cfg.get("path_reobserve_max_path_cells", 160)),
            path_corridor_m=float(
                cfg.get("path_reobserve_path_corridor_m", 0.35)
            ),
            min_path_progress_m=float(
                cfg.get("path_reobserve_min_path_progress_m", 0.25)
            ),
            max_local_subgoal_m=float(
                cfg.get("path_reobserve_max_local_subgoal_m", 3.0)
            ),
            max_initial_heading_error_deg=float(
                cfg.get("path_reobserve_max_heading_error_deg", 40.0)
            ),
            reorient_lookahead_m=float(
                cfg.get("path_reobserve_lookahead_m", 0.75)
            ),
        )

    def _stage23c_semantic_scene_audit(self, memory) -> dict:
        """Audit semantic anchors against Habitat scene annotations.

        This is deliberately episode-end and read-only.  Anchor coordinates
        live in the memory frame, while Habitat objects live in world space;
        the initial base transform is used only for this comparison.  The
        result is an availability/nearest-object audit, not voxel GT.
        """
        result = {
            "enabled": bool(self._stage23c_semantic_scene_audit_enabled),
            "valid": False,
            "gt_reference": "Habitat semantic_scene objects/regions",
            "annotation_available": False,
            "reason": None,
            "object_count": 0,
            "region_count": 0,
            "anchor_count": int(len(getattr(memory, "semantic_anchors", []) or [])),
            "matched_anchor_count": 0,
            "category_agreement_count": 0,
            "category_agreement_rate": None,
            "nearest_distance_m": {"count": 0, "median": None, "p95": None, "max": None},
            "nearest_surface_distance_m": {"count": 0, "median": None, "p95": None, "max": None},
            "anchors": [],
        }
        if not result["enabled"]:
            result["reason"] = "disabled"
            return result
        try:
            sim = self.env._env.sim
            scene = getattr(sim, "semantic_scene", None)
        except Exception as exc:
            result["reason"] = f"semantic_scene_access_error:{type(exc).__name__}"
            return result
        if scene is None:
            result["reason"] = "semantic_scene_unavailable"
            return result

        def _name(category):
            if category is None:
                return ""
            try:
                value = category.name()
            except Exception:
                value = getattr(category, "name", "")
            return str(value or "").strip()

        def _center(item):
            box = getattr(item, "aabb", None) or getattr(item, "obb", None)
            center = getattr(box, "center", None) if box is not None else None
            if center is None:
                center = getattr(item, "center", None)
            if center is None:
                return None
            try:
                value = np.asarray(center, dtype=np.float32).reshape(3)
            except Exception:
                return None
            return value if np.all(np.isfinite(value)) else None

        def _bounds(item, center):
            box = getattr(item, "aabb", None) or getattr(item, "obb", None)
            sizes = getattr(box, "sizes", None) if box is not None else None
            try:
                sizes = np.asarray(sizes, dtype=np.float32).reshape(3)
            except Exception:
                return center.copy(), center.copy()
            if not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
                return center.copy(), center.copy()
            return center - sizes / 2.0, center + sizes / 2.0

        objects = list(getattr(scene, "objects", None) or [])
        regions = list(getattr(scene, "regions", None) or [])
        limit = self._stage23c_semantic_scene_audit_max_objects
        if limit > 0:
            objects = objects[:limit]
            regions = regions[:limit]
        entries = []
        for kind, items in (("object", objects), ("region", regions)):
            for item in items:
                center = _center(item)
                if center is None:
                    continue
                category = _name(getattr(item, "category", None))
                if not category and kind == "region":
                    category = _name(getattr(item, "region", None))
                lower, upper = _bounds(item, center)
                entries.append(
                    {
                        "kind": kind,
                        "id": str(getattr(item, "id", "")),
                        "category": category,
                        "center": center,
                        "lower": lower,
                        "upper": upper,
                    }
                )
        result["object_count"] = int(len(objects))
        result["region_count"] = int(len(regions))
        result["annotation_available"] = bool(entries)
        if not entries:
            result["reason"] = "semantic_scene_has_no_usable_centers"
            return result

        if self._stage23a_initial_agent_matrix is None:
            result["reason"] = "missing_initial_agent_matrix"
            return result
        habitat_to_map = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        map_to_habitat_tf = np.eye(4, dtype=np.float32)
        map_to_habitat_tf[:3, :3] = habitat_to_map.T
        map_to_world_tf = (
            np.asarray(self._stage23a_initial_agent_matrix, dtype=np.float32)
            @ map_to_habitat_tf
        )
        distances = []
        anchors = list(getattr(memory, "semantic_anchors", []) or [])
        for anchor in anchors:
            xy = anchor.get("xy")
            if not xy or len(xy) < 2:
                continue
            local_z = float(anchor.get("world_z") or 0.0)
            local = np.array(
                [float(xy[0]), float(xy[1]), local_z, 1.0], dtype=np.float32
            )
            world = (map_to_world_tf @ local)[:3]
            if not np.all(np.isfinite(world)):
                continue
            nearest = min(entries, key=lambda item: float(np.linalg.norm(item["center"] - world)))
            distance = float(np.linalg.norm(nearest["center"] - world))
            surface_delta = np.maximum(
                np.maximum(nearest["lower"] - world, 0.0),
                world - nearest["upper"],
            )
            surface_distance = float(np.linalg.norm(surface_delta))
            term = str(anchor.get("semantic_top_match") or "").strip().lower()
            category = str(nearest.get("category") or "").strip().lower()
            term_tokens = {token for token in re.split(r"[^a-z0-9]+", term) if len(token) > 2}
            category_tokens = {token for token in re.split(r"[^a-z0-9]+", category) if len(token) > 2}
            agreement = bool(term and category and (term in category or category in term or term_tokens.intersection(category_tokens)))
            distances.append(distance)
            result["matched_anchor_count"] += 1
            result["category_agreement_count"] += int(agreement)
            result["anchors"].append(
                {
                    "anchor_id": anchor.get("anchor_id"),
                    "semantic_top_match": anchor.get("semantic_top_match"),
                    "semantic_kind": anchor.get("semantic_kind"),
                    "grid": anchor.get("grid"),
                    "memory_xy": [float(xy[0]), float(xy[1])],
                    "world_xyz": [float(value) for value in world],
                    "nearest_kind": nearest["kind"],
                    "nearest_id": nearest["id"],
                    "nearest_category": nearest["category"],
                    "nearest_center_xyz": [float(value) for value in nearest["center"]],
                    "nearest_distance_m": distance,
                    "nearest_surface_distance_m": surface_distance,
                    "category_agreement": agreement,
                }
            )
        if distances:
            values = np.asarray(distances, dtype=np.float64)
            result["nearest_distance_m"] = {
                "count": int(values.size),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
            result["category_agreement_rate"] = float(
                result["category_agreement_count"] / len(distances)
            )
            surface_values = np.asarray(
                [item["nearest_surface_distance_m"] for item in result["anchors"]],
                dtype=np.float64,
            )
            result["nearest_surface_distance_m"] = {
                "count": int(surface_values.size),
                "median": float(np.median(surface_values)),
                "p95": float(np.percentile(surface_values, 95)),
                "max": float(np.max(surface_values)),
            }
            for threshold in (0.10, 0.25, 0.50, 1.00):
                result[f"surface_distance_le_{str(threshold).replace('.', '_')}m_rate"] = float(
                    np.mean(surface_values <= threshold)
                )
        result["valid"] = True
        result["reason"] = "ok"
        output_root = self._get_vlmap_run_dir()
        if output_root:
            output_root = os.path.join(output_root, "stage23c_semantic_scene_audit")
            os.makedirs(output_root, exist_ok=True)
            path = os.path.join(
                output_root,
                f"{memory.episode_meta.get('scene_id')}_{memory.episode_meta.get('episode_id')}.json",
            )
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            result["json_path"] = path
        return result

    def _stage24a_semantic_scene_snapshot(self) -> dict:
        """Serialize audit-only Habitat semantic bounds for offline replay."""
        result = {
            "available": False,
            "coordinate_frame": "habitat_world",
            "objects": [],
            "regions": [],
            "reason": None,
        }
        if not self.replay_ledger.enabled:
            result["reason"] = "replay_ledger_disabled"
            return result
        try:
            scene = getattr(self.env._env.sim, "semantic_scene", None)
        except Exception as exc:
            result["reason"] = f"semantic_scene_access_error:{type(exc).__name__}"
            return result
        if scene is None:
            result["reason"] = "semantic_scene_unavailable"
            return result

        def _category_name(item) -> str:
            category = getattr(item, "category", None)
            if category is None:
                return ""
            try:
                return str(category.name() or "").strip()
            except Exception:
                return str(getattr(category, "name", "") or "").strip()

        def _entry(item, kind: str):
            box = getattr(item, "aabb", None) or getattr(item, "obb", None)
            center = getattr(box, "center", None) if box is not None else None
            sizes = getattr(box, "sizes", None) if box is not None else None
            try:
                center = np.asarray(center, dtype=np.float32).reshape(3)
                sizes = np.asarray(sizes, dtype=np.float32).reshape(3)
            except Exception:
                return None
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(sizes)):
                return None
            if np.any(sizes < 0):
                return None
            return {
                "kind": kind,
                "id": str(getattr(item, "id", "")),
                "category": _category_name(item),
                "center": center.tolist(),
                "sizes": sizes.tolist(),
                "lower": (center - sizes / 2.0).tolist(),
                "upper": (center + sizes / 2.0).tolist(),
            }

        for key in ("objects", "regions"):
            kind = key[:-1]
            for item in list(getattr(scene, key, None) or []):
                entry = _entry(item, kind)
                if entry is not None:
                    result[key].append(entry)
        result["available"] = bool(result["objects"] or result["regions"])
        result["reason"] = "ok" if result["available"] else "no_usable_bounds"
        return result

    def _stage23a_mesh_raycast_audit(
        self, depth_m: np.ndarray, context: dict
    ) -> None:
        """Compare sampled GT-sensor RGB-D endpoints with Habitat mesh hits."""
        if not self._stage23a_mesh_raycast_enabled:
            return
        sensor_position = context.get("stage23a_sensor_position")
        sensor_rotation = context.get("stage23a_sensor_rotation_wxyz")
        if sensor_position is None or sensor_rotation is None:
            return
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return
        intrinsic = np.asarray(self.occ_memory.camera_intrinsic, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            return
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
        rows, cols = np.where(valid)
        if rows.size == 0:
            return
        max_rays = self._stage23a_mesh_raycast_max_rays
        if max_rays > 0 and rows.size > max_rays:
            ids = np.linspace(0, rows.size - 1, max_rays).astype(np.int64)
            rows, cols = rows[ids], cols[ids]
        try:
            sensor_rot = quaternion.as_rotation_matrix(
                quaternion.from_float_array(np.asarray(sensor_rotation, dtype=np.float64))
            ).astype(np.float32)
            sensor_pos = np.asarray(sensor_position, dtype=np.float32).reshape(3)
            optical_to_habitat = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            for row, col in zip(rows.tolist(), cols.tolist()):
                z = float(depth[row, col])
                cam_point = np.array(
                    [
                        (float(col) - cx) * z / fx,
                        (float(row) - cy) * z / fy,
                        z,
                    ],
                    dtype=np.float32,
                )
                direction = sensor_rot @ (optical_to_habitat @ cam_point)
                norm = float(np.linalg.norm(direction))
                if not np.isfinite(norm) or norm <= 1e-6:
                    continue
                ray = habitat_sim.geo.Ray(
                    Vector3(*sensor_pos.tolist()),
                    Vector3(*(direction / norm).tolist()),
                )
                self._stage23a_mesh_raycast_total += 1
                result = self.env._env.sim.cast_ray(ray, max_distance=10.0)
                if not bool(getattr(result, "has_hits", False)):
                    self._stage23a_mesh_raycast_misses += 1
                    continue
                hits = getattr(result, "hits", None) or []
                if not hits:
                    self._stage23a_mesh_raycast_misses += 1
                    continue
                hit_record = hits[0]
                hit_value = getattr(hit_record, "point", None)
                if hit_value is None:
                    hit_value = getattr(hit_record, "hit_pos", None)
                if hit_value is None:
                    self._stage23a_mesh_raycast_misses += 1
                    continue
                hit_pos = np.asarray(hit_value, dtype=np.float32).reshape(3)
                expected = sensor_pos + sensor_rot @ (
                    optical_to_habitat @ cam_point
                )
                error = float(np.linalg.norm(hit_pos - expected))
                if np.isfinite(error):
                    self._stage23a_mesh_raycast_errors.append(error)
                    hit_distance = float(np.linalg.norm(hit_pos - sensor_pos))
                    signed_error = float(norm - hit_distance)
                    if np.isfinite(signed_error):
                        self._stage23a_mesh_raycast_signed_errors.append(
                            signed_error
                        )
                    self._stage23a_mesh_raycast_hits += 1
                    if self._stage23a_initial_agent_matrix is not None:
                        habitat_to_map = np.eye(4, dtype=np.float32)
                        habitat_to_map[:3, :3] = np.array(
                            [
                                [0.0, 0.0, -1.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                            dtype=np.float32,
                        )
                        world_to_map = habitat_to_map @ np.linalg.inv(
                            np.asarray(
                                self._stage23a_initial_agent_matrix,
                                dtype=np.float32,
                            )
                        )
                        origin_map = (
                            world_to_map
                            @ np.array(
                                [sensor_pos[0], sensor_pos[1], sensor_pos[2], 1.0],
                                dtype=np.float32,
                            )
                        )[:3]
                        hit_map = (
                            world_to_map
                            @ np.array(
                                [hit_pos[0], hit_pos[1], hit_pos[2], 1.0],
                                dtype=np.float32,
                            )
                        )[:3]
                        endpoint_voxel = self.occ_memory._xyz_to_grid(hit_map)
                        if endpoint_voxel is not None:
                            self._stage23a_mesh_gt_occ_voxels.add(endpoint_voxel)
                        ray_delta = hit_map - origin_map
                        ray_length = float(np.linalg.norm(ray_delta))
                        if ray_length > self.occ_memory.cs:
                            direction_map = ray_delta / ray_length
                            sample_count = int(ray_length / self.occ_memory.cs)
                            for sample_index in range(1, sample_count):
                                point = origin_map + direction_map * (
                                    sample_index * self.occ_memory.cs
                                )
                                voxel = self.occ_memory._xyz_to_grid(point)
                                if voxel is not None and voxel != endpoint_voxel:
                                    self._stage23a_mesh_gt_free_voxels.add(voxel)
        except Exception as exc:
            print(f"[Stage23A][mesh_raycast] disabled for frame: {type(exc).__name__}: {exc}")

    def _stage23a_mesh_raycast_summary(self) -> dict:
        values = np.asarray(self._stage23a_mesh_raycast_errors, dtype=np.float64)
        values = values[np.isfinite(values)]
        signed = np.asarray(
            self._stage23a_mesh_raycast_signed_errors, dtype=np.float64
        )
        signed = signed[np.isfinite(signed)]
        return {
            "enabled": bool(self._stage23a_mesh_raycast_enabled),
            "total_rays": int(self._stage23a_mesh_raycast_total),
            "hit_count": int(self._stage23a_mesh_raycast_hits),
            "miss_count": int(self._stage23a_mesh_raycast_misses),
            "hit_rate": (
                float(self._stage23a_mesh_raycast_hits / self._stage23a_mesh_raycast_total)
                if self._stage23a_mesh_raycast_total
                else None
            ),
            "endpoint_error_count": int(values.size),
            "endpoint_error_mean_m": float(np.mean(values)) if values.size else None,
            "endpoint_error_median_m": float(np.median(values)) if values.size else None,
            "endpoint_error_p95_m": float(np.percentile(values, 95)) if values.size else None,
            "endpoint_error_p99_m": float(np.percentile(values, 99)) if values.size else None,
            "endpoint_error_max_m": float(np.max(values)) if values.size else None,
            "endpoint_error_le_0_01m_rate": (
                float(np.mean(values <= 0.01)) if values.size else None
            ),
            "endpoint_error_gt_0_10m_rate": (
                float(np.mean(values > 0.10)) if values.size else None
            ),
            "endpoint_error_gt_1m_rate": (
                float(np.mean(values > 1.0)) if values.size else None
            ),
            "signed_error_definition": "depth_endpoint_range_minus_mesh_hit_distance",
            "signed_error_count": int(signed.size),
            "signed_error_mean_m": float(np.mean(signed)) if signed.size else None,
            "signed_error_median_m": (
                float(np.median(signed)) if signed.size else None
            ),
            "surface_match_abs_le_0_05m_rate": (
                float(np.mean(np.abs(signed) <= 0.05)) if signed.size else None
            ),
            "potential_false_free_gt_0_05m_rate": (
                float(np.mean(signed > 0.05)) if signed.size else None
            ),
            "potential_false_occupied_lt_neg_0_05m_rate": (
                float(np.mean(signed < -0.05)) if signed.size else None
            ),
            "collision_mesh_miss_rate": (
                float(self._stage23a_mesh_raycast_misses / self._stage23a_mesh_raycast_total)
                if self._stage23a_mesh_raycast_total
                else None
            ),
        }

    def _stage23a_mesh_voxel_gt_summary(self, memory) -> dict:
        """Compare SparseOcc labels with collision-mesh ray voxel labels."""
        gt_occ = set(self._stage23a_mesh_gt_occ_voxels)
        gt_free = set(self._stage23a_mesh_gt_free_voxels) - gt_occ
        observed = gt_occ | gt_free
        pred_occ = set(memory.occ_counts.keys()) & observed
        pred_free = (set(memory.free_counts.keys()) - set(memory.occ_counts.keys())) & observed

        def metrics(predicted, positive):
            tp = len(predicted & positive)
            fp = len(predicted - positive)
            fn = len(positive - predicted)
            precision = tp / float(tp + fp) if tp + fp else None
            recall = tp / float(tp + fn) if tp + fn else None
            return {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": (
                    2.0 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and precision + recall
                    else None
                ),
                "iou": tp / float(tp + fp + fn) if tp + fp + fn else None,
            }

        tolerance = set()
        for row, col, height in gt_occ:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for dh in (-1, 0, 1):
                        tolerance.add((row + dr, col + dc, height + dh))
        tolerant_tp = len(pred_occ & tolerance)
        return {
            "enabled": bool(self._stage23a_mesh_raycast_enabled),
            "valid": bool(observed and gt_occ and gt_free),
            "gt_reference": "Habitat collision mesh first-hit rays",
            "evaluation_domain": "sampled_collision_mesh_observed_rays",
            "unknown_outside_sampled_rays": True,
            "gt_occupied_voxel_count": int(len(gt_occ)),
            "gt_free_voxel_count": int(len(gt_free)),
            "predicted_unknown_voxel_count": int(len(observed - pred_occ - pred_free)),
            "unknown_coverage": (
                float(len(observed - pred_occ - pred_free) / len(observed))
                if observed else None
            ),
            "occupied_exact": metrics(pred_occ, gt_occ),
            "free_exact": metrics(pred_free, gt_free),
            "occupied_tolerance_1voxel": {
                "tp": int(tolerant_tp),
                "prediction_count": int(len(pred_occ)),
                "precision": float(tolerant_tp / len(pred_occ)) if pred_occ else None,
                "tolerance_m": float(memory.cs),
            },
            "false_free_rate": (
                float(len(pred_free & gt_occ) / len(pred_free)) if pred_free else None
            ),
            "false_occupied_rate": (
                float(len(pred_occ & gt_free) / len(pred_occ)) if pred_occ else None
            ),
        }

    @staticmethod
    def _stage23b_binary_metrics(predicted: set, positive: set) -> dict:
        tp = len(predicted & positive)
        fp = len(predicted - positive)
        fn = len(positive - predicted)
        precision = tp / float(tp + fp) if tp + fp else None
        recall = tp / float(tp + fn) if tp + fn else None
        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision is not None
                and recall is not None
                and precision + recall
                else None
            ),
            "iou": (
                tp / float(tp + fp + fn) if tp + fp + fn else None
            ),
        }

    def _stage23b_navmesh_traversability_audit(
        self,
        memory: SparseOccSemanticMemory,
        reference: SparseOccSemanticMemory,
        *,
        branch_name: str,
        scene_id: str,
        episode_id: int,
        readout_height_max_m: Optional[float] = None,
    ) -> dict:
        """Compare a floor-aligned SparseOcc readout with Habitat navmesh."""
        result = {
            "enabled": bool(self._stage23b_navmesh_audit_enabled),
            "branch": str(branch_name),
            "valid": False,
            "reason": None,
            "shadow_only": True,
            "action_applied": False,
            "gt_fields_used_for_navigation": [],
            "gt_reference": "Habitat pathfinder navmesh",
            "evaluation_domain": "oracle_sensor_observed_xy_cells",
            "unknown_is_free": False,
            "local_floor_source": "nearest oracle sensor pose trace",
            "agent_radius_m": float(self._stage23b_agent_radius_m),
            "readout_height_min_m": float(
                memory.config.obstacle_height_min
            ),
            "readout_height_max_m": float(
                memory.config.obstacle_height_max
                if readout_height_max_m is None
                else readout_height_max_m
            ),
        }
        if not self._stage23b_navmesh_audit_enabled:
            result["reason"] = "disabled"
            return result
        if self._stage23a_initial_agent_matrix is None:
            result["reason"] = "missing_initial_agent_matrix"
            return result
        pathfinder = getattr(self.env._env.sim, "pathfinder", None)
        if pathfinder is None or not bool(getattr(pathfinder, "is_loaded", True)):
            result["reason"] = "pathfinder_unavailable"
            return result
        trace = []
        for node in reference.pose_trace:
            sim_position = node.get("sim_position")
            try:
                trace.append(
                    {
                        "row": int(node["row"]),
                        "col": int(node["col"]),
                        "x": float(node["x"]),
                        "y": float(node["y"]),
                        "z": float(node["z"]),
                        "sim_position": np.asarray(
                            sim_position, dtype=np.float32
                        ).reshape(3),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not trace:
            result["reason"] = "missing_oracle_pose_trace"
            return result
        trace_xy = np.asarray([[p["x"], p["y"]] for p in trace], dtype=np.float32)
        habitat_to_map = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        map_to_habitat_tf = np.eye(4, dtype=np.float32)
        map_to_habitat_tf[:3, :3] = habitat_to_map.T
        map_to_world = (
            np.asarray(self._stage23a_initial_agent_matrix, dtype=np.float32)
            @ map_to_habitat_tf
        )

        floor_cache = {}
        world_cache = {}
        nav_cache = {}
        evidence_cache = {}

        def cell_xy(cell):
            return memory._grid_to_xy(cell)

        def nearest_trace(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell not in floor_cache:
                xy = cell_xy(cell)
                index = int(np.argmin(np.sum((trace_xy - xy[None, :]) ** 2, axis=1)))
                floor_cache[cell] = (float(trace[index]["z"]), index)
            return floor_cache[cell]

        def cell_world(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell not in world_cache:
                floor_z, _ = nearest_trace(cell)
                xy = cell_xy(cell)
                point_map = np.array(
                    [float(xy[0]), float(xy[1]), floor_z, 1.0],
                    dtype=np.float32,
                )
                world_cache[cell] = (map_to_world @ point_map)[:3]
            return world_cache[cell]

        def navmesh_label(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell not in nav_cache:
                point = cell_world(cell)
                try:
                    navigable = bool(pathfinder.is_navigable(point, 0.5))
                except TypeError:
                    navigable = bool(pathfinder.is_navigable(point))
                snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
                snap_distance = (
                    float(np.linalg.norm(snapped - point))
                    if snapped.shape == (3,) and np.all(np.isfinite(snapped))
                    else None
                )
                nav_cache[cell] = (navigable, snap_distance)
            return nav_cache[cell]

        radius_cells = int(
            math.ceil(float(self._stage23b_agent_radius_m) / memory.cs)
        )

        def predicted_state(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell in evidence_cache:
                return evidence_cache[cell]
            floor_z, _ = nearest_trace(cell)
            center = memory.validation_floor_aligned_cell_evidence(
                cell[0],
                cell[1],
                floor_z,
                height_max_m=readout_height_max_m,
            )
            blocked = False
            if radius_cells > 0:
                for dr in range(-radius_cells, radius_cells + 1):
                    for dc in range(-radius_cells, radius_cells + 1):
                        if math.hypot(dr, dc) * memory.cs > float(
                            self._stage23b_agent_radius_m
                        ):
                            continue
                        neighbor = memory.validation_floor_aligned_cell_evidence(
                            cell[0] + dr,
                            cell[1] + dc,
                            floor_z,
                            height_max_m=readout_height_max_m,
                        )
                        if neighbor["state"] == "blocked":
                            blocked = True
                            break
                    if blocked:
                        break
            state = "blocked" if blocked else center["state"]
            evidence_cache[cell] = state
            return state

        route_support_cells = set()
        if self._stage23b_route_support_audit_enabled:
            movement_support = [trace[0]] if trace else []
            for node in trace[1:]:
                if math.hypot(
                    node["x"] - movement_support[-1]["x"],
                    node["y"] - movement_support[-1]["y"],
                ) > 1e-4:
                    movement_support.append(node)
            for first, second in zip(movement_support[:-1], movement_support[1:]):
                distance = math.hypot(
                    second["x"] - first["x"], second["y"] - first["y"]
                )
                route_support_cells.update(
                    memory._rasterize_executed_route_edge(
                        (first["row"], first["col"]),
                        (second["row"], second["col"]),
                        edge_length_m=distance,
                        sample_spacing_m=memory.cs,
                    )
                )
            # A collision-free executed centerline certifies a narrow swept
            # corridor, not only a one-cell digital line.  One-cell dilation
            # also keeps diagonal steps connected under corner-cut checks.
            route_support_cells = {
                (row + dr, col + dc)
                for row, col in route_support_cells
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }

        def route_support_state(cell):
            return "free" if cell in route_support_cells else "unknown"

        def combined_state(cell):
            if cell in route_support_cells:
                return "free"
            state = predicted_state(cell)
            return state

        def evenly_sample(cells, limit):
            ordered = sorted((int(r), int(c)) for r, c in cells)
            if len(ordered) <= limit:
                return ordered
            ids = np.linspace(0, len(ordered) - 1, limit).astype(np.int64)
            return [ordered[int(index)] for index in ids]

        ref_occ = {(int(r), int(c)) for r, c, _ in reference.occ_counts}
        ref_free = {(int(r), int(c)) for r, c, _ in reference.free_counts}
        route_cells = list(
            dict.fromkeys((int(node["row"]), int(node["col"])) for node in trace)
        )
        sample_budget = int(self._stage23b_navmesh_max_cells)
        half = max(1, (sample_budget - len(route_cells)) // 2)
        sampled = list(
            dict.fromkeys(
                route_cells
                + evenly_sample(ref_occ, half)
                + evenly_sample(ref_free, half)
            )
        )
        if len(sampled) > sample_budget:
            keep_route = list(dict.fromkeys(route_cells))
            remaining = [cell for cell in sampled if cell not in set(keep_route)]
            sampled = keep_route + evenly_sample(
                remaining, max(0, sample_budget - len(keep_route))
            )

        gt_free = set()
        gt_blocked = set()
        pred_free = set()
        pred_blocked = set()
        pred_unknown = set()
        snap_distances = []
        for cell in sampled:
            navigable, snap_distance = navmesh_label(cell)
            (gt_free if navigable else gt_blocked).add(cell)
            if snap_distance is not None:
                snap_distances.append(snap_distance)
            state = predicted_state(cell)
            if state == "free":
                pred_free.add(cell)
            elif state == "blocked":
                pred_blocked.add(cell)
            else:
                pred_unknown.add(cell)

        route_pred_free = {
            cell for cell in sampled if route_support_state(cell) == "free"
        }
        combined_pred_free = {
            cell for cell in sampled if combined_state(cell) == "free"
        }
        route_support_blocked_override_count = sum(
            predicted_state(cell) == "blocked" for cell in route_support_cells
        )

        route_state_counts = {"free": 0, "blocked": 0, "unknown": 0}
        route_navmesh_free = 0
        for cell in route_cells:
            route_state_counts[predicted_state(cell)] += 1
            route_navmesh_free += int(navmesh_label(cell)[0])

        movement_nodes = [trace[0]]
        for node in trace[1:]:
            if math.hypot(
                node["x"] - movement_nodes[-1]["x"],
                node["y"] - movement_nodes[-1]["y"],
            ) > 1e-4:
                movement_nodes.append(node)
        edge_records = []
        strict_edge_count = 0
        nonblocked_edge_count = 0
        for first, second in zip(movement_nodes[:-1], movement_nodes[1:]):
            start = (first["row"], first["col"])
            end = (second["row"], second["col"])
            distance = math.hypot(second["x"] - first["x"], second["y"] - first["y"])
            cells = memory._rasterize_executed_route_edge(
                start, end, edge_length_m=distance, sample_spacing_m=memory.cs
            )
            counts = {"free": 0, "blocked": 0, "unknown": 0}
            for cell in dict.fromkeys(cells):
                counts[predicted_state(cell)] += 1
            strict = counts["free"] > 0 and counts["blocked"] == 0 and counts["unknown"] == 0
            nonblocked = counts["blocked"] == 0
            strict_edge_count += int(strict)
            nonblocked_edge_count += int(nonblocked)
            edge_records.append(
                {
                    "source_grid": [int(start[0]), int(start[1])],
                    "target_grid": [int(end[0]), int(end[1])],
                    "length_m": float(distance),
                    "cell_state_counts": counts,
                    "strict_free": bool(strict),
                    "no_false_block": bool(nonblocked),
                }
            )

        state_cache = evidence_cache

        def path_with_state(start, goal, state_fn, max_visited=20000):
            start = (int(start[0]), int(start[1]))
            goal = (int(goal[0]), int(goal[1]))
            if state_fn(start) != "free" or state_fn(goal) != "free":
                return False, None
            queue = deque([start])
            distance = {start: 0.0}
            visited = 0
            while queue and visited < max_visited:
                cell = queue.popleft()
                visited += 1
                if cell == goal:
                    return True, float(distance[cell])
                for dr, dc, scale in (
                    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
                    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
                ):
                    neighbor = (cell[0] + dr, cell[1] + dc)
                    if neighbor in distance or state_fn(neighbor) != "free":
                        continue
                    if dr and dc and (
                        state_fn((cell[0] + dr, cell[1])) != "free"
                        or state_fn((cell[0], cell[1] + dc)) != "free"
                    ):
                        continue
                    if math.hypot(neighbor[0] - start[0], neighbor[1] - start[1]) * memory.cs > 6.0:
                        continue
                    distance[neighbor] = distance[cell] + scale * memory.cs
                    queue.append(neighbor)
            return False, None

        def predicted_path(start, goal, max_visited=20000):
            return path_with_state(start, goal, predicted_state, max_visited)

        def gt_path(start, goal):
            shortest = habitat_sim.ShortestPath()
            shortest.requested_start = cell_world(start)
            shortest.requested_end = cell_world(goal)
            found = bool(pathfinder.find_path(shortest))
            return found, float(shortest.geodesic_distance) if found else None

        pairs = []
        anchors = route_cells[:-1]
        if anchors:
            anchor_ids = np.linspace(
                0, len(anchors) - 1, min(6, len(anchors))
            ).astype(np.int64)
            current = route_cells[-1]
            for index in anchor_ids:
                anchor = anchors[int(index)]
                if anchor != current:
                    pairs.append(("current_to_history", current, anchor))
        nav_cells = [cell for cell in sampled if navmesh_label(cell)[0]]
        pair_budget = max(0, self._stage23b_navmesh_max_pairs - len(pairs))
        for index in range(min(pair_budget, max(0, len(nav_cells) - 1))):
            first = nav_cells[index]
            second = nav_cells[-(index + 1)]
            if first != second and np.linalg.norm(cell_xy(first) - cell_xy(second)) <= 4.0:
                pairs.append(("local_observed_pair", first, second))

        pair_records = []
        reachability_matches = 0
        geodesic_errors = []
        current_anchor_gt = current_anchor_pred = 0
        current_anchor_route = current_anchor_combined = 0
        route_reachability_matches = combined_reachability_matches = 0
        for role, start, goal in pairs[: self._stage23b_navmesh_max_pairs]:
            gt_reachable, gt_distance = gt_path(start, goal)
            pred_reachable, pred_distance = predicted_path(start, goal)
            route_reachable, route_distance = path_with_state(
                start, goal, route_support_state
            )
            combined_reachable, combined_distance = path_with_state(
                start, goal, combined_state
            )
            reachability_matches += int(gt_reachable == pred_reachable)
            route_reachability_matches += int(gt_reachable == route_reachable)
            combined_reachability_matches += int(gt_reachable == combined_reachable)
            if role == "current_to_history" and gt_reachable:
                current_anchor_gt += 1
                current_anchor_pred += int(pred_reachable)
                current_anchor_route += int(route_reachable)
                current_anchor_combined += int(combined_reachable)
            if (
                gt_reachable
                and pred_reachable
                and gt_distance is not None
                and pred_distance is not None
            ):
                geodesic_errors.append(abs(pred_distance - gt_distance))
            pair_records.append(
                {
                    "role": role,
                    "start_grid": [int(start[0]), int(start[1])],
                    "goal_grid": [int(goal[0]), int(goal[1])],
                    "gt_reachable": bool(gt_reachable),
                    "predicted_reachable": bool(pred_reachable),
                    "route_support_reachable": bool(route_reachable),
                    "combined_reachable": bool(combined_reachable),
                    "gt_geodesic_m": gt_distance,
                    "predicted_path_m": pred_distance,
                    "route_support_path_m": route_distance,
                    "combined_path_m": combined_distance,
                }
            )

        floor_levels = []
        route_heights_by_cell = {}
        for node in trace:
            height = float(node["z"])
            if not any(abs(height - existing) < 0.50 for existing in floor_levels):
                floor_levels.append(height)
            cell = (int(node["row"]), int(node["col"]))
            route_heights_by_cell.setdefault(cell, []).append(height)
        cross_floor_route_cell_count = sum(
            max(heights) - min(heights) >= 0.75
            for heights in route_heights_by_cell.values()
        )

        result.update(
            {
                "valid": True,
                "reason": "ok",
                "sampled_cell_count": int(len(sampled)),
                "gt_free_cell_count": int(len(gt_free)),
                "gt_blocked_cell_count": int(len(gt_blocked)),
                "predicted_free_cell_count": int(len(pred_free)),
                "predicted_blocked_cell_count": int(len(pred_blocked)),
                "predicted_unknown_cell_count": int(len(pred_unknown)),
                "unknown_coverage": (
                    float(len(pred_unknown) / len(sampled)) if sampled else None
                ),
                "free_metrics_observed_domain": self._stage23b_binary_metrics(
                    pred_free, gt_free
                ),
                "blocked_metrics_observed_domain": self._stage23b_binary_metrics(
                    pred_blocked, gt_blocked
                ),
                "false_free_rate": (
                    float(len(pred_free & gt_blocked) / len(pred_free))
                    if pred_free else None
                ),
                "false_blocked_rate": (
                    float(len(pred_blocked & gt_free) / len(pred_blocked))
                    if pred_blocked else None
                ),
                "navmesh_snap_distance_m": {
                    "mean": float(np.mean(snap_distances)) if snap_distances else None,
                    "p95": float(np.percentile(snap_distances, 95)) if snap_distances else None,
                },
                "executed_route_cell_count": int(len(route_cells)),
                "executed_route_navmesh_free_recall": (
                    float(route_navmesh_free / len(route_cells)) if route_cells else None
                ),
                "executed_route_predicted_free_recall": (
                    float(route_state_counts["free"] / len(route_cells)) if route_cells else None
                ),
                "executed_route_state_counts": route_state_counts,
                "historical_edge_count": int(len(edge_records)),
                "historical_edge_strict_free_recall": (
                    float(strict_edge_count / len(edge_records)) if edge_records else None
                ),
                "historical_edge_no_false_block_recall": (
                    float(nonblocked_edge_count / len(edge_records)) if edge_records else None
                ),
                "edge_records": edge_records,
                "pair_count": int(len(pair_records)),
                "reachability_agreement": (
                    float(reachability_matches / len(pair_records)) if pair_records else None
                ),
                "current_to_history_anchor_gt_reachable_count": int(current_anchor_gt),
                "current_to_history_anchor_predicted_reachable_count": int(current_anchor_pred),
                "current_to_history_anchor_connectivity_recall": (
                    float(current_anchor_pred / current_anchor_gt) if current_anchor_gt else None
                ),
                "route_support_audit_enabled": bool(
                    self._stage23b_route_support_audit_enabled
                ),
                "route_support_cell_count": int(len(route_support_cells)),
                "route_support_blocked_override_count": int(
                    route_support_blocked_override_count
                ),
                "combined_route_support_precedence": "executed_swept_corridor_overrides_occ_blocked",
                "route_support_free_metrics_observed_domain": (
                    self._stage23b_binary_metrics(route_pred_free, gt_free)
                ),
                "route_support_false_free_rate": (
                    float(len(route_pred_free & gt_blocked) / len(route_pred_free))
                    if route_pred_free else None
                ),
                "combined_free_metrics_observed_domain": (
                    self._stage23b_binary_metrics(combined_pred_free, gt_free)
                ),
                "combined_false_free_rate": (
                    float(len(combined_pred_free & gt_blocked) / len(combined_pred_free))
                    if combined_pred_free else None
                ),
                "route_support_reachability_agreement": (
                    float(route_reachability_matches / len(pair_records))
                    if pair_records else None
                ),
                "combined_reachability_agreement": (
                    float(combined_reachability_matches / len(pair_records))
                    if pair_records else None
                ),
                "current_to_history_anchor_route_support_connectivity_recall": (
                    float(current_anchor_route / current_anchor_gt)
                    if current_anchor_gt else None
                ),
                "current_to_history_anchor_combined_connectivity_recall": (
                    float(current_anchor_combined / current_anchor_gt)
                    if current_anchor_gt else None
                ),
                "geodesic_abs_error_m": {
                    "count": int(len(geodesic_errors)),
                    "mean": float(np.mean(geodesic_errors)) if geodesic_errors else None,
                    "p95": float(np.percentile(geodesic_errors, 95)) if geodesic_errors else None,
                },
                "trace_floor_level_count": int(len(floor_levels)),
                "trace_floor_levels_m": [float(value) for value in floor_levels],
                "cross_floor_route_cell_count": int(
                    cross_floor_route_cell_count
                ),
                "cross_floor_route_cell_rate": (
                    float(
                        cross_floor_route_cell_count
                        / len(route_heights_by_cell)
                    )
                    if route_heights_by_cell else None
                ),
                "pair_records": pair_records,
                "cached_cell_state_count": int(len(state_cache)),
            }
        )

        output_root = self._get_vlmap_run_dir()
        if output_root:
            output_root = os.path.join(output_root, "stage23b_navmesh_traversability")
            os.makedirs(output_root, exist_ok=True)
            json_path = os.path.join(
                output_root,
                f"{scene_id}_{episode_id}_{branch_name}.json",
            )
            with open(json_path, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            result["json_path"] = json_path

            canvas = np.full((768, 768, 3), 245, dtype=np.uint8)
            all_rows = [cell[0] for cell in sampled + route_cells]
            all_cols = [cell[1] for cell in sampled + route_cells]
            if all_rows and all_cols:
                row_min, row_max = min(all_rows), max(all_rows)
                col_min, col_max = min(all_cols), max(all_cols)
                span = max(row_max - row_min + 1, col_max - col_min + 1, 1)
                scale = 720.0 / float(span)
                def pixel(cell):
                    return (
                        int(24 + (cell[1] - col_min) * scale),
                        int(24 + (cell[0] - row_min) * scale),
                    )
                for cell in sampled:
                    gt = navmesh_label(cell)[0]
                    state = predicted_state(cell)
                    color = (190, 190, 190)
                    if state == "free" and gt:
                        color = (70, 170, 70)
                    elif state == "blocked" and not gt:
                        color = (60, 60, 210)
                    elif state == "free" and not gt:
                        color = (210, 70, 210)
                    elif state == "blocked" and gt:
                        color = (30, 150, 235)
                    cv2.circle(canvas, pixel(cell), 2, color, -1)
                route_pixels = [pixel(cell) for cell in route_cells]
                for first, second in zip(route_pixels[:-1], route_pixels[1:]):
                    cv2.line(canvas, first, second, (10, 10, 10), 2)
            image_path = os.path.join(
                output_root,
                f"{scene_id}_{episode_id}_{branch_name}.png",
            )
            cv2.imwrite(image_path, canvas)
            result["visualization_path"] = image_path
        return result

    def _plan_s2_loop_path_reobserve_active(
        self,
        event: Optional[dict],
        active_count: int,
        *,
        observations: Optional[dict] = None,
        depth_m: Optional[np.ndarray] = None,
    ) -> dict:
        """Plan one strict-only path bridge or bounded active reorientation."""
        cfg = self._get_s2_action_loop_cfg()
        candidate = dict((event or {}).get("candidate") or {})
        result = {
            "event_type": "s2_loop_path_reobserve_active",
            "event_schema_version": "stage21c_path_reobserve_active_v1",
            "candidate_source": (event or {}).get(
                "candidate_source", "legacy_semantic"
            ),
            "one_primitive_per_reaudit": bool(
                cfg.get("path_reobserve_one_primitive_per_reaudit")
            ),
            "scene_id": (event or {}).get("scene_id"),
            "episode_id": (event or {}).get("episode_id"),
            "episode_index": (event or {}).get("episode_index"),
            "episode_count": (event or {}).get("episode_count"),
            "episode_eval_seed": (event or {}).get("episode_eval_seed"),
            "trigger_step": (event or {}).get("step_id"),
            "step_id": (event or {}).get("step_id"),
            "failure_type": (event or {}).get("failure_type"),
            "triage_tier": (event or {}).get("triage_tier"),
            "triage_reason": (event or {}).get("triage_reason"),
            "turn_direction": (event or {}).get("turn_direction"),
            "candidate": candidate,
            "enabled": bool(cfg.get("path_reobserve_active_enable")),
            "considered": bool(event),
            "intervention_index": int(active_count + 1),
            "intervention_budget": int(
                cfg.get("path_reobserve_max_interventions_per_episode", 1)
            ),
            "execution_mode": None,
            "execution_pending": False,
            "reobserve_pending": False,
            "reorient_actions": [],
            "reorient_primitive_count": 0,
            "max_reorient_primitives": int(
                cfg.get("path_reobserve_max_turn_steps", 4)
            ),
            "reorient_bearing_history_deg": [],
            "iterative_reorient_enable": bool(
                cfg.get("path_reobserve_iterative_reorient_enable")
            ),
            "selected_pixel_goal": None,
            "action_applied": False,
            "output_rewritten": False,
            "reason": None,
            "path_bridge": None,
            "geometry_preflight": {
                "geometry_safe": bool(candidate.get("geometry_safe")),
                "active_gate_safe": bool(candidate.get("active_gate_safe")),
                "direction_bucket": candidate.get("direction_bucket"),
            },
            "gt_fields_used": [],
        }
        if not result["enabled"]:
            result["reason"] = "disabled"
            return result
        if not event:
            result["reason"] = "missing_loop_event"
            return result
        if str(event.get("triage_tier") or "") != "strict_intervention":
            result["reason"] = "non_strict_hold"
            return result
        if active_count >= int(
            cfg.get("path_reobserve_max_interventions_per_episode", 1)
        ):
            result["reason"] = "budget_exhausted"
            return result
        if not bool(candidate.get("geometry_safe")):
            result["reason"] = "candidate_not_geometry_safe"
            return result
        if not bool(candidate.get("active_gate_safe")):
            result["reason"] = "candidate_not_active_gate_safe"
            return result
        if str(candidate.get("direction_bucket") or "").lower() not in set(
            cfg.get("strict_active_allowed_directions") or ()
        ):
            result["reason"] = "candidate_direction_not_allowed"
            return result

        bridge = self._recovery_path_bridge(
            event,
            observations=observations,
            depth_m=depth_m,
            probe_source="s2_loop_path_reobserve_initial",
        )
        result["path_bridge"] = bridge
        max_active_path_m = float(
            cfg.get("path_reobserve_max_active_path_m", 0.0) or 0.0
        )
        path_m = bridge.get("path_m")
        if not active_path_within_bound(path_m, max_active_path_m):
            result.update(
                {
                    "reason": "candidate_path_beyond_active_bound",
                    "max_active_path_m": float(max_active_path_m),
                    "candidate_path_m": float(path_m),
                }
            )
            return result
        if bridge.get("valid"):
            result.update(
                {
                    "execution_mode": "path_pixel",
                    "execution_pending": True,
                    "selected_pixel_goal": bridge.get("selected_pixel_goal"),
                    "reason": "path_pixel_preflight_pass",
                }
            )
            return result
        if not bridge.get("path_reachable"):
            result["reason"] = str(bridge.get("reason") or "path_not_reachable")
            return result

        try:
            path_angle = float(bridge.get("initial_direction_angle_deg"))
        except (TypeError, ValueError):
            result["reason"] = "missing_path_reorient_angle"
            return result
        while path_angle > 180.0:
            path_angle -= 360.0
        while path_angle <= -180.0:
            path_angle += 360.0
        result["reorient_bearing_history_deg"] = [float(path_angle)]
        turn_angle = max(
            1e-6,
            abs(float(getattr(self.config.habitat.simulator, "turn_angle", 15.0))),
        )
        deadband = float(cfg.get("path_reobserve_turn_deadband_deg", 7.5))
        max_turn_steps = int(cfg.get("path_reobserve_max_turn_steps", 4))
        if abs(path_angle) <= deadband:
            if not cfg.get("path_reobserve_scan_when_aligned"):
                result["reason"] = "path_aligned_but_not_visible"
                return result
            # A one-step scan opposite the repeated policy turn exposes new
            # pixels while avoiding another action in the known loop direction.
            turn_action = (
                action_code.RIGHT
                if str(event.get("turn_direction") or "") == "left"
                else action_code.LEFT
            )
            turn_steps = 1
            scan_reason = "opposite_loop_scan"
        else:
            turn_action = action_code.LEFT if path_angle > 0.0 else action_code.RIGHT
            turn_steps = min(
                max_turn_steps,
                max(1, int(math.ceil(abs(path_angle) / turn_angle))),
            )
            scan_reason = "turn_to_path_lookahead"
        if turn_steps <= 0:
            result["reason"] = "empty_reorient_plan"
            return result
        actions = [int(turn_action)] * int(turn_steps)
        if cfg.get("path_reobserve_one_primitive_per_reaudit"):
            actions = actions[:1]
        result.update(
            {
                "execution_mode": "bounded_reorient_reobserve",
                "reobserve_pending": True,
                "reorient_actions": actions,
                "reorient_angle_deg": float(path_angle),
                "habitat_turn_angle_deg": float(turn_angle),
                "scan_reason": scan_reason,
                "reason": "reorient_queued",
            }
        )
        return result

    def _plan_s2_loop_path_reobserve_post_observation(
        self,
        pending: dict,
        *,
        observations: Optional[dict],
        depth_m: Optional[np.ndarray],
        step_id: int,
    ) -> dict:
        result = dict(pending or {})
        planned_actions = [int(item) for item in pending.get("reorient_actions") or []]
        applied_actions = [
            int(item) for item in pending.get("reorient_actions_applied") or []
        ]
        reorient_complete = bool(
            planned_actions and applied_actions == planned_actions
        )
        result.update(
            {
                "event_type": "s2_loop_path_reobserve_post_observation",
                "event_schema_version": "stage21c_path_reobserve_active_v1",
                "step_id": int(step_id),
                "post_reobserve_step": int(step_id),
                "reobserve_pending": False,
                "execution_pending": False,
                "selected_pixel_goal": None,
                "output_rewritten": False,
                # Reaching this query means the bounded turn queue was fully
                # stepped by the environment.
                "action_applied": bool(reorient_complete),
                "reorient_action_applied": bool(reorient_complete),
                "reorient_actions_applied": list(applied_actions),
                "reason": None,
            }
        )
        if not reorient_complete:
            result["reason"] = "reorient_queue_incomplete_handoff_s2"
            return result
        bridge_event = dict(pending or {})
        bridge_event["step_id"] = int(step_id)
        bridge = self._recovery_path_bridge(
            bridge_event,
            observations=observations,
            depth_m=depth_m,
            probe_source="s2_loop_path_reobserve_post_observation",
        )
        result["post_path_bridge"] = bridge
        if bridge.get("valid"):
            result.update(
                {
                    "execution_mode": "post_reobserve_path_pixel",
                    "execution_pending": True,
                    "selected_pixel_goal": bridge.get("selected_pixel_goal"),
                    "reason": "post_reobserve_path_pixel_preflight_pass",
                }
            )
            return result

        primitive_count = int(result.get("reorient_primitive_count", 0) or 0) + len(
            applied_actions
        )
        result["reorient_primitive_count"] = int(primitive_count)
        bearing_history = [
            float(value)
            for value in list(result.get("reorient_bearing_history_deg") or [])
        ]
        current_bearing = bridge.get("initial_direction_angle_deg")
        if current_bearing is not None:
            bearing_history.append(float(current_bearing))
        result["reorient_bearing_history_deg"] = bearing_history
        if not cfg.get("path_reobserve_iterative_reorient_enable"):
            result["reason"] = str(
                bridge.get("reason") or "post_reobserve_no_path_pixel_handoff_s2"
            )
            return result
        if not bridge.get("path_reachable"):
            result["reason"] = str(bridge.get("reason") or "path_not_reachable")
            return result
        max_active_path_m = float(
            cfg.get("path_reobserve_max_active_path_m", 0.0) or 0.0
        )
        if not active_path_within_bound(bridge.get("path_m"), max_active_path_m):
            result["reason"] = "candidate_path_beyond_active_bound_after_reaudit"
            return result
        previous_bearing = bearing_history[-2] if len(bearing_history) >= 2 else None
        decision = iterative_reorientation_decision(
            previous_bearing,
            current_bearing,
            primitive_count=primitive_count,
            max_primitives=int(cfg.get("path_reobserve_max_turn_steps", 4)),
            deadband_deg=float(cfg.get("path_reobserve_turn_deadband_deg", 7.5)),
        )
        result["iterative_reorient_decision"] = decision
        if not decision.get("continue_reorientation"):
            result["reason"] = str(decision.get("reason"))
            return result
        turn_action = (
            action_code.LEFT
            if decision.get("turn_direction") == "left"
            else action_code.RIGHT
        )
        result.update(
            {
                "execution_mode": "bounded_iterative_reorient_reobserve",
                "reobserve_pending": True,
                "reorient_actions": [int(turn_action)],
                "reorient_angle_deg": float(current_bearing),
                "scan_reason": "continue_turn_to_path_lookahead",
                "reason": "iterative_reorient_queued",
            }
        )
        return result

    def _write_s2_action_loop_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "s2_action_loop_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_s2_loop_executed_route_occ_audit_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "s2_loop_executed_route_occ_audit_events.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_stage27_candidate_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "stage27_m3_candidate_events.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_stage41_executor_contract_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "stage41_executor_contract_events.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_stage43_counterfactual_reobserve_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "stage43_counterfactual_reobserve_events.jsonl")
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    @staticmethod
    def _stage43_memory_fingerprint(memory) -> tuple:
        free_counts = getattr(memory, "free_counts", {}) or {}
        occ_counts = getattr(memory, "occ_counts", {}) or {}
        free2d_counts = getattr(memory, "free2d_counts", {}) or {}
        occ2d_counts = getattr(memory, "occ2d_counts", {}) or {}
        return (
            int(getattr(memory, "observation_count", -1)),
            len(getattr(memory, "pose_trace", []) or []),
            len(free_counts), int(sum(free_counts.values())),
            len(occ_counts), int(sum(occ_counts.values())),
            len(free2d_counts), int(sum(free2d_counts.values())),
            len(occ2d_counts), int(sum(occ2d_counts.values())),
            tuple(getattr(memory, "last_pose_grid", ()) or ()),
        )

    @staticmethod
    def _stage43_pose_equal(first, second) -> bool:
        try:
            positions_equal = bool(np.allclose(
                np.asarray(first.position, dtype=np.float64),
                np.asarray(second.position, dtype=np.float64),
                atol=1e-7,
            ))
            first_q = quaternion.as_float_array(first.rotation)
            second_q = quaternion.as_float_array(second.rotation)
            rotations_equal = abs(float(np.dot(first_q, second_q))) >= 1.0 - 1e-7
            return positions_equal and rotations_equal
        except Exception:
            return False

    def _stage43_counterfactual_observation(
        self, observations: dict, *, relative_bearing_deg: float, plan: dict
    ) -> dict:
        """Read a rotated sensor view and restore all official simulator state."""
        result = {
            "observation": None,
            "depth_m": None,
            "observation_readable": False,
            "sim_pose_restored": False,
            "selected_yaw_delta_deg": None,
            "observed_compass_delta_deg": None,
            "post_relative_bearing_deg": None,
            "attempt_count": 0,
            "reason": None,
        }
        if not plan.get("valid"):
            result["reason"] = str(plan.get("reason") or "invalid_reorientation_plan")
            return result
        try:
            sim = self.env._env.sim
            before_state = sim.get_agent_state()
            previous_sim_obs = getattr(sim, "_prev_sim_obs", None)
            before_compass = math.degrees(float(np.asarray(observations.get("compass")).reshape(-1)[0]))
            planned_delta = float(plan.get("planned_yaw_delta_deg", 0.0) or 0.0)
            deltas = [planned_delta]
            if abs(planned_delta) > 1e-9:
                deltas.append(-planned_delta)
            attempts = []
            try:
                for delta_deg in deltas:
                    delta_q = quaternion.from_rotation_vector(
                        np.asarray([0.0, math.radians(delta_deg), 0.0], dtype=np.float64)
                    )
                    target_q = delta_q * before_state.rotation
                    rotation_xyzw = [
                        float(target_q.x), float(target_q.y),
                        float(target_q.z), float(target_q.w),
                    ]
                    counterfactual = sim.get_observations_at(
                        position=np.asarray(before_state.position, dtype=np.float64).tolist(),
                        rotation=rotation_xyzw,
                        keep_agent_at_new_pose=False,
                    )
                    result["attempt_count"] += 1
                    if not counterfactual or counterfactual.get("depth") is None:
                        continue
                    # get_observations_at covers simulator RGB-D sensors only.
                    # Read episodic GPS/compass from the task sensors at the
                    # same temporary pose, then restore in the outer finally.
                    sim.set_agent_state(
                        np.asarray(before_state.position, dtype=np.float64).tolist(),
                        rotation_xyzw,
                        reset_sensors=False,
                    )
                    task_sensors = self.env._env.task.sensor_suite.sensors
                    current_episode = self.env._env.current_episode
                    counterfactual["gps"] = task_sensors["gps"].get_observation(
                        counterfactual, current_episode
                    )
                    counterfactual["compass"] = task_sensors["compass"].get_observation(
                        counterfactual, current_episode
                    )
                    after_compass = math.degrees(
                        float(np.asarray(counterfactual.get("compass")).reshape(-1)[0])
                    )
                    compass_delta = normalize_angle_deg(after_compass - before_compass)
                    residual = normalize_angle_deg(float(relative_bearing_deg) - compass_delta)
                    attempts.append((abs(residual), delta_deg, compass_delta, residual, counterfactual))
            finally:
                before_q = before_state.rotation
                sim.set_agent_state(
                    np.asarray(before_state.position, dtype=np.float64).tolist(),
                    [float(before_q.x), float(before_q.y), float(before_q.z), float(before_q.w)],
                    reset_sensors=False,
                )
                if hasattr(sim, "_prev_sim_obs"):
                    sim._prev_sim_obs = previous_sim_obs
            after_state = sim.get_agent_state()
            result["sim_pose_restored"] = self._stage43_pose_equal(before_state, after_state)
            if not attempts:
                result["reason"] = "counterfactual_observation_unavailable"
                return result
            _, selected_delta, compass_delta, residual, selected = min(
                attempts, key=lambda item: (item[0], abs(item[1]))
            )
            raw_depth = np.asarray(selected["depth"])
            depth_m = filter_depth(raw_depth.reshape(raw_depth.shape[:2]), blur_type=None)
            depth_m = depth_m * (self._max_depth - self._min_depth) + self._min_depth
            result.update({
                "observation": selected,
                "depth_m": depth_m,
                "observation_readable": bool(depth_m.size and np.isfinite(depth_m).any()),
                "selected_yaw_delta_deg": float(selected_delta),
                "observed_compass_delta_deg": float(compass_delta),
                "post_relative_bearing_deg": float(residual),
                "reason": "ok" if result["sim_pose_restored"] else "sim_pose_restore_failed",
            })
        except Exception as exc:
            result["reason"] = f"counterfactual_probe_error:{type(exc).__name__}:{exc}"
        return result

    def _stage43_contract_for_candidate(
        self, candidate: dict, *, observations: dict, depth_m: np.ndarray
    ) -> dict:
        bridge = self._recovery_path_bridge(
            {"candidate": candidate},
            observations=observations,
            depth_m=depth_m,
            probe_source="stage43_counterfactual_reobserve",
        )
        edge_audits = []
        for index, cell in enumerate(list(bridge.get("path") or [])[1:], start=1):
            state = self.occ_memory._cell_state(int(cell[0]), int(cell[1]))
            edge_audits.append({
                "edge_index": int(index),
                "grid": list(cell),
                "state": str(state),
                "sparseocc_safe": state == "free",
                "unknown": state == "unknown",
                "occupied": state == "occupied",
                "depth_occlusion_checked": bool(index == 1 and bridge.get("base_projection_bridge") is not None),
                "depth_readable": True,
                "depth_clear": bool(index == 1 and bridge.get("valid")),
            })
        contract = validate_executor_contract(
            sensor={
                "hfov_deg": float(self.sim_sensors_config.depth_sensor.hfov),
                "hfov_source": "habitat_counterfactual_depth_sensor_config",
                "depth_readable": True,
            },
            edge_audits=edge_audits,
            candidate_safety=candidate,
        )
        return {
            "candidate_id": candidate.get("candidate_id"),
            "bridge": bridge,
            "edge_audits": edge_audits,
            "contract": contract,
        }

    def _stage43_zero_history_bearing(self) -> dict:
        trace = list(getattr(self.occ_memory, "pose_trace", []) or [])
        if len(trace) < 2:
            return {"valid": False, "reason": "insufficient_pose_trace"}
        current = trace[-1]
        min_distance_m = float(
            self._stage27_candidate_audit_cfg.get("stage43_zero_history_distance_m", 0.50)
        )
        target = next((
            item for item in reversed(trace[:-1])
            if math.hypot(
                float(item.get("x", 0.0)) - float(current.get("x", 0.0)),
                float(item.get("y", 0.0)) - float(current.get("y", 0.0)),
            ) >= min_distance_m
        ), None)
        if target is None:
            return {"valid": False, "reason": "no_translated_history_target"}
        direction = self.occ_memory._direction_to_cell(
            (int(current["row"]), int(current["col"])),
            (int(target["row"]), int(target["col"])),
            float(current.get("yaw", 0.0) or 0.0),
        )
        return {
            "valid": direction.get("angle_deg") is not None,
            "reason": "history_bearing_only",
            "relative_bearing_deg": direction.get("angle_deg"),
            "target_step": target.get("step_id"),
            "target_grid": [int(target["row"]), int(target["col"])],
            "history_grants_safety": False,
        }

    def _stage43_counterfactual_reobserve_shadow(
        self, event: dict, *, observations: Optional[dict], depth_m: Optional[np.ndarray]
    ) -> None:
        cfg = self._stage27_candidate_audit_cfg
        if not bool(cfg.get("stage43_counterfactual_reobserve_enable", False)):
            return
        pre_pool = list(event.get("ablation", {}).get(
            "route_occ_clearance_frontier", {}
        ).get("candidates") or [])
        record = {
            "schema_version": STAGE43_SCHEMA_VERSION,
            "event_type": "stage43_counterfactual_reobserve_shadow",
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "step_id": event.get("step_id"),
            "pre_candidate_count": len(pre_pool),
            "post_candidate_count_max": 0,
            "probes": [],
            "shadow_only": True,
            "action_applied": False,
            "unknown_is_free": False,
            "gt_fields_used": [],
        }
        if observations is None or depth_m is None:
            record["reason"] = "missing_current_observation"
            self._write_stage43_counterfactual_reobserve_event(record)
            return

        targets = []
        for candidate in pre_pool[: max(1, int(cfg.get("stage43_max_candidate_probes", 3)))]:
            bridge = self._recovery_path_bridge(
                {**event, "candidate": candidate},
                observations=observations,
                depth_m=depth_m,
                probe_source="stage43_pre_counterfactual",
            )
            angle = bridge.get("initial_direction_angle_deg")
            if bridge.get("path_reachable") and angle is not None:
                targets.append({
                    "source": "safe_candidate_first_edge",
                    "candidate": candidate,
                    "relative_bearing_deg": float(angle),
                    "pre_bridge": bridge,
                })
        if not targets and not pre_pool:
            history = self._stage43_zero_history_bearing()
            if history.get("valid"):
                targets.append({
                    "source": "history_bearing_only",
                    "candidate": None,
                    "relative_bearing_deg": float(history["relative_bearing_deg"]),
                    "history_target": history,
                })
            else:
                record["reason"] = history.get("reason")

        for target in targets:
            plan = plan_bounded_reorientation(
                target["relative_bearing_deg"],
                hfov_deg=float(self.sim_sensors_config.depth_sensor.hfov),
                turn_angle_deg=float(getattr(self.config.habitat.simulator, "turn_angle", 15.0)),
                center_margin_deg=float(cfg.get("stage43_center_margin_deg", 10.0)),
                max_turn_steps=int(cfg.get("stage43_max_turn_steps", 12)),
            )
            official_before = self._stage43_memory_fingerprint(self.occ_memory)
            counterfactual = self._stage43_counterfactual_observation(
                observations,
                relative_bearing_deg=float(target["relative_bearing_deg"]),
                plan=plan,
            )
            probe = {
                "source": target["source"],
                "candidate_id": (target.get("candidate") or {}).get("candidate_id"),
                "history_target": target.get("history_target"),
                "plan": plan,
                "observation_readable": bool(counterfactual.get("observation_readable")),
                "sim_pose_restored": bool(counterfactual.get("sim_pose_restored")),
                "selected_yaw_delta_deg": counterfactual.get("selected_yaw_delta_deg"),
                "observed_compass_delta_deg": counterfactual.get("observed_compass_delta_deg"),
                "post_relative_bearing_deg": counterfactual.get("post_relative_bearing_deg"),
                "attempt_count": counterfactual.get("attempt_count"),
                "reason": counterfactual.get("reason"),
                "post_candidate_count": 0,
                "post_contracts": [],
                "official_memory_mutated": False,
                "safety_authority": "temporary_current_sparseocc_reaudit",
                "action_emitted": False,
                "action_applied": False,
                "shadow_only": True,
                "unknown_is_free": False,
                "gt_fields_used": [],
            }
            try:
                if counterfactual.get("observation_readable") and counterfactual.get("sim_pose_restored"):
                    candidate = target.get("candidate")
                    if candidate is not None:
                        probe["post_candidate_count"] = 1
                        probe["post_contracts"] = [self._stage43_contract_for_candidate(
                            candidate,
                            observations=counterfactual["observation"],
                            depth_m=counterfactual["depth_m"],
                        )]
                    else:
                        temporary_memory = copy.deepcopy(self.occ_memory)
                        cf_observation = counterfactual["observation"]
                        temporary_memory.update_observation(
                            {
                                "rgb": cf_observation.get("rgb"),
                                "depth": counterfactual["depth_m"],
                                "gps": cf_observation.get("gps"),
                                "compass": cf_observation.get("compass"),
                            },
                            counterfactual["depth_m"],
                            rgb=cf_observation.get("rgb"),
                            context={
                                "step_id": int(event.get("step_id", -1)),
                                "scene_id": event.get("scene_id"),
                                "episode_id": event.get("episode_id"),
                                "camera_pitch_deg": 0.0,
                            },
                        )
                        post_event = generate_from_sparse_memory(
                            temporary_memory,
                            trigger_grid=temporary_memory.last_pose_grid,
                            config=cfg,
                            semantic_raw_nodes=[],
                            semantic_filtered_nodes=[],
                            instruction="",
                        )
                        post_pool = list(post_event.get("ablation", {}).get(
                            "route_occ_clearance_frontier", {}
                        ).get("candidates") or [])
                        probe["post_candidate_count"] = len(post_pool)
                        official_memory = self.occ_memory
                        try:
                            self.occ_memory = temporary_memory
                            probe["post_contracts"] = [
                                self._stage43_contract_for_candidate(
                                    candidate,
                                    observations=cf_observation,
                                    depth_m=counterfactual["depth_m"],
                                )
                                for candidate in post_pool[: max(1, int(cfg.get("stage43_max_candidate_probes", 3)))]
                            ]
                        finally:
                            self.occ_memory = official_memory
            except Exception as exc:
                probe["reason"] = f"temporary_reaudit_error:{type(exc).__name__}:{exc}"
            official_after = self._stage43_memory_fingerprint(self.occ_memory)
            probe["official_memory_mutated"] = official_after != official_before
            record["post_candidate_count_max"] = max(
                int(record["post_candidate_count_max"]), int(probe["post_candidate_count"])
            )
            record["probes"].append(probe)
        self._write_stage43_counterfactual_reobserve_event(record)

    def _stage41_executor_contract_shadow(
        self, event: dict, *, observations: Optional[dict], depth_m: Optional[np.ndarray]
    ) -> None:
        if not bool(self._stage27_candidate_audit_cfg.get("stage41_executor_contract_enable", False)):
            return
        pool = list(
            event.get("ablation", {}).get(
                "route_occ_clearance_frontier", {}
            ).get("candidates") or []
        )
        depth_readable = bool(
            isinstance(depth_m, np.ndarray)
            and depth_m.size > 0
            and np.isfinite(depth_m).any()
        )
        sensor = {
            "hfov_deg": float(self.sim_sensors_config.depth_sensor.hfov),
            "hfov_source": "habitat_depth_sensor_config",
            "depth_readable": depth_readable,
        }
        contracts = []
        for candidate in pool:
            bridge = self._recovery_path_bridge(
                {**event, "candidate": candidate},
                observations=observations,
                depth_m=depth_m,
                probe_source="stage41_executor_contract_shadow",
            )
            path = list(bridge.get("path") or [])
            edge_audits = []
            for index, cell in enumerate(path[1:], start=1):
                state = self.occ_memory._cell_state(int(cell[0]), int(cell[1]))
                edge_audits.append({
                    "edge_index": int(index),
                    "grid": list(cell),
                    "state": str(state),
                    "sparseocc_safe": state == "free",
                    "unknown": state == "unknown",
                    "occupied": state == "occupied",
                    "depth_occlusion_checked": bool(index == 1 and bridge.get("base_projection_bridge") is not None),
                    "depth_readable": bool(depth_readable),
                    "depth_clear": bool(index == 1 and bridge.get("valid")),
                })
            report = validate_executor_contract(
                sensor=sensor,
                edge_audits=edge_audits,
                candidate_safety=candidate,
            )
            contracts.append({
                "candidate_id": candidate.get("candidate_id"),
                "bridge": bridge,
                "edge_audits": edge_audits,
                "contract": report,
            })
        self._write_stage41_executor_contract_event({
            "event_type": "stage41_executor_contract_shadow",
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "step_id": event.get("step_id"),
            "sensor": sensor,
            "candidate_count": len(pool),
            "contracts": contracts,
            "shadow_only": True,
            "action_applied": False,
            "unknown_is_free": False,
            "gt_fields_used": [],
        })

    def _maybe_write_stage27_candidate_audit(
        self, *, scene_id: str, episode_id: int, episode_index: int,
        episode_count: int, episode_eval_seed: Optional[int], step_id: int,
        observations: Optional[dict] = None,
        depth_m: Optional[np.ndarray] = None,
        allow_unscheduled: bool = False,
    ) -> Optional[dict]:
        if not self._stage27_candidate_audit_enabled:
            return None
        key = (str(scene_id), int(episode_id), int(step_id))
        entry = self._stage27_candidate_audit_entries.get(key)
        if entry is None and not allow_unscheduled:
            return None
        entry = entry or {"selection": "all_detector_shadow_starts"}
        trigger_grid = getattr(self.occ_memory, "last_pose_grid", None)
        if trigger_grid is None:
            return None
        semantic_candidate_enabled = bool(
            self._stage27_candidate_audit_cfg.get("semantic_candidate_enable", False)
        )
        semantic_raw_nodes = []
        semantic_filtered_nodes = []
        instruction = ""
        if semantic_candidate_enabled:
            semantic_raw_nodes = self.online_lseg_shadow.snapshot_nodes(filtered=False)
            semantic_filtered_nodes = self.online_lseg_shadow.snapshot_nodes(filtered=True)
            instruction = str(
                (self.online_lseg_shadow.episode_meta or {}).get("instruction") or ""
            )
        event = generate_from_sparse_memory(
            self.occ_memory,
            trigger_grid=trigger_grid,
            config=self._stage27_candidate_audit_cfg,
            semantic_raw_nodes=semantic_raw_nodes,
            semantic_filtered_nodes=semantic_filtered_nodes,
            instruction=instruction,
        )
        record = {
            "event_type": "stage27_m3_candidate_generation",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "episode_eval_seed": episode_eval_seed,
            "step_id": int(step_id),
            "audit_selection": entry,
            "audit_selection_fields_used": ["pre_registered_event_step"],
            "candidate_feature_gt_fields_used": [],
            **event,
        }
        if self._stage45_candidate_rejection_truth_enable:
            record["offline_rejection_truth_audit"] = (
                self._stage45_candidate_rejection_truth_audit(
                    record, scene_id=scene_id, episode_id=int(episode_id)
                )
            )
        if bool(self._stage27_candidate_audit_cfg.get("recovery_bev_snapshot_enable", False)):
            semantic_cells = []
            final_pool = record.get("ablation", {}).get(
                "route_occ_clearance_frontier_semantic_filtered", {}
            ).get("candidates", []) or []
            for candidate in final_pool:
                grid = (candidate.get("semantic_evidence") or {}).get("grid") or []
                if len(grid) >= 2:
                    semantic_cells.append(grid)
            record["recovery_bev_spatial"] = build_recovery_bev_spatial_snapshot(
                center_grid=trigger_grid,
                free_cells=getattr(self.occ_memory, "free2d_counts", {}).keys(),
                occupied_cells=getattr(self.occ_memory, "occ2d_counts", {}).keys(),
                pose_trace=getattr(self.occ_memory, "pose_trace", []),
                semantic_cells=semantic_cells,
                radius_cells=int(self._stage27_candidate_audit_cfg.get("recovery_bev_radius_cells", 24)),
            )
        self._write_stage27_candidate_event(record)
        self._stage41_executor_contract_shadow(
            record, observations=observations, depth_m=depth_m
        )
        self._stage43_counterfactual_reobserve_shadow(
            record, observations=observations, depth_m=depth_m
        )
        self._stage27_candidate_audit_records[key] = record
        return record

    def _stage45_candidate_rejection_truth_audit(
        self, record: dict, *, scene_id: str, episode_id: int
    ) -> dict:
        """Run a read-only navmesh audit for frozen route-only candidates."""
        result = {
            "event_schema_version": "stage45_candidate_rejection_truth_event_v1",
            "enabled": True,
            "scene_id": str(scene_id),
            "episode_id": int(episode_id),
            "shadow_only": True,
            "action_applied": False,
            "gt_used_for_navigation": False,
            "unknown_is_free": False,
            "candidate_pool": "route_only",
            "audits": [],
            "summary": summarize_event_audits([]),
        }
        route_pool = (
            record.get("ablation", {})
            .get("route_only", {})
            .get("candidates", [])
            or []
        )
        if not route_pool:
            result["reason"] = "no_route_only_candidates"
            return result
        pathfinder = getattr(self.env._env.sim, "pathfinder", None)
        memory = self.occ_memory
        if pathfinder is None or not bool(getattr(pathfinder, "is_loaded", True)):
            result["reason"] = "pathfinder_unavailable"
            return result
        reference_memory = self.occ_memory_oracle_sensor_pose or memory
        pose_trace = list(getattr(reference_memory, "pose_trace", []) or [])
        trace_xy = []
        trace_z = []
        for node in pose_trace:
            try:
                trace_xy.append((float(node["x"]), float(node["y"])))
                trace_z.append(float(node.get("z", 0.0) or 0.0))
            except (KeyError, TypeError, ValueError):
                continue
        if self._stage23a_initial_agent_matrix is None:
            result["reason"] = "missing_initial_agent_matrix"
            return result
        habitat_to_map = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        map_to_habitat_tf = np.eye(4, dtype=np.float32)
        map_to_habitat_tf[:3, :3] = habitat_to_map.T
        map_to_world_tf = (
            np.asarray(self._stage23a_initial_agent_matrix, dtype=np.float32)
            @ map_to_habitat_tf
        )
        cell_size = float(getattr(memory, "cs", 0.05))
        footprint_radius = float(
            self._stage45_candidate_rejection_truth_cfg.get(
                "footprint_radius_m", 0.18
            )
        )
        radius_cells = max(0, int(math.ceil(footprint_radius / cell_size)))
        max_edge_geodesic_ratio = float(
            self._stage45_candidate_rejection_truth_cfg.get(
                "max_edge_geodesic_ratio", 2.0
            )
        )
        world_cache = {}
        footprint_cache = {}

        def floor_z_for_cell(cell):
            if not trace_xy:
                return 0.0
            x, y = memory._grid_to_xy(cell)
            index = int(
                np.argmin(
                    np.sum(
                        (np.asarray(trace_xy, dtype=np.float32)
                         - np.asarray([[x, y]], dtype=np.float32)) ** 2,
                        axis=1,
                    )
                )
            )
            return float(trace_z[index])

        def world_for_cell(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell not in world_cache:
                xy = memory._grid_to_xy(cell)
                point = np.array(
                    [float(xy[0]), float(xy[1]), floor_z_for_cell(cell), 1.0],
                    dtype=np.float32,
                )
                world_cache[cell] = (map_to_world_tf @ point)[:3]
            return world_cache[cell]

        def navmesh_cell(cell):
            cell = (int(cell[0]), int(cell[1]))
            if cell not in footprint_cache:
                point = world_for_cell(cell)
                try:
                    navigable = bool(pathfinder.is_navigable(point, 0.5))
                except TypeError:
                    navigable = bool(pathfinder.is_navigable(point))
                try:
                    snapped = np.asarray(
                        pathfinder.snap_point(point), dtype=np.float32
                    ).reshape(3)
                    snap_distance = float(np.linalg.norm(snapped - point))
                except Exception:
                    snapped, snap_distance = None, None
                clearance = None
                if snapped is not None and hasattr(
                    pathfinder, "distance_to_closest_obstacle"
                ):
                    try:
                        clearance = float(
                            pathfinder.distance_to_closest_obstacle(
                                snapped, max(2.0, footprint_radius * 2.0)
                            )
                        )
                    except TypeError:
                        try:
                            clearance = float(
                                pathfinder.distance_to_closest_obstacle(snapped)
                            )
                        except Exception:
                            clearance = None
                    except Exception:
                        clearance = None
                footprint_cache[cell] = {
                    "navigable": navigable,
                    "footprint_safe": (
                        bool(clearance + 1e-6 >= footprint_radius)
                        if clearance is not None
                        else None
                    ),
                    "clearance_m": clearance,
                    "snap_distance_m": snap_distance,
                    "footprint_check_method": (
                        "navmesh_distance_to_closest_obstacle"
                    ),
                }
            return footprint_cache[cell]

        def navmesh_edge(first, second):
            start = world_for_cell(first)
            end = world_for_cell(second)
            direct = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
            shortest = habitat_sim.ShortestPath()
            shortest.requested_start = start
            shortest.requested_end = end
            try:
                connected = bool(pathfinder.find_path(shortest))
                geodesic = (
                    float(shortest.geodesic_distance) if connected else None
                )
            except Exception:
                connected, geodesic = False, None
            ratio = (
                float(geodesic / direct)
                if geodesic is not None and direct > 1e-6
                else None
            )
            return {
                "connected": bool(
                    connected
                    and ratio is not None
                    and ratio <= max_edge_geodesic_ratio
                ),
                "direct_m": direct,
                "geodesic_m": geodesic,
                "geodesic_ratio": ratio,
            }

        def sparse_floor_footprint_state(row, col, floor):
            all_free = True
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if math.hypot(dr, dc) > radius_cells:
                        continue
                    state = memory.validation_floor_aligned_cell_evidence(
                        int(row) + dr,
                        int(col) + dc,
                        float(floor),
                        height_max_m=1.5,
                    )["state"]
                    if state == "blocked":
                        return "occupied"
                    all_free = all_free and state == "free"
            return "free" if all_free else "unknown"

        for candidate in route_pool:
            result["audits"].append(
                audit_candidate_rejection_truth(
                    candidate,
                    sparse_2d_state=memory._cell_state,
                    sparse_floor_footprint_state=sparse_floor_footprint_state,
                    navmesh_cell=navmesh_cell,
                    navmesh_edge=navmesh_edge,
                    footprint_radius_m=footprint_radius,
                )
            )
        result["summary"] = summarize_event_audits(result["audits"])
        result["reason"] = "ok"
        return result

    def _write_s2_loop_fixed_route_occ_audit_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "s2_loop_fixed_route_occ_audit_events.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_s2_recovery_context_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "s2_recovery_context_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_s2_loop_strict_active_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "s2_loop_strict_active_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_s2_loop_projection_bridge_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "s2_loop_projection_bridge_events.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_s2_loop_path_reobserve_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "s2_loop_path_reobserve_active_events.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _save_s2_loop_path_reobserve_snapshot(
        self, event: dict, observations: Optional[dict]
    ) -> Optional[str]:
        if not observations or observations.get("rgb") is None:
            return None
        log_dir = self._get_vlmap_run_dir() or self.output_path
        if not log_dir:
            return None
        snapshot_dir = os.path.join(log_dir, "s2_loop_path_reobserve_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        name = (
            f"{event.get('scene_id')}_{int(event.get('episode_id'))}_"
            f"trigger{int(event.get('trigger_step', -1))}_"
            f"post{int(event.get('post_reobserve_step', event.get('step_id', -1)))}.jpg"
        )
        path = os.path.join(snapshot_dir, name)
        image = Image.fromarray(
            np.asarray(observations.get("rgb"), dtype=np.uint8)
        ).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            (
                f"path reobserve trigger={event.get('trigger_step')} "
                f"post={event.get('post_reobserve_step')} "
                f"reason={event.get('reason')}"
            ),
            fill=(255, 255, 0),
        )
        pixel_goal = list(event.get("selected_pixel_goal") or [])
        if len(pixel_goal) == 2:
            x, y = int(pixel_goal[0]), int(pixel_goal[1])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 0, 0))
        image.save(path, quality=92)
        return os.path.relpath(path, log_dir)

    def _observe_s2_action_loop_shadow(
        self,
        *,
        state: dict,
        output: str,
        observations: dict,
        depth_m: Optional[np.ndarray],
        step_id: int,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        episode_eval_seed: Optional[int],
    ) -> Optional[dict]:
        cfg = self._get_s2_action_loop_cfg()
        transition = observe_s2_action_query(
            state,
            output=output,
            step_id=step_id,
            gps=observations.get("gps"),
            compass=observations.get("compass"),
            config=cfg,
        )
        if not transition or transition.get("transition") != "start":
            return transition

        candidate_event = self.occ_memory.generate_query_candidates(
            obs={
                "gps": observations.get("gps"),
                "compass": observations.get("compass"),
            },
            current_waypoint_decision={},
            context={
                "step_id": int(step_id),
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "episode_index": int(episode_index),
                "episode_count": int(episode_count),
                "episode_eval_seed": episode_eval_seed,
                "s2_action_loop_detected": True,
                "s2_action_loop_direction": transition.get("turn_direction"),
                "s2_action_loop_generation_streak": transition.get(
                    "same_turn_generation_streak"
                ),
            },
        )
        # An empty audit manifest is a smoke-only mode: record only actual D0
        # loop starts, never every environment step.
        stage27_key = (str(scene_id), int(episode_id), int(step_id))
        stage27_record = self._stage27_candidate_audit_records.get(stage27_key)
        if not self._stage27_candidate_audit_entries:
            stage27_record = self._maybe_write_stage27_candidate_audit(
                scene_id=scene_id,
                episode_id=int(episode_id),
                episode_index=int(episode_index),
                episode_count=int(episode_count),
                episode_eval_seed=episode_eval_seed,
                step_id=int(step_id),
                observations=observations,
                depth_m=depth_m,
                allow_unscheduled=True,
            )
        candidate = self._best_semantic_resilience_backtrack_candidate(candidate_event)
        current_free_ratio = float(
            (candidate or {}).get("current_visible_free_ratio", 1.0) or 0.0
        )
        current_exit_count = int(
            (candidate or {}).get("current_executable_exit_count", 0) or 0
        )
        obstructed = bool(
            candidate is not None
            and (
                current_exit_count <= 1
                or current_free_ratio < 0.45
            )
        )
        failure_type = (
            "s2_turn_loop_obstructed" if obstructed else "s2_turn_loop_semantic"
        )
        recommended_primitive = "reorient_reobserve" if obstructed else "reobserve"
        trigger_reasons = ["s2_repeated_turn_generation", "s2_low_translation"]
        context_tags = ["s2_policy_loop", "decision_state_restoration"]
        if obstructed:
            trigger_reasons.append("local_trap")
            context_tags.append("spatial_constriction")
        triage = classify_semantic_recovery_triage(
            candidate,
            self._get_semantic_resilience_active_lite_cfg(),
            failure_type=failure_type,
            recommended_primitive=recommended_primitive,
            trigger_reasons=trigger_reasons,
            context_tags=context_tags,
            step_id=int(step_id),
        )
        event = {
            "event_type": "s2_action_loop_shadow",
            "event_schema_version": "stage21a_s2_loop_v1",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "episode_eval_seed": episode_eval_seed,
            "step_id": int(step_id),
            "enabled": bool(cfg.get("enable")),
            "shadow_only": bool(cfg.get("shadow_only", True)),
            "applied": False,
            **transition,
            "failure_type": failure_type,
            "recommended_primitive": recommended_primitive,
            "trigger_reasons": trigger_reasons,
            "recovery_context_tags": context_tags,
            "candidate_event_reason": candidate_event.get("reason"),
            "candidate": candidate,
            "triage": triage,
            "triage_tier": triage.get("tier"),
            "triage_reason": triage.get("reason"),
            "gt_fields_used": [],
        }
        if cfg.get("path_reobserve_candidate_source") == "stage27_frozen_m3":
            if stage27_record is None:
                event.update(
                    {
                        "candidate": None,
                        "candidate_source": "stage27_frozen_m3",
                        "triage_tier": "hold",
                        "triage_reason": "missing_same_step_stage27_event",
                    }
                )
            else:
                event = bind_candidate_to_loop_event(event, stage27_record)
        if (
            bool(cfg.get("executed_route_occ_audit_enable"))
            and str(event.get("triage_tier") or "") == "strict_intervention"
        ):
            route_audit = self.occ_memory.audit_executed_route_to_candidate(
                candidate or {},
                current_step=int(step_id),
                max_edge_m=float(cfg["executed_route_occ_audit_max_edge_m"]),
                sample_spacing_m=float(
                    cfg["executed_route_occ_audit_sample_spacing_m"]
                ),
                max_path_cells=int(
                    cfg["executed_route_occ_audit_max_path_cells"]
                ),
                max_visited_cells=int(
                    cfg["executed_route_occ_audit_max_visited_cells"]
                ),
            )
            route_audit_event = {
                "event_type": "s2_loop_executed_route_occ_audit",
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "episode_index": int(episode_index),
                "episode_count": int(episode_count),
                "episode_eval_seed": episode_eval_seed,
                "step_id": int(step_id),
                "loop_index": transition.get("loop_index"),
                "failure_type": failure_type,
                "triage_tier": triage.get("tier"),
                "triage_reason": triage.get("reason"),
                "candidate_id": (candidate or {}).get("candidate_id"),
                "candidate_source": (candidate or {}).get(
                    "semantic_resilience_source"
                ),
                "audit": route_audit,
                "shadow_only": True,
                "action_applied": False,
                "output_rewritten": False,
                "gt_fields_used": [],
            }
            event["executed_route_occ_audit_valid"] = bool(
                route_audit.get("valid")
            )
            event["executed_route_occ_audit_reason"] = route_audit.get("reason")
            self._write_s2_loop_executed_route_occ_audit_event(route_audit_event)

        fixed_route_reference = next(
            (
                entry
                for entry in cfg.get("fixed_route_occ_audit_entries", ())
                if str(entry.get("scene_id")) == str(scene_id)
                and int(entry.get("episode_id", -1)) == int(episode_id)
                and int(entry.get("step_id", -1)) == int(step_id)
            ),
            None,
        )
        if bool(cfg.get("fixed_route_occ_audit_enable")) and fixed_route_reference:
            fixed_candidate = {
                "candidate_id": fixed_route_reference.get("candidate_id"),
                "grid": list(fixed_route_reference.get("anchor_grid") or []),
                "semantic_resilience_source_step_id": fixed_route_reference.get(
                    "source_step"
                ),
                "semantic_resilience_source": "stage22a_fixed_reference",
            }
            fixed_route_audit = self.occ_memory.audit_executed_route_to_candidate(
                fixed_candidate,
                current_step=int(step_id),
                max_edge_m=float(cfg["executed_route_occ_audit_max_edge_m"]),
                sample_spacing_m=float(
                    cfg["executed_route_occ_audit_sample_spacing_m"]
                ),
                max_path_cells=int(
                    cfg["executed_route_occ_audit_max_path_cells"]
                ),
                max_visited_cells=int(
                    cfg["executed_route_occ_audit_max_visited_cells"]
                ),
            )
            if bool(cfg.get("fixed_route_occ_evidence_audit_enable")):
                fixed_route_audit["route_cell_evidence"] = (
                    self.occ_memory.audit_route_cell_evidence(
                        fixed_route_audit,
                        include_height_aligned=bool(
                            cfg.get("fixed_route_height_evidence_audit_enable")
                        ),
                    )
                )
            selected_candidate = candidate or {}
            fixed_route_event = {
                "event_type": "s2_loop_fixed_route_occ_audit",
                "event_schema_version": "stage22c_fixed_route_occ_audit_v1",
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "episode_index": int(episode_index),
                "episode_count": int(episode_count),
                "episode_eval_seed": episode_eval_seed,
                "step_id": int(step_id),
                "loop_index": transition.get("loop_index"),
                "fixed_reference": dict(fixed_route_reference),
                "current_selected_candidate": {
                    "candidate_id": selected_candidate.get("candidate_id"),
                    "grid": selected_candidate.get("grid"),
                    "source_step": selected_candidate.get(
                        "semantic_resilience_source_step_id"
                    ),
                    "source": selected_candidate.get(
                        "semantic_resilience_source"
                    ),
                },
                "current_triage_tier": triage.get("tier"),
                "current_triage_reason": triage.get("reason"),
                "audit": fixed_route_audit,
                "shadow_only": True,
                "action_applied": False,
                "output_rewritten": False,
                "gt_fields_used": [],
            }
            event["fixed_route_occ_audit_valid"] = bool(
                fixed_route_audit.get("valid")
            )
            event["fixed_route_occ_audit_reason"] = fixed_route_audit.get(
                "reason"
            )
            self._write_s2_loop_fixed_route_occ_audit_event(fixed_route_event)
        max_snapshots = max(0, int(cfg.get("max_snapshots_per_episode", 2)))
        snapshot_expected = bool(
            int(transition.get("loop_index", 0) or 0) <= max_snapshots
        )
        event["rgb_snapshot_expected"] = snapshot_expected
        if snapshot_expected:
            log_dir = self._get_vlmap_run_dir() or self.output_path
            snapshot_dir = os.path.join(log_dir, "s2_action_loop_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_name = (
                f"{scene_id}_{int(episode_id)}_step{int(step_id)}_"
                f"loop{int(transition.get('loop_index', 0) or 0)}.jpg"
            )
            snapshot_path = os.path.join(snapshot_dir, snapshot_name)
            snapshot_image = Image.fromarray(
                np.asarray(observations["rgb"], dtype=np.uint8)
            ).convert("RGB")
            draw = ImageDraw.Draw(snapshot_image)
            draw.rectangle((0, 0, snapshot_image.width, 30), fill=(0, 0, 0))
            draw.text(
                (8, 8),
                (
                    f"S2 loop step={int(step_id)} "
                    f"turn={transition.get('turn_direction')} "
                    f"queries={transition.get('same_turn_generation_streak')}"
                ),
                fill=(255, 255, 0),
            )
            snapshot_image.save(snapshot_path, quality=92)
            event["rgb_file"] = os.path.relpath(snapshot_path, log_dir)
        self._write_s2_action_loop_event(event)
        print(
            "[S2ActionLoop][Shadow] "
            f"episode={scene_id}/{episode_id} step={step_id} "
            f"direction={transition.get('turn_direction')} "
            f"generations={transition.get('same_turn_generation_streak')} "
            f"turns={transition.get('cumulative_turn_actions')} "
            f"tier={triage.get('tier')}",
            flush=True,
        )
        return event

    def _maybe_save_stuck_snapshot(
        self,
        *,
        state: dict,
        event: Optional[dict],
        rgb: np.ndarray,
        step_id: int,
        scene_id: str,
        episode_id: int,
        instruction: str,
        action,
        pixel_goal,
        local_actions,
        action_seq,
        llm_outputs: str,
        action_source: str = "unknown",
        pre_safety_action=None,
        vlmap_safety_decision: Optional[dict] = None,
        last_s2_query_step: Optional[int] = None,
        episode_eval_seed: Optional[int] = None,
        environment_step_applied: bool = True,
        force: bool = False,
    ) -> Optional[dict]:
        """Save one representative RGB and S2 decision for a stuck episode."""
        cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        if not bool(cfg.get("stuck_snapshot_enable", False)) or state.get("stuck_snapshot_saved"):
            return None
        event = dict(event or {})
        goal_key = None
        if isinstance(pixel_goal, (list, tuple)) and len(pixel_goal) >= 2:
            goal_key = (int(pixel_goal[0]), int(pixel_goal[1]))
        if goal_key != state.get("stuck_snapshot_last_pixel_goal"):
            state["stuck_snapshot_last_pixel_goal"] = goal_key
            state["stuck_snapshot_pixel_goal_start_step"] = (
                None if goal_key is None else int(step_id)
            )
            state["stuck_snapshot_pixel_goal_change_count"] = int(
                state.get("stuck_snapshot_pixel_goal_change_count", 0) or 0
            ) + 1
        if last_s2_query_step is not None:
            state["stuck_snapshot_last_s2_query_step"] = int(last_s2_query_step)
        action_value = None if action is None else int(action)
        history = state.setdefault("stuck_snapshot_action_history", [])
        source_history = state.setdefault("stuck_snapshot_action_source_history", [])
        if action_value is not None:
            history.append(action_value)
            source_history.append(str(action_source or "unknown"))
        window = max(4, int(cfg.get("stuck_snapshot_action_window_steps", 32)))
        if len(history) > window:
            del history[:-window]
        if len(source_history) > window:
            del source_history[:-window]
        min_step = max(0, int(cfg.get("stuck_snapshot_min_step", 30)))
        repeated_ratio = 0.0
        dominant_action = None
        if history:
            dominant_action = max(set(history), key=history.count)
            repeated_ratio = float(history.count(dominant_action) / len(history))
        repeat_trigger = bool(
            len(history) >= window
            and repeated_ratio >= float(cfg.get("stuck_snapshot_repeat_ratio", 0.90))
        )
        stagnation_trigger = bool(
            event.get("map_stagnation_recovery_gate")
            or event.get("total_map_stagnation_trigger")
            or (
                event.get("low_displacement")
                and int(event.get("total_stagnation_streak", 0) or 0) >= window
            )
        )
        if not force and (
            int(step_id) < min_step or not (repeat_trigger or stagnation_trigger)
        ):
            return None
        pixel_goal_start_step = state.get("stuck_snapshot_pixel_goal_start_step")
        pixel_goal_age_steps = (
            None
            if pixel_goal_start_step is None
            else max(0, int(step_id) - int(pixel_goal_start_step))
        )
        effective_last_s2_query_step = state.get("stuck_snapshot_last_s2_query_step")
        s2_query_age_steps = (
            None
            if effective_last_s2_query_step is None
            else max(0, int(step_id) - int(effective_last_s2_query_step))
        )
        dominant_action_source = None
        if source_history:
            dominant_action_source = max(set(source_history), key=source_history.count)
        run_dir = self._get_vlmap_run_dir() or self.output_path
        snapshot_dir = os.path.join(run_dir, "stuck_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        stem = f"{scene_id}_{int(episode_id)}_step{int(step_id)}"
        image_path = os.path.join(snapshot_dir, f"{stem}.jpg")
        metadata_path = os.path.join(snapshot_dir, f"{stem}.json")
        snapshot_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(snapshot_image)
        if isinstance(pixel_goal, (list, tuple)) and len(pixel_goal) >= 2:
            goal_x, goal_y = int(pixel_goal[0]), int(pixel_goal[1])
            draw.ellipse(
                (goal_x - 8, goal_y - 8, goal_x + 8, goal_y + 8),
                outline=(255, 0, 0),
                width=4,
            )
        draw.rectangle((0, 0, snapshot_image.width, 30), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            f"step={int(step_id)} action={action_value} repeat={repeated_ratio:.2f}",
            fill=(255, 255, 0),
        )
        snapshot_image.save(image_path, quality=92)
        metadata = {
            "event_type": "stuck_snapshot",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "instruction": instruction,
            "trigger_reasons": [
                reason for reason, enabled in (
                    ("repeated_action", repeat_trigger),
                    ("map_or_pose_stagnation", stagnation_trigger),
                    ("forced_episode_end", force),
                ) if enabled
            ],
            "current_action": action_value,
            "pre_safety_action": (
                None if pre_safety_action is None else int(pre_safety_action)
            ),
            "action_source": str(action_source or "unknown"),
            "action_name_map": {
                "0": "STOP",
                "1": "FORWARD",
                "2": "LEFT",
                "3": "RIGHT",
                "4": "LOOKUP",
                "5": "LOOKDOWN",
            },
            "dominant_action_source": dominant_action_source,
            "action_source_window": list(source_history),
            "environment_step_applied": bool(environment_step_applied),
            "dominant_action": dominant_action,
            "action_window": list(history),
            "dominant_action_ratio": repeated_ratio,
            "pixel_goal": pixel_goal,
            "pixel_goal_age_steps": pixel_goal_age_steps,
            "pixel_goal_change_count": int(
                state.get("stuck_snapshot_pixel_goal_change_count", 0) or 0
            ),
            "last_s2_query_step": effective_last_s2_query_step,
            "s2_query_age_steps": s2_query_age_steps,
            "episode_eval_seed": episode_eval_seed,
            "local_actions": list(local_actions or []),
            "action_seq": list(action_seq or []),
            "local_action_queue_length": len(local_actions or []),
            "system2_action_queue_length": len(action_seq or []),
            "s2_output": str(llm_outputs or ""),
            "s2_decision": {
                "pixel_goal": pixel_goal,
                "queued_local_actions": list(local_actions or []),
                "queued_system2_actions": list(action_seq or []),
                "raw_output": str(llm_outputs or ""),
            },
            "vlmap_safety_decision": dict(vlmap_safety_decision or {}),
            "recovery_shadow_event": event,
            "rgb_file": os.path.basename(image_path),
        }
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(self._jsonable(metadata), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        state["stuck_snapshot_saved"] = True
        state["stuck_snapshot_path"] = metadata_path
        print(f"[StuckSnapshot] saved {metadata_path}")
        return metadata

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._jsonable(value.tolist())
        if isinstance(value, torch.Tensor):
            return self._jsonable(value.detach().cpu().tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return self._jsonable(vars(value))
        return str(value)

    def _extract_collision_summary(self, metrics: dict, *, steps: Optional[int] = None) -> dict:
        raw = (metrics or {}).get("collisions")
        count = 0.0
        is_collision = False
        if isinstance(raw, dict):
            for key in ("count", "collision_count", "num_collisions", "collisions"):
                if key in raw and raw[key] is not None:
                    try:
                        count = float(raw[key])
                    except (TypeError, ValueError):
                        count = 0.0
                    break
            if "is_collision" in raw:
                is_collision = bool(raw.get("is_collision"))
            else:
                is_collision = count > 0.0
        elif raw is not None:
            for attr in ("count", "collision_count", "num_collisions"):
                if hasattr(raw, attr):
                    try:
                        count = float(getattr(raw, attr))
                    except (TypeError, ValueError):
                        count = 0.0
                    break
            else:
                try:
                    count = float(raw)
                except (TypeError, ValueError):
                    count = 0.0
            if hasattr(raw, "is_collision"):
                is_collision = bool(getattr(raw, "is_collision"))
            else:
                is_collision = count > 0.0

        steps_int = max(1, int(steps or 0))
        success = float((metrics or {}).get("success", 0.0) or 0.0)
        spl = float((metrics or {}).get("spl", 0.0) or 0.0)
        collision_free = 1.0 if count <= 0.0 else 0.0
        return {
            "collision_raw": self._jsonable(raw),
            "collision_count": float(count),
            "collision_is_collision": bool(is_collision),
            "collision_free": float(collision_free),
            "collision_rate_per_step": float(count / steps_int),
            "cf_success": float(success * collision_free),
            "cf_spl": float(spl * collision_free),
        }

    def _seed_eval_rng(self, seed: int, label: str = "") -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if label:
            print(f"[HabitatVLN] fixed eval random seed ({label}): {seed}")

    def _get_eval_episode_seed(
        self,
        episode_index: int,
        episode_id: int,
        scene_id: Optional[str] = None,
    ) -> Optional[int]:
        overrides = dict(getattr(self.model_args, "eval_episode_seed_overrides", None) or {})
        if overrides:
            keys = []
            if scene_id is not None:
                keys.append(f"{scene_id}/{int(episode_id)}")
            keys.extend((str(int(episode_id)), int(episode_id)))
            for key in keys:
                if key in overrides:
                    return int(overrides[key])
        base_seed = getattr(self.model_args, "eval_random_seed", None)
        if base_seed is None or not bool(getattr(self.model_args, "eval_seed_per_episode", False)):
            return None

        mode = getattr(self.model_args, "eval_episode_seed_mode", "episode_index")
        if mode == "episode_id":
            episode_offset = int(episode_id)
        elif mode == "episode_index":
            episode_offset = int(episode_index)
        else:
            raise ValueError(f"Invalid eval_episode_seed_mode: {mode}")

        rank_offset = int(getattr(self, "rank", 0)) * 100000
        return int(base_seed) + episode_offset + rank_offset

    def _seed_eval_rng_for_episode(
        self,
        episode_index: int,
        episode_id: int,
        scene_id: Optional[str] = None,
    ) -> Optional[int]:
        episode_seed = self._get_eval_episode_seed(episode_index, episode_id, scene_id)
        if episode_seed is None:
            return None
        self._seed_eval_rng(episode_seed, f"episode_index={episode_index}, episode_id={episode_id}")
        return episode_seed

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "collisions" in global_metrics:
            collisions_all = global_metrics["collisions"]
            result_all["collision_count_sum"] = float(collisions_all.sum().item()) if denom > 0 else 0.0
            result_all["collision_count_mean"] = float(collisions_all.mean().item()) if denom > 0 else 0.0
            result_all["collision_episode_rate"] = (
                float((collisions_all > 0).float().mean().item()) if denom > 0 else 0.0
            )
        if "collision_free" in global_metrics:
            collision_free_all = global_metrics["collision_free"]
            result_all["collision_free_rate"] = (
                float(collision_free_all.mean().item()) if denom > 0 else 0.0
            )
        if "cf_sucs" in global_metrics:
            cf_sucs_all = global_metrics["cf_sucs"]
            result_all["cf_sucs_all"] = float(cf_sucs_all.mean().item()) if denom > 0 else 0.0
        if "cf_spls" in global_metrics:
            cf_spls_all = global_metrics["cf_spls"]
            result_all["cf_spls_all"] = float(cf_spls_all.mean().item()) if denom > 0 else 0.0

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def _select_s2_prompt_prefix(self) -> str:
        prompt_index = getattr(self.model_args, "s2_prompt_conjunction_index", None)
        if prompt_index is not None:
            return self.conjunctions[int(prompt_index) % len(self.conjunctions)]
        return random.choice(self.conjunctions)

    def _postprocess_habitat_action_with_vlmap_safety(
        self,
        action,
        observations: dict,
        depth_m: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        episode_index: Optional[int] = None,
        episode_count: Optional[int] = None,
        pixel_goal=None,
    ):
        if not hasattr(self, "vlmap_safety"):
            return action, False, {}
        safety_obs = {
            "depth": depth_m,
            "rgb": rgb,
            "gps": observations.get("gps"),
            "compass": observations.get("compass"),
            "last_nav_action": self._vlmap_last_nav_action,
            "debug_context": {
                "step_id": step_id,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "episode_count": episode_count,
                "pixel_goal": pixel_goal,
            },
        }
        if safety_obs["gps"] is None or safety_obs["compass"] is None:
            return action, False, {}
        safe_action, changed = self.vlmap_safety.postprocess(safety_obs, int(action))
        decision = dict(getattr(self.vlmap_safety, "last_decision", {}) or {})
        if changed:
            print(f"[VLMapSafety][Habitat] replace action {int(action)} -> {int(safe_action)}")
        elif decision.get("budget_suppressed"):
            print(
                "[VLMapSafety][Habitat] keep action "
                f"{int(action)} because {decision.get('budget_suppressed_reason')}"
            )
        if decision.get("replan_required"):
            print(
                "[VLMapSafety][Habitat] repeated block; "
                "clear local goal and request S2 replan"
            )
        if decision.get("waypoint_repair_required"):
            print(
                "[VLMapSafety][Habitat] stuck cluster; "
                "clear local goal and mark waypoint-level repair required"
            )
        return int(safe_action), changed, decision

    def _evaluate_pixel_goal_with_vlmap(
        self,
        pixel_goal,
        observations: dict,
        depth_m: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        episode_index: Optional[int] = None,
        episode_count: Optional[int] = None,
        camera_pitch_deg: float = 0.0,
    ) -> dict:
        if not hasattr(self, "vlmap_safety"):
            return {}
        evaluate = getattr(self.vlmap_safety, "evaluate_pixel_goal", None)
        if evaluate is None:
            return {}
        safety_obs = {
            "depth": depth_m,
            "rgb": rgb,
            "gps": observations.get("gps"),
            "compass": observations.get("compass"),
        }
        if safety_obs["gps"] is None or safety_obs["compass"] is None:
            return {}
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        depth_h, depth_w = depth_m.shape[:2]
        source_image_width = int(vlmap_safety_cfg.get("waypoint_source_image_width") or depth_w)
        source_image_height = int(vlmap_safety_cfg.get("waypoint_source_image_height") or depth_h)
        context = {
            "step_id": step_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "image_width": source_image_width,
            "image_height": source_image_height,
            "camera_pitch_deg": float(camera_pitch_deg),
        }
        decision = evaluate(safety_obs, pixel_goal, context=context)
        if decision.get("waypoint_recovery_required"):
            print(
                "[VLMapSafety][Habitat][Waypoint] queue VLMap recovery "
                f"for pixel goal {pixel_goal}; "
                f"reason={decision.get('waypoint_recovery_reason')} "
                f"actions={decision.get('waypoint_recovery_actions')} "
                f"risk={decision.get('waypoint_risk_score')}"
            )
        elif decision.get("waypoint_recovery_suppressed_reason"):
            print(
                "[VLMapSafety][Habitat][Waypoint] suppress VLMap recovery "
                f"for pixel goal {pixel_goal}; "
                f"reason={decision.get('waypoint_recovery_suppressed_reason')} "
                f"risk={decision.get('waypoint_risk_score')}"
            )
        if decision.get("requery_required"):
            print(
                "[VLMapSafety][Habitat][Waypoint] request S2 requery "
                f"for pixel goal {pixel_goal}; "
                f"reason={decision.get('waypoint_requery_reason')} "
                f"risk={decision.get('waypoint_risk_score')}"
            )
        elif decision.get("waypoint_requery_suppressed_reason"):
            print(
                "[VLMapSafety][Habitat][Waypoint] suppress S2 requery "
                f"for pixel goal {pixel_goal}; "
                f"reason={decision.get('waypoint_requery_suppressed_reason')} "
                f"risk={decision.get('waypoint_risk_score')}"
            )
        if decision.get("valid") and decision.get("path_free") is False:
            print(
                "[VLMapSafety][Habitat][Waypoint] blocked pixel goal "
                f"{pixel_goal}; shadow={decision.get('shadow_only')} "
                f"requery={decision.get('requery_required')}"
            )
        return dict(decision or {})

    def _validate_local_actions_with_vlmap(
        self,
        local_actions,
        observations: dict,
        depth_m: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        episode_index: Optional[int] = None,
        episode_count: Optional[int] = None,
        pixel_goal=None,
    ):
        if not hasattr(self, "vlmap_safety"):
            return False, {}
        validate = getattr(self.vlmap_safety, "validate_trajectory", None)
        if validate is None:
            return False, {}
        safety_obs = {
            "depth": depth_m,
            "rgb": rgb,
            "gps": observations.get("gps"),
            "compass": observations.get("compass"),
        }
        if safety_obs["gps"] is None or safety_obs["compass"] is None:
            return False, {}
        context = {
            "step_id": step_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "pixel_goal": pixel_goal,
        }
        decision = validate(safety_obs, local_actions, context=context)
        if decision.get("would_reject"):
            mode = "shadow" if decision.get("shadow_only") else "active"
            print(
                "[VLMapSafety][Habitat][Trajectory] "
                f"{mode} reject candidate actions={decision.get('actions')} "
                f"blocked={decision.get('blocked_steps')}/{decision.get('checked_forward_steps')} "
                f"reason={decision.get('reject_reason')} "
                f"suppressed={decision.get('reject_suppressed_reason')}"
            )
        if decision.get("reject_required"):
            print(
                "[VLMapSafety][Habitat][Trajectory] reject local trajectory; "
                "clear current goal and request new S2 observation"
            )
        return bool(decision.get("reject_required")), dict(decision or {})

    def _evaluate_semantic_match_with_vlmap(
        self,
        semantic_image,
        instruction: str,
        pixel_goal,
        observations: dict,
        step_id: Optional[int] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        episode_index: Optional[int] = None,
        episode_count: Optional[int] = None,
        observation_source: str = "current",
    ) -> dict:
        if not hasattr(self, "vlmap_semantic"):
            return {}
        context = {
            "step_id": step_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "instruction": instruction,
            "pixel_goal": pixel_goal,
            "observation_source": observation_source,
            "gps": observations.get("gps"),
            "compass": observations.get("compass"),
        }
        try:
            image_arr = np.asarray(semantic_image)
            if image_arr.ndim >= 2:
                context["image_height"] = int(image_arr.shape[0])
                context["image_width"] = int(image_arr.shape[1])
        except Exception:
            pass
        decision = self.vlmap_semantic.match_observation(semantic_image, context=context)
        if decision.get("status") == "ok" and decision.get("top_match"):
            print(
                "[VLMapSemantic][Habitat] "
                f"top={decision.get('top_match')} "
                f"score={float(decision.get('top_score', 0.0)):.3f} "
                f"hits={decision.get('threshold_hits', [])}"
            )
            if decision.get("stagnation_would_requery"):
                if decision.get("stagnation_requery_required"):
                    mode = "active-reobserve"
                elif decision.get("stagnation_hint_required"):
                    mode = "active-hint"
                else:
                    mode = "shadow"
                print(
                    "[VLMapSemantic][Habitat][Stagnation] "
                    f"{mode} triggered; "
                    f"unique={decision.get('stagnation_recent_unique_count')} "
                    f"recent={decision.get('stagnation_recent_terms')}"
                )
        elif decision.get("status") in ("model_unavailable", "score_error", "text_feature_unavailable"):
            print(
                "[VLMapSemantic][Habitat] "
                f"status={decision.get('status')} "
                f"reason={decision.get('disabled_reason')}"
            )
        return dict(decision or {})

    def _format_vlmap_waypoint_feedback(self, pixel_goal, decision: dict) -> str:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        if not bool(vlmap_safety_cfg.get("waypoint_requery_feedback_enable", True)):
            return ""

        stats = decision.get("line_stats") or {}
        checked = int(stats.get("checked", 0) or 0)
        blocked = int(stats.get("blocked", 0) or 0)
        risk_score = float(decision.get("waypoint_risk_score", 0.0) or 0.0)
        reason = decision.get("waypoint_requery_reason") or decision.get("reason") or "unsafe"

        if reason == "blocked":
            risk_text = "the straight route to it is blocked by mapped obstacles"
        elif reason == "high_risk":
            risk_text = (
                "the straight route to it crosses many mapped obstacle cells "
                f"({blocked}/{checked}, risk {risk_score:.2f})"
            )
        else:
            risk_text = f"it is considered unsafe by the local VLMap ({reason})"

        try:
            x, y = int(pixel_goal[0]), int(pixel_goal[1])
            coord_text = f"column={x}, row={y}"
        except (TypeError, ValueError, IndexError):
            coord_text = str(pixel_goal)

        return (
            "Navigation safety feedback: the previous waypoint you selected "
            f"({coord_text}) was rejected because {risk_text}. "
            "Select a different waypoint on visible open floor, away from furniture, walls, "
            "doorframes, and narrow obstacle bands. Do not repeat the rejected waypoint. "
            "Still output only the next waypoint coordinates or STOP."
        )

    def _format_vlmap_semantic_stagnation_hint(self, decision: dict) -> str:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        configured_hint = vlmap_safety_cfg.get("semantic_stagnation_prompt_hint")
        if configured_hint:
            return str(configured_hint)
        return (
            "Navigation note: your recent observations look similar. "
            "Re-check the instruction and choose a waypoint that makes progress "
            "toward the next landmark. Output only the next waypoint coordinates or STOP."
        )

    def _get_occ_memory_guidance_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("occ_memory_guidance_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("occ_memory_guidance_shadow_only", True)),
            "min_dead_zone_score": float(vlmap_safety_cfg.get("occ_memory_guidance_min_dead_zone_score", 0.65)),
            "require_no_recent_high_conf": bool(
                vlmap_safety_cfg.get("occ_memory_guidance_require_no_recent_high_conf", True)
            ),
            "min_frontier_count": max(1, int(vlmap_safety_cfg.get("occ_memory_guidance_min_frontier_count", 1))),
            "cooldown_steps": max(0, int(vlmap_safety_cfg.get("occ_memory_guidance_cooldown_steps", 24))),
            "max_hints_per_episode": int(vlmap_safety_cfg.get("occ_memory_guidance_max_hints_per_episode", 2)),
            "requery_on_trigger": bool(vlmap_safety_cfg.get("occ_memory_guidance_requery_on_trigger", False)),
            "prompt_hint": vlmap_safety_cfg.get("occ_memory_guidance_prompt_hint"),
            "counterfactual_enable": bool(
                vlmap_safety_cfg.get("occ_memory_guidance_counterfactual_enable", False)
            ),
            "counterfactual_max_queries_per_episode": int(
                vlmap_safety_cfg.get("occ_memory_guidance_counterfactual_max_queries_per_episode", 2)
            ),
            "counterfactual_pixel_shift_threshold": float(
                vlmap_safety_cfg.get("occ_memory_guidance_counterfactual_pixel_shift_threshold", 40.0)
            ),
        }

    def _format_occ_memory_guidance_hint(self, decision: dict) -> str:
        cfg = self._get_occ_memory_guidance_cfg()
        configured_hint = cfg.get("prompt_hint")
        direction = str(decision.get("frontier_dominant_direction") or "unknown")
        direction_text = {
            "front": "ahead of you",
            "left": "to your left",
            "right": "to your right",
            "back": "behind you, so consider turning around or choosing a visible side opening",
        }.get(direction, "toward a different visible opening")
        recent_terms = decision.get("semantic_recent_terms") or []
        recent_text = ", ".join(str(item) for item in recent_terms[-3:]) if recent_terms else "similar views"
        frontier_count = int(decision.get("frontier_dominant_count", 0) or 0)
        score = float(decision.get("semantic_dead_zone_score", 0.0) or 0.0)
        if configured_hint:
            try:
                return str(configured_hint).format(
                    direction=direction,
                    direction_text=direction_text,
                    recent_terms=recent_text,
                    frontier_count=frontier_count,
                    dead_zone_score=score,
                )
            except Exception:
                return str(configured_hint)
        return (
            "Navigation memory hint: your recent observations are semantically repetitive "
            f"({recent_text}) and the current waypoint is in a low-confidence semantic zone "
            f"(score {score:.2f}). The sparse 3D memory shows the largest nearby unexplored "
            f"frontier is {direction_text} ({frontier_count} frontier cells). "
            "Use this only if it matches the instruction: choose a visible waypoint toward "
            "open floor, a doorway, or an unexplored opening in that direction, and avoid "
            "continuing through the same semantic area. Output only the next waypoint "
            "coordinates or STOP."
        )

    def _occ_memory_guidance_trigger_reason(
        self,
        decision: dict,
        *,
        step_id: int,
        hint_set_count: int,
        last_hint_step: Optional[int],
    ) -> tuple[bool, str]:
        cfg = self._get_occ_memory_guidance_cfg()
        if not cfg.get("enable"):
            return False, "disabled"
        if not decision or not decision.get("valid"):
            return False, "invalid_waypoint_probe"
        score = float(decision.get("semantic_dead_zone_score", 0.0) or 0.0)
        if not decision.get("semantic_dead_zone") or score < float(cfg["min_dead_zone_score"]):
            return False, "not_dead_zone"
        if cfg["require_no_recent_high_conf"] and int(decision.get("semantic_recent_high_conf_count", 0) or 0) > 0:
            return False, "recent_high_conf_semantic"
        if int(decision.get("frontier_dominant_count", 0) or 0) < int(cfg["min_frontier_count"]):
            return False, "no_dominant_frontier"
        if decision.get("frontier_dominant_direction") == "back":
            return False, "frontier_only_behind"
        max_hints = int(cfg["max_hints_per_episode"])
        if max_hints >= 0 and int(hint_set_count) >= max_hints:
            return False, "max_hints_per_episode"
        if last_hint_step is not None and int(step_id) - int(last_hint_step) < int(cfg["cooldown_steps"]):
            return False, "cooldown"
        return True, "semantic_dead_zone_with_frontier"

    def _get_occ_memory_candidate_probe_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("occ_memory_candidate_probe_enable", False)),
            "selection_enable": bool(
                vlmap_safety_cfg.get("occ_memory_candidate_selection_enable", False)
            ),
            "selection_max_queries_per_episode": int(
                vlmap_safety_cfg.get("occ_memory_candidate_selection_max_queries_per_episode", 2)
            ),
            "selection_max_new_tokens": int(
                vlmap_safety_cfg.get("occ_memory_candidate_selection_max_new_tokens", 32)
            ),
            "selection_parse_coordinates": bool(
                vlmap_safety_cfg.get("occ_memory_candidate_selection_parse_coordinates", False)
            ),
            "selection_coordinate_threshold_px": float(
                vlmap_safety_cfg.get("occ_memory_candidate_selection_coordinate_threshold_px", 90.0)
            ),
        }

    def _format_occ_memory_candidate_selection_prompt(self, candidate_event: dict) -> str:
        cfg = self._get_occ_memory_candidate_probe_cfg()
        candidates = candidate_event.get("candidates") or []
        lines = [
            "Sparse 3D memory proposes several candidate navigation targets.",
            "Choose the single candidate that best matches the instruction and makes spatial progress.",
            "If none of the candidates are useful, answer NONE.",
            "Answer only one token: A, B, C, D, or NONE.",
        ]
        if cfg.get("selection_parse_coordinates"):
            lines[-1] = (
                "Answer A, B, C, D, or NONE. If you prefer pointing on the final top-down memory map, "
                "output one coordinate on that final map."
            )
        current_state = candidate_event.get("current_waypoint_goal_state")
        current_dead_zone = candidate_event.get("current_waypoint_semantic_dead_zone")
        if current_state is not None or current_dead_zone is not None:
            lines.append(
                "Current S2 waypoint memory status: "
                f"geometry={current_state}, semantic_dead_zone={current_dead_zone}."
            )
        if candidate_event.get("goal_progress_enabled"):
            sequence = candidate_event.get("goal_progress_landmark_sequence") or []
            completed = candidate_event.get("goal_progress_completed_landmarks") or []
            next_landmark = candidate_event.get("goal_progress_next_landmark")
            lines.append(
                "Goal progress memory: "
                f"ordered_landmarks={sequence}, completed={completed}, next={next_landmark}."
            )
        for item in candidates:
            label = item.get("candidate_id")
            semantic = item.get("semantic_evidence") or {}
            semantic_text = ""
            if semantic.get("semantic_top_match"):
                semantic_text = (
                    f", semantic={semantic.get('semantic_top_match')} "
                    f"score={semantic.get('semantic_top_score')} "
                    f"instruction_relevance={float(item.get('semantic_relevance_score') or 0.0):.2f} "
                    f"novelty={float(item.get('semantic_novelty_score') or 0.0):.2f} "
                    f"landmark_status={item.get('landmark_status')} "
                    f"matched_landmark={item.get('matched_landmark')} "
                    f"next_relevance={float(item.get('next_landmark_relevance') or 0.0):.2f} "
                    f"completed_penalty={float(item.get('completed_landmark_penalty') or 0.0):.2f} "
                    f"repeated_penalty={float(item.get('repeated_semantic_penalty') or 0.0):.2f}"
                )
            angle_to_current = item.get("angle_to_current_waypoint_deg")
            angle_text = "unknown" if angle_to_current is None else f"{float(angle_to_current):.1f}deg"
            target_text = ""
            if item.get("target_frontier_enabled"):
                target_text = (
                    f", target_frontier={float(item.get('target_frontier_score') or 0.0):.2f} "
                    f"escape={item.get('target_frontier_escape_candidate')} "
                    f"doorway={float(item.get('target_frontier_doorway_like_score') or 0.0):.2f} "
                    f"cluster={float(item.get('target_frontier_cluster_score') or 0.0):.2f} "
                    f"corridor={float(item.get('target_frontier_corridor_continuation_score') or 0.0):.2f} "
                    f"intent_safe={item.get('target_frontier_intent_safe')}"
                )
            lines.append(
                f"{label}: type={item.get('candidate_type')}, "
                f"direction={item.get('direction_bucket')} "
                f"angle={float(item.get('direction_angle_deg') or 0.0):.1f}deg, "
                f"distance={float(item.get('distance_m') or 0.0):.2f}m, "
                f"geometry={item.get('goal_state')}, "
                f"frontier_progress={float(item.get('frontier_progress_score') or 0.0):.2f}, "
                f"topology_novelty={float(item.get('topology_novelty_score') or 0.0):.2f}, "
                f"revisit_risk={float(item.get('revisit_risk') or 0.0):.2f}, "
                f"angle_to_original={angle_text}"
                f"{semantic_text}"
                f"{target_text}."
            )
        return " ".join(lines)

    def _parse_occ_memory_candidate_choice_output(self, text: str, candidates: list) -> dict:
        cfg = self._get_occ_memory_candidate_probe_cfg()
        text = "" if text is None else str(text)
        upper_text = text.upper()
        labels = {
            str(item.get("candidate_id")).upper(): item
            for item in candidates
            if item.get("candidate_id")
        }
        if re.search(r"\bNONE\b", upper_text):
            return {
                "valid": True,
                "none": True,
                "choice": None,
                "selected_candidate": None,
                "reason": "none",
            }
        for label in re.findall(r"\b([A-Z])\b", upper_text):
            if label in labels:
                return {
                    "valid": True,
                    "none": False,
                    "choice": label,
                    "selected_candidate": labels[label],
                    "reason": "matched_label",
                }
        if cfg.get("selection_parse_coordinates"):
            numbers = [int(item) for item in re.findall(r"\d+", text)]
            if len(numbers) >= 2:
                # Existing S2 waypoint convention is "row col"; test both
                # conventions against BEV candidate centers and keep the closer
                # one because this is only a shadow parser.
                raw_a = [float(numbers[1]), float(numbers[0])]
                raw_b = [float(numbers[0]), float(numbers[1])]
                best = None
                best_distance = None
                best_convention = None
                for convention, point in (("row_col", raw_a), ("x_y", raw_b)):
                    for item in candidates:
                        bev_pixel = item.get("bev_pixel")
                        if not bev_pixel or len(bev_pixel) < 2:
                            continue
                        dist = float(
                            np.hypot(
                                float(point[0]) - float(bev_pixel[0]),
                                float(point[1]) - float(bev_pixel[1]),
                            )
                        )
                        if best_distance is None or dist < best_distance:
                            best = item
                            best_distance = dist
                            best_convention = convention
                threshold = float(cfg.get("selection_coordinate_threshold_px", 90.0) or 90.0)
                if best is not None and best_distance is not None and best_distance <= threshold:
                    return {
                        "valid": True,
                        "none": False,
                        "choice": best.get("candidate_id"),
                        "selected_candidate": best,
                        "reason": "nearest_bev_coordinate",
                        "coordinate_numbers": numbers[:2],
                        "coordinate_distance_px": best_distance,
                        "coordinate_convention": best_convention,
                        "coordinate_threshold_px": threshold,
                    }
                return {
                    "valid": False,
                    "none": False,
                    "choice": None,
                    "selected_candidate": None,
                    "reason": "coordinate_too_far",
                    "coordinate_numbers": numbers[:2],
                    "coordinate_distance_px": best_distance,
                    "coordinate_threshold_px": threshold,
                }
            arrow_direction = None
            if "←" in text:
                arrow_direction = "left"
            elif "→" in text:
                arrow_direction = "right"
            elif "↑" in text:
                arrow_direction = "front"
            elif "↓" in text:
                arrow_direction = "back"
            if arrow_direction:
                options = [
                    item
                    for item in candidates
                    if item.get("direction_bucket") == arrow_direction
                ]
                if options:
                    selected = max(options, key=lambda item: float(item.get("score", 0.0) or 0.0))
                    return {
                        "valid": True,
                        "none": False,
                        "choice": selected.get("candidate_id"),
                        "selected_candidate": selected,
                        "reason": "direction_token",
                        "direction_token": arrow_direction,
                    }
        return {
            "valid": False,
            "none": False,
            "choice": None,
            "selected_candidate": None,
            "reason": "no_candidate_label",
        }

    def _run_occ_memory_candidate_selection_probe(
        self,
        *,
        base_prompt_body: str,
        input_images: list,
        messages_prefix: Optional[list],
        candidate_event: dict,
        context: dict,
    ) -> dict:
        cfg = self._get_occ_memory_candidate_probe_cfg()
        candidates = candidate_event.get("candidates") or []
        candidate_prompt = self._format_occ_memory_candidate_selection_prompt(candidate_event)
        selection_images = list(input_images)
        candidate_bev_path = candidate_event.get("candidate_bev_path")
        image_prompt = DEFAULT_IMAGE_TOKEN
        if candidate_bev_path and os.path.exists(candidate_bev_path):
            try:
                with Image.open(candidate_bev_path) as bev_img:
                    selection_images.append(bev_img.convert("RGB").copy())
                image_prompt = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}"
                candidate_prompt += (
                    " The next image is the current RGB observation. "
                    "The final image is a top-down sparse memory map with candidates marked A-D."
                )
            except Exception:
                pass
        prompt_instruction = f"{base_prompt_body} {candidate_prompt} {image_prompt}."
        rng_state = self._capture_torch_rng_state()
        event = {
            "status": "ok",
            "output": "",
            "valid": False,
            "none": False,
            "choice": None,
            "reason": None,
            "selected_candidate": None,
            "candidate_bev_path": candidate_bev_path,
        }
        try:
            output = self._generate_s2_text_from_prompt_instruction(
                prompt_instruction,
                selection_images,
                messages_prefix=messages_prefix,
                max_new_tokens=int(cfg.get("selection_max_new_tokens", 32) or 32),
            )
            parse = self._parse_occ_memory_candidate_choice_output(output, candidates)
            event.update({"output": output, **parse})
        except Exception as exc:
            event.update(
                {
                    "status": "error",
                    "reason": "exception",
                    "error": str(exc),
                }
            )
        finally:
            self._restore_torch_rng_state(rng_state)
        self.occ_memory.record_candidate_selection_event(
            candidate_event=candidate_event,
            selection=event,
            context=context,
        )
        return event

    def _get_s2_candidate_probe_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("s2_candidate_probe_enable", False)),
            "count": max(0, int(vlmap_safety_cfg.get("s2_candidate_count", 3))),
            "temperature": float(vlmap_safety_cfg.get("s2_candidate_temperature", 0.7)),
            "top_p": float(vlmap_safety_cfg.get("s2_candidate_top_p", 0.9)),
            "min_pixel_distance": float(vlmap_safety_cfg.get("s2_candidate_min_pixel_distance", 50.0)),
            "max_queries_per_episode": int(vlmap_safety_cfg.get("s2_candidate_max_queries_per_episode", 0)),
            "max_new_tokens": int(vlmap_safety_cfg.get("s2_candidate_max_new_tokens", 128)),
        }

    def _parse_s2_candidate_output(self, text: str, image_width: Optional[int] = None) -> dict:
        text = "" if text is None else str(text)
        upper_text = text.upper()
        numbers = [int(item) for item in re.findall(r"\d+", text)]
        result = {
            "text": text,
            "valid": False,
            "is_stop": "STOP" in upper_text,
            "pixel_goal": None,
            "direction_bucket": "stop" if "STOP" in upper_text else "invalid",
            "number_count": len(numbers),
        }
        if len(numbers) < 2:
            return result

        # Match the existing evaluator convention: model text "row col" becomes
        # pixel_goal [col, row].
        pixel_goal = [int(numbers[1]), int(numbers[0])]
        result["valid"] = True
        result["pixel_goal"] = pixel_goal
        width = int(image_width or getattr(self.model_args, "resize_w", 384) or 384)
        if pixel_goal[0] < width / 3:
            result["direction_bucket"] = "left"
        elif pixel_goal[0] > 2 * width / 3:
            result["direction_bucket"] = "right"
        else:
            result["direction_bucket"] = "center"
        return result

    def _s2_candidate_pairwise_distances(self, candidates: list) -> list:
        valid_goals = [item.get("pixel_goal") for item in candidates if item.get("valid")]
        distances = []
        for idx, goal_a in enumerate(valid_goals):
            for goal_b in valid_goals[idx + 1 :]:
                distances.append(
                    float(np.hypot(float(goal_a[0]) - float(goal_b[0]), float(goal_a[1]) - float(goal_b[1])))
                )
        return distances

    def _s2_candidate_unique_count(self, candidates: list, min_pixel_distance: float) -> int:
        representatives = []
        for item in candidates:
            if not item.get("valid"):
                continue
            goal = item.get("pixel_goal")
            if all(
                float(np.hypot(float(goal[0]) - float(rep[0]), float(goal[1]) - float(rep[1])))
                >= min_pixel_distance
                for rep in representatives
            ):
                representatives.append(goal)
        return len(representatives)

    def _capture_torch_rng_state(self) -> dict:
        state = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_torch_rng_state(self, state: dict) -> None:
        cpu_state = state.get("cpu")
        if cpu_state is not None:
            torch.random.set_rng_state(cpu_state)
        cuda_state = state.get("cuda_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _generate_s2_text_from_prompt_instruction(
        self,
        prompt_instruction: str,
        input_images: list,
        *,
        messages_prefix: Optional[list] = None,
        max_new_tokens: int = 128,
    ) -> str:
        parts = split_and_clean(prompt_instruction)
        content = []
        input_img_id = 0
        for part in parts:
            if part == DEFAULT_IMAGE_TOKEN:
                if input_img_id >= len(input_images):
                    raise ValueError(
                        "S2 prompt contains more image tokens than provided images: "
                        f"{len(input_images)} images"
                    )
                content.append({"type": "image", "image": input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": part})

        messages = list(messages_prefix or [])
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences

        return self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

    def _run_occ_memory_guidance_counterfactual(
        self,
        *,
        base_prompt_body: str,
        final_prompt: str,
        input_images: list,
        messages_prefix: Optional[list],
        base_output: str,
        hint: str,
        decision: dict,
        context: dict,
        image_width: int,
    ) -> dict:
        cfg = self._get_occ_memory_guidance_cfg()
        threshold = float(cfg.get("counterfactual_pixel_shift_threshold", 40.0) or 40.0)
        event = {
            "counterfactual_status": "ok",
            "counterfactual_base_output": base_output,
        }
        rng_state = self._capture_torch_rng_state()
        try:
            hinted_prompt_instruction = f"{base_prompt_body} {hint} {final_prompt}."
            hinted_output = self._generate_s2_text_from_prompt_instruction(
                hinted_prompt_instruction,
                input_images,
                messages_prefix=messages_prefix,
                max_new_tokens=128,
            )
        except Exception as exc:
            hinted_output = ""
            event["counterfactual_status"] = "error"
            event["counterfactual_error"] = str(exc)
        finally:
            self._restore_torch_rng_state(rng_state)

        base_parse = self._parse_s2_candidate_output(base_output, image_width=image_width)
        hinted_parse = self._parse_s2_candidate_output(hinted_output, image_width=image_width)
        event.update(
            {
                "counterfactual_hinted_output": hinted_output,
                "counterfactual_base_valid": bool(base_parse.get("valid")),
                "counterfactual_hinted_valid": bool(hinted_parse.get("valid")),
                "counterfactual_base_is_stop": bool(base_parse.get("is_stop")),
                "counterfactual_hinted_is_stop": bool(hinted_parse.get("is_stop")),
                "counterfactual_base_pixel_goal": base_parse.get("pixel_goal"),
                "counterfactual_hinted_pixel_goal": hinted_parse.get("pixel_goal"),
                "counterfactual_base_image_direction": base_parse.get("direction_bucket"),
                "counterfactual_hinted_image_direction": hinted_parse.get("direction_bucket"),
            }
        )

        if base_parse.get("valid") and hinted_parse.get("valid"):
            base_goal = base_parse["pixel_goal"]
            hinted_goal = hinted_parse["pixel_goal"]
            dx = float(hinted_goal[0]) - float(base_goal[0])
            dy = float(hinted_goal[1]) - float(base_goal[1])
            pixel_shift = float(np.hypot(dx, dy))
            frontier_direction = str(decision.get("frontier_dominant_direction") or "unknown")
            if frontier_direction == "right":
                follows_hint = dx >= threshold
            elif frontier_direction == "left":
                follows_hint = dx <= -threshold
            else:
                follows_hint = None
            event.update(
                {
                    "counterfactual_pixel_shift": pixel_shift,
                    "counterfactual_pixel_dx": dx,
                    "counterfactual_pixel_dy": dy,
                    "counterfactual_changed_pixel": pixel_shift >= threshold,
                    "counterfactual_changed_image_direction": (
                        base_parse.get("direction_bucket") != hinted_parse.get("direction_bucket")
                    ),
                    "counterfactual_follows_left_right_hint": follows_hint,
                }
            )
        else:
            event.update(
                {
                    "counterfactual_pixel_shift": None,
                    "counterfactual_pixel_dx": None,
                    "counterfactual_pixel_dy": None,
                    "counterfactual_changed_pixel": False,
                    "counterfactual_changed_image_direction": False,
                    "counterfactual_follows_left_right_hint": None,
                }
            )

        self.occ_memory.record_guidance_event(
            action="counterfactual",
            hint=hint,
            reason="shadow_steerability_probe",
            decision=decision,
            context=context,
            extra=event,
        )
        return event

    def _get_som_counterfactual_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("som_counterfactual_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("som_counterfactual_shadow_only", True)),
            "max_queries_per_episode": int(
                vlmap_safety_cfg.get("som_counterfactual_max_queries_per_episode", 30)
            ),
            "overlay_type": str(
                vlmap_safety_cfg.get("som_counterfactual_overlay_type", "frontier_direction")
            ),
            "pixel_shift_threshold": float(
                vlmap_safety_cfg.get("som_counterfactual_pixel_shift_threshold", 40.0)
            ),
            "min_unsafe_signal": float(
                vlmap_safety_cfg.get("som_counterfactual_min_unsafe_signal", 0.50)
            ),
            "active_min_unsafe_signal": float(
                vlmap_safety_cfg.get("som_counterfactual_active_min_unsafe_signal", 0.50)
            ),
            "active_max_per_episode": int(
                vlmap_safety_cfg.get("som_counterfactual_active_max_per_episode", 3)
            ),
            # Stage14c-v2: gate active replacement on OccMem goal_state.
            # "any"  -> original Stage14c behaviour (no geometry gate)
            # "occupied" -> only replace when S2 goal lands on occupied cell (v2a)
            # "occupied_or_free_follows" -> occupied OR (free AND follows_frontier) (v2b)
            "active_goal_state_gate": str(
                vlmap_safety_cfg.get("som_counterfactual_active_goal_state_gate", "any")
            ),
            "frontier_alpha": max(
                0,
                min(255, int(vlmap_safety_cfg.get("som_counterfactual_frontier_alpha", 80))),
            ),
            "unsafe_alpha": max(
                0,
                min(255, int(vlmap_safety_cfg.get("som_counterfactual_unsafe_alpha", 120))),
            ),
            "draw_frontier": bool(vlmap_safety_cfg.get("som_counterfactual_draw_frontier", True)),
            "draw_unsafe_goal": bool(vlmap_safety_cfg.get("som_counterfactual_draw_unsafe_goal", True)),
            "draw_base_goal": bool(vlmap_safety_cfg.get("som_counterfactual_draw_base_goal", False)),
            "max_new_tokens": int(vlmap_safety_cfg.get("som_counterfactual_max_new_tokens", 128)),
        }

    def _write_som_counterfactual_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "som_counterfactual_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _get_stage15_repair_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "active": bool(vlmap_safety_cfg.get("stage15_repair_active", False)),
            "gate_mode": str(vlmap_safety_cfg.get("stage15_repair_gate_mode", "consecutive")),
            "gate_min_count": max(
                1, int(vlmap_safety_cfg.get("stage15_repair_gate_min_count", 3))
            ),
            "active_max_per_episode": max(
                0,
                int(vlmap_safety_cfg.get("stage15_repair_active_max_per_episode", 5)),
            ),
        }

    def _write_stage15_repair_active_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "stage15_repair_active_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _get_stage_d_bfs_escape_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("stage_d_bfs_escape_shadow_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("stage_d_bfs_escape_shadow_only", True)),
            "active": bool(vlmap_safety_cfg.get("stage_d_bfs_escape_active_enable", False)),
            "active_max_per_episode": max(
                0, int(vlmap_safety_cfg.get("stage_d_bfs_escape_active_max_per_episode", 2))
            ),
            "active_require_target_frontier": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_require_target_frontier", True)
            ),
            "active_path_edge_steps": max(
                1, int(vlmap_safety_cfg.get("stage_d_bfs_escape_active_path_edge_steps", 8))
            ),
            "active_goal_world_z": float(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_goal_world_z", 0.0)
            ),
            "active_require_pixel_in_bounds": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_require_pixel_in_bounds", True)
            ),
            "active_pixel_goal_mode": str(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_pixel_goal_mode", "projection")
                or "projection"
            ).lower(),
            "active_direction_y_ratio": float(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_direction_y_ratio", 0.75)
            ),
            "active_direction_front_x_ratio": float(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_direction_front_x_ratio", 0.50)
            ),
            "active_direction_left_x_ratio": float(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_direction_left_x_ratio", 0.25)
            ),
            "active_direction_right_x_ratio": float(
                vlmap_safety_cfg.get("stage_d_bfs_escape_active_direction_right_x_ratio", 0.75)
            ),
            "min_step": max(0, int(vlmap_safety_cfg.get("stage_d_bfs_escape_min_step", 30))),
            "compass_window_steps": max(
                1, int(vlmap_safety_cfg.get("stage_d_bfs_escape_compass_window_steps", 20))
            ),
            "compass_reversal_threshold": max(
                0.0,
                float(vlmap_safety_cfg.get("stage_d_bfs_escape_compass_reversal_threshold", 0.07)),
            ),
            "compass_reversal_metric": str(
                vlmap_safety_cfg.get("stage_d_bfs_escape_compass_reversal_metric", "sign")
                or "sign"
            ).lower(),
            "consecutive_occupied_min": max(
                1, int(vlmap_safety_cfg.get("stage_d_bfs_escape_consecutive_occupied_min", 3))
            ),
            "use_compass_reversal": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_use_compass_reversal", True)
            ),
            "use_consecutive_occupied": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_use_consecutive_occupied", True)
            ),
            "use_semantic_stagnation": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_use_semantic_stagnation", True)
            ),
            "max_action_steps": max(
                1, int(vlmap_safety_cfg.get("stage_d_bfs_escape_max_action_steps", 8))
            ),
            "forward_step_m": max(
                0.05,
                float(
                    vlmap_safety_cfg.get(
                        "stage_d_bfs_escape_forward_step_m",
                        getattr(self.config.habitat.simulator, "forward_step_size", 0.25),
                    )
                ),
            ),
            "frontier_sample_limit": max(
                0, int(vlmap_safety_cfg.get("stage_d_bfs_escape_frontier_sample_limit", 5000))
            ),
            "require_instruction_relevant": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_require_instruction_relevant", True)
            ),
            "allow_fallback_target_frontier": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_allow_fallback_target_frontier", False)
            ),
            "max_events_per_episode": int(
                vlmap_safety_cfg.get("stage_d_bfs_escape_max_events_per_episode", -1)
            ),
            "log_non_trigger_steps": bool(
                vlmap_safety_cfg.get("stage_d_bfs_escape_log_non_trigger_steps", False)
            ),
        }

    def _write_stage_d_bfs_escape_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "bfs_escape_shadow_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_stage_d_bfs_escape_active_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "bfs_escape_active_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _stage_d_compass_reversal_rate(
        self,
        cache: list,
        step_id: int,
        window_steps: int,
        metric: str = "sign",
    ) -> dict:
        min_step = int(step_id) - int(window_steps)
        events = []
        for item in list(cache or []):
            try:
                item_step = int(item.get("eval_step", -1))
            except (TypeError, ValueError):
                continue
            if min_step <= item_step <= int(step_id):
                events.append(item)
        events.sort(key=lambda item: int(item.get("eval_step", -1)))
        compass_vals = []
        for item in events:
            compass = item.get("compass")
            if not compass or len(compass) < 1:
                continue
            try:
                compass_vals.append(float(compass[0]))
            except (TypeError, ValueError):
                continue
        sign_reversal_count = 0
        angle_reversal_count = 0
        for idx in range(1, len(compass_vals)):
            prev = float(compass_vals[idx - 1])
            cur = float(compass_vals[idx])
            if prev * cur < 0:
                sign_reversal_count += 1
            diff = abs((cur - prev + math.pi) % (2.0 * math.pi) - math.pi)
            if diff >= (0.75 * math.pi):
                angle_reversal_count += 1
        denom = max(1, len(compass_vals) - 1)
        metric = str(metric or "sign").lower()
        if metric == "angle":
            reversal_count = angle_reversal_count
        else:
            metric = "sign"
            reversal_count = sign_reversal_count
        return {
            "trajectory_event_count_w": int(len(events)),
            "compass_sample_count_w": int(len(compass_vals)),
            "compass_reversal_metric": metric,
            "compass_sign_reversal_count_w": int(sign_reversal_count),
            "compass_sign_reversal_rate_w": float(sign_reversal_count / denom),
            "compass_angle_reversal_count_w": int(angle_reversal_count),
            "compass_angle_reversal_rate_w": float(angle_reversal_count / denom),
            "compass_reversal_count_w": int(reversal_count),
            "compass_reversal_rate_w": float(reversal_count / denom),
        }

    def _update_stage_d_bfs_escape_shadow(
        self,
        *,
        cfg: dict,
        trajectory_cache: list,
        step_id: int,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        observations: dict,
        occ_waypoint_decision: dict,
        consecutive_occupied_count: int,
    ) -> Optional[dict]:
        if not cfg.get("enable"):
            return None
        compass = self._stage_d_compass_reversal_rate(
            trajectory_cache,
            step_id,
            int(cfg.get("compass_window_steps", 20)),
            str(cfg.get("compass_reversal_metric", "sign") or "sign"),
        )
        semantic_stagnation = bool(
            (occ_waypoint_decision or {}).get("semantic_stagnation_active")
            or (occ_waypoint_decision or {}).get("semantic_last_stagnation")
        )
        compass_trigger = bool(
            cfg.get("use_compass_reversal")
            and compass.get("compass_reversal_rate_w", 0.0)
            > float(cfg.get("compass_reversal_threshold", 0.07))
        )
        occupied_trigger = bool(
            cfg.get("use_consecutive_occupied")
            and int(consecutive_occupied_count) >= int(cfg.get("consecutive_occupied_min", 3))
        )
        stagnation_trigger = bool(cfg.get("use_semantic_stagnation") and semantic_stagnation)
        trigger_conditions = []
        if compass_trigger:
            trigger_conditions.append("compass_reversal")
        if occupied_trigger:
            trigger_conditions.append("consecutive_occupied")
        if stagnation_trigger:
            trigger_conditions.append("semantic_stagnation")
        triggered = bool(step_id >= int(cfg.get("min_step", 30)) and trigger_conditions)
        event = {
            "event_type": "stage_d_bfs_escape_shadow",
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "step_id": int(step_id),
            "shadow_only": bool(cfg.get("shadow_only", True)),
            "triggered": bool(triggered),
            "trigger_conditions": trigger_conditions,
            "trigger_condition": "+".join(trigger_conditions) if trigger_conditions else None,
            "min_step": int(cfg.get("min_step", 30)),
            "goal_state": (occ_waypoint_decision or {}).get("goal_state"),
            "consecutive_occupied_count": int(consecutive_occupied_count),
            "semantic_stagnation_active": bool(semantic_stagnation),
            **compass,
        }
        if not triggered:
            event["reason"] = "not_triggered"
            if bool(cfg.get("log_non_trigger_steps", False)):
                self._write_stage_d_bfs_escape_event(event)
            return event
        try:
            bfs = self.occ_memory.bfs_to_semantic_frontier(
                obs={
                    "gps": (observations or {}).get("gps"),
                    "compass": (observations or {}).get("compass"),
                },
                current_waypoint_decision=occ_waypoint_decision,
                max_action_steps=int(cfg.get("max_action_steps", 8)),
                forward_step_m=float(cfg.get("forward_step_m", 0.25)),
                frontier_sample_limit=int(cfg.get("frontier_sample_limit", 5000)),
                require_instruction_relevant=bool(cfg.get("require_instruction_relevant", True)),
                allow_fallback_target_frontier=bool(cfg.get("allow_fallback_target_frontier", False)),
            )
            event.update(bfs)
        except Exception as exc:
            event.update(
                {
                    "valid": False,
                    "reason": "bfs_exception",
                    "bfs_reachable": False,
                    "error": str(exc),
                }
            )
        self._write_stage_d_bfs_escape_event(event)
        return event

    def _som_direction_bucket(self, pixel_goal, image_width: int) -> str:
        try:
            x = float(pixel_goal[0])
        except (TypeError, ValueError, IndexError):
            return "invalid"
        width = max(1.0, float(image_width))
        if x < width / 3.0:
            return "left"
        if x > 2.0 * width / 3.0:
            return "right"
        return "center"

    def _som_unsafe_signal_from_decision(self, decision: dict) -> float:
        decision = dict(decision or {})
        signal = 0.0
        goal_state = str(decision.get("goal_state") or "")
        if goal_state in ("occupied", "unknown"):
            signal = max(signal, 1.0)
        if bool(decision.get("semantic_dead_zone")):
            signal = max(signal, 0.75)
        try:
            signal = max(signal, float(decision.get("semantic_dead_zone_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            if bool(decision.get("points_to_revisited_region")):
                signal = max(signal, float(decision.get("revisit_score", 0.0) or 0.25))
        except (TypeError, ValueError):
            signal = max(signal, 0.25)
        return min(1.0, max(0.0, float(signal)))

    def _render_som_overlay_v1(
        self,
        image_pil: Image.Image,
        *,
        frontier_direction: Optional[str],
        unsafe_signal: float,
        base_pixel_goal,
        cfg: dict,
    ):
        img = image_pil.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        marks = []

        direction = str(frontier_direction or "").lower()
        if direction == "forward":
            direction = "front"
        if bool(cfg.get("draw_frontier", True)) and direction in ("left", "right", "front", "center", "back"):
            alpha = int(cfg.get("frontier_alpha", 80))
            green = (0, 200, 60, alpha)
            if direction == "left":
                box = [0, h // 4, w // 3, (3 * h) // 4]
            elif direction == "right":
                box = [(2 * w) // 3, h // 4, w - 1, (3 * h) // 4]
            elif direction in ("front", "center"):
                box = [w // 3, h // 5, (2 * w) // 3, (3 * h) // 5]
            else:
                box = [w // 3, (3 * h) // 5, (2 * w) // 3, h - 1]
            draw.rectangle(box, fill=green)
            marks.append({"type": "frontier_direction", "direction": direction, "box": box})

        unsafe_signal = min(1.0, max(0.0, float(unsafe_signal or 0.0)))
        if (
            bool(cfg.get("draw_unsafe_goal", True))
            and unsafe_signal >= float(cfg.get("min_unsafe_signal", 0.50))
            and base_pixel_goal is not None
        ):
            try:
                x = int(float(base_pixel_goal[0]))
                y = int(float(base_pixel_goal[1]))
                radius = max(16, min(w, h) // 12)
                alpha = int(cfg.get("unsafe_alpha", 120))
                red = (255, 40, 40, alpha)
                box = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(box, fill=red, outline=(255, 0, 0, min(255, alpha + 60)), width=3)
                marks.append(
                    {
                        "type": "unsafe_goal",
                        "unsafe_signal": float(unsafe_signal),
                        "center": [int(x), int(y)],
                        "radius": int(radius),
                    }
                )
            except (TypeError, ValueError, IndexError):
                pass

        if bool(cfg.get("draw_base_goal", False)) and base_pixel_goal is not None:
            try:
                x = int(float(base_pixel_goal[0]))
                y = int(float(base_pixel_goal[1]))
                radius = 8
                draw.line([x - radius, y, x + radius, y], fill=(30, 120, 255, 200), width=3)
                draw.line([x, y - radius, x, y + radius], fill=(30, 120, 255, 200), width=3)
                marks.append({"type": "base_goal_cross", "center": [int(x), int(y)]})
            except (TypeError, ValueError, IndexError):
                pass

        if not marks:
            return image_pil.copy(), {"marks": [], "rendered": False}
        return Image.alpha_composite(img, overlay).convert("RGB"), {"marks": marks, "rendered": True}

    def _run_som_counterfactual(
        self,
        *,
        base_prompt_body: str,
        final_prompt: str,
        input_images: list,
        messages_prefix: Optional[list],
        base_output: str,
        base_pixel_goal,
        occ_decision: dict,
        context: dict,
        image_width: int,
    ) -> dict:
        cfg = self._get_som_counterfactual_cfg()
        threshold = float(cfg.get("pixel_shift_threshold", 40.0) or 40.0)
        frontier_direction = (occ_decision or {}).get("frontier_dominant_direction")
        unsafe_signal = self._som_unsafe_signal_from_decision(occ_decision or {})
        event = {
            "event_type": "som_counterfactual_shadow",
            "status": "ok",
            "shadow_only": bool(cfg.get("shadow_only", True)),
            "overlay_type": cfg.get("overlay_type"),
            "base_output": base_output,
            "overlay_output": "",
            "frontier_dominant_direction": frontier_direction,
            "frontier_dominant_count": (occ_decision or {}).get("frontier_dominant_count"),
            "frontier_direction_counts": (occ_decision or {}).get("frontier_direction_counts"),
            "semantic_dead_zone_score": (occ_decision or {}).get("semantic_dead_zone_score"),
            "goal_state": (occ_decision or {}).get("goal_state"),
            "unsafe_signal": float(unsafe_signal),
            **dict(context or {}),
        }
        if not input_images:
            event.update({"status": "skipped_no_images", "overlay_valid": False})
            return event

        current_image = input_images[-1]
        overlay_image, overlay_info = self._render_som_overlay_v1(
            current_image,
            frontier_direction=frontier_direction,
            unsafe_signal=unsafe_signal,
            base_pixel_goal=base_pixel_goal,
            cfg=cfg,
        )
        event["overlay_info"] = overlay_info
        if not overlay_info.get("rendered"):
            event.update({"status": "skipped_no_overlay", "overlay_valid": False})
            return event

        overlay_images = list(input_images)
        overlay_images[-1] = overlay_image
        rng_state = self._capture_torch_rng_state()
        try:
            prompt_instruction = f"{base_prompt_body} {final_prompt}."
            overlay_output = self._generate_s2_text_from_prompt_instruction(
                prompt_instruction,
                overlay_images,
                messages_prefix=messages_prefix,
                max_new_tokens=int(cfg.get("max_new_tokens", 128) or 128),
            )
        except Exception as exc:
            overlay_output = ""
            event["status"] = "error"
            event["error"] = str(exc)
        finally:
            self._restore_torch_rng_state(rng_state)

        base_parse = self._parse_s2_candidate_output(base_output, image_width=image_width)
        overlay_parse = self._parse_s2_candidate_output(overlay_output, image_width=image_width)
        event.update(
            {
                "overlay_output": overlay_output,
                "base_valid": bool(base_parse.get("valid")),
                "overlay_valid": bool(overlay_parse.get("valid")),
                "base_is_stop": bool(base_parse.get("is_stop")),
                "overlay_is_stop": bool(overlay_parse.get("is_stop")),
                "base_pixel_goal": base_parse.get("pixel_goal"),
                "overlay_pixel_goal": overlay_parse.get("pixel_goal"),
                "base_direction_bucket": base_parse.get("direction_bucket"),
                "overlay_direction_bucket": overlay_parse.get("direction_bucket"),
            }
        )

        if base_parse.get("valid") and overlay_parse.get("valid"):
            base_goal = base_parse["pixel_goal"]
            overlay_goal = overlay_parse["pixel_goal"]
            dx = float(overlay_goal[0]) - float(base_goal[0])
            dy = float(overlay_goal[1]) - float(base_goal[1])
            pixel_shift = float(np.hypot(dx, dy))
            direction = str(frontier_direction or "").lower()
            if direction == "forward":
                direction = "front"
            if direction == "right":
                follows_frontier = dx >= threshold
            elif direction == "left":
                follows_frontier = dx <= -threshold
            elif direction in ("front", "center"):
                follows_frontier = (
                    overlay_parse.get("direction_bucket") == "center"
                    and base_parse.get("direction_bucket") != "center"
                )
            else:
                follows_frontier = None
            unsafe_shift_proxy = (
                pixel_shift >= threshold
                if unsafe_signal >= float(cfg.get("min_unsafe_signal", 0.50))
                else None
            )
            event.update(
                {
                    "pixel_shift": pixel_shift,
                    "pixel_dx": dx,
                    "pixel_dy": dy,
                    "changed_pixel": bool(pixel_shift >= threshold),
                    "changed_direction_bucket": bool(
                        base_parse.get("direction_bucket") != overlay_parse.get("direction_bucket")
                    ),
                    "follows_frontier_direction": follows_frontier,
                    "unsafe_shift_proxy": unsafe_shift_proxy,
                    "unsafe_shift_proxy_note": (
                        "proxy only: significant pixel shift when unsafe signal is present; "
                        "does not prove motion away from the unsafe region"
                    ),
                    "moved_away_from_unsafe": None,
                    "moved_away_from_unsafe_reason": "not_computable_without_projected_unsafe_region",
                }
            )
        else:
            event.update(
                {
                    "pixel_shift": None,
                    "pixel_dx": None,
                    "pixel_dy": None,
                    "changed_pixel": False,
                    "changed_direction_bucket": False,
                    "follows_frontier_direction": None,
                    "unsafe_shift_proxy": None,
                    "unsafe_shift_proxy_note": (
                        "proxy only: significant pixel shift when unsafe signal is present; "
                        "does not prove motion away from the unsafe region"
                    ),
                    "moved_away_from_unsafe": None,
                    "moved_away_from_unsafe_reason": "invalid_base_or_overlay_pixel_goal",
                }
            )
        return event

    def _write_s2_candidate_probe_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "s2_candidate_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _probe_s2_candidate_diversity(
        self,
        inputs,
        input_token_length: int,
        greedy_text: str,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        step_id: int,
        query_index: int,
        instruction: str,
    ) -> dict:
        cfg = self._get_s2_candidate_probe_cfg()
        if not cfg["enable"] or cfg["count"] <= 0:
            return {}

        rng_state = self._capture_torch_rng_state()
        candidates = []
        try:
            with torch.no_grad():
                for candidate_index in range(cfg["count"]):
                    sampled_output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=cfg["max_new_tokens"],
                        do_sample=True,
                        temperature=cfg["temperature"],
                        top_p=cfg["top_p"],
                        use_cache=True,
                        past_key_values=None,
                        return_dict_in_generate=True,
                    ).sequences
                    candidate_text = self.processor.tokenizer.decode(
                        sampled_output_ids[0][input_token_length:], skip_special_tokens=True
                    )
                    candidate = self._parse_s2_candidate_output(
                        candidate_text,
                        image_width=getattr(self.model_args, "resize_w", 384),
                    )
                    candidate["candidate_index"] = candidate_index
                    candidates.append(candidate)
        finally:
            # The active System1 generator samples trajectories with torch RNG.
            # Restore state so this shadow-only probe does not alter behavior.
            self._restore_torch_rng_state(rng_state)

        distances = self._s2_candidate_pairwise_distances(candidates)
        valid_candidate_count = sum(1 for item in candidates if item.get("valid"))
        unique_candidate_count = self._s2_candidate_unique_count(candidates, cfg["min_pixel_distance"])
        direction_buckets = [item.get("direction_bucket") for item in candidates]
        event = {
            "event_type": "s2_candidate_probe",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "step_id": int(step_id),
            "query_index": int(query_index),
            "instruction": instruction,
            "candidate_count": int(cfg["count"]),
            "valid_candidate_count": int(valid_candidate_count),
            "unique_candidate_count": int(unique_candidate_count),
            "min_pixel_distance": float(cfg["min_pixel_distance"]),
            "max_pairwise_pixel_distance": float(max(distances)) if distances else 0.0,
            "mean_pairwise_pixel_distance": float(np.mean(distances)) if distances else 0.0,
            "direction_buckets": direction_buckets,
            "unique_direction_bucket_count": len(set(direction_buckets)),
            "greedy": self._parse_s2_candidate_output(
                greedy_text,
                image_width=getattr(self.model_args, "resize_w", 384),
            ),
            "candidates": candidates,
        }
        self._write_s2_candidate_probe_event(event)
        print(
            "[S2CandidateProbe] "
            f"query={query_index} valid={valid_candidate_count}/{cfg['count']} "
            f"unique={unique_candidate_count} "
            f"max_dist={event['max_pairwise_pixel_distance']:.1f} "
            f"buckets={direction_buckets}"
        )
        return event

    def _get_nextdit_candidate_probe_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("nextdit_candidate_probe_enable", False)),
            "max_candidates": max(0, int(vlmap_safety_cfg.get("nextdit_candidate_max_candidates", 32))),
            "max_events_per_episode": int(
                vlmap_safety_cfg.get("nextdit_candidate_max_events_per_episode", 12)
            ),
            "min_endpoint_grid_distance": float(
                vlmap_safety_cfg.get("nextdit_candidate_min_endpoint_grid_distance", 4.0)
            ),
            "action_horizon": max(1, int(vlmap_safety_cfg.get("nextdit_candidate_action_horizon", MAX_LOCAL_STEPS))),
            "active_enable": bool(vlmap_safety_cfg.get("nextdit_candidate_active_enable", False)),
            "active_strategy": str(
                vlmap_safety_cfg.get("nextdit_candidate_active_strategy", "vlmap_obstacle")
            ),
            "active_max_interventions_per_episode": int(
                vlmap_safety_cfg.get("nextdit_candidate_active_max_interventions_per_episode", 2)
            ),
            "active_require_current_reject": bool(
                vlmap_safety_cfg.get("nextdit_candidate_active_require_current_reject", True)
            ),
            "active_occ_current_min_occupied_hits": max(
                1,
                int(vlmap_safety_cfg.get("nextdit_candidate_active_occ_current_min_occupied_hits", 1)),
            ),
            "active_occ_max_direction_deviation_deg": float(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_max_direction_deviation_deg", 45.0)
            ),
            "active_occ_unknown_weight": float(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_unknown_weight", 0.15)
            ),
            "active_occ_direction_weight": float(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_direction_weight", 0.30)
            ),
            "active_occ_forward_progress_weight": float(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_forward_progress_weight", 0.05)
            ),
            "active_occ_require_action_diff": bool(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_require_action_diff", True)
            ),
            "active_occ_reject_all_unknown": bool(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_reject_all_unknown", True)
            ),
            "active_occ_require_vlmap_nonreject": bool(
                vlmap_safety_cfg.get("nextdit_candidate_active_occ_require_vlmap_nonreject", False)
            ),
            "occ_memory_score_enable": bool(
                vlmap_safety_cfg.get("nextdit_candidate_occ_memory_score_enable", False)
            ),
            "occ_memory_score_max_points": max(
                2,
                int(vlmap_safety_cfg.get("nextdit_candidate_occ_memory_score_max_points", 33)),
            ),
        }

    def _write_nextdit_candidate_probe_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "nextdit_candidate_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_nextdit_active_rerank_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "nextdit_active_rerank_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _get_occ_memory_recovery_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("occ_memory_recovery_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("occ_memory_recovery_shadow_only", True)),
            "min_step": max(0, int(vlmap_safety_cfg.get("occ_memory_recovery_min_step", 30))),
            "occupied_stagnation_window_steps": max(
                1,
                int(vlmap_safety_cfg.get("occ_memory_recovery_occupied_stagnation_window_steps", 20)),
            ),
            "total_stagnation_window_steps": max(
                1,
                int(vlmap_safety_cfg.get("occ_memory_recovery_total_stagnation_window_steps", 20)),
            ),
            "displacement_window_steps": max(
                1,
                int(vlmap_safety_cfg.get("occ_memory_recovery_displacement_window_steps", 20)),
            ),
            "low_displacement_threshold_m": max(
                0.0,
                float(vlmap_safety_cfg.get("occ_memory_recovery_low_displacement_threshold_m", 0.35)),
            ),
            "require_low_displacement_for_map_stagnation": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_require_low_displacement_for_map_stagnation", True)
            ),
            "collision_trigger_enable": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_collision_trigger_enable", True)
            ),
            "total_map_stagnation_trigger_enable": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_total_map_stagnation_trigger_enable", False)
            ),
            "max_interventions_per_episode": int(
                vlmap_safety_cfg.get("occ_memory_recovery_max_interventions_per_episode", 1)
            ),
            "active_use_map_stagnation": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_active_use_map_stagnation", True)
            ),
            "active_use_collision_trigger": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_active_use_collision_trigger", False)
            ),
            "arrival_like_protection_enable": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_arrival_like_protection_enable", True)
            ),
            "arrival_like_radius_cells": max(
                1,
                int(vlmap_safety_cfg.get("occ_memory_recovery_arrival_like_radius_cells", 8)),
            ),
            "arrival_like_min_free_ratio": float(
                vlmap_safety_cfg.get("occ_memory_recovery_arrival_like_min_free_ratio", 0.35)
            ),
            "arrival_like_max_occupied_ratio": float(
                vlmap_safety_cfg.get("occ_memory_recovery_arrival_like_max_occupied_ratio", 0.04)
            ),
            "escape_probe_distance_m": max(
                0.10,
                float(vlmap_safety_cfg.get("occ_memory_recovery_escape_probe_distance_m", 0.75)),
            ),
            "escape_candidate_angles_deg": list(
                vlmap_safety_cfg.get("occ_memory_recovery_escape_candidate_angles_deg", [45.0, -45.0, 60.0, -60.0])
            ),
            "escape_max_turn_steps": max(
                1,
                int(vlmap_safety_cfg.get("occ_memory_recovery_escape_max_turn_steps", 3)),
            ),
            "escape_forward_steps": max(
                0,
                int(vlmap_safety_cfg.get("occ_memory_recovery_escape_forward_steps", 1)),
            ),
            "escape_allow_forward_only_if_free": bool(
                vlmap_safety_cfg.get("occ_memory_recovery_escape_allow_forward_only_if_free", True)
            ),
            "escape_clear_goal": bool(vlmap_safety_cfg.get("occ_memory_recovery_escape_clear_goal", True)),
            "log_every_step": bool(vlmap_safety_cfg.get("occ_memory_recovery_log_every_step", True)),
        }

    def _init_occ_memory_recovery_state(self) -> dict:
        return {
            "event_count": 0,
            "logged_event_count": 0,
            "recovery_trigger_event_count": 0,
            "recovery_trigger_start_count": 0,
            "map_stagnation_event_count": 0,
            "map_stagnation_start_count": 0,
            "total_map_stagnation_event_count": 0,
            "low_displacement_event_count": 0,
            "collision_trigger_event_count": 0,
            "collision_trigger_start_count": 0,
            "first_recovery_trigger_step": None,
            "first_map_stagnation_step": None,
            "first_collision_trigger_step": None,
            "max_occupied_stagnation_streak": 0,
            "max_total_stagnation_streak": 0,
            "max_collision_delta": 0.0,
            "max_pose_window_displacement_m": 0.0,
            "min_pose_window_displacement_m": None,
            "prev_occupied_cell_count": None,
            "prev_free_cell_count": None,
            "prev_collision_count": 0.0,
            "occupied_stagnation_streak": 0,
            "total_stagnation_streak": 0,
            "pose_history": [],
            "last_recovery_trigger": False,
            "last_map_stagnation": False,
            "last_collision_trigger": False,
            "active_intervention_count": 0,
            "active_applied_count": 0,
            "active_suppressed_count": 0,
            "active_first_step": None,
            "active_reason_counts": {},
            "last_event": None,
        }

    def _write_occ_memory_recovery_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "occ_memory_recovery_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _occ_memory_cell_state(self, row: int, col: int) -> str:
        try:
            return str(self.occ_memory._cell_state(int(row), int(col)))
        except Exception:
            return "unknown"

    def _occ_memory_local_surround_summary(self, cfg: dict) -> dict:
        pose_state = None
        try:
            pose_state = self.occ_memory._current_pose_state({})
        except Exception:
            pose_state = None
        result = {
            "valid": False,
            "reason": None,
        }
        if pose_state is None:
            result["reason"] = "missing_pose"
            return result
        grid = pose_state.get("grid") or []
        if len(grid) < 2:
            result["reason"] = "missing_grid"
            return result
        row0, col0 = int(grid[0]), int(grid[1])
        radius = max(1, int(cfg.get("arrival_like_radius_cells", 8)))
        occupied = 0
        free = 0
        unknown = 0
        checked = 0
        for row in range(row0 - radius, row0 + radius + 1):
            for col in range(col0 - radius, col0 + radius + 1):
                if row < 0 or row >= int(self.occ_memory.gs) or col < 0 or col >= int(self.occ_memory.gs):
                    continue
                if (row - row0) * (row - row0) + (col - col0) * (col - col0) > radius * radius:
                    continue
                checked += 1
                state = self._occ_memory_cell_state(row, col)
                if state == "occupied":
                    occupied += 1
                elif state == "free":
                    free += 1
                else:
                    unknown += 1
        free_ratio = float(free / checked) if checked else 0.0
        occupied_ratio = float(occupied / checked) if checked else 0.0
        unknown_ratio = float(unknown / checked) if checked else 1.0
        arrival_like = bool(
            checked > 0
            and free_ratio >= float(cfg.get("arrival_like_min_free_ratio", 0.35))
            and occupied_ratio <= float(cfg.get("arrival_like_max_occupied_ratio", 0.04))
        )
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "grid": [int(row0), int(col0)],
                "radius_cells": int(radius),
                "checked_cell_count": int(checked),
                "occupied_cell_count": int(occupied),
                "free_cell_count": int(free),
                "unknown_cell_count": int(unknown),
                "free_ratio": float(free_ratio),
                "occupied_ratio": float(occupied_ratio),
                "unknown_ratio": float(unknown_ratio),
                "arrival_like_free_space": bool(arrival_like),
            }
        )
        return result

    def _occ_memory_probe_line_summary(self, start_xy, yaw: float, rel_angle_deg: float, distance_m: float) -> dict:
        cs = max(0.05, float(getattr(self.occ_memory, "cs", 0.05)))
        sample_count = max(1, int(math.ceil(float(distance_m) / cs)))
        angle = float(yaw) + math.radians(float(rel_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cells = []
        seen = set()
        for idx in range(1, sample_count + 1):
            dist = float(distance_m) * float(idx) / float(sample_count)
            x = float(start_xy[0]) + cos_a * dist
            y = float(start_xy[1]) + sin_a * dist
            try:
                row, col = self.occ_memory._xy_to_grid_cell(x, y)
            except Exception:
                continue
            if row < 0 or row >= int(self.occ_memory.gs) or col < 0 or col >= int(self.occ_memory.gs):
                continue
            cell = (int(row), int(col))
            if cell in seen:
                continue
            seen.add(cell)
            cells.append(cell)
        occupied = 0
        free = 0
        unknown = 0
        occupied_preview = []
        for row, col in cells:
            state = self._occ_memory_cell_state(row, col)
            if state == "occupied":
                occupied += 1
                if len(occupied_preview) < 8:
                    occupied_preview.append([int(row), int(col)])
            elif state == "free":
                free += 1
            else:
                unknown += 1
        checked = len(cells)
        return {
            "checked_cell_count": int(checked),
            "occupied_hit_count": int(occupied),
            "free_hit_count": int(free),
            "unknown_hit_count": int(unknown),
            "occupied_hit_ratio": float(occupied / checked) if checked else 0.0,
            "free_hit_ratio": float(free / checked) if checked else 0.0,
            "unknown_hit_ratio": float(unknown / checked) if checked else 1.0,
            "occupied_cells_preview": occupied_preview,
        }

    def _occ_memory_escape_plan(self, cfg: dict) -> dict:
        pose_state = None
        try:
            pose_state = self.occ_memory._current_pose_state({})
        except Exception:
            pose_state = None
        result = {
            "valid": False,
            "reason": None,
            "actions": [],
            "candidates": [],
        }
        if pose_state is None:
            result["reason"] = "missing_pose"
            return result
        start_xy = pose_state.get("xy")
        if not start_xy or len(start_xy) < 2:
            result["reason"] = "missing_xy"
            return result
        yaw = float(pose_state.get("yaw", 0.0))
        turn_angle = max(1e-6, abs(float(getattr(self.config.habitat.simulator, "turn_angle", 15.0))))
        max_turn_steps = max(1, int(cfg.get("escape_max_turn_steps", 3)))
        probe_distance = float(cfg.get("escape_probe_distance_m", 0.75))
        forward_steps_cfg = max(0, int(cfg.get("escape_forward_steps", 1)))
        allow_forward_only_if_free = bool(cfg.get("escape_allow_forward_only_if_free", True))
        candidates = []
        seen_keys = set()
        for raw_angle in list(cfg.get("escape_candidate_angles_deg") or []):
            try:
                requested_angle = float(raw_angle)
            except (TypeError, ValueError):
                continue
            if abs(requested_angle) < 1e-6:
                continue
            turn_steps = int(math.ceil(abs(requested_angle) / turn_angle))
            turn_steps = max(1, min(max_turn_steps, turn_steps))
            sign = 1.0 if requested_angle > 0.0 else -1.0
            actual_angle = sign * float(turn_steps) * turn_angle
            turn_action = action_code.LEFT if actual_angle > 0.0 else action_code.RIGHT
            key = (int(turn_action), int(turn_steps))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            probe = self._occ_memory_probe_line_summary(start_xy, yaw, actual_angle, probe_distance)
            path_free = bool(probe.get("checked_cell_count", 0) > 0 and probe.get("occupied_hit_count", 0) == 0)
            forward_steps = int(forward_steps_cfg)
            if allow_forward_only_if_free and not path_free:
                forward_steps = 0
            actions = [int(turn_action)] * int(turn_steps)
            if forward_steps > 0:
                actions.extend([int(action_code.FORWARD)] * int(forward_steps))
            score = (
                10.0 * float(probe.get("occupied_hit_count", 0) or 0)
                + 0.40 * float(probe.get("unknown_hit_count", 0) or 0)
                - 0.15 * float(probe.get("free_hit_count", 0) or 0)
                + 0.05 * float(turn_steps)
            )
            candidates.append(
                {
                    "requested_angle_deg": float(requested_angle),
                    "actual_angle_deg": float(actual_angle),
                    "turn_action": int(turn_action),
                    "turn_steps": int(turn_steps),
                    "forward_steps": int(forward_steps),
                    "actions": actions,
                    "path_free": bool(path_free),
                    "score": float(score),
                    **probe,
                }
            )
        if not candidates:
            result["reason"] = "no_candidates"
            return result
        best = min(candidates, key=lambda item: float(item.get("score", 0.0)))
        actions = [int(item) for item in best.get("actions", []) if int(item) in (1, 2, 3)]
        if not actions:
            result["reason"] = "selected_empty_actions"
            result["candidates"] = candidates
            result["selected"] = best
            return result
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "actions": actions,
                "selected": best,
                "candidates": candidates,
                "probe_distance_m": float(probe_distance),
            }
        )
        return result

    def _update_occ_memory_recovery_shadow(
        self,
        state: dict,
        *,
        update_event: Optional[dict],
        metrics: Optional[dict],
        step_id: int,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        last_action,
        pixel_goal,
        local_actions,
        action_seq,
        vlmap_recovery_actions,
    ) -> Optional[dict]:
        cfg = self._get_occ_memory_recovery_cfg()
        if not cfg["enable"]:
            return None

        update_event = dict(update_event or {})
        try:
            occupied_count = int(update_event.get("occupied_cell_count"))
        except (TypeError, ValueError):
            occupied_count = int(len(getattr(self.occ_memory, "occ2d_counts", {}) or {}))
        try:
            free_count = int(update_event.get("free_cell_count"))
        except (TypeError, ValueError):
            free_count = int(len(getattr(self.occ_memory, "free2d_counts", {}) or {}))

        prev_occ = state.get("prev_occupied_cell_count")
        prev_free = state.get("prev_free_cell_count")
        if prev_occ is None:
            occupied_delta = 0
            free_delta = 0
            total_delta = 0
            state["occupied_stagnation_streak"] = 0
            state["total_stagnation_streak"] = 0
        else:
            occupied_delta = int(occupied_count - int(prev_occ))
            free_delta = int(free_count - int(prev_free or 0))
            total_delta = int(occupied_delta + free_delta)
            if occupied_delta <= 0:
                state["occupied_stagnation_streak"] = int(state.get("occupied_stagnation_streak", 0)) + 1
            else:
                state["occupied_stagnation_streak"] = 0
            if total_delta <= 0:
                state["total_stagnation_streak"] = int(state.get("total_stagnation_streak", 0)) + 1
            else:
                state["total_stagnation_streak"] = 0
        state["prev_occupied_cell_count"] = int(occupied_count)
        state["prev_free_cell_count"] = int(free_count)

        pose = None
        if getattr(self.occ_memory, "pose_trace", None):
            pose = self.occ_memory.pose_trace[-1]
        pose_xy = None
        pose_grid = None
        pose_yaw = None
        if pose:
            pose_xy = [float(pose.get("x", 0.0)), float(pose.get("y", 0.0))]
            pose_grid = [int(pose.get("row", 0)), int(pose.get("col", 0))]
            pose_yaw = float(pose.get("yaw", 0.0))
            state["pose_history"].append(
                {
                    "step_id": int(step_id),
                    "x": float(pose_xy[0]),
                    "y": float(pose_xy[1]),
                }
            )
        max_pose_history = int(cfg["displacement_window_steps"]) + 1
        if len(state["pose_history"]) > max_pose_history:
            state["pose_history"] = state["pose_history"][-max_pose_history:]

        pose_window_displacement = None
        pose_window_ready = len(state["pose_history"]) >= max_pose_history
        if pose_window_ready:
            first_pose = state["pose_history"][0]
            last_pose = state["pose_history"][-1]
            pose_window_displacement = float(
                math.hypot(float(last_pose["x"]) - float(first_pose["x"]), float(last_pose["y"]) - float(first_pose["y"]))
            )
            state["max_pose_window_displacement_m"] = max(
                float(state.get("max_pose_window_displacement_m", 0.0) or 0.0),
                float(pose_window_displacement),
            )
            min_disp = state.get("min_pose_window_displacement_m")
            if min_disp is None or pose_window_displacement < float(min_disp):
                state["min_pose_window_displacement_m"] = float(pose_window_displacement)
        low_displacement = bool(
            pose_window_ready
            and pose_window_displacement is not None
            and pose_window_displacement <= float(cfg["low_displacement_threshold_m"])
        )

        collision_summary = self._extract_collision_summary(metrics or {}, steps=max(1, int(step_id)))
        collision_count = float(collision_summary.get("collision_count", 0.0) or 0.0)
        prev_collision_count = float(state.get("prev_collision_count", 0.0) or 0.0)
        collision_delta = max(0.0, float(collision_count - prev_collision_count))
        state["prev_collision_count"] = float(collision_count)

        late_enough = int(step_id) >= int(cfg["min_step"])
        occupied_streak = int(state.get("occupied_stagnation_streak", 0) or 0)
        total_streak = int(state.get("total_stagnation_streak", 0) or 0)
        map_stagnation = bool(
            late_enough and occupied_streak >= int(cfg["occupied_stagnation_window_steps"])
        )
        total_map_stagnation = bool(
            late_enough and total_streak >= int(cfg["total_stagnation_window_steps"])
        )
        map_stagnation_recovery_gate = bool(
            map_stagnation
            and (
                not bool(cfg["require_low_displacement_for_map_stagnation"])
                or low_displacement
            )
        )
        collision_trigger = bool(cfg["collision_trigger_enable"] and collision_delta > 0.0)
        total_map_stagnation_trigger = bool(
            cfg["total_map_stagnation_trigger_enable"]
            and total_map_stagnation
            and (
                not bool(cfg["require_low_displacement_for_map_stagnation"])
                or low_displacement
            )
        )
        recovery_trigger = bool(
            collision_trigger
            or map_stagnation_recovery_gate
            or total_map_stagnation_trigger
        )
        local_surround = self._occ_memory_local_surround_summary(cfg)
        arrival_like_free_space = bool(
            cfg.get("arrival_like_protection_enable", True)
            and local_surround.get("valid")
            and local_surround.get("arrival_like_free_space")
            and map_stagnation
            and collision_count <= 0.0
            and not collision_trigger
        )
        escape_plan = self._occ_memory_escape_plan(cfg)
        active_signal_allowed = bool(
            (
                bool(cfg.get("active_use_map_stagnation", True))
                and (map_stagnation_recovery_gate or total_map_stagnation_trigger)
            )
            or (
                bool(cfg.get("active_use_collision_trigger", False))
                and collision_trigger
            )
        )
        active_recovery_allowed = bool(
            recovery_trigger
            and active_signal_allowed
            and not arrival_like_free_space
            and escape_plan.get("valid")
            and bool(escape_plan.get("actions"))
        )
        trigger_started = bool(recovery_trigger and not state.get("last_recovery_trigger", False))
        map_stagnation_started = bool(map_stagnation and not state.get("last_map_stagnation", False))
        collision_trigger_started = bool(collision_trigger and not state.get("last_collision_trigger", False))

        state["event_count"] = int(state.get("event_count", 0)) + 1
        if recovery_trigger:
            state["recovery_trigger_event_count"] = int(state.get("recovery_trigger_event_count", 0)) + 1
            if state.get("first_recovery_trigger_step") is None:
                state["first_recovery_trigger_step"] = int(step_id)
        if trigger_started:
            state["recovery_trigger_start_count"] = int(state.get("recovery_trigger_start_count", 0)) + 1
        if map_stagnation:
            state["map_stagnation_event_count"] = int(state.get("map_stagnation_event_count", 0)) + 1
            if state.get("first_map_stagnation_step") is None:
                state["first_map_stagnation_step"] = int(step_id)
        if map_stagnation_started:
            state["map_stagnation_start_count"] = int(state.get("map_stagnation_start_count", 0)) + 1
        if total_map_stagnation:
            state["total_map_stagnation_event_count"] = int(state.get("total_map_stagnation_event_count", 0)) + 1
        if low_displacement:
            state["low_displacement_event_count"] = int(state.get("low_displacement_event_count", 0)) + 1
        if collision_trigger:
            state["collision_trigger_event_count"] = int(state.get("collision_trigger_event_count", 0)) + 1
            if state.get("first_collision_trigger_step") is None:
                state["first_collision_trigger_step"] = int(step_id)
        if collision_trigger_started:
            state["collision_trigger_start_count"] = int(state.get("collision_trigger_start_count", 0)) + 1
        state["max_occupied_stagnation_streak"] = max(
            int(state.get("max_occupied_stagnation_streak", 0) or 0),
            int(occupied_streak),
        )
        state["max_total_stagnation_streak"] = max(
            int(state.get("max_total_stagnation_streak", 0) or 0),
            int(total_streak),
        )
        state["max_collision_delta"] = max(
            float(state.get("max_collision_delta", 0.0) or 0.0),
            float(collision_delta),
        )

        event = {
            "event_type": "occ_memory_recovery_shadow",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "step_id": int(step_id),
            "enabled": bool(cfg["enable"]),
            "shadow_only": bool(cfg["shadow_only"]),
            "late_enough": bool(late_enough),
            "occupied_cell_count": int(occupied_count),
            "free_cell_count": int(free_count),
            "occupied_cell_delta": int(occupied_delta),
            "free_cell_delta": int(free_delta),
            "total_cell_delta": int(total_delta),
            "occupied_stagnation_streak": int(occupied_streak),
            "total_stagnation_streak": int(total_streak),
            "map_stagnation": bool(map_stagnation),
            "map_stagnation_started": bool(map_stagnation_started),
            "total_map_stagnation": bool(total_map_stagnation),
            "low_displacement": bool(low_displacement),
            "pose_window_ready": bool(pose_window_ready),
            "pose_window_displacement_m": (
                None if pose_window_displacement is None else float(pose_window_displacement)
            ),
            "pose_grid": pose_grid,
            "pose_xy": pose_xy,
            "pose_yaw": pose_yaw,
            "collision_count": float(collision_count),
            "collision_delta": float(collision_delta),
            "collision_trigger": bool(collision_trigger),
            "collision_trigger_started": bool(collision_trigger_started),
            "map_stagnation_recovery_gate": bool(map_stagnation_recovery_gate),
            "total_map_stagnation_trigger": bool(total_map_stagnation_trigger),
            "recovery_trigger": bool(recovery_trigger),
            "recovery_trigger_started": bool(trigger_started),
            "arrival_like_free_space": bool(arrival_like_free_space),
            "local_surround": local_surround,
            "escape_plan": escape_plan,
            "active_signal_allowed": bool(active_signal_allowed),
            "active_recovery_allowed": bool(active_recovery_allowed),
            "last_action": None if last_action is None else int(last_action),
            "pixel_goal_active": pixel_goal is not None,
            "local_action_count": len(list(local_actions or [])),
            "action_seq_count": len(list(action_seq or [])),
            "vlmap_recovery_action_count": len(list(vlmap_recovery_actions or [])),
            "config": {
                "min_step": int(cfg["min_step"]),
                "occupied_stagnation_window_steps": int(cfg["occupied_stagnation_window_steps"]),
                "total_stagnation_window_steps": int(cfg["total_stagnation_window_steps"]),
                "displacement_window_steps": int(cfg["displacement_window_steps"]),
                "low_displacement_threshold_m": float(cfg["low_displacement_threshold_m"]),
                "require_low_displacement_for_map_stagnation": bool(
                    cfg["require_low_displacement_for_map_stagnation"]
                ),
                "collision_trigger_enable": bool(cfg["collision_trigger_enable"]),
                "total_map_stagnation_trigger_enable": bool(cfg["total_map_stagnation_trigger_enable"]),
                "active_use_map_stagnation": bool(cfg["active_use_map_stagnation"]),
                "active_use_collision_trigger": bool(cfg["active_use_collision_trigger"]),
                "arrival_like_protection_enable": bool(cfg["arrival_like_protection_enable"]),
            },
        }
        state["last_recovery_trigger"] = bool(recovery_trigger)
        state["last_map_stagnation"] = bool(map_stagnation)
        state["last_collision_trigger"] = bool(collision_trigger)
        state["last_event"] = event
        if bool(cfg["log_every_step"]) or recovery_trigger or trigger_started:
            self._write_occ_memory_recovery_event(event)
            state["logged_event_count"] = int(state.get("logged_event_count", 0)) + 1
        return event

    def _summarize_occ_memory_recovery_state(self, state: dict) -> dict:
        if not state:
            return {}
        return {
            "event_count": int(state.get("event_count", 0) or 0),
            "logged_event_count": int(state.get("logged_event_count", 0) or 0),
            "recovery_trigger_event_count": int(state.get("recovery_trigger_event_count", 0) or 0),
            "recovery_trigger_start_count": int(state.get("recovery_trigger_start_count", 0) or 0),
            "first_recovery_trigger_step": state.get("first_recovery_trigger_step"),
            "map_stagnation_event_count": int(state.get("map_stagnation_event_count", 0) or 0),
            "map_stagnation_start_count": int(state.get("map_stagnation_start_count", 0) or 0),
            "first_map_stagnation_step": state.get("first_map_stagnation_step"),
            "total_map_stagnation_event_count": int(state.get("total_map_stagnation_event_count", 0) or 0),
            "low_displacement_event_count": int(state.get("low_displacement_event_count", 0) or 0),
            "collision_trigger_event_count": int(state.get("collision_trigger_event_count", 0) or 0),
            "collision_trigger_start_count": int(state.get("collision_trigger_start_count", 0) or 0),
            "first_collision_trigger_step": state.get("first_collision_trigger_step"),
            "max_occupied_stagnation_streak": int(state.get("max_occupied_stagnation_streak", 0) or 0),
            "max_total_stagnation_streak": int(state.get("max_total_stagnation_streak", 0) or 0),
            "max_collision_delta": float(state.get("max_collision_delta", 0.0) or 0.0),
            "max_pose_window_displacement_m": float(
                state.get("max_pose_window_displacement_m", 0.0) or 0.0
            ),
            "min_pose_window_displacement_m": state.get("min_pose_window_displacement_m"),
            "active_intervention_count": int(state.get("active_intervention_count", 0) or 0),
            "active_applied_count": int(state.get("active_applied_count", 0) or 0),
            "active_suppressed_count": int(state.get("active_suppressed_count", 0) or 0),
            "active_first_step": state.get("active_first_step"),
            "active_reason_counts": dict(state.get("active_reason_counts") or {}),
        }

    def _get_failure_prediction_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(vlmap_safety_cfg.get("failure_prediction_enable", False)),
            "shadow_only": bool(vlmap_safety_cfg.get("failure_prediction_shadow_only", True)),
            "min_step": max(0, int(vlmap_safety_cfg.get("failure_prediction_min_step", 30))),
            "window_steps": max(1, int(vlmap_safety_cfg.get("failure_prediction_window_steps", 20))),
            "threshold": float(vlmap_safety_cfg.get("failure_prediction_threshold", 0.65)),
            "stagnation_weight": float(vlmap_safety_cfg.get("failure_prediction_stagnation_score_weight", 0.40)),
            "semantic_weight": float(vlmap_safety_cfg.get("failure_prediction_semantic_weight", 0.30)),
            "collision_weight": float(vlmap_safety_cfg.get("failure_prediction_collision_weight", 0.20)),
            "displacement_weight": float(vlmap_safety_cfg.get("failure_prediction_displacement_weight", 0.10)),
            "map_growth_weight": float(vlmap_safety_cfg.get("failure_prediction_map_growth_weight", 0.15)),
            "unsafe_waypoint_weight": float(vlmap_safety_cfg.get("failure_prediction_unsafe_waypoint_weight", 0.15)),
            "direction_enable": bool(vlmap_safety_cfg.get("failure_prediction_direction_enable", False)),
            "pg_ecc_weight": float(vlmap_safety_cfg.get("failure_prediction_pg_ecc_weight", 0.0)),
            "compass_reversal_weight": float(
                vlmap_safety_cfg.get("failure_prediction_compass_reversal_weight", 0.0)
            ),
            "heading_var_weight": float(vlmap_safety_cfg.get("failure_prediction_heading_var_weight", 0.0)),
            "pg_ecc_threshold": float(vlmap_safety_cfg.get("failure_prediction_pg_ecc_threshold", 0.30)),
            "pg_ecc_norm": max(
                1e-6,
                float(vlmap_safety_cfg.get("failure_prediction_pg_ecc_norm", 0.30)),
            ),
            "compass_reversal_max": max(
                1.0,
                float(vlmap_safety_cfg.get("failure_prediction_compass_reversal_max", 4.0)),
            ),
            "direction_image_width": max(
                1.0,
                float(vlmap_safety_cfg.get("failure_prediction_direction_image_width", 640.0)),
            ),
            "direction_cache_max_events": max(
                16,
                int(vlmap_safety_cfg.get("failure_prediction_direction_cache_max_events", 256)),
            ),
            "stagnation_streak_scale": max(
                1.0,
                float(vlmap_safety_cfg.get("failure_prediction_stagnation_streak_scale", 30.0)),
            ),
            "low_map_growth_norm": max(
                1.0,
                float(vlmap_safety_cfg.get("failure_prediction_low_map_growth_norm", 120.0)),
            ),
            "displacement_norm_m": max(
                0.01,
                float(vlmap_safety_cfg.get("failure_prediction_displacement_norm_m", 1.25)),
            ),
            "collision_norm": max(
                1.0,
                float(vlmap_safety_cfg.get("failure_prediction_collision_norm", 2.0)),
            ),
            "min_explore_efficiency": max(
                0.0,
                float(vlmap_safety_cfg.get("failure_prediction_min_explore_efficiency", 20.0)),
            ),
            "log_every_step": bool(vlmap_safety_cfg.get("failure_prediction_log_every_step", True)),
            "predictor_version": str(vlmap_safety_cfg.get("failure_prediction_version", "stage14a_rule_v1")),
        }

    def _init_failure_prediction_state(self) -> dict:
        return {
            "event_count": 0,
            "logged_event_count": 0,
            "predicted_event_count": 0,
            "prediction_start_count": 0,
            "first_predicted_step": None,
            "max_failure_score": 0.0,
            "max_stagnation_score": 0.0,
            "max_semantic_score": 0.0,
            "max_collision_score": 0.0,
            "max_displacement_score": 0.0,
            "max_pg_ecc_score": 0.0,
            "max_compass_reversal_score": 0.0,
            "max_heading_var_score": 0.0,
            "mode_hint_counts": {},
            "samples": [],
            "traj_cache": [],
            "last_predicted": False,
            "last_event": None,
        }

    def _write_failure_prediction_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "failure_prediction_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _failure_prediction_semantic_snapshot(self, step_id: int) -> dict:
        semantic_events = list(getattr(self.occ_memory, "semantic_events", []) or [])
        high_conf_events = [event for event in semantic_events if event.get("high_conf_semantic")]
        latest = dict(getattr(self.occ_memory, "last_semantic_decision", {}) or {})
        recent_terms = latest.get("stagnation_recent_terms")
        if recent_terms is None and semantic_events:
            recent_terms = semantic_events[-1].get("stagnation_recent_terms")

        waypoint_events = list(getattr(self.occ_memory, "waypoint_events", []) or [])
        latest_waypoint = waypoint_events[-1] if waypoint_events else {}
        dead_zone_score = latest_waypoint.get("semantic_dead_zone_score")
        if dead_zone_score is None:
            dead_zone_score = latest.get("semantic_dead_zone_score")
        try:
            dead_zone_score = float(dead_zone_score)
        except (TypeError, ValueError):
            dead_zone_score = 0.0

        return {
            "step_id": int(step_id),
            "semantic_event_count": int(len(semantic_events)),
            "semantic_high_conf_count": int(len(high_conf_events)),
            "latest_top_match": latest.get("top_match"),
            "latest_top_score": latest.get("top_score"),
            "latest_high_conf_semantic": bool(latest.get("high_conf_semantic")),
            "semantic_stagnation": bool(
                latest.get("stagnation_would_requery")
                or latest_waypoint.get("semantic_stagnation_active")
                or latest_waypoint.get("semantic_last_stagnation")
            ),
            "semantic_recent_unique_count": latest.get("stagnation_recent_unique_count"),
            "semantic_recent_terms": recent_terms,
            "semantic_dead_zone": bool(latest_waypoint.get("semantic_dead_zone")),
            "semantic_dead_zone_score": float(dead_zone_score),
        }

    def _failure_prediction_window_samples(self, samples: list, step_id: int, window_steps: int) -> list:
        min_step = int(step_id) - int(window_steps)
        return [sample for sample in samples if int(sample.get("step_id", 0)) >= min_step]

    def _failure_prediction_float_list(
        self,
        value,
        *,
        max_len: Optional[int] = None,
    ) -> Optional[list]:
        value = self._jsonable(value)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = [value]
        if not isinstance(value, list):
            return None
        if max_len is not None:
            value = value[:max_len]
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out or None

    def _append_failure_prediction_trajectory_cache(
        self,
        state: Optional[dict],
        *,
        step_id: int,
        pixel_goal,
        observations: dict,
    ) -> None:
        if not state or pixel_goal is None:
            return
        cfg = self._get_failure_prediction_cfg()
        if not cfg.get("enable") or not cfg.get("direction_enable"):
            return

        pg = self._failure_prediction_float_list(pixel_goal, max_len=2)
        if not pg or len(pg) < 2:
            return
        gps = self._failure_prediction_float_list((observations or {}).get("gps"), max_len=2)
        compass = self._failure_prediction_float_list((observations or {}).get("compass"), max_len=1)
        event = {
            "eval_step": int(step_id),
            "pixel_goal": [float(pg[0]), float(pg[1])],
            "gps": gps,
            "compass": compass,
        }
        cache = list(state.get("traj_cache") or [])
        cache.append(event)
        max_events = int(cfg.get("direction_cache_max_events", 256))
        state["traj_cache"] = cache[-max_events:]

    def _failure_prediction_direction_features(
        self,
        *,
        step_id: int,
        state: dict,
        cfg: dict,
    ) -> dict:
        if not cfg.get("direction_enable"):
            return {
                "trajectory_event_count_w": 0,
                "pg_ecc_mean_w": 0.0,
                "compass_reversal_count_w": 0,
                "heading_variance_w": 0.0,
            }

        window_steps = int(cfg.get("window_steps", 20))
        min_step = int(step_id) - window_steps
        traj_cache = list(state.get("traj_cache") or [])
        traj_window = []
        for event in traj_cache:
            try:
                event_step = int(event.get("eval_step", -1))
            except (TypeError, ValueError):
                continue
            if min_step <= event_step <= int(step_id):
                traj_window.append(event)
        traj_window.sort(key=lambda item: int(item.get("eval_step", -1)))

        image_width = float(cfg.get("direction_image_width", 640.0))
        center_x = image_width * 0.5
        half_width = max(1.0, image_width * 0.5)
        pg_eccentricities = []
        for event in traj_window:
            pg = event.get("pixel_goal")
            if not pg or len(pg) < 1:
                continue
            try:
                pg_x = float(pg[0])
            except (TypeError, ValueError):
                continue
            pg_eccentricities.append(min(1.0, max(0.0, abs(pg_x - center_x) / half_width)))
        pg_ecc_mean = (
            float(sum(pg_eccentricities) / len(pg_eccentricities))
            if pg_eccentricities
            else 0.0
        )

        compass_vals = []
        for event in traj_window:
            compass = event.get("compass")
            if not compass or len(compass) < 1:
                continue
            try:
                compass_vals.append(float(compass[0]))
            except (TypeError, ValueError):
                continue
        compass_reversals = 0
        for i in range(1, len(compass_vals)):
            if compass_vals[i - 1] * compass_vals[i] < 0:
                compass_reversals += 1

        gps_pairs = []
        for event in traj_window:
            gps = event.get("gps")
            if not gps or len(gps) < 2:
                continue
            try:
                gps_pairs.append(
                    (
                        int(event.get("eval_step", -1)),
                        [float(gps[0]), float(gps[1])],
                    )
                )
            except (TypeError, ValueError):
                continue
        gps_pairs.sort(key=lambda item: item[0])
        headings = []
        for i in range(1, len(gps_pairs)):
            dx = gps_pairs[i][1][0] - gps_pairs[i - 1][1][0]
            dz = gps_pairs[i][1][1] - gps_pairs[i - 1][1][1]
            if math.hypot(dx, dz) > 1e-4:
                headings.append(math.atan2(dx, dz))
        heading_variance = 0.0
        if len(headings) >= 2:
            sin_m = sum(math.sin(h) for h in headings) / len(headings)
            cos_m = sum(math.cos(h) for h in headings) / len(headings)
            heading_variance = 1.0 - math.sqrt(sin_m * sin_m + cos_m * cos_m)
            heading_variance = min(1.0, max(0.0, float(heading_variance)))

        return {
            "trajectory_event_count_w": int(len(traj_window)),
            "pg_ecc_mean_w": float(pg_ecc_mean),
            "compass_reversal_count_w": int(compass_reversals),
            "heading_variance_w": float(heading_variance),
        }

    def _failure_prediction_score_from_history(
        self,
        *,
        step_id: int,
        state: dict,
        cfg: dict,
    ) -> dict:
        samples = list(state.get("samples") or [])
        if not samples:
            return {
                "failure_score": 0.0,
                "failure_predicted": False,
                "failure_mode_hint": "insufficient_history",
                "signal_breakdown": {},
                "features": {},
            }

        window_steps = int(cfg.get("window_steps", 20))
        window = self._failure_prediction_window_samples(samples, step_id, window_steps)
        if not window:
            window = [samples[-1]]
        first = window[0]
        last = window[-1]
        window_span = max(1, int(last.get("step_id", step_id)) - int(first.get("step_id", step_id)))

        occ_growth = int(last.get("occupied_cell_count", 0) or 0) - int(
            first.get("occupied_cell_count", 0) or 0
        )
        free_growth = int(last.get("free_cell_count", 0) or 0) - int(first.get("free_cell_count", 0) or 0)
        total_growth = occ_growth + free_growth
        collision_sum = float(sum(float(item.get("collision_delta", 0.0) or 0.0) for item in window))
        collision_rate = float(collision_sum / max(1, len(window)))

        displacement_total = 0.0
        prev_xy = None
        for item in window:
            xy = item.get("pose_xy")
            if not xy or len(xy) < 2:
                continue
            xy = [float(xy[0]), float(xy[1])]
            if prev_xy is not None:
                displacement_total += float(math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]))
            prev_xy = xy

        semantic_new_events = int(last.get("semantic_event_count", 0) or 0) - int(
            first.get("semantic_event_count", 0) or 0
        )
        high_conf_count = int(last.get("semantic_high_conf_count", 0) or 0)
        semantic_dead_zone_score = float(last.get("semantic_dead_zone_score", 0.0) or 0.0)
        semantic_stagnation = bool(last.get("semantic_stagnation"))

        waypoint_window = []
        try:
            min_step = int(step_id) - window_steps
            for event in list(getattr(self.occ_memory, "waypoint_events", []) or []):
                event_step = event.get("step_id")
                if event_step is None or int(event_step) < min_step or int(event_step) > int(step_id):
                    continue
                waypoint_window.append(event)
        except Exception:
            waypoint_window = []
        unsafe_waypoint_count = 0
        for event in waypoint_window:
            goal_state = str(event.get("goal_state") or "")
            if goal_state in ("occupied", "unknown") or event.get("semantic_dead_zone"):
                unsafe_waypoint_count += 1
        unsafe_waypoint_ratio = float(unsafe_waypoint_count / max(1, len(waypoint_window)))

        stagnation_streak = int(last.get("occupied_stagnation_streak", 0) or 0)
        stagnation_score = min(
            1.0,
            max(0.0, float(stagnation_streak) / float(cfg.get("stagnation_streak_scale", 30.0))),
        )
        low_growth_score = 1.0 - min(
            1.0,
            max(0.0, float(max(0, total_growth)) / float(cfg.get("low_map_growth_norm", 120.0))),
        )
        displacement_score = 1.0 - min(
            1.0,
            max(0.0, float(displacement_total) / float(cfg.get("displacement_norm_m", 1.25))),
        )
        collision_score = min(
            1.0,
            max(0.0, float(collision_sum) / float(cfg.get("collision_norm", 2.0))),
        )
        semantic_no_new_score = 1.0 if int(step_id) >= int(cfg.get("min_step", 30)) and semantic_new_events <= 0 else 0.0
        semantic_score = max(
            1.0 if semantic_stagnation else 0.0,
            float(semantic_dead_zone_score),
            0.50 * semantic_no_new_score if high_conf_count <= 0 else 0.0,
        )
        explore_efficiency = float(max(0, occ_growth) / max(0.01, displacement_total))
        inefficient_explore_score = 1.0 - min(
            1.0,
            explore_efficiency / max(1e-6, float(cfg.get("min_explore_efficiency", 20.0))),
        )
        map_signal = max(float(low_growth_score), float(inefficient_explore_score))
        direction_features = self._failure_prediction_direction_features(
            step_id=step_id,
            state=state,
            cfg=cfg,
        )
        pg_ecc_mean = float(direction_features.get("pg_ecc_mean_w", 0.0) or 0.0)
        pg_ecc_score = min(
            1.0,
            max(
                0.0,
                (pg_ecc_mean - float(cfg.get("pg_ecc_threshold", 0.30)))
                / float(cfg.get("pg_ecc_norm", 0.30)),
            ),
        )
        compass_reversal_count = int(direction_features.get("compass_reversal_count_w", 0) or 0)
        compass_reversal_score = min(
            1.0,
            max(
                0.0,
                float(compass_reversal_count) / float(cfg.get("compass_reversal_max", 4.0)),
            ),
        )
        heading_var_score = min(
            1.0,
            max(0.0, float(direction_features.get("heading_variance_w", 0.0) or 0.0)),
        )
        direction_signal = max(
            float(pg_ecc_score),
            float(compass_reversal_score),
            float(heading_var_score),
        )

        raw_score = (
            float(cfg.get("stagnation_weight", 0.40)) * float(stagnation_score)
            + float(cfg.get("semantic_weight", 0.30)) * float(semantic_score)
            + float(cfg.get("collision_weight", 0.20)) * float(collision_score)
            + float(cfg.get("displacement_weight", 0.10)) * float(displacement_score)
            + float(cfg.get("map_growth_weight", 0.15)) * float(map_signal)
            + float(cfg.get("unsafe_waypoint_weight", 0.15)) * float(unsafe_waypoint_ratio)
            + float(cfg.get("pg_ecc_weight", 0.0)) * float(pg_ecc_score)
            + float(cfg.get("compass_reversal_weight", 0.0)) * float(compass_reversal_score)
            + float(cfg.get("heading_var_weight", 0.0)) * float(heading_var_score)
        )
        normalizer = max(
            1e-6,
            float(cfg.get("stagnation_weight", 0.40))
            + float(cfg.get("semantic_weight", 0.30))
            + float(cfg.get("collision_weight", 0.20))
            + float(cfg.get("displacement_weight", 0.10))
            + float(cfg.get("map_growth_weight", 0.15))
            + float(cfg.get("unsafe_waypoint_weight", 0.15))
            + float(cfg.get("pg_ecc_weight", 0.0))
            + float(cfg.get("compass_reversal_weight", 0.0))
            + float(cfg.get("heading_var_weight", 0.0)),
        )
        failure_score = min(1.0, max(0.0, float(raw_score / normalizer)))
        failure_predicted = bool(
            int(step_id) >= int(cfg.get("min_step", 30))
            and failure_score >= float(cfg.get("threshold", 0.65))
        )

        if stagnation_score >= 0.65 and displacement_score >= 0.50:
            mode_hint = "stuck"
        elif collision_score >= 0.50:
            mode_hint = "collision_risk"
        elif direction_signal >= 0.65:
            mode_hint = "direction_confusion"
        elif semantic_score >= 0.60 or unsafe_waypoint_ratio >= 0.50:
            mode_hint = "lost_or_wrong_direction"
        elif map_signal >= 0.65 and displacement_score >= 0.35:
            mode_hint = "low_progress"
        else:
            mode_hint = "navigating"

        features = {
            "window_steps": int(window_steps),
            "window_span_steps": int(window_span),
            "occ_growth_last_w": int(occ_growth),
            "free_growth_last_w": int(free_growth),
            "total_growth_last_w": int(total_growth),
            "displacement_total_w": float(displacement_total),
            "collision_sum_w": float(collision_sum),
            "collision_rate_w": float(collision_rate),
            "semantic_new_events_w": int(semantic_new_events),
            "semantic_high_conf_count_t": int(high_conf_count),
            "semantic_dead_zone_score_t": float(semantic_dead_zone_score),
            "semantic_stagnation_t": bool(semantic_stagnation),
            "unsafe_waypoint_count_w": int(unsafe_waypoint_count),
            "waypoint_count_w": int(len(waypoint_window)),
            "unsafe_waypoint_ratio_w": float(unsafe_waypoint_ratio),
            "step_fraction": float(step_id / max(1, int(self.max_steps_per_episode))),
            "explore_efficiency": float(explore_efficiency),
            "trajectory_event_count_w": int(direction_features.get("trajectory_event_count_w", 0) or 0),
            "pg_ecc_mean_w": float(pg_ecc_mean),
            "compass_reversal_count_w": int(compass_reversal_count),
            "heading_variance_w": float(direction_features.get("heading_variance_w", 0.0) or 0.0),
        }
        signal_breakdown = {
            "stagnation_score": float(stagnation_score),
            "semantic_score": float(semantic_score),
            "collision_score": float(collision_score),
            "displacement_score": float(displacement_score),
            "low_map_growth_score": float(low_growth_score),
            "inefficient_explore_score": float(inefficient_explore_score),
            "map_signal": float(map_signal),
            "unsafe_waypoint_score": float(unsafe_waypoint_ratio),
            "pg_ecc_score": float(pg_ecc_score),
            "compass_reversal_score": float(compass_reversal_score),
            "heading_var_score": float(heading_var_score),
            "direction_signal": float(direction_signal),
        }
        return {
            "failure_score": float(failure_score),
            "failure_predicted": bool(failure_predicted),
            "failure_mode_hint": mode_hint,
            "signal_breakdown": signal_breakdown,
            "features": features,
        }

    def _update_failure_prediction_shadow(
        self,
        *,
        state: dict,
        occ_event: Optional[dict],
        step_id: int,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
    ) -> Optional[dict]:
        cfg = self._get_failure_prediction_cfg()
        if not cfg.get("enable"):
            return None
        occ_event = dict(occ_event or {})
        semantic_snapshot = self._failure_prediction_semantic_snapshot(step_id)
        sample = {
            "step_id": int(step_id),
            "occupied_cell_count": int(occ_event.get("occupied_cell_count", 0) or 0),
            "free_cell_count": int(occ_event.get("free_cell_count", 0) or 0),
            "occupied_stagnation_streak": int(occ_event.get("occupied_stagnation_streak", 0) or 0),
            "total_stagnation_streak": int(occ_event.get("total_stagnation_streak", 0) or 0),
            "collision_delta": float(occ_event.get("collision_delta", 0.0) or 0.0),
            "collision_count": float(occ_event.get("collision_count", 0.0) or 0.0),
            "pose_xy": occ_event.get("pose_xy"),
            **semantic_snapshot,
        }
        samples = list(state.get("samples") or [])
        samples.append(sample)
        max_samples = max(64, int(cfg.get("window_steps", 20)) * 4 + 8)
        state["samples"] = samples[-max_samples:]

        score = self._failure_prediction_score_from_history(step_id=step_id, state=state, cfg=cfg)
        predicted = bool(score.get("failure_predicted"))
        started = bool(predicted and not bool(state.get("last_predicted")))
        state["event_count"] = int(state.get("event_count", 0) or 0) + 1
        if predicted:
            state["predicted_event_count"] = int(state.get("predicted_event_count", 0) or 0) + 1
        if started:
            state["prediction_start_count"] = int(state.get("prediction_start_count", 0) or 0) + 1
            if state.get("first_predicted_step") is None:
                state["first_predicted_step"] = int(step_id)

        failure_score = float(score.get("failure_score", 0.0) or 0.0)
        breakdown = dict(score.get("signal_breakdown") or {})
        state["max_failure_score"] = max(float(state.get("max_failure_score", 0.0) or 0.0), failure_score)
        state["max_stagnation_score"] = max(
            float(state.get("max_stagnation_score", 0.0) or 0.0),
            float(breakdown.get("stagnation_score", 0.0) or 0.0),
        )
        state["max_semantic_score"] = max(
            float(state.get("max_semantic_score", 0.0) or 0.0),
            float(breakdown.get("semantic_score", 0.0) or 0.0),
        )
        state["max_collision_score"] = max(
            float(state.get("max_collision_score", 0.0) or 0.0),
            float(breakdown.get("collision_score", 0.0) or 0.0),
        )
        state["max_displacement_score"] = max(
            float(state.get("max_displacement_score", 0.0) or 0.0),
            float(breakdown.get("displacement_score", 0.0) or 0.0),
        )
        state["max_pg_ecc_score"] = max(
            float(state.get("max_pg_ecc_score", 0.0) or 0.0),
            float(breakdown.get("pg_ecc_score", 0.0) or 0.0),
        )
        state["max_compass_reversal_score"] = max(
            float(state.get("max_compass_reversal_score", 0.0) or 0.0),
            float(breakdown.get("compass_reversal_score", 0.0) or 0.0),
        )
        state["max_heading_var_score"] = max(
            float(state.get("max_heading_var_score", 0.0) or 0.0),
            float(breakdown.get("heading_var_score", 0.0) or 0.0),
        )
        mode_hint = str(score.get("failure_mode_hint") or "unknown")
        mode_counts = dict(state.get("mode_hint_counts") or {})
        mode_counts[mode_hint] = int(mode_counts.get(mode_hint, 0) or 0) + 1
        state["mode_hint_counts"] = mode_counts

        event = {
            "event_type": "failure_prediction_shadow",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "step_id": int(step_id),
            "enabled": True,
            "shadow_only": bool(cfg.get("shadow_only", True)),
            "predictor_version": cfg.get("predictor_version"),
            "min_step": int(cfg.get("min_step", 30)),
            "window_steps": int(cfg.get("window_steps", 20)),
            "threshold": float(cfg.get("threshold", 0.65)),
            "failure_score": failure_score,
            "failure_predicted": bool(predicted),
            "failure_prediction_started": bool(started),
            "failure_mode_hint": mode_hint,
            "signal_breakdown": breakdown,
            "features": score.get("features") or {},
            "source": {
                "occ_event_valid": bool(occ_event),
                "semantic_event_count": int(semantic_snapshot.get("semantic_event_count", 0) or 0),
                "semantic_high_conf_count": int(semantic_snapshot.get("semantic_high_conf_count", 0) or 0),
            },
        }
        state["last_predicted"] = bool(predicted)
        state["last_event"] = event
        if bool(cfg.get("log_every_step", True)) or predicted or started:
            self._write_failure_prediction_event(event)
            state["logged_event_count"] = int(state.get("logged_event_count", 0) or 0) + 1
        return event

    def _summarize_failure_prediction_state(self, state: dict) -> dict:
        if not state:
            return {}
        return {
            "event_count": int(state.get("event_count", 0) or 0),
            "logged_event_count": int(state.get("logged_event_count", 0) or 0),
            "predicted_event_count": int(state.get("predicted_event_count", 0) or 0),
            "prediction_start_count": int(state.get("prediction_start_count", 0) or 0),
            "first_predicted_step": state.get("first_predicted_step"),
            "max_failure_score": float(state.get("max_failure_score", 0.0) or 0.0),
            "max_stagnation_score": float(state.get("max_stagnation_score", 0.0) or 0.0),
            "max_semantic_score": float(state.get("max_semantic_score", 0.0) or 0.0),
            "max_collision_score": float(state.get("max_collision_score", 0.0) or 0.0),
            "max_displacement_score": float(state.get("max_displacement_score", 0.0) or 0.0),
            "max_pg_ecc_score": float(state.get("max_pg_ecc_score", 0.0) or 0.0),
            "max_compass_reversal_score": float(
                state.get("max_compass_reversal_score", 0.0) or 0.0
            ),
            "max_heading_var_score": float(state.get("max_heading_var_score", 0.0) or 0.0),
            "mode_hint_counts": dict(state.get("mode_hint_counts") or {}),
        }

    def _record_occ_memory_recovery_active_reason(self, state: dict, reason: str) -> None:
        counts = dict(state.get("active_reason_counts") or {})
        counts[str(reason)] = int(counts.get(str(reason), 0)) + 1
        state["active_reason_counts"] = counts

    def _maybe_apply_occ_memory_recovery(
        self,
        state: dict,
        event: Optional[dict],
        *,
        step_id: int,
        active_count: int,
    ) -> dict:
        cfg = self._get_occ_memory_recovery_cfg()
        status = {
            "event_type": "occ_memory_recovery_active",
            "step_id": int(step_id),
            "enabled": bool(cfg.get("enable")),
            "shadow_only": bool(cfg.get("shadow_only")),
            "considered": False,
            "applied": False,
            "reason": None,
            "actions": [],
            "active_intervention_index": int(active_count + 1),
            "active_intervention_budget": int(cfg.get("max_interventions_per_episode", 1)),
        }
        if not cfg.get("enable"):
            status["reason"] = "disabled"
            return status
        if not event:
            status["reason"] = "missing_recovery_event"
            return status
        if cfg.get("shadow_only"):
            status["considered"] = bool(event.get("recovery_trigger"))
            status["reason"] = "shadow_only"
            return status
        if not event.get("recovery_trigger"):
            status["reason"] = "no_recovery_trigger"
            return status

        status["considered"] = True
        max_interventions = int(cfg.get("max_interventions_per_episode", 1))
        if max_interventions >= 0 and active_count >= max_interventions:
            status["reason"] = "budget_exhausted"
        elif event.get("arrival_like_free_space"):
            status["reason"] = "arrival_like_free_space"
        elif not event.get("active_signal_allowed"):
            status["reason"] = "active_signal_disabled"
        elif not event.get("active_recovery_allowed"):
            status["reason"] = "active_recovery_not_allowed"
        else:
            escape_plan = event.get("escape_plan") or {}
            actions = [int(item) for item in list(escape_plan.get("actions") or []) if int(item) in (1, 2, 3)]
            if not actions:
                status["reason"] = "empty_escape_actions"
            else:
                status.update(
                    {
                        "applied": True,
                        "reason": "applied",
                        "actions": actions,
                        "escape_plan": escape_plan,
                        "recovery_event": event,
                    }
                )

        if status["applied"]:
            state["active_intervention_count"] = int(state.get("active_intervention_count", 0) or 0) + 1
            state["active_applied_count"] = int(state.get("active_applied_count", 0) or 0) + 1
            if state.get("active_first_step") is None:
                state["active_first_step"] = int(step_id)
        else:
            if status["considered"]:
                state["active_suppressed_count"] = int(state.get("active_suppressed_count", 0) or 0) + 1
        self._record_occ_memory_recovery_active_reason(state, status.get("reason") or "unknown")
        self._write_occ_memory_recovery_event(status)
        return status

    def _get_semantic_resilience_active_lite_cfg(self) -> dict:
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        return {
            "enable": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_enable",
                    False,
                )
            ),
            "shadow_only": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_shadow_only",
                    True,
                )
            ),
            "min_step": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_min_step",
                        30,
                    )
                ),
            ),
            "max_interventions_per_episode": int(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_max_interventions_per_episode",
                    1,
                )
            ),
            "cooldown_steps": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_cooldown_steps",
                        45,
                    )
                ),
            ),
            "utility_threshold": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_utility_threshold",
                    0.58,
                )
            ),
            "local_trap_utility_threshold": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold",
                    0.62,
                )
            ),
            "open_threshold": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_open_threshold",
                    0.65,
                )
            ),
            "min_backtrack_m": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_min_backtrack_m",
                    1.0,
                )
            ),
            "max_backtrack_m": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_max_backtrack_m",
                    3.5,
                )
            ),
            "max_step_gap": int(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_max_step_gap",
                    45,
                )
            ),
            "require_current_problem": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_require_current_problem",
                    True,
                )
            ),
            "require_geometry_safe": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_require_geometry_safe",
                    True,
                )
            ),
            "require_active_gate_safe": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_require_active_gate_safe",
                    False,
                )
            ),
            "evaluate_gate_when_shadow_only": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only",
                    False,
                )
            ),
            "allowed_recommended_primitives": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "require_any_trigger_reasons": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_require_any_trigger_reasons",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "require_all_trigger_reasons": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_require_all_trigger_reasons",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "require_any_context_tags": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_require_any_context_tags",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "require_all_context_tags": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_require_all_context_tags",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "require_target_frontier_intent_safe": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_require_target_frontier_intent_safe",
                    False,
                )
            ),
            "min_target_frontier_score": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_min_target_frontier_score",
                    0.0,
                )
            ),
            "max_completed_landmark_penalty": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_max_completed_landmark_penalty",
                    1.0,
                )
            ),
            "max_turn_steps": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_max_turn_steps",
                        4,
                    )
                ),
            ),
            "forward_steps": max(
                0,
                int(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_forward_steps",
                        0,
                    )
                ),
            ),
            "allow_forward_to_backtrack": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack",
                    False,
                )
            ),
            "forward_open_threshold": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_forward_open_threshold",
                    0.80,
                )
            ),
            "append_reobserve_action": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_append_reobserve_action",
                    True,
                )
            ),
            "clear_goal": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_clear_goal",
                    True,
                )
            ),
            "log_all_considered": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_log_all_considered",
                    True,
                )
            ),
            "allowed_failure_types": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_allowed_failure_types",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "v2_evidence_gate_enable": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_enable",
                    False,
                )
            ),
            "v2_evidence_gate_require_strict_intervention": bool(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_require_strict_intervention",
                    True,
                )
            ),
            "allowed_v2_evidence_tiers": tuple(
                str(item).strip()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_allowed_v2_evidence_tiers",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "allowed_direction_buckets": tuple(
                str(item).strip().lower()
                for item in list(
                    vlmap_safety_cfg.get(
                        "occ_memory_semantic_resilience_active_lite_allowed_direction_buckets",
                        [],
                    )
                    or []
                )
                if str(item).strip()
            ),
            "execution_mode": str(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_execution_mode",
                    "action_sequence",
                )
                or "action_sequence"
            ).lower(),
            "direction_y_ratio": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_direction_y_ratio",
                    0.75,
                )
            ),
            "direction_front_x_ratio": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_direction_front_x_ratio",
                    0.50,
                )
            ),
            "direction_left_x_ratio": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_direction_left_x_ratio",
                    0.25,
                )
            ),
            "direction_right_x_ratio": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_direction_right_x_ratio",
                    0.75,
                )
            ),
            "v2_evidence_gate_min_open_score": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_open_score",
                    0.70,
                )
            ),
            "v2_evidence_gate_min_doorway_score": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_doorway_score",
                    0.60,
                )
            ),
            "v2_evidence_gate_min_target_frontier_score": float(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_target_frontier_score",
                    0.10,
                )
            ),
            "v2_evidence_gate_min_step_gap": int(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_step_gap",
                    20,
                )
            ),
            "v2_evidence_gate_min_nearby_visits": int(
                vlmap_safety_cfg.get(
                    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_nearby_visits",
                    3,
                )
            ),
        }

    def _write_semantic_resilience_active_lite_event(self, event: dict) -> None:
        run_dir = self._get_vlmap_run_dir()
        log_dir = run_dir or self.output_path
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "stage19_semantic_resilience_active_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _best_semantic_resilience_backtrack_candidate(self, candidate_event: Optional[dict]) -> Optional[dict]:
        candidates = []
        for item in list((candidate_event or {}).get("candidates") or []):
            if str(item.get("candidate_type") or "") != "resilience_backtrack":
                continue
            if not bool(item.get("semantic_resilience_candidate")):
                continue
            candidates.append(dict(item))
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                bool(item.get("geometry_safe")),
                float(item.get("semantic_resilience_score", 0.0) or 0.0),
                float(item.get("semantic_resilience_open_score", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return candidates[0]

    def _semantic_resilience_active_lite_utility_threshold(
        self,
        cfg: dict,
        *,
        trigger_reasons: Optional[list] = None,
        context_tags: Optional[list] = None,
    ) -> Tuple[float, str]:
        base_threshold = float(cfg.get("utility_threshold", 0.58))
        reasons = {str(item) for item in list(trigger_reasons or [])}
        tags = {str(item) for item in list(context_tags or [])}

        # Stage19a relaxed smoke showed that pure local-trap / spatial
        # constriction triggers can be false positives on otherwise successful
        # S2 trajectories.  Keep those cases stricter, while allowing
        # semantic-stagnation + policy-memory-conflict candidates to use the
        # base recovery threshold.
        local_trap_like = bool("local_trap" in reasons or "spatial_constriction" in tags)
        stronger_failure_signal = bool(
            reasons.intersection(
                {
                    "current_waypoint_occupied",
                    "current_waypoint_not_active_safe",
                    "current_points_to_revisited_region",
                    "semantic_obstacle_near_trap",
                }
            )
            or tags.intersection(
                {
                    "policy_memory_conflict",
                    "revisit_loop_risk",
                    "semantic_obstacle_context",
                }
            )
        )
        if local_trap_like and not stronger_failure_signal:
            return (
                max(
                    base_threshold,
                    float(cfg.get("local_trap_utility_threshold", 0.62)),
                ),
                "local_trap_strict",
            )
        return base_threshold, "default"

    def _semantic_resilience_failure_profile(
        self,
        candidate: Optional[dict],
        *,
        trigger_reasons: Optional[list] = None,
        context_tags: Optional[list] = None,
        current_problem: bool = False,
    ) -> dict:
        candidate = dict(candidate or {})
        reasons = {str(item) for item in list(trigger_reasons or [])}
        tags = {str(item) for item in list(context_tags or [])}

        failure_type = "unknown"
        recommended_primitive = "hold_s2"
        failure_risk = "low"
        rationale = []

        revisit_signal = bool(
            reasons.intersection({"current_points_to_revisited_region"})
            or tags.intersection({"revisit_loop_risk"})
        )
        hard_stuck_signal = bool(
            reasons.intersection({"current_waypoint_occupied", "current_waypoint_not_active_safe"})
            or str(candidate.get("goal_state") or "") == "occupied"
        )
        obstacle_term_count = float(
            candidate.get("semantic_resilience_obstacle_term_count", 0.0) or 0.0
        )
        passage_term_count = float(
            candidate.get("semantic_resilience_passage_term_count", 0.0) or 0.0
        )
        semantic_obstacle_signal = bool(
            "semantic_obstacle_near_trap" in reasons
            or tags.intersection({"semantic_obstacle_context"})
            or obstacle_term_count > max(1.0, passage_term_count)
        )
        local_trap_signal = bool("local_trap" in reasons or "spatial_constriction" in tags)
        stagnation_signal = bool(
            reasons.intersection({"semantic_dead_zone", "semantic_stagnation"})
            or tags.intersection({"semantic_uncertainty_or_stagnation"})
        )

        strong_open_safe = bool(
            bool(candidate.get("active_gate_safe"))
            and float(candidate.get("semantic_resilience_open_score", 0.0) or 0.0) >= 0.80
        )

        if hard_stuck_signal or (local_trap_signal and semantic_obstacle_signal):
            failure_type = "stuck_collision"
            recommended_primitive = (
                "one_safe_forward_reobserve" if strong_open_safe else "reorient_reobserve"
            )
            failure_risk = "high"
            rationale.append("hard_stuck_or_obstacle_trap")
        elif local_trap_signal:
            failure_type = "local_trap"
            recommended_primitive = "reorient_reobserve"
            failure_risk = "medium"
            rationale.append("local_trap")
        elif revisit_signal:
            failure_type = "semantic_drift_revisit"
            recommended_primitive = "backtrack_reobserve"
            failure_risk = "high"
            rationale.append("revisit_signal")
        elif stagnation_signal:
            failure_type = "semantic_stagnation"
            recommended_primitive = "reobserve"
            failure_risk = "medium"
            rationale.append("semantic_stagnation")
        elif semantic_obstacle_signal:
            failure_type = "stuck_collision"
            recommended_primitive = (
                "one_safe_forward_reobserve" if strong_open_safe else "reorient_reobserve"
            )
            failure_risk = "high"
            rationale.append("semantic_obstacle")
        elif current_problem:
            failure_type = "policy_conflict"
            recommended_primitive = "reobserve"
            failure_risk = "medium"
            rationale.append("current_problem")

        if failure_type == "stuck_collision" and strong_open_safe:
            rationale.append("strong_open_safe")

        return {
            "failure_type": failure_type,
            "recommended_primitive": recommended_primitive,
            "failure_risk": failure_risk,
            "failure_rationale": rationale,
        }

    def _semantic_resilience_v2_evidence_gate(
        self,
        candidate: Optional[dict],
        cfg: dict,
        *,
        failure_type: str,
        recommended_primitive: str,
        trigger_reasons: Optional[list] = None,
        context_tags: Optional[list] = None,
        step_id: int = 0,
    ) -> dict:
        return classify_semantic_recovery_triage(
            candidate,
            cfg,
            failure_type=failure_type,
            recommended_primitive=recommended_primitive,
            trigger_reasons=trigger_reasons,
            context_tags=context_tags,
            step_id=step_id,
        )

    def _summarize_stage19_episode_failure_type(
        self,
        metrics: dict,
        *,
        failure_type_counts: Optional[dict] = None,
        recommended_primitive_counts: Optional[dict] = None,
        step_id: Optional[int] = None,
        collision_count: Optional[float] = None,
    ) -> tuple:
        failure_type_counts = dict(failure_type_counts or {})
        recommended_primitive_counts = dict(recommended_primitive_counts or {})
        if bool(metrics.get("success")):
            return "success", "none"

        distance = float(metrics.get("distance_to_goal", float("inf")) or float("inf"))
        oracle_success = bool(metrics.get("oracle_success"))
        if oracle_success and distance <= 1.5:
            return "near_goal_no_stop", "stop_advisor"

        if failure_type_counts:
            priority = {
                "stuck_collision": 4,
                "semantic_drift_revisit": 3,
                "local_trap": 2,
                "semantic_stagnation": 1,
                "policy_conflict": 0,
                "unknown": -1,
            }
            dominant_type = max(
                failure_type_counts.items(),
                key=lambda item: (int(item[1]), priority.get(str(item[0]), -1)),
            )[0]
            primitive_map = {
                "stuck_collision": "reorient_reobserve",
                "semantic_drift_revisit": "backtrack_reobserve",
                "local_trap": "reorient_reobserve",
                "semantic_stagnation": "reobserve",
                "policy_conflict": "reobserve",
                "unknown": "hold_s2",
            }
            if recommended_primitive_counts:
                dominant_primitive = max(
                    recommended_primitive_counts.items(),
                    key=lambda item: int(item[1]),
                )[0]
                return str(dominant_type), str(dominant_primitive)
            return str(dominant_type), primitive_map.get(str(dominant_type), "hold_s2")

        if collision_count is not None and float(collision_count) >= 25.0:
            return "stuck_collision", "reorient_reobserve"
        if step_id is not None and int(step_id) >= 500:
            return "semantic_stagnation", "reobserve"
        return "unknown", "hold_s2"

    def _semantic_resilience_active_lite_actions(self, candidate: dict, cfg: dict) -> list:
        actions = []
        angle = candidate.get("direction_angle_deg")
        if angle is None:
            bucket = str(candidate.get("direction_bucket") or "unknown")
            angle = {
                "front": 0.0,
                "left": 90.0,
                "right": -90.0,
                "back": 180.0,
            }.get(bucket, 0.0)
        try:
            angle = float(angle)
        except (TypeError, ValueError):
            angle = 0.0
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0

        turn_angle = max(
            1e-6,
            abs(float(getattr(self.config.habitat.simulator, "turn_angle", 15.0))),
        )
        max_turn_steps = max(0, int(cfg.get("max_turn_steps", 4)))
        turn_steps = min(max_turn_steps, int(math.ceil(abs(angle) / turn_angle)))
        if turn_steps > 0:
            turn_action = action_code.LEFT if angle > 0.0 else action_code.RIGHT
            actions.extend([int(turn_action)] * int(turn_steps))

        forward_steps = int(cfg.get("forward_steps", 0) or 0)
        open_score = float(candidate.get("semantic_resilience_open_score", 0.0) or 0.0)
        can_forward = bool(
            forward_steps > 0
            and bool(candidate.get("geometry_safe"))
            and open_score >= float(cfg.get("forward_open_threshold", 0.80))
            and (
                bool(candidate.get("active_gate_safe"))
                or bool(cfg.get("allow_forward_to_backtrack", False))
            )
        )
        if can_forward:
            actions.extend([int(action_code.FORWARD)] * int(forward_steps))

        if bool(cfg.get("append_reobserve_action", True)):
            actions.append(int(action_code.LOOKDOWN))

        return [int(item) for item in actions if int(item) in (1, 2, 3, 5)]

    def _semantic_resilience_active_lite_directional_pixel_goal(
        self, candidate: dict, cfg: dict
    ) -> dict:
        direction = str(candidate.get("direction_bucket") or "").lower()
        x_ratios = {
            "front": float(cfg.get("direction_front_x_ratio", 0.50)),
            "left": float(cfg.get("direction_left_x_ratio", 0.25)),
            "right": float(cfg.get("direction_right_x_ratio", 0.75)),
        }
        if direction not in x_ratios:
            return {
                "valid": False,
                "reason": "direction_not_actionable",
                "direction": direction,
                "pixel_goal": None,
            }
        image_w = max(1, int(getattr(self.model_args, "resize_w", 384) or 384))
        image_h = max(1, int(getattr(self.model_args, "resize_h", 384) or 384))
        pixel_goal = [
            int(
                round(
                    max(0.0, min(1.0, x_ratios[direction]))
                    * float(image_w - 1)
                )
            ),
            int(
                round(
                    max(0.0, min(1.0, float(cfg.get("direction_y_ratio", 0.75))))
                    * float(image_h - 1)
                )
            ),
        ]
        return {
            "valid": True,
            "reason": "ok",
            "direction": direction,
            "pixel_goal": pixel_goal,
            "image_width": int(image_w),
            "image_height": int(image_h),
        }

    def _maybe_apply_semantic_resilience_active_lite(
        self,
        candidate_event: Optional[dict],
        *,
        step_id: int,
        active_count: int,
        last_active_step: Optional[int],
        scene_id: Optional[str] = None,
        episode_id: Optional[int] = None,
    ) -> dict:
        cfg = self._get_semantic_resilience_active_lite_cfg()
        state = dict((candidate_event or {}).get("semantic_resilience_state") or {})
        trigger_reasons = list(
            (candidate_event or {}).get("semantic_resilience_trigger_reasons")
            or state.get("trigger_reasons")
            or []
        )
        context_tags = list(
            (candidate_event or {}).get("semantic_resilience_recovery_context_tags")
            or state.get("recovery_context_tags")
            or []
        )
        recovery_trigger = bool(
            (candidate_event or {}).get("semantic_resilience_recovery_trigger")
            or state.get("recovery_trigger")
        )
        current_problem = bool(
            state.get("current_policy_problem")
            or any(
                reason
                in {
                    "current_waypoint_occupied",
                    "current_waypoint_not_active_safe",
                    "semantic_dead_zone",
                    "semantic_stagnation",
                    "current_points_to_revisited_region",
                    "local_trap",
                    "semantic_obstacle_near_trap",
                }
                for reason in trigger_reasons
            )
        )
        candidate = self._best_semantic_resilience_backtrack_candidate(candidate_event)
        status = {
            "event_type": "stage19_semantic_resilience_active",
            "scene_id": scene_id,
            "episode_id": episode_id,
            "step_id": int(step_id),
            "enabled": bool(cfg.get("enable")),
            "shadow_only": bool(cfg.get("shadow_only")),
            "considered": False,
            "applied": False,
            "reason": None,
            "actions": [],
            "active_intervention_index": int(active_count + 1),
            "active_intervention_budget": int(cfg.get("max_interventions_per_episode", 1)),
            "recovery_trigger": bool(recovery_trigger),
            "current_problem": bool(current_problem),
            "trigger_reasons": trigger_reasons,
            "recovery_context_tags": context_tags,
            "candidate": candidate,
        }
        status.update(
            self._semantic_resilience_failure_profile(
                candidate,
                trigger_reasons=trigger_reasons,
                context_tags=context_tags,
                current_problem=current_problem,
            )
        )
        if not cfg.get("enable"):
            status["reason"] = "disabled"
            return status
        if not candidate_event:
            status["reason"] = "missing_candidate_event"
            return status
        status["considered"] = bool(recovery_trigger or candidate is not None)
        if not status["considered"]:
            status["reason"] = "no_recovery_trigger"
            return status
        v2_evidence_gate = self._semantic_resilience_v2_evidence_gate(
            candidate,
            cfg,
            failure_type=str(status.get("failure_type") or "unknown"),
            recommended_primitive=str(status.get("recommended_primitive") or "hold_s2"),
            trigger_reasons=trigger_reasons,
            context_tags=context_tags,
            step_id=int(step_id),
        )
        if bool(v2_evidence_gate.get("enabled")):
            status["v2_evidence_gate"] = v2_evidence_gate
            status["v2_evidence_tier"] = str(v2_evidence_gate.get("tier") or "unknown")
            status["v2_evidence_reason"] = str(v2_evidence_gate.get("reason") or "unknown")
        execution_mode = str(cfg.get("execution_mode") or "action_sequence").lower()
        status["execution_mode"] = execution_mode
        allowed_failure_types = {
            str(item)
            for item in list(cfg.get("allowed_failure_types") or [])
            if str(item)
        }
        if allowed_failure_types and str(status.get("failure_type") or "unknown") not in allowed_failure_types:
            status["reason"] = "failure_type_not_allowed"
            if bool(cfg.get("shadow_only")) or bool(cfg.get("log_all_considered", True)):
                self._write_semantic_resilience_active_lite_event(status)
            return status
        allowed_primitives = {
            str(item)
            for item in list(cfg.get("allowed_recommended_primitives") or [])
            if str(item)
        }
        if allowed_primitives and str(status.get("recommended_primitive") or "hold_s2") not in allowed_primitives:
            status["reason"] = "recommended_primitive_not_allowed"
            if bool(cfg.get("shadow_only")) or bool(cfg.get("log_all_considered", True)):
                self._write_semantic_resilience_active_lite_event(status)
            return status
        evaluate_shadow_gate = bool(
            cfg.get("shadow_only") and cfg.get("evaluate_gate_when_shadow_only")
        )
        if cfg.get("shadow_only") and not evaluate_shadow_gate:
            status["reason"] = "shadow_only"
            self._write_semantic_resilience_active_lite_event(status)
            return status
        if int(step_id) < int(cfg.get("min_step", 30)):
            status["reason"] = "too_early"
        elif candidate is None:
            status["reason"] = "missing_backtrack_candidate"
        elif int(cfg.get("max_interventions_per_episode", 1)) >= 0 and active_count >= int(
            cfg.get("max_interventions_per_episode", 1)
        ):
            status["reason"] = "budget_exhausted"
        elif last_active_step is not None and int(step_id) - int(last_active_step) < int(
            cfg.get("cooldown_steps", 45)
        ):
            status["reason"] = "cooldown"
        elif bool(cfg.get("require_current_problem", True)) and not current_problem:
            status["reason"] = "current_policy_not_problematic"
        elif bool(cfg.get("require_geometry_safe", True)) and not bool(candidate.get("geometry_safe")):
            status["reason"] = "candidate_not_geometry_safe"
        elif cfg.get("allowed_direction_buckets") and str(
            candidate.get("direction_bucket") or "unknown"
        ).lower() not in set(cfg.get("allowed_direction_buckets") or ()):
            status["reason"] = "candidate_direction_not_allowed"
        elif bool(cfg.get("require_active_gate_safe", False)) and not bool(candidate.get("active_gate_safe")):
            status["reason"] = "candidate_not_active_gate_safe"
        else:
            trigger_reason_set = {str(item) for item in list(trigger_reasons or [])}
            context_tag_set = {str(item) for item in list(context_tags or [])}
            require_any_trigger_reasons = {
                str(item) for item in list(cfg.get("require_any_trigger_reasons") or [])
            }
            require_all_trigger_reasons = {
                str(item) for item in list(cfg.get("require_all_trigger_reasons") or [])
            }
            require_any_context_tags = {
                str(item) for item in list(cfg.get("require_any_context_tags") or [])
            }
            require_all_context_tags = {
                str(item) for item in list(cfg.get("require_all_context_tags") or [])
            }
            utility = float(candidate.get("semantic_resilience_score", 0.0) or 0.0)
            open_score = float(candidate.get("semantic_resilience_open_score", 0.0) or 0.0)
            distance = float(candidate.get("semantic_resilience_backtrack_distance_m", -1.0) or -1.0)
            step_gap = candidate.get("semantic_resilience_step_gap")
            target_frontier_score = float(candidate.get("target_frontier_score", 0.0) or 0.0)
            completed_landmark_penalty = float(candidate.get("completed_landmark_penalty", 0.0) or 0.0)
            utility_threshold, utility_threshold_context = (
                self._semantic_resilience_active_lite_utility_threshold(
                    cfg,
                    trigger_reasons=trigger_reasons,
                    context_tags=context_tags,
                )
            )
            v2_evidence_gate = dict(status.get("v2_evidence_gate") or {})
            v2_tier = str(v2_evidence_gate.get("tier") or "")
            allowed_v2_tiers = {
                str(item) for item in tuple(cfg.get("allowed_v2_evidence_tiers") or ())
            }
            status["utility_threshold_used"] = float(utility_threshold)
            status["utility_threshold_context"] = str(utility_threshold_context)
            if (
                bool(v2_evidence_gate.get("enabled"))
                and allowed_v2_tiers
                and v2_tier not in allowed_v2_tiers
            ):
                if v2_tier == "adapter_candidate":
                    status["reason"] = "v2_adapter_candidate_hold"
                else:
                    status["reason"] = "v2_evidence_abstain"
            elif (
                bool(v2_evidence_gate.get("enabled"))
                and not allowed_v2_tiers
                and bool(cfg.get("v2_evidence_gate_require_strict_intervention", True))
                and v2_tier != "strict_intervention"
            ):
                if v2_tier == "adapter_candidate":
                    status["reason"] = "v2_adapter_candidate_hold"
                else:
                    status["reason"] = "v2_evidence_abstain"
            elif require_any_trigger_reasons and not trigger_reason_set.intersection(
                require_any_trigger_reasons
            ):
                status["reason"] = "missing_required_trigger_reason"
            elif require_all_trigger_reasons and not require_all_trigger_reasons.issubset(
                trigger_reason_set
            ):
                status["reason"] = "missing_required_trigger_reason"
            elif require_any_context_tags and not context_tag_set.intersection(
                require_any_context_tags
            ):
                status["reason"] = "missing_required_context_tag"
            elif require_all_context_tags and not require_all_context_tags.issubset(
                context_tag_set
            ):
                status["reason"] = "missing_required_context_tag"
            elif bool(cfg.get("require_target_frontier_intent_safe", False)) and not bool(
                candidate.get("target_frontier_intent_safe")
            ):
                status["reason"] = "target_frontier_intent_not_safe"
            elif target_frontier_score < float(cfg.get("min_target_frontier_score", 0.0)):
                status["reason"] = "low_target_frontier_score"
            elif completed_landmark_penalty > float(
                cfg.get("max_completed_landmark_penalty", 1.0)
            ):
                status["reason"] = "completed_landmark_penalty_too_high"
            elif utility < float(utility_threshold):
                status["reason"] = "low_recovery_utility"
            elif open_score < float(cfg.get("open_threshold", 0.65)):
                status["reason"] = "low_open_score"
            elif distance < float(cfg.get("min_backtrack_m", 1.0)) or distance > float(
                cfg.get("max_backtrack_m", 3.5)
            ):
                status["reason"] = "backtrack_distance_out_of_range"
            elif step_gap is not None and int(step_gap) > int(cfg.get("max_step_gap", 45)):
                status["reason"] = "backtrack_step_gap_too_large"
            else:
                pixel_goal_plan = None
                actions = []
                if execution_mode == "directional_pixel_goal":
                    pixel_goal_plan = self._semantic_resilience_active_lite_directional_pixel_goal(
                        candidate, cfg
                    )
                    if not pixel_goal_plan.get("valid"):
                        status["reason"] = str(
                            pixel_goal_plan.get("reason") or "invalid_directional_pixel_goal"
                        )
                else:
                    actions = self._semantic_resilience_active_lite_actions(candidate, cfg)
                    if not actions:
                        status["reason"] = "empty_recovery_actions"
                if status.get("reason") is not None:
                    pass
                elif evaluate_shadow_gate:
                    status.update(
                        {
                            "would_apply": True,
                            "reason": "shadow_gate_pass",
                            "shadow_actions": actions,
                            "pixel_goal_plan": pixel_goal_plan,
                            "action_plan": {
                                "mode": execution_mode,
                                "clear_goal": bool(cfg.get("clear_goal", True)),
                                "forward_steps": int(cfg.get("forward_steps", 0) or 0),
                                "append_reobserve_action": bool(
                                    cfg.get("append_reobserve_action", True)
                                ),
                            },
                        }
                    )
                else:
                    status.update(
                        {
                            "would_apply": True,
                            "applied": execution_mode != "directional_pixel_goal",
                            "execution_pending": execution_mode
                            == "directional_pixel_goal",
                            "reason": (
                                "execution_pending"
                                if execution_mode == "directional_pixel_goal"
                                else "applied"
                            ),
                            "actions": actions,
                            "pixel_goal_plan": pixel_goal_plan,
                            "action_plan": {
                                "mode": execution_mode,
                                "clear_goal": bool(cfg.get("clear_goal", True)),
                                "forward_steps": int(cfg.get("forward_steps", 0) or 0),
                                "append_reobserve_action": bool(
                                    cfg.get("append_reobserve_action", True)
                                ),
                            },
                        }
                    )

        if (
            not status.get("execution_pending")
            and (bool(cfg.get("log_all_considered", True)) or status.get("applied"))
        ):
            self._write_semantic_resilience_active_lite_event(status)
        return status

    def _normalize_candidate_actions(self, actions, horizon: Optional[int] = None) -> list:
        normalized = [int(item) for item in list(actions or [])]
        if len(normalized) < MAX_STEPS:
            normalized += [0] * (MAX_STEPS - len(normalized))
        if horizon is not None:
            normalized = normalized[: int(horizon)]
        return normalized

    def _actions_from_nextdit_sample(self, dp_actions, sample_index: int, horizon: int) -> list:
        sample = dp_actions[int(sample_index)]
        if hasattr(sample, "detach"):
            sample = sample.detach().clone()
            if sample.dim() == 2:
                sample = sample.unsqueeze(0)
        else:
            sample = torch.as_tensor(np.asarray(sample)).clone()
            if sample.dim() == 2:
                sample = sample.unsqueeze(0)
        try:
            actions = traj_to_actions(sample)
        except Exception as exc:
            print(f"[NextDiTCandidateProbe] failed to convert sample {sample_index}: {exc}")
            actions = []
        return self._normalize_candidate_actions(actions, horizon=horizon)

    def _local_xy_from_nextdit_sample(self, dp_actions, sample_index: int, max_points: int) -> Optional[np.ndarray]:
        try:
            sample = dp_actions[int(sample_index)]
            if hasattr(sample, "detach"):
                arr = sample.detach().float().cpu().numpy().copy()
            else:
                arr = np.asarray(sample, dtype=np.float32).copy()
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
                return None
            # Match traj_to_actions(): dx/dy are stored at 4x scale.
            arr[:, :2] /= 4.0
            cumsum_xy = np.cumsum(arr[:, :2], axis=0)
            local_xy = np.zeros((arr.shape[0] + 1, 2), dtype=np.float32)
            local_xy[1:] = cumsum_xy
            if max_points > 0 and local_xy.shape[0] > max_points:
                ids = np.linspace(0, local_xy.shape[0] - 1, int(max_points)).astype(np.int64)
                local_xy = local_xy[ids]
            return local_xy
        except Exception as exc:
            print(f"[NextDiTCandidateProbe] failed to reconstruct raw sample {sample_index}: {exc}")
            return None

    def _local_xy_from_nextdit_average(self, dp_actions, max_points: int) -> Optional[np.ndarray]:
        try:
            if hasattr(dp_actions, "detach"):
                samples = dp_actions.detach().float().cpu().clone()
            else:
                samples = torch.as_tensor(np.asarray(dp_actions), dtype=torch.float32).clone()
            if samples.dim() != 3 or samples.shape[0] < 1 or samples.shape[1] < 1 or samples.shape[2] < 2:
                return None
            # traj_to_actions(..., use_discrate_action=False) performs the single /4 unnormalization.
            local_xy = np.asarray(traj_to_actions(samples, use_discrate_action=False), dtype=np.float32)
            if local_xy.ndim != 2 or local_xy.shape[0] < 2 or local_xy.shape[1] < 2:
                return None
            if max_points > 0 and local_xy.shape[0] > max_points:
                ids = np.linspace(0, local_xy.shape[0] - 1, int(max_points)).astype(np.int64)
                local_xy = local_xy[ids]
            return local_xy
        except Exception as exc:
            print(f"[NextDiTCandidateProbe] failed to reconstruct averaged trajectory: {exc}")
            return None

    def _score_nextdit_averaged_trajectory_with_occ_memory(
        self,
        dp_actions,
        observations: dict,
        *,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        step_id: int,
        query_index: int,
    ) -> Optional[dict]:
        cfg = self._get_nextdit_candidate_probe_cfg()
        if not cfg.get("occ_memory_score_enable") or dp_actions is None:
            return None
        local_xy = self._local_xy_from_nextdit_average(
            dp_actions,
            int(cfg.get("occ_memory_score_max_points", 33)),
        )
        if local_xy is None:
            return {
                "enabled": True,
                "valid": False,
                "reason": "averaged_trajectory_reconstruction_failed",
            }
        return self.occ_memory.score_local_xy_trajectory(
            local_xy,
            {
                "gps": observations.get("gps"),
                "compass": observations.get("compass"),
            },
            context={
                "step_id": step_id,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "episode_count": episode_count,
                "query_index": query_index,
                "candidate_probe": "nextdit_averaged_trajectory",
            },
        )

    def _trajectory_blocked_fraction_sum(self, decision: dict) -> float:
        blocked_fraction_sum = 0.0
        for step in (decision or {}).get("step_details", []) or []:
            stats = step.get("stats") or {}
            blocked_fraction_sum += float(stats.get("blocked_fraction", 0.0) or 0.0)
        return float(blocked_fraction_sum)

    def _trajectory_obstacle_score(self, decision: dict) -> float:
        if not decision or not decision.get("valid"):
            return float("inf")
        return (
            (1000.0 if decision.get("would_reject") else 0.0)
            + 100.0 * int(decision.get("blocked_steps", 0) or 0)
            + self._trajectory_blocked_fraction_sum(decision)
        )

    def _trajectory_decision_score(self, decision: dict) -> float:
        if not decision or not decision.get("valid"):
            return float("inf")
        checked = int(decision.get("checked_forward_steps", 0) or 0)
        return self._trajectory_obstacle_score(decision) - 0.01 * checked

    def _unique_action_sequence_count(self, candidates: list) -> int:
        return len({tuple(item.get("actions") or []) for item in candidates})

    def _unique_endpoint_count(self, candidates: list, min_grid_distance: float) -> int:
        representatives = []
        for item in candidates:
            end_grid = item.get("decision", {}).get("end_grid")
            if end_grid is None:
                continue
            if all(
                float(np.hypot(float(end_grid[0]) - float(rep[0]), float(end_grid[1]) - float(rep[1])))
                >= min_grid_distance
                for rep in representatives
            ):
                representatives.append(end_grid)
        return len(representatives)

    def _is_nextdit_occ_memory_active_strategy(self, cfg: Optional[dict] = None) -> bool:
        cfg = cfg or self._get_nextdit_candidate_probe_cfg()
        return str(cfg.get("active_strategy", "")).lower() in {
            "occ_memory_conservative",
            "occ_memory",
            "occmem_conservative",
        }

    def _local_endpoint_angle_deg_from_occ_score(self, score: Optional[dict]) -> Optional[float]:
        if not score or not score.get("valid"):
            return None
        endpoint = score.get("local_endpoint_xy")
        if endpoint is None:
            return None
        try:
            x = float(endpoint[0])
            y = float(endpoint[1])
        except (TypeError, ValueError, IndexError):
            return None
        if float(np.hypot(x, y)) < 1e-4:
            return None
        return float(math.degrees(math.atan2(y, x)))

    def _angle_distance_deg(self, angle_a: Optional[float], angle_b: Optional[float]) -> Optional[float]:
        if angle_a is None or angle_b is None:
            return None
        try:
            diff = (float(angle_a) - float(angle_b) + 180.0) % 360.0 - 180.0
        except (TypeError, ValueError):
            return None
        return float(abs(diff))

    def _occ_memory_candidate_deviation_deg(self, current_score: Optional[dict], candidate_score: Optional[dict]):
        current_angle = self._local_endpoint_angle_deg_from_occ_score(current_score)
        candidate_angle = self._local_endpoint_angle_deg_from_occ_score(candidate_score)
        return self._angle_distance_deg(current_angle, candidate_angle)

    def _occ_memory_candidate_active_score(
        self,
        occ_score: dict,
        direction_deviation_deg: Optional[float],
        cfg: dict,
    ) -> float:
        occupied_hit_count = int(occ_score.get("occupied_hit_count", 0) or 0)
        unknown_hit_count = int(occ_score.get("unknown_hit_count", 0) or 0)
        endpoint = occ_score.get("local_endpoint_xy") or [0.0, 0.0]
        try:
            forward_progress = max(0.0, float(endpoint[0]))
        except (TypeError, ValueError, IndexError):
            forward_progress = 0.0
        max_deviation = max(1e-6, float(cfg.get("active_occ_max_direction_deviation_deg", 45.0)))
        deviation_ratio = (
            float(direction_deviation_deg) / max_deviation
            if direction_deviation_deg is not None
            else 1.0
        )
        return float(
            1.0 * occupied_hit_count
            + float(cfg.get("active_occ_unknown_weight", 0.15)) * unknown_hit_count
            + float(cfg.get("active_occ_direction_weight", 0.30)) * deviation_ratio
            - float(cfg.get("active_occ_forward_progress_weight", 0.05)) * forward_progress
        )

    def _probe_nextdit_trajectory_candidates(
        self,
        dp_actions,
        current_actions: list,
        current_decision: dict,
        observations: dict,
        depth_m: np.ndarray,
        rgb: Optional[np.ndarray],
        *,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        step_id: int,
        query_index: int,
        pixel_goal,
        current_occ_memory_score: Optional[dict] = None,
    ) -> dict:
        cfg = self._get_nextdit_candidate_probe_cfg()
        if not (cfg["enable"] or cfg["active_enable"]) or dp_actions is None:
            return {}
        if not hasattr(dp_actions, "shape") or len(dp_actions.shape) < 3:
            return {}
        validate = getattr(getattr(self, "vlmap_safety", None), "validate_trajectory", None)
        if validate is None:
            return {}

        sample_count = int(dp_actions.shape[0])
        candidate_count = sample_count if cfg["max_candidates"] <= 0 else min(sample_count, cfg["max_candidates"])
        horizon = int(cfg["action_horizon"])
        safety_obs = {
            "depth": depth_m,
            "rgb": rgb,
            "gps": observations.get("gps"),
            "compass": observations.get("compass"),
        }
        if safety_obs["gps"] is None or safety_obs["compass"] is None:
            return {}

        candidates = []
        occ_valid_count = 0
        occ_invalid_count = 0
        occ_would_reject_count = 0
        occ_unknown_candidate_count = 0
        occ_checked_cell_sum = 0
        occ_occupied_hit_sum = 0
        occ_unknown_hit_sum = 0
        occ_memory_score_enabled = bool(cfg.get("occ_memory_score_enable"))
        if occ_memory_score_enabled and current_occ_memory_score is None:
            current_occ_memory_score = self._score_nextdit_averaged_trajectory_with_occ_memory(
                dp_actions,
                observations,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_index=episode_index,
                episode_count=episode_count,
                step_id=step_id,
                query_index=query_index,
            )
        for candidate_index in range(candidate_count):
            actions = self._actions_from_nextdit_sample(dp_actions, candidate_index, horizon=horizon)
            decision = validate(
                safety_obs,
                actions,
                context={
                    "step_id": step_id,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "episode_count": episode_count,
                    "pixel_goal": pixel_goal,
                    "candidate_index": candidate_index,
                    "candidate_probe": "nextdit",
                    "skip_map_update": True,
                    "suppress_trajectory_event": True,
                },
            )
            occ_memory_score = None
            if occ_memory_score_enabled:
                local_xy = self._local_xy_from_nextdit_sample(
                    dp_actions,
                    candidate_index,
                    int(cfg.get("occ_memory_score_max_points", 33)),
                )
                if local_xy is None:
                    occ_memory_score = {
                        "enabled": True,
                        "valid": False,
                        "reason": "raw_sample_reconstruction_failed",
                    }
                    occ_invalid_count += 1
                else:
                    occ_memory_score = self.occ_memory.score_local_xy_trajectory(
                        local_xy,
                        safety_obs,
                        context={
                            "step_id": step_id,
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "episode_index": episode_index,
                            "episode_count": episode_count,
                            "candidate_index": candidate_index,
                            "candidate_probe": "nextdit_raw_trajectory",
                        },
                    )
                    if occ_memory_score.get("valid"):
                        occ_valid_count += 1
                        if occ_memory_score.get("would_reject"):
                            occ_would_reject_count += 1
                        if occ_memory_score.get("has_unknown"):
                            occ_unknown_candidate_count += 1
                        occ_checked_cell_sum += int(occ_memory_score.get("checked_cell_count", 0) or 0)
                        occ_occupied_hit_sum += int(occ_memory_score.get("occupied_hit_count", 0) or 0)
                        occ_unknown_hit_sum += int(occ_memory_score.get("unknown_hit_count", 0) or 0)
                    else:
                        occ_invalid_count += 1
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "actions": actions,
                    "score": float(self._trajectory_decision_score(decision)),
                    "obstacle_score": float(self._trajectory_obstacle_score(decision)),
                    "decision": decision,
                    "occ_memory_score": occ_memory_score,
                }
            )

        if not candidates:
            return {}

        current_actions = self._normalize_candidate_actions(current_actions, horizon=horizon)
        current_score = float(self._trajectory_decision_score(current_decision))
        current_obstacle_score = float(self._trajectory_obstacle_score(current_decision))
        selected = min(candidates, key=lambda item: item["score"])
        safer_candidate_count = sum(1 for item in candidates if item["score"] + 1e-6 < current_score)
        would_reject_count = sum(1 for item in candidates if item.get("decision", {}).get("would_reject"))
        current_rank = 1 + sum(1 for item in candidates if item["score"] + 1e-6 < current_score)
        event = {
            "event_type": "nextdit_candidate_probe",
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "episode_index": int(episode_index),
            "episode_count": int(episode_count),
            "step_id": int(step_id),
            "query_index": int(query_index),
            "sample_count": int(sample_count),
            "candidate_count": int(candidate_count),
            "action_horizon": int(horizon),
            "current_actions": current_actions,
            "current_score": current_score,
            "current_obstacle_score": current_obstacle_score,
            "current_would_reject": bool((current_decision or {}).get("would_reject")),
            "current_decision": current_decision,
            "current_occ_memory_score": current_occ_memory_score,
            "current_occ_memory_valid": bool(
                (current_occ_memory_score or {}).get("valid")
            ),
            "current_occ_memory_would_reject": bool(
                (current_occ_memory_score or {}).get("would_reject")
            ),
            "current_occ_memory_occupied_hit_count": int(
                (current_occ_memory_score or {}).get("occupied_hit_count", 0) or 0
            ),
            "current_occ_memory_unknown_hit_count": int(
                (current_occ_memory_score or {}).get("unknown_hit_count", 0) or 0
            ),
            "current_occ_memory_unknown_hit_ratio": float(
                (current_occ_memory_score or {}).get("unknown_hit_ratio", 0.0) or 0.0
            ),
            "selected_candidate_index": int(selected["candidate_index"]),
            "selected_actions": selected["actions"],
            "selected_score": float(selected["score"]),
            "selected_obstacle_score": float(selected["obstacle_score"]),
            "selected_differs_from_current": bool(selected["actions"] != current_actions),
            "safer_candidate_count": int(safer_candidate_count),
            "current_rank": int(current_rank),
            "would_reject_candidate_count": int(would_reject_count),
            "unique_action_sequence_count": int(self._unique_action_sequence_count(candidates)),
            "unique_endpoint_count": int(
                self._unique_endpoint_count(candidates, cfg["min_endpoint_grid_distance"])
            ),
            "min_endpoint_grid_distance": float(cfg["min_endpoint_grid_distance"]),
            "occ_memory_score_enabled": bool(occ_memory_score_enabled),
            "occ_memory_score_valid_candidate_count": int(occ_valid_count),
            "occ_memory_score_invalid_candidate_count": int(occ_invalid_count),
            "occ_memory_score_would_reject_candidate_count": int(occ_would_reject_count),
            "occ_memory_score_unknown_candidate_count": int(occ_unknown_candidate_count),
            "occ_memory_score_mean_checked_cell_count": float(
                occ_checked_cell_sum / max(1, occ_valid_count)
            ),
            "occ_memory_score_mean_occupied_hit_count": float(
                occ_occupied_hit_sum / max(1, occ_valid_count)
            ),
            "occ_memory_score_mean_unknown_hit_count": float(
                occ_unknown_hit_sum / max(1, occ_valid_count)
            ),
            "pixel_goal": None if pixel_goal is None else [int(pixel_goal[0]), int(pixel_goal[1])],
            "candidates": candidates,
        }
        self._write_nextdit_candidate_probe_event(event)
        print(
            "[NextDiTCandidateProbe] "
            f"query={query_index} candidates={candidate_count}/{sample_count} "
            f"unique_actions={event['unique_action_sequence_count']} "
            f"unique_end={event['unique_endpoint_count']} "
            f"safer={safer_candidate_count} "
            f"selected={event['selected_candidate_index']} "
            f"diff={event['selected_differs_from_current']}"
        )
        return event

    def _nextdit_active_probe_needed(
        self,
        current_decision: dict,
        active_intervention_count: int,
        current_occ_memory_score: Optional[dict] = None,
    ) -> bool:
        cfg = self._get_nextdit_candidate_probe_cfg()
        if not cfg.get("active_enable"):
            return False
        max_interventions = int(cfg.get("active_max_interventions_per_episode", 2))
        if max_interventions >= 0 and active_intervention_count >= max_interventions:
            return False
        if self._is_nextdit_occ_memory_active_strategy(cfg):
            if not cfg.get("occ_memory_score_enable"):
                return False
            if not current_occ_memory_score or not current_occ_memory_score.get("valid"):
                return False
            occupied_hits = int(current_occ_memory_score.get("occupied_hit_count", 0) or 0)
            return occupied_hits >= int(cfg.get("active_occ_current_min_occupied_hits", 1))
        if cfg.get("active_require_current_reject", True) and not bool(
            (current_decision or {}).get("would_reject")
        ):
            return False
        return True

    def _select_nextdit_occ_memory_active_candidate(self, event: dict):
        cfg = self._get_nextdit_candidate_probe_cfg()
        current_occ = event.get("current_occ_memory_score") or {}
        if not current_occ.get("valid"):
            return None, "current_occ_invalid"
        current_occupied_hits = int(current_occ.get("occupied_hit_count", 0) or 0)
        if current_occupied_hits < int(cfg.get("active_occ_current_min_occupied_hits", 1)):
            return None, "current_occ_not_occupied"

        horizon = int(event.get("action_horizon", cfg.get("action_horizon", MAX_LOCAL_STEPS)))
        current_actions = self._normalize_candidate_actions(event.get("current_actions"), horizon=horizon)
        max_deviation = float(cfg.get("active_occ_max_direction_deviation_deg", 45.0))
        eligible = []
        for item in event.get("candidates") or []:
            decision = item.get("decision") or {}
            if cfg.get("active_occ_require_vlmap_nonreject", False):
                if not decision.get("valid") or decision.get("would_reject"):
                    continue
            candidate_actions = self._normalize_candidate_actions(item.get("actions"), horizon=horizon)
            if not candidate_actions or int(candidate_actions[0]) == int(action_code.STOP):
                continue
            if cfg.get("active_occ_require_action_diff", True) and candidate_actions == current_actions:
                continue
            occ_score = item.get("occ_memory_score") or {}
            if not occ_score.get("valid"):
                continue
            if int(occ_score.get("occupied_hit_count", 0) or 0) > 0:
                continue
            checked = int(occ_score.get("checked_cell_count", 0) or 0)
            unknown_hits = int(occ_score.get("unknown_hit_count", 0) or 0)
            if cfg.get("active_occ_reject_all_unknown", True) and checked > 0 and unknown_hits >= checked:
                continue
            deviation = self._occ_memory_candidate_deviation_deg(current_occ, occ_score)
            if deviation is None:
                continue
            if deviation > max_deviation:
                continue
            active_score = self._occ_memory_candidate_active_score(occ_score, deviation, cfg)
            enriched = dict(item)
            enriched["active_occ_score"] = float(active_score)
            enriched["active_direction_deviation_deg"] = float(deviation)
            enriched["active_selected_actions"] = candidate_actions
            eligible.append(enriched)

        if not eligible:
            return None, "no_occ_safe_intent_aligned_candidate"

        def candidate_key(item):
            occ_score = item.get("occ_memory_score") or {}
            return (
                float(item.get("active_occ_score", float("inf"))),
                int(occ_score.get("unknown_hit_count", 0) or 0),
                float(item.get("active_direction_deviation_deg", 180.0) or 180.0),
                int(item.get("candidate_index", 0)),
            )

        return min(eligible, key=candidate_key), "selected_occ_memory_conservative"

    def _select_nextdit_active_candidate(self, event: dict):
        cfg = self._get_nextdit_candidate_probe_cfg()
        if self._is_nextdit_occ_memory_active_strategy(cfg):
            return self._select_nextdit_occ_memory_active_candidate(event)

        if cfg.get("active_require_current_reject", True) and not bool(
            event.get("current_would_reject")
        ):
            return None, "current_not_reject"

        current_obstacle_score = float(event.get("current_obstacle_score", float("inf")))
        eligible = []
        for item in event.get("candidates") or []:
            decision = item.get("decision") or {}
            if not decision.get("valid"):
                continue
            if decision.get("would_reject"):
                continue
            obstacle_score = float(item.get("obstacle_score", self._trajectory_obstacle_score(decision)))
            if obstacle_score + 1e-6 >= current_obstacle_score:
                continue
            eligible.append(item)

        if not eligible:
            return None, "no_nonreject_improving_candidate"

        def candidate_key(item):
            decision = item.get("decision") or {}
            obstacle_score = float(item.get("obstacle_score", self._trajectory_obstacle_score(decision)))
            checked = int(decision.get("checked_forward_steps", 0) or 0)
            return (obstacle_score, -checked, int(item.get("candidate_index", 0)))

        return min(eligible, key=candidate_key), "selected"

    def _maybe_apply_nextdit_active_candidate(
        self,
        event: dict,
        *,
        active_intervention_count: int,
    ) -> dict:
        cfg = self._get_nextdit_candidate_probe_cfg()
        status = {
            "considered": False,
            "applied": False,
            "reason": "disabled",
        }
        if not cfg.get("active_enable") or not event:
            return status

        max_interventions = int(cfg.get("active_max_interventions_per_episode", 2))
        if max_interventions >= 0 and active_intervention_count >= max_interventions:
            status["reason"] = "episode_intervention_budget"
            return status
        occ_memory_strategy = self._is_nextdit_occ_memory_active_strategy(cfg)
        if (
            not occ_memory_strategy
            and cfg.get("active_require_current_reject", True)
            and not bool(event.get("current_would_reject"))
        ):
            status["reason"] = "current_not_reject"
            return status
        if occ_memory_strategy:
            current_occ = event.get("current_occ_memory_score") or {}
            if not current_occ.get("valid"):
                status["reason"] = "current_occ_invalid"
                return status
            occupied_hits = int(current_occ.get("occupied_hit_count", 0) or 0)
            if occupied_hits < int(cfg.get("active_occ_current_min_occupied_hits", 1)):
                status["reason"] = "current_occ_not_occupied"
                return status

        status["considered"] = True
        horizon = int(event.get("action_horizon", cfg.get("action_horizon", MAX_LOCAL_STEPS)))
        current_actions = self._normalize_candidate_actions(event.get("current_actions"), horizon=horizon)
        selected, reason = self._select_nextdit_active_candidate(event)
        if selected is None:
            active_event = {
                "event_type": "nextdit_active_rerank",
                "scene_id": event.get("scene_id"),
                "episode_id": event.get("episode_id"),
                "episode_index": event.get("episode_index"),
                "episode_count": event.get("episode_count"),
                "step_id": event.get("step_id"),
                "query_index": event.get("query_index"),
                "active_intervention_index": int(active_intervention_count + 1),
                "active_intervention_budget": int(max_interventions),
                "action_horizon": int(horizon),
                "current_actions": current_actions,
                "current_score": event.get("current_score"),
                "current_obstacle_score": event.get("current_obstacle_score"),
                "current_decision": event.get("current_decision"),
                "current_occ_memory_score": event.get("current_occ_memory_score"),
                "selected_candidate_index": None,
                "selected_actions": None,
                "selected_score": None,
                "selected_obstacle_score": None,
                "selected_decision": None,
                "selected_occ_memory_score": None,
                "selected_active_occ_score": None,
                "selected_direction_deviation_deg": None,
                "selected_differs_from_current": False,
                "candidate_count": event.get("candidate_count"),
                "would_reject_candidate_count": event.get("would_reject_candidate_count"),
                "occ_memory_score_valid_candidate_count": event.get(
                    "occ_memory_score_valid_candidate_count"
                ),
                "occ_memory_score_would_reject_candidate_count": event.get(
                    "occ_memory_score_would_reject_candidate_count"
                ),
                "unique_action_sequence_count": event.get("unique_action_sequence_count"),
                "unique_endpoint_count": event.get("unique_endpoint_count"),
                "applied": False,
                "reason": reason,
            }
            self._write_nextdit_active_rerank_event(active_event)
            status.update(active_event)
            return status

        selected_actions = self._normalize_candidate_actions(selected.get("actions"), horizon=horizon)
        active_event = {
            "event_type": "nextdit_active_rerank",
            "scene_id": event.get("scene_id"),
            "episode_id": event.get("episode_id"),
            "episode_index": event.get("episode_index"),
            "episode_count": event.get("episode_count"),
            "step_id": event.get("step_id"),
            "query_index": event.get("query_index"),
            "active_intervention_index": int(active_intervention_count + 1),
            "active_intervention_budget": int(max_interventions),
            "action_horizon": int(horizon),
            "current_actions": current_actions,
            "current_score": event.get("current_score"),
            "current_obstacle_score": event.get("current_obstacle_score"),
            "current_decision": event.get("current_decision"),
            "current_occ_memory_score": event.get("current_occ_memory_score"),
            "selected_candidate_index": int(selected.get("candidate_index", -1)),
            "selected_actions": selected_actions,
            "selected_score": selected.get("score"),
            "selected_obstacle_score": selected.get("obstacle_score"),
            "selected_decision": selected.get("decision"),
            "selected_occ_memory_score": selected.get("occ_memory_score"),
            "selected_active_occ_score": selected.get("active_occ_score"),
            "selected_direction_deviation_deg": selected.get("active_direction_deviation_deg"),
            "selected_differs_from_current": bool(selected_actions != current_actions),
            "candidate_count": event.get("candidate_count"),
            "would_reject_candidate_count": event.get("would_reject_candidate_count"),
            "occ_memory_score_valid_candidate_count": event.get(
                "occ_memory_score_valid_candidate_count"
            ),
            "occ_memory_score_would_reject_candidate_count": event.get(
                "occ_memory_score_would_reject_candidate_count"
            ),
            "unique_action_sequence_count": event.get("unique_action_sequence_count"),
            "unique_endpoint_count": event.get("unique_endpoint_count"),
            "applied": True,
            "reason": reason,
        }
        self._write_nextdit_active_rerank_event(active_event)
        print(
            "[NextDiTActiveRerank] "
            f"episode={event.get('episode_id')} step={event.get('step_id')} "
            f"candidate={active_event['selected_candidate_index']} "
            f"current={current_actions} selected={selected_actions} "
            f"score={active_event['current_obstacle_score']}->{active_event['selected_obstacle_score']} "
            f"occ={int((event.get('current_occ_memory_score') or {}).get('occupied_hit_count', 0) or 0)}"
            f"->{int((selected.get('occ_memory_score') or {}).get('occupied_hit_count', 0) or 0)} "
            f"dev={active_event.get('selected_direction_deviation_deg')}"
        )
        status.update(active_event)
        status["applied"] = True
        return status

    def _run_nextdit_candidate_probe_or_active(
        self,
        dp_actions,
        current_actions: list,
        current_decision: dict,
        observations: dict,
        depth_m: np.ndarray,
        rgb: Optional[np.ndarray],
        *,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
        step_id: int,
        query_index: int,
        pixel_goal,
        probe_event_count: int,
        active_intervention_count: int,
    ):
        cfg = self._get_nextdit_candidate_probe_cfg()
        probe_enabled = bool(cfg.get("enable"))
        current_occ_memory_score = None
        if cfg.get("occ_memory_score_enable"):
            current_occ_memory_score = self._score_nextdit_averaged_trajectory_with_occ_memory(
                dp_actions,
                observations,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_index=episode_index,
                episode_count=episode_count,
                step_id=step_id,
                query_index=query_index,
            )
        active_needed = self._nextdit_active_probe_needed(
            current_decision,
            active_intervention_count,
            current_occ_memory_score,
        )
        if not (probe_enabled or active_needed):
            return {}, {}, False

        max_events = int(cfg.get("max_events_per_episode", 0) or 0)
        should_record_probe = probe_enabled and (
            max_events <= 0 or probe_event_count < max_events
        )
        if not should_record_probe and not active_needed:
            return {}, {}, True

        event = self._probe_nextdit_trajectory_candidates(
            dp_actions,
            current_actions,
            current_decision,
            observations,
            depth_m,
            rgb,
            scene_id=scene_id,
            episode_id=episode_id,
            episode_index=episode_index,
            episode_count=episode_count,
            step_id=step_id,
            query_index=query_index,
            pixel_goal=pixel_goal,
            current_occ_memory_score=current_occ_memory_score,
        )
        active_status = {}
        if active_needed:
            if event:
                active_status = self._maybe_apply_nextdit_active_candidate(
                    event,
                    active_intervention_count=active_intervention_count,
                )
            else:
                active_status = {
                    "considered": True,
                    "applied": False,
                    "reason": "candidate_event_unavailable",
                }
        return event, active_status, False

    def _vlmap_goal_grid_from_decision(self, decision: dict):
        goal_grid = decision.get("goal_grid")
        if goal_grid is None:
            return None
        try:
            return (int(goal_grid[0]), int(goal_grid[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def _match_rejected_vlmap_goal(self, decision: dict, rejected_goal_grids: list):
        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
        if not bool(vlmap_safety_cfg.get("waypoint_requery_duplicate_suppression", True)):
            return None
        goal_grid = self._vlmap_goal_grid_from_decision(decision)
        if goal_grid is None:
            return None
        radius = max(0, int(vlmap_safety_cfg.get("waypoint_requery_repeat_grid_radius", 2)))
        for rejected_goal in rejected_goal_grids:
            dx = abs(goal_grid[0] - rejected_goal[0])
            dy = abs(goal_grid[1] - rejected_goal[1])
            if max(dx, dy) <= radius:
                return {
                    "goal_grid": goal_grid,
                    "rejected_goal_grid": rejected_goal,
                    "chebyshev_distance": max(dx, dy),
                }
        return None

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        collisions, collision_free, cf_sucs, cf_spls = [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw, collisions, collision_free, cf_sucs, cf_spls

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
                    collision_count = float(res.get("collision_count", 0.0) or 0.0)
                    collision_free_value = float(res.get("collision_free", 1.0 if collision_count <= 0.0 else 0.0))
                    collisions.append(collision_count)
                    collision_free.append(collision_free_value)
                    cf_sucs.append(float(res.get("cf_success", float(res["success"]) * collision_free_value)))
                    cf_spls.append(float(res.get("cf_spl", float(res["spl"]) * collision_free_value)))
        return sucs, spls, oss, nes, ndtw, collisions, collision_free, cf_sucs, cf_spls

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()

        # resume from previous results
        (
            sucs,
            spls,
            oss,
            nes,
            ndtw,
            collisions,
            collision_free,
            cf_sucs,
            cf_spls,
        ) = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        episode_count = len(self.env.episodes)
        process_bar = tqdm.tqdm(total=episode_count, desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break
            episode_index = max(0, getattr(self.env, "_current_episode_index", 1) - 1)

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            episode_eval_seed = self._seed_eval_rng_for_episode(
                episode_index, episode_id, scene_id
            )
            self._stage23a_initial_sim_position = None
            self._stage23a_initial_agent_matrix = None
            self._stage23a_mesh_raycast_errors = []
            self._stage23a_mesh_raycast_signed_errors = []
            self._stage23a_mesh_raycast_total = 0
            self._stage23a_mesh_raycast_hits = 0
            self._stage23a_mesh_raycast_misses = 0
            self._stage23a_mesh_gt_occ_voxels = set()
            self._stage23a_mesh_gt_free_voxels = set()
            self._stage23a_sim_pose_context(initialize=True)
            print("episode start", episode_instruction)
            self.vlmap_safety.reset()
            self._vlmap_last_nav_action = None
            self.vlmap_semantic.reset_episode(
                instruction=episode_instruction,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_index=episode_index,
                episode_count=episode_count,
            )
            self.occ_memory.reset_episode(
                instruction=episode_instruction,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_index=episode_index,
                episode_count=episode_count,
            )
            self.replay_ledger.reset_episode(
                instruction=episode_instruction,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_eval_seed=episode_eval_seed,
                episode_index=episode_index,
                episode_count=episode_count,
                rank=getattr(self, "rank", 0),
                world_size=getattr(self, "world_size", 1),
                camera_model={
                    "intrinsic": np.asarray(
                        self.occ_memory.camera_intrinsic, dtype=np.float32
                    ).tolist(),
                    "width": int(self.sim_sensors_config.depth_sensor.width),
                    "height": int(self.sim_sensors_config.depth_sensor.height),
                    "hfov_deg": float(self.sim_sensors_config.depth_sensor.hfov),
                    "depth_min_m": float(self._min_depth),
                    "depth_max_m": float(self._max_depth),
                    "depth_convention": "metric_z_depth_optical_camera",
                    "optical_axes": "+x_right,+y_down,+z_forward",
                },
                semantic_scene_gt=self._stage24a_semantic_scene_snapshot(),
                coordinate_transforms={
                    "map_to_habitat_world": (
                        (
                            np.asarray(self._stage23a_initial_agent_matrix, dtype=np.float32)
                            @ np.asarray(
                                [
                                    [0.0, 0.0, -1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ], dtype=np.float32
                            ).T
                        ).tolist()
                        if self._stage23a_initial_agent_matrix is not None else None
                    ),
                    "map_frame": "sparse_occ_map",
                    "gt_frame": "habitat_world",
                },
            )
            self.online_lseg_shadow.reset_episode(
                instruction=episode_instruction,
                scene_id=scene_id,
                episode_id=episode_id,
                episode_eval_seed=episode_eval_seed,
                episode_index=episode_index,
                episode_count=episode_count,
                rank=getattr(self, "rank", 0),
                world_size=getattr(self, "world_size", 1),
                semantic_scene_gt=self._stage24a_semantic_scene_snapshot(),
                coordinate_transforms={
                    "map_to_habitat_world": (
                        (
                            np.asarray(
                                self._stage23a_initial_agent_matrix,
                                dtype=np.float32,
                            )
                            @ np.asarray(
                                [
                                    [0.0, 0.0, -1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            ).T
                        ).tolist()
                        if self._stage23a_initial_agent_matrix is not None
                        else None
                    ),
                    "map_frame": "sparse_occ_map",
                    "gt_frame": "habitat_world",
                },
            )
            if self.occ_memory_oracle_pose is not None:
                self.occ_memory_oracle_pose.reset_episode(
                    instruction=episode_instruction,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    episode_count=episode_count,
                )
            if self.occ_memory_oracle_sensor_pose is not None:
                self.occ_memory_oracle_sensor_pose.reset_episode(
                    instruction=episode_instruction,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    episode_count=episode_count,
                )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            rgb_list = []
            rgb_frame_records = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            last_s2_query_step = None
            action = None
            action_source = "not_selected"
            pre_safety_action = None
            last_action_applied = False
            replay_observation_index = 0
            replay_query_index = 0
            replay_previous_collision_count = 0.0
            replay_previous_action_collision_count = 0.0

            def replay_action_audit_metrics():
                nonlocal replay_previous_action_collision_count
                replay_action_metrics = self.env.get_metrics()
                replay_action_collision_summary = self._extract_collision_summary(
                    replay_action_metrics, steps=step_id
                )
                replay_action_collision_count = float(
                    replay_action_collision_summary.get("collision_count", 0.0)
                    or 0.0
                )
                result = {
                    "distance_to_goal": replay_action_metrics.get(
                        "distance_to_goal"
                    ),
                    "success": replay_action_metrics.get("success"),
                    "collision_count": replay_action_collision_count,
                    "collision_delta": max(
                        0.0,
                        replay_action_collision_count
                        - replay_previous_action_collision_count,
                    ),
                }
                replay_previous_action_collision_count = (
                    replay_action_collision_count
                )
                return result
            history_id = []
            vlmap_safety_decision = {}
            messages = []
            local_actions = []
            vlmap_recovery_actions = []
            pending_vlmap_waypoint_feedback = ""
            pending_vlmap_semantic_hint = ""
            pending_occ_memory_guidance_hint = ""
            pending_s2_recovery_context = None
            s2_recovery_context_set_count = 0
            s2_recovery_context_injected_count = 0
            s2_recovery_context_counterfactual_count = 0
            s2_recovery_context_changed_count = 0
            s2_recovery_context_expired_count = 0
            s2_loop_strict_active_event_count = 0
            s2_loop_strict_active_rewrite_count = 0
            s2_loop_strict_active_applied_count = 0
            s2_loop_strict_active_first_step = None
            pending_s2_loop_strict_active_execution = None
            s2_loop_projection_bridge_event_count = 0
            s2_loop_projection_bridge_strict_count = 0
            s2_loop_projection_bridge_valid_count = 0
            s2_loop_path_reobserve_event_count = 0
            s2_loop_path_reobserve_intervention_count = 0
            s2_loop_path_reobserve_reorient_count = 0
            s2_loop_path_reobserve_post_query_count = 0
            s2_loop_path_reobserve_pixel_rewrite_count = 0
            s2_loop_path_reobserve_applied_count = 0
            s2_loop_path_reobserve_first_step = None
            pending_s2_loop_path_reobserve = None
            pending_s2_loop_path_execution = None
            semantic_hint_set_count = 0
            semantic_hint_injected_count = 0
            semantic_hint_detection_step = None
            semantic_hint_injection_step = None
            semantic_hint_not_injected_reason = None
            occ_memory_guidance_trigger_count = 0
            occ_memory_guidance_hint_set_count = 0
            occ_memory_guidance_hint_injected_count = 0
            occ_memory_guidance_requery_count = 0
            occ_memory_guidance_shadow_skip_count = 0
            occ_memory_guidance_blocked_count = 0
            occ_memory_guidance_detection_step = None
            occ_memory_guidance_injection_step = None
            occ_memory_guidance_last_set_step = None
            occ_memory_guidance_not_injected_reason = None
            occ_memory_guidance_counterfactual_count = 0
            occ_memory_guidance_counterfactual_valid_count = 0
            occ_memory_guidance_counterfactual_changed_count = 0
            occ_memory_guidance_counterfactual_direction_changed_count = 0
            occ_memory_guidance_counterfactual_left_right_follow_count = 0
            occ_memory_guidance_counterfactual_pixel_shift_sum = 0.0
            som_counterfactual_count = 0
            som_counterfactual_valid_count = 0
            som_counterfactual_changed_count = 0
            som_counterfactual_direction_changed_count = 0
            som_counterfactual_frontier_follow_count = 0
            som_counterfactual_unsafe_shift_proxy_count = 0
            som_counterfactual_skipped_count = 0
            som_counterfactual_error_count = 0
            som_counterfactual_pixel_shift_sum = 0.0
            som_counterfactual_active_applied_count = 0
            stage15_repair_consecutive_count = 0
            stage15_repair_cumulative_count = 0
            stage15_repair_active_event_count = 0
            stage15_repair_active_applied_count = 0
            stage15_repair_active_first_step = None
            stage15_repair_active_reason_counts = {}
            stage_d_bfs_escape_event_count = 0
            stage_d_bfs_escape_trigger_count = 0
            stage_d_bfs_escape_reachable_count = 0
            stage_d_bfs_escape_first_trigger_step = None
            stage_d_bfs_escape_reason_counts = {}
            stage_d_bfs_escape_trigger_reason_counts = {}
            stage_d_bfs_trajectory_cache = []
            stage_d_bfs_escape_active_event_count = 0
            stage_d_bfs_escape_active_applied_count = 0
            stage_d_bfs_escape_active_first_step = None
            stage_d_bfs_escape_active_reason_counts = {}
            occ_memory_candidate_probe_event_count = 0
            occ_memory_candidate_probe_valid_event_count = 0
            occ_memory_candidate_probe_skipped_count = 0
            occ_memory_candidate_probe_candidate_sum = 0
            occ_memory_candidate_probe_geometry_safe_sum = 0
            occ_memory_candidate_probe_active_gate_safe_sum = 0
            occ_memory_candidate_probe_current_aligned_sum = 0
            occ_memory_candidate_probe_next_landmark_relevant_sum = 0
            occ_memory_candidate_probe_completed_landmark_sum = 0
            occ_memory_candidate_probe_repeated_semantic_sum = 0
            occ_memory_candidate_probe_unknown_target_frontier_bonus_sum = 0
            occ_memory_candidate_probe_target_frontier_sum = 0
            occ_memory_candidate_probe_target_frontier_escape_sum = 0
            occ_memory_candidate_probe_target_frontier_intent_safe_sum = 0
            occ_memory_candidate_probe_target_frontier_doorway_like_sum = 0
            occ_memory_candidate_selection_query_count = 0
            occ_memory_candidate_selection_valid_count = 0
            occ_memory_candidate_selection_none_count = 0
            occ_memory_candidate_selection_active_gate_safe_count = 0
            occ_memory_candidate_selection_current_aligned_count = 0
            occ_memory_candidate_selection_error_count = 0
            occ_memory_candidate_selection_label_count = 0
            occ_memory_candidate_selection_coordinate_count = 0
            occ_memory_candidate_selection_direction_count = 0
            occ_memory_candidate_selection_semanticized_count = 0
            occ_memory_candidate_selection_instruction_relevant_count = 0
            occ_memory_candidate_selection_next_landmark_relevant_count = 0
            occ_memory_candidate_selection_completed_landmark_count = 0
            occ_memory_candidate_selection_repeated_semantic_count = 0
            rejected_vlmap_goal_grids = []
            s2_candidate_probe_s2_query_count = 0
            s2_candidate_probe_event_count = 0
            s2_candidate_probe_skipped_query_count = 0
            s2_candidate_probe_valid_query_count = 0
            s2_candidate_probe_diverse_query_count = 0
            s2_candidate_probe_valid_candidate_sum = 0
            s2_candidate_probe_unique_candidate_sum = 0
            s2_candidate_probe_mean_pairwise_distance_sum = 0.0
            s2_candidate_probe_max_pairwise_distance = 0.0
            s2_action_loop_state = init_s2_action_loop_state()
            nextdit_candidate_probe_event_count = 0
            nextdit_candidate_probe_skipped_count = 0
            nextdit_candidate_probe_candidate_sum = 0
            nextdit_candidate_probe_unique_action_sum = 0
            nextdit_candidate_probe_unique_endpoint_sum = 0
            nextdit_candidate_probe_safer_event_count = 0
            nextdit_candidate_probe_selected_diff_count = 0
            nextdit_candidate_probe_current_reject_count = 0
            nextdit_candidate_probe_would_reject_candidate_sum = 0
            nextdit_candidate_occ_valid_candidate_sum = 0
            nextdit_candidate_occ_invalid_candidate_sum = 0
            nextdit_candidate_occ_would_reject_candidate_sum = 0
            nextdit_candidate_occ_unknown_candidate_sum = 0
            nextdit_candidate_occ_checked_cell_sum = 0.0
            nextdit_candidate_occ_occupied_hit_sum = 0.0
            nextdit_candidate_occ_unknown_hit_sum = 0.0
            nextdit_candidate_occ_current_valid_event_count = 0
            nextdit_candidate_occ_current_would_reject_event_count = 0
            nextdit_candidate_occ_current_occupied_hit_sum = 0.0
            nextdit_candidate_occ_current_unknown_hit_sum = 0.0
            nextdit_candidate_active_considered_count = 0
            nextdit_candidate_active_intervention_count = 0
            nextdit_candidate_active_changed_count = 0
            nextdit_candidate_active_no_candidate_count = 0
            occ_memory_recovery_state = self._init_occ_memory_recovery_state()
            occ_memory_recovery_active_count = 0
            failure_prediction_state = self._init_failure_prediction_state()
            stage19_semantic_resilience_active_considered_count = 0
            stage19_semantic_resilience_active_applied_count = 0
            stage19_semantic_resilience_active_suppressed_count = 0
            stage19_semantic_resilience_active_first_step = None
            stage19_semantic_resilience_active_last_step = None
            stage19_semantic_resilience_active_action_sum = 0
            stage19_semantic_resilience_active_reason_counts = {}
            stage19_semantic_resilience_failure_type_counts = {}
            stage19_semantic_resilience_recommended_primitive_counts = {}

            done = False
            flag = False
            pixel_goal = None
            forward_action = 0

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                current_depth_m = depth.copy()
                depth = current_depth_m * 1000
                occ_memory_obs = {
                    "rgb": rgb,
                    "depth": current_depth_m,
                    "gps": observations.get("gps"),
                    "compass": observations.get("compass"),
                }
                occ_memory_context = {
                    "step_id": step_id,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "episode_count": episode_count,
                    # At loop entry, `action` is the previous Habitat
                    # action. LOOKDOWN is applied twice per visual tilt,
                    # so this observation is approximately 2*tilt below
                    # horizon; normal observations are horizon-facing.
                    "camera_pitch_deg": (
                        2.0 * self._tilt_angle_deg
                        if action == action_code.LOOKDOWN
                        else 0.0
                    ),
                    **self._stage23a_sim_pose_context(),
                }
                self._stage23a_mesh_raycast_audit(
                    current_depth_m, occ_memory_context
                )
                occ_memory_update_event = self.occ_memory.update_observation(
                    occ_memory_obs,
                    current_depth_m,
                    rgb=rgb,
                    context=occ_memory_context,
                )
                # Stage45 needs the current oracle floor pose before it writes
                # the offline rejection audit. This isolated branch is never
                # read by navigation and is updated exactly once per frame.
                if (
                    self._stage45_candidate_rejection_truth_enable
                    and self.occ_memory_oracle_sensor_pose is not None
                ):
                    self.occ_memory_oracle_sensor_pose.update_observation(
                        occ_memory_obs,
                        current_depth_m,
                        rgb=rgb,
                        context={
                            **occ_memory_context,
                            "stage23a_map_branch": "oracle_sensor_pose",
                            "gt_fields_used": [
                                "habitat_sensor_state_position",
                                "habitat_sensor_state_rotation",
                            ],
                        },
                    )
                # Stage27 is invoked only for the pre-registered detector
                # event steps. It serializes shadow evidence and cannot affect
                # the frozen action path.
                self._maybe_write_stage27_candidate_audit(
                    scene_id=scene_id,
                    episode_id=int(episode_id),
                    episode_index=int(episode_index),
                    episode_count=int(episode_count),
                    episode_eval_seed=episode_eval_seed,
                    step_id=int(step_id),
                    observations=observations,
                    depth_m=current_depth_m,
                )
                # Stage24A is audit-only.  Keep the raw RGB-D in the ledger while
                # indexing only compact map/pose state in JSONL; no ledger field is
                # read by the action-selection path.
                replay_pose = {
                    "gps": observations.get("gps"),
                    "compass": observations.get("compass"),
                    "camera_pitch_deg": occ_memory_context.get("camera_pitch_deg"),
                    "stage23a_sim_position": occ_memory_context.get(
                        "stage23a_sim_position"
                    ),
                    "stage23a_sim_rotation_wxyz": occ_memory_context.get(
                        "stage23a_sim_rotation_wxyz"
                    ),
                    "stage23a_gt_relative_height_m": occ_memory_context.get(
                        "stage23a_gt_relative_height_m"
                    ),
                    "stage23a_sensor_position": occ_memory_context.get(
                        "stage23a_sensor_position"
                    ),
                    "stage23a_sensor_rotation_wxyz": occ_memory_context.get(
                        "stage23a_sensor_rotation_wxyz"
                    ),
                    "stage23_gt_camera_pose_map": occ_memory_context.get(
                        "stage23_gt_camera_pose_map"
                    ),
                    "stage23_gt_base_pose_map": occ_memory_context.get(
                        "stage23_gt_base_pose_map"
                    ),
                }
                replay_occ_summary = {
                    key: occ_memory_update_event.get(key)
                    for key in (
                        "event_type",
                        "valid",
                        "reason",
                        "update_count",
                        "sampled_point_count",
                        "occupied_added",
                        "free_added",
                        "occupied_voxel_count",
                        "free_voxel_count",
                        "occupied_cell_count",
                        "free_cell_count",
                        "frontier_count",
                        "pose_grid",
                        "requested_camera_pitch_deg",
                        "applied_camera_pitch_deg",
                        "validation_endpoint_total_count",
                        "validation_endpoint_mapped_count",
                    )
                    if key in occ_memory_update_event
                }
                replay_route_node = {
                    "pose_grid": occ_memory_update_event.get("pose_grid"),
                    "step_id": int(step_id),
                    "gps": observations.get("gps"),
                    "compass": observations.get("compass"),
                }
                replay_metrics = self.env.get_metrics()
                replay_collision_summary = self._extract_collision_summary(
                    replay_metrics, steps=step_id
                )
                replay_collision_count = float(
                    replay_collision_summary.get("collision_count", 0.0) or 0.0
                )
                replay_audit_metrics = {
                    "distance_to_goal": replay_metrics.get("distance_to_goal"),
                    "success": replay_metrics.get("success"),
                    "oracle_success": replay_metrics.get("oracle_success"),
                    "spl": replay_metrics.get("spl"),
                    "collision_count": replay_collision_count,
                    "collision_delta": max(
                        0.0,
                        replay_collision_count - replay_previous_collision_count,
                    ),
                }
                replay_previous_collision_count = replay_collision_count
                self.replay_ledger.record_observation(
                    step_id=step_id,
                    observation_index=replay_observation_index,
                    rgb=rgb,
                    depth=current_depth_m,
                    pose=replay_pose,
                    camera_pitch_deg=occ_memory_context.get("camera_pitch_deg"),
                    previous_action=action,
                    previous_action_source=action_source,
                    previous_pre_safety_action=pre_safety_action,
                    previous_action_applied=last_action_applied,
                    route_node=replay_route_node,
                    occ_summary=replay_occ_summary,
                    semantic_state=dict(self.occ_memory.last_semantic_decision or {}),
                    audit_metrics=replay_audit_metrics,
                )
                replay_observation_index += 1
                if self.occ_memory_oracle_pose is not None:
                    self.occ_memory_oracle_pose.update_observation(
                        occ_memory_obs,
                        current_depth_m,
                        rgb=rgb,
                        context={
                            **occ_memory_context,
                            "stage23a_map_branch": "oracle_relative_height",
                            "gt_fields_used": ["habitat_sim_agent_position_y"],
                        },
                    )
                if (
                    self.occ_memory_oracle_sensor_pose is not None
                    and not self._stage45_candidate_rejection_truth_enable
                ):
                    self.occ_memory_oracle_sensor_pose.update_observation(
                        occ_memory_obs,
                        current_depth_m,
                        rgb=rgb,
                        context={
                            **occ_memory_context,
                            "stage23a_map_branch": "oracle_sensor_pose",
                            "gt_fields_used": [
                                "habitat_sensor_state_position",
                                "habitat_sensor_state_rotation",
                            ],
                        },
                    )
                occ_memory_recovery_event = self._update_occ_memory_recovery_shadow(
                    occ_memory_recovery_state,
                    update_event=occ_memory_update_event,
                    metrics=self.env.get_metrics(),
                    step_id=step_id,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    episode_count=episode_count,
                    last_action=action,
                    pixel_goal=pixel_goal,
                    local_actions=local_actions,
                    action_seq=action_seq,
                    vlmap_recovery_actions=vlmap_recovery_actions,
                )
                failure_prediction_event = self._update_failure_prediction_shadow(
                    state=failure_prediction_state,
                    occ_event=occ_memory_recovery_event,
                    step_id=step_id,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    episode_count=episode_count,
                )
                occ_memory_recovery_active_status = self._maybe_apply_occ_memory_recovery(
                    occ_memory_recovery_state,
                    occ_memory_recovery_event,
                    step_id=step_id,
                    active_count=occ_memory_recovery_active_count,
                )
                if occ_memory_recovery_active_status.get("applied"):
                    occ_memory_recovery_active_count += 1
                    vlmap_recovery_actions = list(occ_memory_recovery_active_status.get("actions") or [])
                    if self._get_occ_memory_recovery_cfg().get("escape_clear_goal"):
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        input_images = []
                        llm_outputs = ""
                        local_actions = []
                        action_seq = []
                        forward_action = 0
                        draw_pixel_goal = False
                        flag = False
                    print(
                        "[OccMemoryRecovery][Habitat] queue recovery actions "
                        f"{vlmap_recovery_actions} at step {step_id}"
                    )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)
                    rgb_frame_records.append(
                        {"step_id": int(step_id), "image": image}
                    )

                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)

                if len(vlmap_recovery_actions) == 0 and len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    recovery_context = pending_s2_recovery_context
                    recovery_cfg = self._get_s2_action_loop_cfg()
                    if recovery_context and not recovery_cfg.get("recovery_context_shadow_only"):
                        sources[0]["value"] += (
                            " "
                            + self._recovery_context_prompt(
                                recovery_context, "text_images"
                            )
                        )
                        input_images.extend(list(recovery_context.get("images") or []))
                        s2_recovery_context_injected_count += 1
                        remaining = int(
                            recovery_context.get("remaining_queries", 1) or 1
                        ) - 1
                        recovery_context["remaining_queries"] = remaining
                        if remaining <= 0:
                            pending_s2_recovery_context = None
                            s2_recovery_context_expired_count += 1

                    if pending_vlmap_waypoint_feedback:
                        sources[0]["value"] += f" {pending_vlmap_waypoint_feedback}"
                        print(
                            "[VLMapSafety][Habitat][Waypoint] inject S2 feedback: "
                            f"{pending_vlmap_waypoint_feedback}"
                        )
                        pending_vlmap_waypoint_feedback = ""

                    if pending_occ_memory_guidance_hint:
                        sources[0]["value"] += f" {pending_occ_memory_guidance_hint}"
                        occ_memory_guidance_hint_injected_count += 1
                        if occ_memory_guidance_injection_step is None:
                            occ_memory_guidance_injection_step = step_id
                        self.occ_memory.record_guidance_event(
                            action="injected",
                            hint=pending_occ_memory_guidance_hint,
                            context={
                                "step_id": step_id,
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                            },
                        )
                        print(
                            "[OccMemory][Habitat][Guidance] inject S2 hint: "
                            f"{pending_occ_memory_guidance_hint}"
                        )
                        pending_occ_memory_guidance_hint = ""
                        occ_memory_guidance_not_injected_reason = None

                    if pending_vlmap_semantic_hint:
                        sources[0]["value"] += f" {pending_vlmap_semantic_hint}"
                        semantic_hint_injected_count += 1
                        if semantic_hint_injection_step is None:
                            semantic_hint_injection_step = step_id
                        print(
                            "[VLMapSemantic][Habitat][Stagnation] inject S2 hint: "
                            f"{pending_vlmap_semantic_hint}"
                        )
                        pending_vlmap_semantic_hint = ""
                        semantic_hint_not_injected_reason = None

                    s2_prompt_body_before_final_prompt = copy.deepcopy(sources[0]["value"])
                    prompt = self._select_s2_prompt_prefix() + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    s2_counterfactual_messages_prefix = list(messages)
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    last_s2_query_step = int(step_id)
                    query_pixel_goal = None
                    query_coordinates = [
                        int(value) for value in re.findall(r"\d+", llm_outputs)
                    ]
                    if len(query_coordinates) >= 2:
                        query_pixel_goal = [
                            int(query_coordinates[1]),
                            int(query_coordinates[0]),
                        ]
                    self.replay_ledger.record_query(
                        step_id=step_id,
                        query_id=replay_query_index,
                        output=llm_outputs,
                        pixel_goal=query_pixel_goal,
                        input_steps={
                            "history_rgb_indices": [
                                int(item) for item in (history_id or [])
                            ],
                            "history_steps": [
                                int(rgb_frame_records[item].get("step_id"))
                                for item in (history_id or [])
                                if 0 <= int(item) < len(rgb_frame_records)
                            ],
                            "current_step": int(step_id),
                            "observation_index": int(replay_observation_index - 1),
                        },
                        semantic_state=dict(self.occ_memory.last_semantic_decision or {}),
                    )
                    # Stage24D runs only after Frozen S2 has produced this query's
                    # output. Its result is intentionally discarded by every
                    # prompt, gate, candidate, and action-selection branch.
                    self.online_lseg_shadow.process_query_frame(
                        rgb=rgb,
                        depth_m=current_depth_m,
                        camera_pose_map=occ_memory_context.get(
                            "stage23_gt_camera_pose_map"
                        ),
                        step_id=int(step_id),
                        query_id=int(replay_query_index),
                        observation_index=int(replay_observation_index - 1),
                        occ_memory=self.occ_memory,
                    )
                    replay_query_index += 1
                    print('step_id:', step_id, 'output text:', llm_outputs)
                    loop_observer_output = llm_outputs

                    if pending_s2_loop_path_reobserve is not None:
                        path_post_event = (
                            self._plan_s2_loop_path_reobserve_post_observation(
                                pending_s2_loop_path_reobserve,
                                observations=observations,
                                depth_m=current_depth_m,
                                step_id=int(step_id),
                            )
                        )
                        path_post_event["base_s2_output"] = llm_outputs
                        s2_loop_path_reobserve_post_query_count += 1
                        if path_post_event.get("action_applied"):
                            s2_loop_path_reobserve_applied_count += 1
                            if s2_loop_path_reobserve_first_step is None:
                                s2_loop_path_reobserve_first_step = int(
                                    path_post_event.get("trigger_step", step_id)
                                )
                        if path_post_event.get("reobserve_pending"):
                            reorient_actions = [
                                int(item)
                                for item in list(
                                    path_post_event.get("reorient_actions") or []
                                )
                                if int(item) in (
                                    int(action_code.LEFT),
                                    int(action_code.RIGHT),
                                )
                            ]
                            if len(reorient_actions) == 1:
                                path_post_event["reorient_actions"] = list(
                                    reorient_actions
                                )
                                path_post_event["rgb_file"] = (
                                    self._save_s2_loop_path_reobserve_snapshot(
                                        path_post_event, observations
                                    )
                                )
                                self._write_s2_loop_path_reobserve_event(
                                    path_post_event
                                )
                                next_pending = dict(path_post_event)
                                next_pending["reorient_actions_applied"] = []
                                pending_s2_loop_path_reobserve = next_pending
                                s2_loop_path_reobserve_reorient_count += 1
                                vlmap_recovery_actions = list(reorient_actions)
                                pixel_goal = None
                                output_ids = None
                                traj_latents = None
                                pix_goal_image = None
                                pix_goal_depth = None
                                messages = []
                                input_images = []
                                llm_outputs = ""
                                local_actions = []
                                action_seq = []
                                action = None
                                forward_action = 0
                                draw_pixel_goal = False
                                flag = False
                                print(
                                    "[S2LoopPathReobserve] "
                                    f"episode={scene_id}/{episode_id} step={step_id} "
                                    f"iterative_queue={reorient_actions} "
                                    f"primitive_count={path_post_event.get('reorient_primitive_count')} "
                                    f"path_angle={path_post_event.get('reorient_angle_deg')}",
                                    flush=True,
                                )
                                continue
                            path_post_event.update(
                                {
                                    "reason": "invalid_iterative_reorient_queue_hold",
                                    "reobserve_pending": False,
                                }
                            )
                        if path_post_event.get("execution_pending"):
                            planned_goal = list(
                                path_post_event.get("selected_pixel_goal") or []
                            )
                            try:
                                if len(planned_goal) != 2:
                                    raise ValueError("invalid_path_pixel_goal")
                                goal_x, goal_y = int(planned_goal[0]), int(planned_goal[1])
                                replacement_text = f"{goal_y} {goal_x}"
                                replacement_ids = self.processor.tokenizer(
                                    replacement_text,
                                    add_special_tokens=False,
                                    return_tensors="pt",
                                ).input_ids.to(output_ids.device)
                                if replacement_ids.numel() <= 0:
                                    raise ValueError("empty_recovery_pixel_tokens")
                                prompt_len = int(inputs.input_ids.shape[1])
                                output_ids = torch.cat(
                                    [output_ids[:, :prompt_len], replacement_ids], dim=1
                                )
                                llm_outputs = replacement_text
                            except Exception as exc:
                                path_post_event.update(
                                    {
                                        "reason": "post_reobserve_output_rewrite_failed",
                                        "execution_pending": False,
                                        "execution_error_type": type(exc).__name__,
                                        "execution_error": str(exc),
                                    }
                                )
                            else:
                                path_post_event.update(
                                    {
                                        "reason": "post_reobserve_output_rewritten_pending_trajectory",
                                        "output_rewritten": True,
                                        "execution_pending": True,
                                        "recovery_s2_output": replacement_text,
                                        "executed_pixel_goal": [goal_x, goal_y],
                                        "intervention_already_applied": True,
                                    }
                                )
                                s2_loop_path_reobserve_pixel_rewrite_count += 1
                                pending_s2_loop_path_execution = dict(path_post_event)
                        path_post_event["rgb_file"] = (
                            self._save_s2_loop_path_reobserve_snapshot(
                                path_post_event, observations
                            )
                        )
                        self._write_s2_loop_path_reobserve_event(path_post_event)
                        pending_s2_loop_path_reobserve = None

                    s2_loop_event = self._observe_s2_action_loop_shadow(
                        state=s2_action_loop_state,
                        output=loop_observer_output,
                        observations=observations,
                        depth_m=current_depth_m,
                        step_id=step_id,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        episode_index=episode_index,
                        episode_count=episode_count,
                        episode_eval_seed=episode_eval_seed,
                    )

                    if s2_loop_event and s2_loop_event.get("transition") == "start":
                        projection_bridge_event = (
                            self._plan_s2_loop_projection_bridge_shadow(
                                s2_loop_event,
                                observations=observations,
                                depth_m=current_depth_m,
                            )
                        )
                        if projection_bridge_event.get("enabled"):
                            projection_bridge_event["base_s2_output"] = llm_outputs
                            s2_loop_projection_bridge_event_count += 1
                            if (
                                projection_bridge_event.get("triage_tier")
                                == "strict_intervention"
                            ):
                                s2_loop_projection_bridge_strict_count += 1
                            if projection_bridge_event.get("proposal_valid"):
                                s2_loop_projection_bridge_valid_count += 1
                            self._write_s2_loop_projection_bridge_event(
                                projection_bridge_event
                            )

                        path_reobserve_event = (
                            self._plan_s2_loop_path_reobserve_active(
                                s2_loop_event,
                                s2_loop_path_reobserve_intervention_count,
                                observations=observations,
                                depth_m=current_depth_m,
                            )
                        )
                        if path_reobserve_event.get("enabled"):
                            s2_loop_path_reobserve_event_count += 1
                            path_reobserve_event["base_s2_output"] = llm_outputs
                            if path_reobserve_event.get("reobserve_pending"):
                                reorient_actions = [
                                    int(item)
                                    for item in list(
                                        path_reobserve_event.get("reorient_actions") or []
                                    )
                                    if int(item) in (
                                        int(action_code.LEFT),
                                        int(action_code.RIGHT),
                                    )
                                ]
                                if reorient_actions:
                                    path_reobserve_event["reorient_actions"] = list(
                                        reorient_actions
                                    )
                                    path_reobserve_event["reorient_actions_applied"] = []
                                    pending_s2_loop_path_reobserve = dict(
                                        path_reobserve_event
                                    )
                                    s2_loop_path_reobserve_intervention_count += 1
                                    s2_loop_path_reobserve_reorient_count += 1
                                    vlmap_recovery_actions = list(reorient_actions)
                                    pixel_goal = None
                                    output_ids = None
                                    traj_latents = None
                                    pix_goal_image = None
                                    pix_goal_depth = None
                                    messages = []
                                    input_images = []
                                    llm_outputs = ""
                                    local_actions = []
                                    action_seq = []
                                    action = None
                                    forward_action = 0
                                    draw_pixel_goal = False
                                    flag = False
                                    self._write_s2_loop_path_reobserve_event(
                                        path_reobserve_event
                                    )
                                    print(
                                        "[S2LoopPathReobserve] "
                                        f"episode={scene_id}/{episode_id} step={step_id} "
                                        f"queue={reorient_actions} "
                                        f"path_angle={path_reobserve_event.get('reorient_angle_deg')}",
                                        flush=True,
                                    )
                                    continue
                                path_reobserve_event.update(
                                    {
                                        "reason": "empty_reorient_queue_hold",
                                        "reobserve_pending": False,
                                    }
                                )
                            if path_reobserve_event.get("execution_pending"):
                                planned_goal = list(
                                    path_reobserve_event.get("selected_pixel_goal") or []
                                )
                                try:
                                    if len(planned_goal) != 2:
                                        raise ValueError("invalid_path_pixel_goal")
                                    goal_x, goal_y = int(planned_goal[0]), int(planned_goal[1])
                                    replacement_text = f"{goal_y} {goal_x}"
                                    replacement_ids = self.processor.tokenizer(
                                        replacement_text,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    ).input_ids.to(output_ids.device)
                                    if replacement_ids.numel() <= 0:
                                        raise ValueError("empty_recovery_pixel_tokens")
                                    prompt_len = int(inputs.input_ids.shape[1])
                                    output_ids = torch.cat(
                                        [output_ids[:, :prompt_len], replacement_ids], dim=1
                                    )
                                    llm_outputs = replacement_text
                                except Exception as exc:
                                    path_reobserve_event.update(
                                        {
                                            "reason": "path_output_rewrite_failed",
                                            "execution_pending": False,
                                            "execution_error_type": type(exc).__name__,
                                            "execution_error": str(exc),
                                        }
                                    )
                                else:
                                    path_reobserve_event.update(
                                        {
                                            "reason": "path_output_rewritten_pending_trajectory",
                                            "output_rewritten": True,
                                            "execution_pending": True,
                                            "recovery_s2_output": replacement_text,
                                            "executed_pixel_goal": [goal_x, goal_y],
                                            "intervention_already_applied": False,
                                        }
                                    )
                                    s2_loop_path_reobserve_intervention_count += 1
                                    s2_loop_path_reobserve_pixel_rewrite_count += 1
                                    pending_s2_loop_path_execution = dict(
                                        path_reobserve_event
                                    )
                            self._write_s2_loop_path_reobserve_event(
                                path_reobserve_event
                            )

                        strict_active_event = self._plan_s2_loop_strict_active(
                            s2_loop_event,
                            s2_loop_strict_active_rewrite_count,
                            observations=observations,
                            depth_m=current_depth_m,
                        )
                        if strict_active_event.get("enabled"):
                            s2_loop_strict_active_event_count += 1
                            strict_active_event["base_s2_output"] = llm_outputs
                            if strict_active_event.get("execution_pending"):
                                plan = dict(strict_active_event.get("pixel_goal_plan") or {})
                                planned_goal = list(plan.get("pixel_goal") or [])
                                try:
                                    if len(planned_goal) != 2:
                                        raise ValueError("invalid_pixel_goal")
                                    goal_x, goal_y = int(planned_goal[0]), int(planned_goal[1])
                                    replacement_text = f"{goal_y} {goal_x}"
                                    replacement_ids = self.processor.tokenizer(
                                        replacement_text,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    ).input_ids.to(output_ids.device)
                                    if replacement_ids.numel() <= 0:
                                        raise ValueError("empty_recovery_pixel_tokens")
                                    prompt_len = int(inputs.input_ids.shape[1])
                                    output_ids = torch.cat(
                                        [output_ids[:, :prompt_len], replacement_ids], dim=1
                                    )
                                    llm_outputs = replacement_text
                                except Exception as exc:
                                    strict_active_event.update(
                                        {
                                            "reason": "output_rewrite_failed",
                                            "execution_pending": False,
                                            "execution_error_type": type(exc).__name__,
                                            "execution_error": str(exc),
                                        }
                                    )
                                else:
                                    strict_active_event.update(
                                        {
                                            "reason": "output_rewritten_pending_trajectory",
                                            "action_applied": False,
                                            "output_rewritten": True,
                                            "execution_pending": True,
                                            "recovery_s2_output": replacement_text,
                                            "executed_pixel_goal": [goal_x, goal_y],
                                        }
                                    )
                                    s2_loop_strict_active_rewrite_count += 1
                                    pending_s2_loop_strict_active_execution = dict(
                                        strict_active_event
                                    )
                                    print(
                                        "[S2LoopStrictActive] "
                                        f"episode={scene_id}/{episode_id} step={step_id} "
                                        f"base={strict_active_event.get('base_s2_output')} "
                                        f"recovery={replacement_text}",
                                        flush=True,
                                    )
                            self._write_s2_loop_strict_active_event(strict_active_event)

                        recovery_context = self._build_s2_recovery_context(
                            s2_loop_event,
                            frame_records=rgb_frame_records,
                            current_image=image,
                        )
                        if recovery_context is not None:
                            pending_s2_recovery_context = recovery_context
                            s2_recovery_context_set_count += 1
                            self._write_s2_recovery_context_event(
                                {
                                    "event_type": "s2_recovery_context_set",
                                    "event_schema_version": "stage21d_recovery_context_v1",
                                    "scene_id": recovery_context.get("scene_id"),
                                    "episode_id": recovery_context.get("episode_id"),
                                    "episode_index": recovery_context.get("episode_index"),
                                    "episode_count": recovery_context.get("episode_count"),
                                    "episode_eval_seed": recovery_context.get("episode_eval_seed"),
                                    "trigger_step": recovery_context.get("trigger_step"),
                                    "current_query_step": recovery_context.get("current_query_step"),
                                    "first_repeated_decision_step": recovery_context.get(
                                        "first_repeated_decision_step"
                                    ),
                                    "safe_anchor_step": recovery_context.get("safe_anchor_step"),
                                    "failure_type": recovery_context.get("failure_type"),
                                    "triage_tier": recovery_context.get("triage_tier"),
                                    "triage_reason": recovery_context.get("triage_reason"),
                                    "turn_direction": recovery_context.get("turn_direction"),
                                    "base_current_frame_present": recovery_context.get(
                                        "base_current_frame_present"
                                    ),
                                    "image_roles": list(
                                        recovery_context.get("image_roles") or []
                                    ),
                                    "image_steps": list(
                                        recovery_context.get("image_steps") or []
                                    ),
                                    "snapshot_records": list(
                                        recovery_context.get("snapshot_records") or []
                                    ),
                                    "snapshot_error": recovery_context.get("snapshot_error"),
                                    "remaining_queries": recovery_context.get(
                                        "remaining_queries"
                                    ),
                                    "shadow_variants": list(
                                        self._get_s2_action_loop_cfg().get(
                                            "recovery_context_shadow_variants"
                                        )
                                        or []
                                    ),
                                    "action_applied": False,
                                    "gt_fields_used": [],
                                }
                            )

                    recovery_cfg = self._get_s2_action_loop_cfg()
                    if (
                        pending_s2_recovery_context
                        and recovery_cfg.get("recovery_context_shadow_only")
                    ):
                        for variant in recovery_cfg.get(
                            "recovery_context_shadow_variants"
                        ) or ():
                            counterfactual_event = self._run_s2_recovery_context_counterfactual(
                                base_prompt_body=s2_prompt_body_before_final_prompt,
                                final_prompt=prompt,
                                input_images=input_images,
                                messages_prefix=s2_counterfactual_messages_prefix,
                                base_output=llm_outputs,
                                context=pending_s2_recovery_context,
                                image_width=int(self.model_args.resize_w),
                                variant=variant,
                                current_query_step=int(step_id),
                            )
                            self._write_s2_recovery_context_event(counterfactual_event)
                            s2_recovery_context_counterfactual_count += 1
                            if counterfactual_event.get("changed_pixel"):
                                s2_recovery_context_changed_count += 1
                        pending_s2_recovery_context["remaining_queries"] = int(
                            pending_s2_recovery_context.get("remaining_queries", 1)
                            or 1
                        ) - 1
                        if pending_s2_recovery_context["remaining_queries"] <= 0:
                            pending_s2_recovery_context = None
                            s2_recovery_context_expired_count += 1

                    s2_candidate_probe_s2_query_count += 1
                    s2_candidate_probe_cfg = self._get_s2_candidate_probe_cfg()
                    s2_candidate_probe_max_queries = int(
                        s2_candidate_probe_cfg.get("max_queries_per_episode", 0) or 0
                    )
                    if s2_candidate_probe_cfg.get("enable"):
                        if (
                            s2_candidate_probe_max_queries <= 0
                            or s2_candidate_probe_event_count < s2_candidate_probe_max_queries
                        ):
                            candidate_probe_event = self._probe_s2_candidate_diversity(
                                inputs,
                                inputs.input_ids.shape[1],
                                llm_outputs,
                                scene_id,
                                episode_id,
                                episode_index,
                                episode_count,
                                step_id,
                                s2_candidate_probe_s2_query_count,
                                episode_instruction,
                            )
                            if candidate_probe_event:
                                s2_candidate_probe_event_count += 1
                                valid_candidate_count = int(
                                    candidate_probe_event.get("valid_candidate_count", 0) or 0
                                )
                                unique_candidate_count = int(
                                    candidate_probe_event.get("unique_candidate_count", 0) or 0
                                )
                                max_pairwise_distance = float(
                                    candidate_probe_event.get("max_pairwise_pixel_distance", 0.0) or 0.0
                                )
                                mean_pairwise_distance = float(
                                    candidate_probe_event.get("mean_pairwise_pixel_distance", 0.0) or 0.0
                                )
                                min_pixel_distance = float(
                                    candidate_probe_event.get("min_pixel_distance", 50.0) or 50.0
                                )
                                s2_candidate_probe_valid_candidate_sum += valid_candidate_count
                                s2_candidate_probe_unique_candidate_sum += unique_candidate_count
                                s2_candidate_probe_mean_pairwise_distance_sum += mean_pairwise_distance
                                s2_candidate_probe_max_pairwise_distance = max(
                                    s2_candidate_probe_max_pairwise_distance,
                                    max_pairwise_distance,
                                )
                                if valid_candidate_count >= 2:
                                    s2_candidate_probe_valid_query_count += 1
                                if max_pairwise_distance >= min_pixel_distance:
                                    s2_candidate_probe_diverse_query_count += 1
                        else:
                            s2_candidate_probe_skipped_query_count += 1

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True
                        self._append_failure_prediction_trajectory_cache(
                            failure_prediction_state,
                            step_id=step_id,
                            pixel_goal=pixel_goal,
                            observations=observations,
                        )
                        try:
                            stage_d_bfs_trajectory_cache.append(
                                {
                                    "eval_step": int(step_id),
                                    "pixel_goal": [int(pixel_goal[0]), int(pixel_goal[1])],
                                    "gps": self._jsonable(observations.get("gps")),
                                    "compass": self._jsonable(observations.get("compass")),
                                }
                            )
                            stage_d_bfs_trajectory_cache = stage_d_bfs_trajectory_cache[-256:]
                        except Exception:
                            pass

                        semantic_image = input_images[-1] if input_images else image
                        semantic_source = "look_down" if action == action_code.LOOKDOWN else "forward"
                        semantic_decision = self._evaluate_semantic_match_with_vlmap(
                            semantic_image,
                            episode_instruction,
                            pixel_goal,
                            observations,
                            step_id=step_id,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            episode_index=episode_index,
                            episode_count=episode_count,
                            observation_source=semantic_source,
                        )
                        self.occ_memory.record_semantic(
                            semantic_decision,
                            obs={
                                "rgb": rgb,
                                "depth": current_depth_m,
                                "gps": observations.get("gps"),
                                "compass": observations.get("compass"),
                            },
                            depth=current_depth_m,
                            context={
                                "step_id": step_id,
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                                "pixel_goal": pixel_goal,
                                "image_width": int(rgb.shape[1]) if hasattr(rgb, "shape") and len(rgb.shape) >= 2 else None,
                                "image_height": int(rgb.shape[0]) if hasattr(rgb, "shape") and len(rgb.shape) >= 2 else None,
                            },
                        )

                        # Stage 2 VLMap advisor: check the S2 waypoint before asking
                        # NextDiT/System1 to turn it into local trajectory actions.
                        waypoint_camera_pitch_deg = 2.0 * self._tilt_angle_deg if action == action_code.LOOKDOWN else 0.0
                        vlmap_waypoint_decision = self._evaluate_pixel_goal_with_vlmap(
                            pixel_goal,
                            observations,
                            current_depth_m,
                            rgb=rgb,
                            step_id=step_id,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            episode_index=episode_index,
                            episode_count=episode_count,
                            camera_pitch_deg=waypoint_camera_pitch_deg,
                        )
                        vlmap_safety_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
                        depth_h, depth_w = current_depth_m.shape[:2]
                        occ_waypoint_decision = self.occ_memory.evaluate_waypoint(
                            pixel_goal,
                            {
                                "gps": observations.get("gps"),
                                "compass": observations.get("compass"),
                            },
                            current_depth_m,
                            context={
                                "step_id": step_id,
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                                "image_width": int(
                                    vlmap_safety_cfg.get("waypoint_source_image_width") or depth_w
                                ),
                                "image_height": int(
                                    vlmap_safety_cfg.get("waypoint_source_image_height") or depth_h
                                ),
                                "s2_pixel_goal": pixel_goal,
                                "vlmap_waypoint_valid": vlmap_waypoint_decision.get("valid"),
                                "vlmap_waypoint_reason": vlmap_waypoint_decision.get("reason"),
                                "stage15_repair_prev_consecutive_count": int(
                                    stage15_repair_consecutive_count
                                ),
                                "stage15_repair_prev_cumulative_count": int(
                                    stage15_repair_cumulative_count
                                ),
                            },
                        )
                        if occ_waypoint_decision.get("valid"):
                            stage15_repair_consecutive_count = int(
                                occ_waypoint_decision.get(
                                    "stage15_repair_consecutive_count",
                                    stage15_repair_consecutive_count,
                                )
                                or 0
                            )
                            stage15_repair_cumulative_count = int(
                                occ_waypoint_decision.get(
                                    "stage15_repair_cumulative_count",
                                    stage15_repair_cumulative_count,
                                )
                                or 0
                            )
                        else:
                            stage15_repair_consecutive_count = 0
                        stage15_cfg = self._get_stage15_repair_cfg()
                        if bool(stage15_cfg.get("active")):
                            stage15_repair_active_event_count += 1
                            repair_goal = occ_waypoint_decision.get("repair_pixel_goal")
                            repair_valid = bool(occ_waypoint_decision.get("repair_valid"))
                            repair_candidate = bool(occ_waypoint_decision.get("repair_candidate"))
                            gate_mode = str(stage15_cfg.get("gate_mode", "consecutive"))
                            gate_min_count = int(stage15_cfg.get("gate_min_count", 3) or 3)
                            active_max_per_episode = int(
                                stage15_cfg.get("active_max_per_episode", 5) or 0
                            )
                            current_consecutive = int(stage15_repair_consecutive_count)
                            current_cumulative = int(stage15_repair_cumulative_count)
                            if gate_mode == "consecutive":
                                count_gate_ok = current_consecutive >= gate_min_count
                                count_gate_reason = "consecutive_too_low"
                            elif gate_mode == "cumulative":
                                count_gate_ok = current_cumulative >= gate_min_count
                                count_gate_reason = "cumulative_too_low"
                            elif gate_mode == "all":
                                count_gate_ok = True
                                count_gate_reason = "count_gate_ok"
                            else:
                                count_gate_ok = False
                                count_gate_reason = "invalid_gate_mode"
                            stage15_event = {
                                "event_type": "stage15_repair_active",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                                "step_id": step_id,
                                "gate_mode": gate_mode,
                                "gate_min_count": gate_min_count,
                                "active_max_per_episode": active_max_per_episode,
                                "goal_state": occ_waypoint_decision.get("goal_state"),
                                "repair_candidate": repair_candidate,
                                "repair_valid": repair_valid,
                                "repair_reason": occ_waypoint_decision.get("repair_reason"),
                                "repair_active_applied": False,
                                "repair_active_output_ids_rewritten": False,
                                "repair_active_gate_reason": None,
                                "repair_active_original_pixel_goal": (
                                    None if pixel_goal is None else list(pixel_goal)
                                ),
                                "repair_active_replaced_pixel_goal": None,
                                "repair_active_consecutive_count": current_consecutive,
                                "repair_active_cumulative_count": current_cumulative,
                                "repair_active_applied_count_before": int(
                                    stage15_repair_active_applied_count
                                ),
                                "repair_active_backtrack_cells": occ_waypoint_decision.get(
                                    "repair_backtrack_cells"
                                ),
                                "repair_active_pixel_shift": occ_waypoint_decision.get(
                                    "repair_pixel_shift"
                                ),
                                "repair_free_grid": occ_waypoint_decision.get("repair_free_grid"),
                                "repair_pixel_goal": repair_goal,
                            }
                            active_allowed = (
                                bool(occ_waypoint_decision.get("valid"))
                                and repair_candidate
                                and repair_valid
                                and repair_goal is not None
                                and len(repair_goal) == 2
                                and count_gate_ok
                                and stage15_repair_active_applied_count < active_max_per_episode
                            )
                            if active_allowed:
                                old_pixel_goal = None if pixel_goal is None else list(pixel_goal)
                                try:
                                    if output_ids is None or output_ids.ndim != 2:
                                        stage15_event["repair_active_gate_reason"] = (
                                            "missing_base_output_ids"
                                        )
                                    else:
                                        repair_x = int(repair_goal[0])
                                        repair_y = int(repair_goal[1])
                                        # S2 text convention is "row col"; pixel_goal storage is [x, y].
                                        repair_output = f"{repair_y} {repair_x}"
                                        repair_generated_ids = self.processor.tokenizer(
                                            repair_output,
                                            add_special_tokens=False,
                                            return_tensors="pt",
                                        ).input_ids.to(output_ids.device)
                                        prompt_len = int(inputs.input_ids.shape[1])
                                        if repair_generated_ids.numel() <= 0:
                                            stage15_event["repair_active_gate_reason"] = (
                                                "empty_repair_output_ids"
                                            )
                                        else:
                                            output_ids = torch.cat(
                                                [
                                                    output_ids[:, :prompt_len],
                                                    repair_generated_ids,
                                                ],
                                                dim=1,
                                            )
                                            pixel_goal = [repair_x, repair_y]
                                            local_actions = []
                                            traj_latents = None
                                            draw_pixel_goal = True
                                            stage15_repair_active_applied_count += 1
                                            if stage15_repair_active_first_step is None:
                                                stage15_repair_active_first_step = step_id
                                            stage15_event["repair_active_applied"] = True
                                            stage15_event[
                                                "repair_active_output_ids_rewritten"
                                            ] = True
                                            stage15_event["repair_active_gate_reason"] = "applied"
                                            stage15_event[
                                                "repair_active_original_pixel_goal"
                                            ] = old_pixel_goal
                                            stage15_event[
                                                "repair_active_replaced_pixel_goal"
                                            ] = list(pixel_goal)
                                            stage15_event["repair_active_coordinate_text"] = (
                                                repair_output
                                            )
                                            try:
                                                traj_cache = list(
                                                    failure_prediction_state.get("traj_cache") or []
                                                )
                                                if traj_cache and int(
                                                    traj_cache[-1].get("eval_step", -1)
                                                ) == int(step_id):
                                                    traj_cache[-1]["pixel_goal"] = [
                                                        float(pixel_goal[0]),
                                                        float(pixel_goal[1]),
                                                    ]
                                                    traj_cache[-1]["pixel_goal_source"] = (
                                                        "stage15_repair_active"
                                                    )
                                                    failure_prediction_state["traj_cache"] = traj_cache
                                            except (TypeError, ValueError):
                                                pass
                                            print(
                                                "[OccMemory][Habitat][Stage15][Active] "
                                                f"repair pixel_goal {old_pixel_goal} -> {pixel_goal} "
                                                f"mode={gate_mode} "
                                                f"consecutive={current_consecutive} "
                                                f"cumulative={current_cumulative} "
                                                f"count={stage15_repair_active_applied_count}/"
                                                f"{active_max_per_episode}"
                                            )
                                except Exception as exc:
                                    stage15_event["repair_active_gate_reason"] = (
                                        "output_id_rewrite_error"
                                    )
                                    stage15_event["repair_active_error"] = str(exc)
                            elif not occ_waypoint_decision.get("valid"):
                                stage15_event["repair_active_gate_reason"] = "invalid_waypoint"
                            elif not repair_candidate:
                                stage15_event["repair_active_gate_reason"] = (
                                    "goal_state_not_occupied"
                                )
                            elif not repair_valid:
                                stage15_event["repair_active_gate_reason"] = "repair_invalid"
                            elif repair_goal is None or len(repair_goal) != 2:
                                stage15_event["repair_active_gate_reason"] = "invalid_repair_goal"
                            elif not count_gate_ok:
                                stage15_event["repair_active_gate_reason"] = count_gate_reason
                            elif stage15_repair_active_applied_count >= active_max_per_episode:
                                stage15_event["repair_active_gate_reason"] = "cap_reached"
                            else:
                                stage15_event["repair_active_gate_reason"] = "not_applied"
                            stage15_reason = str(
                                stage15_event.get("repair_active_gate_reason") or "unknown"
                            )
                            stage15_repair_active_reason_counts[stage15_reason] = (
                                int(stage15_repair_active_reason_counts.get(stage15_reason, 0))
                                + 1
                            )
                            stage15_event["repair_active_applied_count_after"] = int(
                                stage15_repair_active_applied_count
                            )
                            self._write_stage15_repair_active_event(stage15_event)
                        stage_d_cfg = self._get_stage_d_bfs_escape_cfg()
                        stage_d_max_events = int(
                            stage_d_cfg.get("max_events_per_episode", -1) or -1
                        )
                        stage_d_allowed = bool(stage_d_cfg.get("enable")) and (
                            stage_d_max_events < 0
                            or stage_d_bfs_escape_trigger_count < stage_d_max_events
                        )
                        if stage_d_allowed:
                            stage_d_event = self._update_stage_d_bfs_escape_shadow(
                                cfg=stage_d_cfg,
                                trajectory_cache=stage_d_bfs_trajectory_cache,
                                step_id=step_id,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                episode_index=episode_index,
                                episode_count=episode_count,
                                observations=observations,
                                occ_waypoint_decision=occ_waypoint_decision,
                                consecutive_occupied_count=stage15_repair_consecutive_count,
                            )
                            if stage_d_event is not None:
                                if (
                                    stage_d_event.get("triggered")
                                    or stage_d_cfg.get("log_non_trigger_steps")
                                ):
                                    stage_d_bfs_escape_event_count += 1
                                if stage_d_event.get("triggered"):
                                    stage_d_bfs_escape_trigger_count += 1
                                    if stage_d_bfs_escape_first_trigger_step is None:
                                        stage_d_bfs_escape_first_trigger_step = step_id
                                    if stage_d_event.get("bfs_reachable"):
                                        stage_d_bfs_escape_reachable_count += 1
                                    for condition in list(
                                        stage_d_event.get("trigger_conditions") or []
                                    ):
                                        stage_d_bfs_escape_trigger_reason_counts[condition] = (
                                            int(
                                                stage_d_bfs_escape_trigger_reason_counts.get(
                                                    condition, 0
                                                )
                                            )
                                            + 1
                                        )
                                reason = str(stage_d_event.get("reason") or "unknown")
                                stage_d_bfs_escape_reason_counts[reason] = (
                                    int(stage_d_bfs_escape_reason_counts.get(reason, 0)) + 1
                                )
                                stage_d_active_enabled = (
                                    bool(stage_d_cfg.get("active"))
                                    and not bool(stage_d_cfg.get("shadow_only", True))
                                    and bool(stage_d_event.get("triggered"))
                                )
                                if stage_d_active_enabled:
                                    stage_d_bfs_escape_active_event_count += 1
                                    active_max_per_episode = int(
                                        stage_d_cfg.get("active_max_per_episode", 2) or 0
                                    )
                                    active_path_edge_steps = int(
                                        stage_d_cfg.get("active_path_edge_steps", 8) or 8
                                    )
                                    target_candidate = dict(
                                        stage_d_event.get("bfs_target_candidate") or {}
                                    )
                                    path = list(stage_d_event.get("bfs_path") or [])
                                    require_target_frontier = bool(
                                        stage_d_cfg.get("active_require_target_frontier", True)
                                    )
                                    target_frontier_ok = (
                                        bool(target_candidate.get("target_frontier_candidate"))
                                        if require_target_frontier
                                        else True
                                    )
                                    active_event = {
                                        "event_type": "stage_d_bfs_escape_active",
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "episode_index": episode_index,
                                        "episode_count": episode_count,
                                        "step_id": int(step_id),
                                        "trigger_conditions": list(
                                            stage_d_event.get("trigger_conditions") or []
                                        ),
                                        "trigger_condition": stage_d_event.get(
                                            "trigger_condition"
                                        ),
                                        "bfs_reachable": bool(
                                            stage_d_event.get("bfs_reachable")
                                        ),
                                        "bfs_reason": stage_d_event.get("reason"),
                                        "bfs_path_edge_count": stage_d_event.get(
                                            "bfs_path_edge_count"
                                        ),
                                        "bfs_path_m": stage_d_event.get("bfs_path_m"),
                                        "bfs_action_steps_estimate": stage_d_event.get(
                                            "bfs_action_steps_estimate"
                                        ),
                                        "bfs_target_grid": stage_d_event.get(
                                            "bfs_target_grid"
                                        ),
                                        "bfs_target_direction": stage_d_event.get(
                                            "bfs_target_direction"
                                        ),
                                        "bfs_target_direction_angle_deg": stage_d_event.get(
                                            "bfs_target_direction_angle_deg"
                                        ),
                                        "bfs_target_instruction_relevant": bool(
                                            stage_d_event.get(
                                                "bfs_target_instruction_relevant"
                                            )
                                        ),
                                        "target_frontier_candidate": bool(
                                            target_candidate.get("target_frontier_candidate")
                                        ),
                                        "target_frontier_escape_candidate": bool(
                                            target_candidate.get(
                                                "target_frontier_escape_candidate"
                                            )
                                        ),
                                        "target_matched_landmark": target_candidate.get(
                                            "matched_landmark"
                                        ),
                                        "target_landmark_status": target_candidate.get(
                                            "landmark_status"
                                        ),
                                        "target_next_landmark_relevance": target_candidate.get(
                                            "next_landmark_relevance"
                                        ),
                                        "target_semantic_progress_score": target_candidate.get(
                                            "semantic_progress_score"
                                        ),
                                        "target_goal_progress_score": target_candidate.get(
                                            "goal_progress_score"
                                        ),
                                        "active_require_target_frontier": bool(
                                            require_target_frontier
                                        ),
                                        "active_target_frontier_gate_ok": bool(
                                            target_frontier_ok
                                        ),
                                        "active_max_per_episode": int(active_max_per_episode),
                                        "active_applied_count_before": int(
                                            stage_d_bfs_escape_active_applied_count
                                        ),
                                        "active_applied": False,
                                        "active_output_ids_rewritten": False,
                                        "active_original_pixel_goal": (
                                            None if pixel_goal is None else list(pixel_goal)
                                        ),
                                        "active_replaced_pixel_goal": None,
                                        "active_selected_grid": None,
                                        "active_selected_path_index": None,
                                        "active_projection": None,
                                        "active_gate_reason": None,
                                    }
                                    active_allowed = (
                                        bool(stage_d_event.get("bfs_reachable"))
                                        and target_frontier_ok
                                        and stage_d_bfs_escape_active_applied_count
                                        < active_max_per_episode
                                    )
                                    if active_allowed and len(path) >= 2:
                                        path_index = min(
                                            max(1, active_path_edge_steps),
                                            len(path) - 1,
                                        )
                                        selected_grid = path[path_index]
                                        active_event["active_selected_grid"] = selected_grid
                                        active_event["active_selected_path_index"] = int(
                                            path_index
                                        )
                                        active_pixel_goal_mode = str(
                                            stage_d_cfg.get(
                                                "active_pixel_goal_mode", "projection"
                                            )
                                            or "projection"
                                        ).lower()
                                        active_context = {
                                            "step_id": step_id,
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "episode_index": episode_index,
                                            "episode_count": episode_count,
                                            "image_width": int(
                                                vlmap_safety_cfg.get(
                                                    "waypoint_source_image_width"
                                                )
                                                or depth_w
                                            ),
                                            "image_height": int(
                                                vlmap_safety_cfg.get(
                                                    "waypoint_source_image_height"
                                                )
                                                or depth_h
                                            ),
                                        }
                                        active_event["active_pixel_goal_mode"] = (
                                            active_pixel_goal_mode
                                        )
                                        active_event["active_direction"] = stage_d_event.get(
                                            "bfs_target_direction"
                                        )
                                        active_event["active_direction_pixel_goal"] = None
                                        if active_pixel_goal_mode == "directional":
                                            image_w = max(
                                                1,
                                                int(
                                                    getattr(
                                                        self.model_args,
                                                        "resize_w",
                                                        active_context["image_width"],
                                                    )
                                                    or active_context["image_width"]
                                                ),
                                            )
                                            image_h = max(
                                                1,
                                                int(
                                                    getattr(
                                                        self.model_args,
                                                        "resize_h",
                                                        active_context["image_height"],
                                                    )
                                                    or active_context["image_height"]
                                                ),
                                            )
                                            active_context["image_width"] = int(image_w)
                                            active_context["image_height"] = int(image_h)
                                            direction = str(
                                                stage_d_event.get("bfs_target_direction") or ""
                                            ).lower()
                                            x_ratios = {
                                                "front": float(
                                                    stage_d_cfg.get(
                                                        "active_direction_front_x_ratio", 0.50
                                                    )
                                                ),
                                                "left": float(
                                                    stage_d_cfg.get(
                                                        "active_direction_left_x_ratio", 0.25
                                                    )
                                                ),
                                                "right": float(
                                                    stage_d_cfg.get(
                                                        "active_direction_right_x_ratio", 0.75
                                                    )
                                                ),
                                            }
                                            if direction not in x_ratios:
                                                projection = {
                                                    "valid": False,
                                                    "reason": "direction_not_actionable",
                                                    "pixel_goal": None,
                                                    "direction": direction,
                                                }
                                            else:
                                                y_ratio = float(
                                                    stage_d_cfg.get(
                                                        "active_direction_y_ratio", 0.75
                                                    )
                                                )
                                                dir_goal = [
                                                    int(
                                                        round(
                                                            max(0.0, min(1.0, x_ratios[direction]))
                                                            * float(image_w - 1)
                                                        )
                                                    ),
                                                    int(
                                                        round(
                                                            max(0.0, min(1.0, y_ratio))
                                                            * float(image_h - 1)
                                                        )
                                                    ),
                                                ]
                                                projection = {
                                                    "valid": True,
                                                    "reason": "ok",
                                                    "pixel_goal": dir_goal,
                                                    "direction": direction,
                                                    "image_width": int(image_w),
                                                    "image_height": int(image_h),
                                                }
                                                active_event[
                                                    "active_direction_pixel_goal"
                                                ] = list(dir_goal)
                                        else:
                                            projection = self.occ_memory.project_grid_to_pixel_goal(
                                                selected_grid,
                                                {
                                                    "gps": observations.get("gps"),
                                                    "compass": observations.get("compass"),
                                                },
                                                current_depth_m,
                                                context=active_context,
                                                goal_world_z=float(
                                                    stage_d_cfg.get("active_goal_world_z", 0.0)
                                                    or 0.0
                                                ),
                                            )
                                        active_event["active_projection"] = projection
                                        projected_goal = (
                                            projection.get("pixel_goal")
                                            if isinstance(projection, dict)
                                            else None
                                        )
                                        if not projection.get("valid"):
                                            active_event["active_gate_reason"] = str(
                                                projection.get("reason")
                                                or "projection_failed"
                                            )
                                        elif not projected_goal or len(projected_goal) != 2:
                                            active_event["active_gate_reason"] = (
                                                "invalid_projected_goal"
                                            )
                                        else:
                                            proj_x = int(projected_goal[0])
                                            proj_y = int(projected_goal[1])
                                            in_bounds = (
                                                0 <= proj_x < int(active_context["image_width"])
                                                and 0
                                                <= proj_y
                                                < int(active_context["image_height"])
                                            )
                                            active_event["active_projected_in_bounds"] = bool(
                                                in_bounds
                                            )
                                            if (
                                                stage_d_cfg.get(
                                                    "active_require_pixel_in_bounds", True
                                                )
                                                and not in_bounds
                                            ):
                                                active_event["active_gate_reason"] = (
                                                    "projected_goal_out_of_bounds"
                                                )
                                            elif output_ids is None or output_ids.ndim != 2:
                                                active_event["active_gate_reason"] = (
                                                    "missing_base_output_ids"
                                                )
                                            else:
                                                try:
                                                    active_output = f"{proj_y} {proj_x}"
                                                    active_generated_ids = (
                                                        self.processor.tokenizer(
                                                            active_output,
                                                            add_special_tokens=False,
                                                            return_tensors="pt",
                                                        ).input_ids.to(output_ids.device)
                                                    )
                                                    prompt_len = int(inputs.input_ids.shape[1])
                                                    if active_generated_ids.numel() <= 0:
                                                        active_event[
                                                            "active_gate_reason"
                                                        ] = "empty_active_output_ids"
                                                    else:
                                                        old_pixel_goal = (
                                                            None
                                                            if pixel_goal is None
                                                            else list(pixel_goal)
                                                        )
                                                        output_ids = torch.cat(
                                                            [
                                                                output_ids[:, :prompt_len],
                                                                active_generated_ids,
                                                            ],
                                                            dim=1,
                                                        )
                                                        pixel_goal = [proj_x, proj_y]
                                                        local_actions = []
                                                        traj_latents = None
                                                        draw_pixel_goal = True
                                                        stage_d_bfs_escape_active_applied_count += 1
                                                        if (
                                                            stage_d_bfs_escape_active_first_step
                                                            is None
                                                        ):
                                                            stage_d_bfs_escape_active_first_step = (
                                                                step_id
                                                            )
                                                        active_event["active_applied"] = True
                                                        active_event[
                                                            "active_output_ids_rewritten"
                                                        ] = True
                                                        active_event[
                                                            "active_original_pixel_goal"
                                                        ] = old_pixel_goal
                                                        active_event[
                                                            "active_replaced_pixel_goal"
                                                        ] = list(pixel_goal)
                                                        active_event[
                                                            "active_coordinate_text"
                                                        ] = active_output
                                                        active_event[
                                                            "active_gate_reason"
                                                        ] = "applied"
                                                        try:
                                                            if (
                                                                stage_d_bfs_trajectory_cache
                                                                and int(
                                                                    stage_d_bfs_trajectory_cache[
                                                                        -1
                                                                    ].get("eval_step", -1)
                                                                )
                                                                == int(step_id)
                                                            ):
                                                                stage_d_bfs_trajectory_cache[-1][
                                                                    "pixel_goal"
                                                                ] = [
                                                                    int(pixel_goal[0]),
                                                                    int(pixel_goal[1]),
                                                                ]
                                                                stage_d_bfs_trajectory_cache[-1][
                                                                    "pixel_goal_source"
                                                                ] = "stage_d_bfs_active"
                                                            traj_cache = list(
                                                                failure_prediction_state.get(
                                                                    "traj_cache"
                                                                )
                                                                or []
                                                            )
                                                            if traj_cache and int(
                                                                traj_cache[-1].get(
                                                                    "eval_step", -1
                                                                )
                                                            ) == int(step_id):
                                                                traj_cache[-1][
                                                                    "pixel_goal"
                                                                ] = [
                                                                    float(pixel_goal[0]),
                                                                    float(pixel_goal[1]),
                                                                ]
                                                                traj_cache[-1][
                                                                    "pixel_goal_source"
                                                                ] = "stage_d_bfs_active"
                                                                failure_prediction_state[
                                                                    "traj_cache"
                                                                ] = traj_cache
                                                        except (TypeError, ValueError):
                                                            pass
                                                        print(
                                                            "[OccMemory][Habitat][StageD][Active] "
                                                            f"replace pixel_goal {old_pixel_goal} "
                                                            f"-> {pixel_goal} "
                                                            f"target_frontier={target_frontier_ok} "
                                                            f"count={stage_d_bfs_escape_active_applied_count}/"
                                                            f"{active_max_per_episode}"
                                                        )
                                                except Exception as exc:
                                                    active_event[
                                                        "active_gate_reason"
                                                    ] = "output_id_rewrite_error"
                                                    active_event["active_error"] = str(exc)
                                    elif stage_d_bfs_escape_active_applied_count >= active_max_per_episode:
                                        active_event["active_gate_reason"] = "cap_reached"
                                    elif not stage_d_event.get("bfs_reachable"):
                                        active_event["active_gate_reason"] = "bfs_not_reachable"
                                    elif not target_frontier_ok:
                                        active_event["active_gate_reason"] = (
                                            "target_frontier_gate_failed"
                                        )
                                    else:
                                        active_event["active_gate_reason"] = "invalid_bfs_path"
                                    active_reason = str(
                                        active_event.get("active_gate_reason") or "unknown"
                                    )
                                    stage_d_bfs_escape_active_reason_counts[active_reason] = (
                                        int(
                                            stage_d_bfs_escape_active_reason_counts.get(
                                                active_reason, 0
                                            )
                                        )
                                        + 1
                                    )
                                    active_event["active_applied_count_after"] = int(
                                        stage_d_bfs_escape_active_applied_count
                                    )
                                    self._write_stage_d_bfs_escape_active_event(active_event)
                        som_cfg = self._get_som_counterfactual_cfg()
                        som_max_queries = int(som_cfg.get("max_queries_per_episode", 30) or 0)
                        som_allowed = (
                            bool(som_cfg.get("enable"))
                            and (
                                som_max_queries <= 0
                                or som_counterfactual_count < som_max_queries
                            )
                        )
                        if som_allowed:
                            som_context = {
                                "step_id": step_id,
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                            }
                            som_event = self._run_som_counterfactual(
                                base_prompt_body=s2_prompt_body_before_final_prompt,
                                final_prompt=prompt,
                                input_images=input_images,
                                messages_prefix=s2_counterfactual_messages_prefix,
                                base_output=llm_outputs,
                                base_pixel_goal=pixel_goal,
                                occ_decision=occ_waypoint_decision,
                                context=som_context,
                                image_width=int(self.model_args.resize_w),
                            )
                            som_event["active_applied"] = False
                            som_event["active_original_pixel_goal"] = None
                            som_event["active_replaced_pixel_goal"] = None
                            som_event["active_output_ids_rewritten"] = False
                            som_event["active_replacement_mode"] = None
                            som_event["active_gate_reason"] = "shadow_only"
                            if not bool(som_cfg.get("shadow_only", True)):
                                active_min_unsafe_signal = float(
                                    som_cfg.get("active_min_unsafe_signal", 0.50) or 0.50
                                )
                                active_max_per_episode = max(
                                    0, int(som_cfg.get("active_max_per_episode", 3))
                                )
                                unsafe_signal = float(som_event.get("unsafe_signal", 0.0) or 0.0)
                                overlay_goal = som_event.get("overlay_pixel_goal")
                                # Stage14c-v2: geometry gate on OccMem goal_state.
                                # post-hoc analysis of Stage14c showed all 4 succ->fail
                                # regressions were goal_state=="unknown" replacements,
                                # while both recoveries (266,286) carried occupied events.
                                active_goal_state_gate = str(
                                    som_cfg.get("active_goal_state_gate", "any")
                                )
                                som_goal_state = str(som_event.get("goal_state") or "")
                                som_follows_frontier = bool(
                                    som_event.get("follows_frontier_direction")
                                )
                                if active_goal_state_gate == "occupied":
                                    goal_state_gate_ok = som_goal_state == "occupied"
                                elif active_goal_state_gate == "occupied_or_free_follows":
                                    goal_state_gate_ok = (
                                        som_goal_state == "occupied"
                                        or (
                                            som_goal_state == "free"
                                            and som_follows_frontier
                                        )
                                    )
                                else:
                                    goal_state_gate_ok = True
                                active_allowed = (
                                    som_event.get("status") == "ok"
                                    and bool(som_event.get("overlay_valid"))
                                    and bool(som_event.get("changed_pixel"))
                                    and unsafe_signal >= active_min_unsafe_signal
                                    and goal_state_gate_ok
                                    and som_counterfactual_active_applied_count < active_max_per_episode
                                )
                                # always log geometry gate fields for post-hoc analysis
                                som_event["active_goal_state_gate"] = active_goal_state_gate
                                som_event["active_goal_state_gate_ok"] = goal_state_gate_ok
                                som_event["active_som_goal_state"] = som_goal_state
                                som_event["active_som_follows_frontier"] = som_follows_frontier
                                if active_allowed and overlay_goal and len(overlay_goal) == 2:
                                    old_pixel_goal = None if pixel_goal is None else list(pixel_goal)
                                    overlay_output = str(som_event.get("overlay_output") or "").strip()
                                    try:
                                        if output_ids is None or output_ids.ndim != 2:
                                            som_event["active_gate_reason"] = "missing_base_output_ids"
                                        else:
                                            overlay_generated_ids = self.processor.tokenizer(
                                                overlay_output,
                                                add_special_tokens=False,
                                                return_tensors="pt",
                                            ).input_ids.to(output_ids.device)
                                            prompt_len = int(inputs.input_ids.shape[1])
                                            if overlay_generated_ids.numel() <= 0:
                                                som_event["active_gate_reason"] = "empty_overlay_output_ids"
                                            else:
                                                output_ids = torch.cat(
                                                    [
                                                        output_ids[:, :prompt_len],
                                                        overlay_generated_ids,
                                                    ],
                                                    dim=1,
                                                )
                                                pixel_goal = [
                                                    int(overlay_goal[0]),
                                                    int(overlay_goal[1]),
                                                ]
                                                local_actions = []
                                                traj_latents = None
                                                draw_pixel_goal = True
                                                som_counterfactual_active_applied_count += 1
                                                som_event["active_applied"] = True
                                                som_event["active_original_pixel_goal"] = old_pixel_goal
                                                som_event["active_replaced_pixel_goal"] = list(pixel_goal)
                                                som_event["active_output_ids_rewritten"] = True
                                                som_event["active_replacement_mode"] = (
                                                    "base_prompt_overlay_coordinate_tokens"
                                                )
                                                som_event["active_gate_reason"] = "applied"
                                                try:
                                                    traj_cache = list(
                                                        failure_prediction_state.get("traj_cache") or []
                                                    )
                                                    if traj_cache and int(
                                                        traj_cache[-1].get("eval_step", -1)
                                                    ) == int(step_id):
                                                        traj_cache[-1]["pixel_goal"] = [
                                                            float(pixel_goal[0]),
                                                            float(pixel_goal[1]),
                                                        ]
                                                        traj_cache[-1]["pixel_goal_source"] = (
                                                            "som_active_overlay"
                                                        )
                                                        failure_prediction_state["traj_cache"] = traj_cache
                                                except (TypeError, ValueError):
                                                    pass
                                                print(
                                                    "[OccMemory][Habitat][SoM][Active] replace pixel_goal "
                                                    f"{old_pixel_goal} -> {pixel_goal} "
                                                    f"shift={som_event.get('pixel_shift')} "
                                                    f"unsafe={unsafe_signal:.2f} "
                                                    f"count={som_counterfactual_active_applied_count}/"
                                                    f"{active_max_per_episode}"
                                                )
                                    except Exception as exc:
                                        som_event["active_gate_reason"] = "output_id_rewrite_error"
                                        som_event["active_error"] = str(exc)
                                elif som_counterfactual_active_applied_count >= active_max_per_episode:
                                    som_event["active_gate_reason"] = "max_per_episode"
                                elif som_event.get("status") != "ok":
                                    som_event["active_gate_reason"] = "status_not_ok"
                                elif not som_event.get("overlay_valid"):
                                    som_event["active_gate_reason"] = "invalid_overlay"
                                elif not som_event.get("changed_pixel"):
                                    som_event["active_gate_reason"] = "unchanged_pixel"
                                elif unsafe_signal < active_min_unsafe_signal:
                                    som_event["active_gate_reason"] = "unsafe_signal_low"
                                elif not goal_state_gate_ok:
                                    som_event["active_gate_reason"] = (
                                        f"goal_state_gate_blocked"
                                        f"[gate={active_goal_state_gate}"
                                        f",state={som_goal_state}]"
                                    )
                                else:
                                    som_event["active_gate_reason"] = "invalid_overlay_goal"
                            self._write_som_counterfactual_event(som_event)
                            som_counterfactual_count += 1
                            som_status = str(som_event.get("status", "ok"))
                            if som_status.startswith("skipped"):
                                som_counterfactual_skipped_count += 1
                            if som_status == "error":
                                som_counterfactual_error_count += 1
                            if som_event.get("overlay_valid"):
                                som_counterfactual_valid_count += 1
                            if som_event.get("changed_pixel"):
                                som_counterfactual_changed_count += 1
                            if som_event.get("changed_direction_bucket"):
                                som_counterfactual_direction_changed_count += 1
                            if som_event.get("follows_frontier_direction"):
                                som_counterfactual_frontier_follow_count += 1
                            if som_event.get("unsafe_shift_proxy"):
                                som_counterfactual_unsafe_shift_proxy_count += 1
                            som_shift = som_event.get("pixel_shift")
                            if isinstance(som_shift, (int, float)):
                                som_counterfactual_pixel_shift_sum += float(som_shift)
                            print(
                                "[OccMemory][Habitat][SoM] counterfactual overlay; "
                                f"status={som_event.get('status')} "
                                f"frontier={som_event.get('frontier_dominant_direction')} "
                                f"base={som_event.get('base_pixel_goal')} "
                                f"overlay={som_event.get('overlay_pixel_goal')} "
                                f"shift={som_event.get('pixel_shift')}"
                            )
                        candidate_probe_cfg = self._get_occ_memory_candidate_probe_cfg()
                        stage19_candidate_event = None
                        if candidate_probe_cfg.get("enable"):
                            base_eval_seed = getattr(self.model_args, "eval_random_seed", None)
                            habitat_dataset_cfg = getattr(
                                getattr(self.config, "habitat", None),
                                "dataset",
                                None,
                            )
                            candidate_context = {
                                "split": getattr(habitat_dataset_cfg, "split", None),
                                "rank": int(getattr(self, "rank", 0)),
                                "local_rank": int(getattr(self, "local_rank", 0)),
                                "world_size": int(getattr(self, "world_size", 1)),
                                "eval_random_seed": base_eval_seed,
                                "episode_eval_seed": episode_eval_seed,
                                "step_id": step_id,
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_index": episode_index,
                                "episode_count": episode_count,
                                "s2_pixel_goal": pixel_goal,
                                "s2_output": llm_outputs,
                                "vlmap_waypoint_valid": vlmap_waypoint_decision.get("valid"),
                                "vlmap_waypoint_reason": vlmap_waypoint_decision.get("reason"),
                                "occ_waypoint_valid": occ_waypoint_decision.get("valid"),
                                "occ_waypoint_reason": occ_waypoint_decision.get("reason"),
                            }
                            candidate_event = self.occ_memory.generate_query_candidates(
                                obs={
                                    "gps": observations.get("gps"),
                                    "compass": observations.get("compass"),
                                },
                                current_waypoint_decision=occ_waypoint_decision,
                                context=candidate_context,
                            )
                            stage19_candidate_event = candidate_event
                            if candidate_event.get("reason") == "max_events_per_episode":
                                occ_memory_candidate_probe_skipped_count += 1
                            elif candidate_event.get("enabled"):
                                occ_memory_candidate_probe_event_count += 1
                                if candidate_event.get("valid"):
                                    occ_memory_candidate_probe_valid_event_count += 1
                                occ_memory_candidate_probe_candidate_sum += int(
                                    candidate_event.get("candidate_count", 0) or 0
                                )
                                occ_memory_candidate_probe_geometry_safe_sum += int(
                                    candidate_event.get("candidate_geometry_safe_count", 0) or 0
                                )
                                occ_memory_candidate_probe_active_gate_safe_sum += int(
                                    candidate_event.get("candidate_active_gate_safe_count", 0) or 0
                                )
                                occ_memory_candidate_probe_current_aligned_sum += int(
                                    candidate_event.get("candidate_current_aligned_count", 0) or 0
                                )
                                occ_memory_candidate_probe_next_landmark_relevant_sum += int(
                                    candidate_event.get("candidate_next_landmark_relevant_count", 0) or 0
                                )
                                occ_memory_candidate_probe_completed_landmark_sum += int(
                                    candidate_event.get("candidate_completed_landmark_count", 0) or 0
                                )
                                occ_memory_candidate_probe_repeated_semantic_sum += int(
                                    candidate_event.get("candidate_repeated_semantic_count", 0) or 0
                                )
                                occ_memory_candidate_probe_unknown_target_frontier_bonus_sum += int(
                                    candidate_event.get(
                                        "candidate_unknown_target_frontier_bonus_count", 0
                                    ) or 0
                                )
                                occ_memory_candidate_probe_target_frontier_sum += int(
                                    candidate_event.get("candidate_target_frontier_count", 0) or 0
                                )
                                occ_memory_candidate_probe_target_frontier_escape_sum += int(
                                    candidate_event.get(
                                        "candidate_target_frontier_escape_count", 0
                                    ) or 0
                                )
                                occ_memory_candidate_probe_target_frontier_intent_safe_sum += int(
                                    candidate_event.get(
                                        "candidate_target_frontier_intent_safe_count", 0
                                    ) or 0
                                )
                                occ_memory_candidate_probe_target_frontier_doorway_like_sum += int(
                                    candidate_event.get(
                                        "candidate_target_frontier_doorway_like_count", 0
                                    ) or 0
                                )
                                selection_max_queries = int(
                                    candidate_probe_cfg.get("selection_max_queries_per_episode", 2) or 0
                                )
                                selection_allowed = (
                                    bool(candidate_probe_cfg.get("selection_enable"))
                                    and bool(candidate_event.get("valid"))
                                    and (
                                        selection_max_queries <= 0
                                        or occ_memory_candidate_selection_query_count < selection_max_queries
                                    )
                                )
                                if selection_allowed:
                                    selection_event = self._run_occ_memory_candidate_selection_probe(
                                        base_prompt_body=s2_prompt_body_before_final_prompt,
                                        input_images=input_images,
                                        messages_prefix=s2_counterfactual_messages_prefix,
                                        candidate_event=candidate_event,
                                        context=candidate_context,
                                    )
                                    occ_memory_candidate_selection_query_count += 1
                                    if selection_event.get("status") == "error":
                                        occ_memory_candidate_selection_error_count += 1
                                    if selection_event.get("valid"):
                                        occ_memory_candidate_selection_valid_count += 1
                                        reason = selection_event.get("reason")
                                        if reason == "matched_label":
                                            occ_memory_candidate_selection_label_count += 1
                                        elif reason == "nearest_bev_coordinate":
                                            occ_memory_candidate_selection_coordinate_count += 1
                                        elif reason == "direction_token":
                                            occ_memory_candidate_selection_direction_count += 1
                                    if selection_event.get("none"):
                                        occ_memory_candidate_selection_none_count += 1
                                    selected_candidate = selection_event.get("selected_candidate") or {}
                                    if selected_candidate.get("active_gate_safe"):
                                        occ_memory_candidate_selection_active_gate_safe_count += 1
                                    if selected_candidate.get("aligned_with_current_waypoint"):
                                        occ_memory_candidate_selection_current_aligned_count += 1
                                    if selected_candidate.get("semanticized_candidate"):
                                        occ_memory_candidate_selection_semanticized_count += 1
                                    if selected_candidate.get("instruction_relevant"):
                                        occ_memory_candidate_selection_instruction_relevant_count += 1
                                    if float(selected_candidate.get("next_landmark_relevance", 0.0) or 0.0) > 0.0:
                                        occ_memory_candidate_selection_next_landmark_relevant_count += 1
                                    if float(selected_candidate.get("completed_landmark_penalty", 0.0) or 0.0) > 0.0:
                                        occ_memory_candidate_selection_completed_landmark_count += 1
                                    if float(selected_candidate.get("repeated_semantic_penalty", 0.0) or 0.0) > 0.0:
                                        occ_memory_candidate_selection_repeated_semantic_count += 1
                        if occ_waypoint_decision.get("valid") and occ_waypoint_decision.get("goal_state") != "free":
                            print(
                                "[OccMemory][Habitat][Waypoint] "
                                f"goal_state={occ_waypoint_decision.get('goal_state')} "
                                f"frontier_m={occ_waypoint_decision.get('frontier_distance_m')} "
                                f"revisit={occ_waypoint_decision.get('points_to_revisited_region')}"
                            )
                        repeated_goal_match = self._match_rejected_vlmap_goal(
                            vlmap_waypoint_decision, rejected_vlmap_goal_grids
                        )
                        if repeated_goal_match is not None:
                            repeated_msg = (
                                "[VLMapSafety][Habitat][Waypoint] repeated rejected grid "
                                f"goal={repeated_goal_match['goal_grid']} "
                                f"previous={repeated_goal_match['rejected_goal_grid']} "
                                f"dist={repeated_goal_match['chebyshev_distance']}"
                            )
                            if vlmap_waypoint_decision.get("requery_required"):
                                vlmap_waypoint_decision["requery_required"] = False
                                vlmap_waypoint_decision["waypoint_requery_suppressed_reason"] = (
                                    "repeated_rejected_grid"
                                )
                                print(repeated_msg + "; suppress duplicate S2 requery")
                            else:
                                print(repeated_msg)

                        # Always restore the camera to the normal forward view before
                        # either handing the goal to System1 or asking System2 again.
                        _, _, lookup_done, _ = self.env.step(action_code.LOOKUP)
                        observations, _, lookup_done_2, _ = self.env.step(action_code.LOOKUP)
                        done = done or lookup_done or lookup_done_2
                        stage19_active_status = self._maybe_apply_semantic_resilience_active_lite(
                            stage19_candidate_event,
                            step_id=step_id,
                            active_count=stage19_semantic_resilience_active_applied_count,
                            last_active_step=stage19_semantic_resilience_active_last_step,
                            scene_id=scene_id,
                            episode_id=episode_id,
                        )
                        if stage19_active_status.get("considered"):
                            stage19_semantic_resilience_active_considered_count += 1
                            failure_type = str(stage19_active_status.get("failure_type") or "unknown")
                            stage19_semantic_resilience_failure_type_counts[failure_type] = (
                                int(stage19_semantic_resilience_failure_type_counts.get(failure_type, 0))
                                + 1
                            )
                            recommended_primitive = str(
                                stage19_active_status.get("recommended_primitive") or "hold_s2"
                            )
                            stage19_semantic_resilience_recommended_primitive_counts[
                                recommended_primitive
                            ] = (
                                int(
                                    stage19_semantic_resilience_recommended_primitive_counts.get(
                                        recommended_primitive, 0
                                    )
                                )
                                + 1
                            )
                            if stage19_active_status.get(
                                "applied"
                            ) or stage19_active_status.get("execution_pending"):
                                execution_mode = str(
                                    stage19_active_status.get("execution_mode")
                                    or "action_sequence"
                                ).lower()
                                recovery_actions = [
                                    int(item)
                                    for item in list(stage19_active_status.get("actions") or [])
                                    if int(item) in (1, 2, 3, 5)
                                ]
                                execution_succeeded = True
                                if execution_mode == "directional_pixel_goal":
                                    pixel_goal_plan = dict(
                                        stage19_active_status.get("pixel_goal_plan") or {}
                                    )
                                    planned_goal = list(pixel_goal_plan.get("pixel_goal") or [])
                                    try:
                                        if len(planned_goal) != 2:
                                            raise ValueError("invalid_pixel_goal")
                                        if output_ids is None or output_ids.ndim != 2:
                                            raise ValueError("missing_base_s2_output_ids")
                                        if inputs is None or inputs.input_ids.ndim != 2:
                                            raise ValueError("missing_base_s2_input_ids")
                                        goal_x, goal_y = (
                                            int(planned_goal[0]),
                                            int(planned_goal[1]),
                                        )
                                        generated_ids = self.processor.tokenizer(
                                            f"{goal_y} {goal_x}",
                                            add_special_tokens=False,
                                            return_tensors="pt",
                                        ).input_ids.to(output_ids.device)
                                        if generated_ids.numel() <= 0:
                                            raise ValueError("empty_directional_goal_tokens")
                                        prompt_len = int(inputs.input_ids.shape[1])
                                        output_ids = torch.cat(
                                            [output_ids[:, :prompt_len], generated_ids],
                                            dim=1,
                                        )
                                        pixel_goal = [goal_x, goal_y]
                                        local_actions = []
                                        action_seq = []
                                        vlmap_recovery_actions = []
                                        traj_latents = None
                                        draw_pixel_goal = True
                                        pending_vlmap_waypoint_feedback = ""
                                        pending_vlmap_semantic_hint = ""
                                        pending_occ_memory_guidance_hint = ""
                                        action = None
                                        forward_action = 0
                                        flag = False
                                    except Exception as exc:
                                        execution_succeeded = False
                                        stage19_active_status.update(
                                            {
                                                "applied": False,
                                                "execution_pending": False,
                                                "reason": "directional_pixel_goal_execution_failed",
                                                "execution_error_type": type(exc).__name__,
                                                "execution_error": str(exc),
                                            }
                                        )
                                    else:
                                        stage19_active_status.update(
                                            {
                                                "applied": True,
                                                "execution_pending": False,
                                                "reason": "applied",
                                                "executed_pixel_goal": list(pixel_goal),
                                            }
                                        )
                                    self._write_semantic_resilience_active_lite_event(
                                        stage19_active_status
                                    )
                                else:
                                    vlmap_recovery_actions = recovery_actions
                                    if self._get_semantic_resilience_active_lite_cfg().get(
                                        "clear_goal"
                                    ):
                                        pixel_goal = None
                                        output_ids = None
                                        traj_latents = None
                                        pix_goal_image = None
                                        pix_goal_depth = None
                                        messages = []
                                        input_images = []
                                        llm_outputs = ""
                                        local_actions = []
                                        action_seq = []
                                        pending_vlmap_waypoint_feedback = ""
                                        pending_vlmap_semantic_hint = ""
                                        pending_occ_memory_guidance_hint = ""
                                        action = None
                                        forward_action = 0
                                        draw_pixel_goal = False
                                        flag = False
                                if execution_succeeded:
                                    stage19_semantic_resilience_active_applied_count += 1
                                    stage19_semantic_resilience_active_last_step = int(step_id)
                                    if stage19_semantic_resilience_active_first_step is None:
                                        stage19_semantic_resilience_active_first_step = int(step_id)
                                    stage19_semantic_resilience_active_action_sum += int(
                                        len(recovery_actions)
                                    )
                                    reason = str(
                                        stage19_active_status.get("reason") or "unknown"
                                    )
                                    stage19_semantic_resilience_active_reason_counts[reason] = (
                                        int(
                                            stage19_semantic_resilience_active_reason_counts.get(
                                                reason, 0
                                            )
                                        )
                                        + 1
                                    )
                                    print(
                                        "[Stage19][SemanticResilience][ActiveLite] "
                                        f"apply {execution_mode} "
                                        f"actions={vlmap_recovery_actions} pixel_goal={pixel_goal} "
                                        f"reason={stage19_active_status.get('reason')} "
                                        f"candidate={((stage19_active_status.get('candidate') or {}).get('candidate_id'))}"
                                    )
                                    continue
                                print(
                                    "[Stage19][SemanticResilience][ActiveLite] "
                                    f"suppress {execution_mode}; "
                                    f"reason={stage19_active_status.get('reason')} "
                                    f"error={stage19_active_status.get('execution_error')}"
                                )
                            reason = str(stage19_active_status.get("reason") or "unknown")
                            stage19_semantic_resilience_active_reason_counts[reason] = (
                                int(stage19_semantic_resilience_active_reason_counts.get(reason, 0))
                                + 1
                            )
                            stage19_semantic_resilience_active_suppressed_count += 1
                        if semantic_decision.get("stagnation_hint_required"):
                            if not pending_vlmap_semantic_hint:
                                pending_vlmap_semantic_hint = (
                                    self._format_vlmap_semantic_stagnation_hint(semantic_decision)
                                )
                                semantic_hint_set_count += 1
                                if semantic_hint_detection_step is None:
                                    semantic_hint_detection_step = step_id
                                semantic_hint_not_injected_reason = "pending_next_s2_query"
                                print(
                                    "[VLMapSemantic][Habitat][Stagnation] "
                                    "queue delayed S2 hint; "
                                    f"reason={semantic_decision.get('stagnation_would_requery_reason')} "
                                    f"recent={semantic_decision.get('stagnation_recent_terms')}"
                                )
                        if semantic_decision.get("stagnation_requery_required"):
                            pixel_goal = None
                            output_ids = None
                            traj_latents = None
                            pix_goal_image = None
                            pix_goal_depth = None
                            messages = []
                            input_images = []
                            llm_outputs = ""
                            local_actions = []
                            action_seq = []
                            vlmap_recovery_actions = []
                            pending_vlmap_waypoint_feedback = ""
                            pending_vlmap_semantic_hint = ""
                            action = None
                            forward_action = 0
                            draw_pixel_goal = False
                            flag = False
                            print(
                                "[VLMapSemantic][Habitat][Stagnation] "
                                "clear current goal and re-observe with S2; "
                                f"reason={semantic_decision.get('stagnation_would_requery_reason')} "
                                f"recent={semantic_decision.get('stagnation_recent_terms')}"
                            )
                            if pending_s2_loop_strict_active_execution is not None:
                                pending_s2_loop_strict_active_execution.update(
                                    {
                                        "event_type": "s2_loop_strict_active_execution",
                                        "reason": "semantic_requery_before_trajectory",
                                        "action_applied": False,
                                        "execution_pending": False,
                                    }
                                )
                                self._write_s2_loop_strict_active_event(
                                    pending_s2_loop_strict_active_execution
                                )
                                pending_s2_loop_strict_active_execution = None
                            if pending_s2_loop_path_execution is not None:
                                pending_s2_loop_path_execution.update(
                                    {
                                        "event_type": "s2_loop_path_reobserve_execution",
                                        "reason": "semantic_requery_before_path_trajectory",
                                        "pixel_action_applied": False,
                                        "execution_pending": False,
                                    }
                                )
                                self._write_s2_loop_path_reobserve_event(
                                    pending_s2_loop_path_execution
                                )
                                pending_s2_loop_path_execution = None
                            continue
                        guidance_cfg = self._get_occ_memory_guidance_cfg()
                        guidance_context = {
                            "step_id": step_id,
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "episode_index": episode_index,
                            "episode_count": episode_count,
                        }
                        guidance_should_trigger, guidance_reason = self._occ_memory_guidance_trigger_reason(
                            occ_waypoint_decision,
                            step_id=step_id,
                            hint_set_count=occ_memory_guidance_hint_set_count,
                            last_hint_step=occ_memory_guidance_last_set_step,
                        )
                        dead_zone_candidate = bool(
                            guidance_cfg.get("enable")
                            and occ_waypoint_decision.get("valid")
                            and occ_waypoint_decision.get("semantic_dead_zone")
                        )
                        if guidance_should_trigger:
                            occ_memory_guidance_trigger_count += 1
                            occ_memory_guidance_hint = self._format_occ_memory_guidance_hint(occ_waypoint_decision)
                            self.occ_memory.record_guidance_event(
                                action="triggered",
                                hint=occ_memory_guidance_hint,
                                reason=guidance_reason,
                                decision=occ_waypoint_decision,
                                context=guidance_context,
                            )
                            counterfactual_max_queries = int(
                                guidance_cfg.get("counterfactual_max_queries_per_episode", 2) or 0
                            )
                            counterfactual_allowed = (
                                bool(guidance_cfg.get("counterfactual_enable"))
                                and (
                                    counterfactual_max_queries <= 0
                                    or occ_memory_guidance_counterfactual_count < counterfactual_max_queries
                                )
                            )
                            if counterfactual_allowed:
                                counterfactual_event = self._run_occ_memory_guidance_counterfactual(
                                    base_prompt_body=s2_prompt_body_before_final_prompt,
                                    final_prompt=prompt,
                                    input_images=input_images,
                                    messages_prefix=s2_counterfactual_messages_prefix,
                                    base_output=llm_outputs,
                                    hint=occ_memory_guidance_hint,
                                    decision=occ_waypoint_decision,
                                    context=guidance_context,
                                    image_width=int(self.model_args.resize_w),
                                )
                                occ_memory_guidance_counterfactual_count += 1
                                if counterfactual_event.get("counterfactual_hinted_valid"):
                                    occ_memory_guidance_counterfactual_valid_count += 1
                                if counterfactual_event.get("counterfactual_changed_pixel"):
                                    occ_memory_guidance_counterfactual_changed_count += 1
                                if counterfactual_event.get("counterfactual_changed_image_direction"):
                                    occ_memory_guidance_counterfactual_direction_changed_count += 1
                                if counterfactual_event.get("counterfactual_follows_left_right_hint"):
                                    occ_memory_guidance_counterfactual_left_right_follow_count += 1
                                pixel_shift = counterfactual_event.get("counterfactual_pixel_shift")
                                if isinstance(pixel_shift, (int, float)):
                                    occ_memory_guidance_counterfactual_pixel_shift_sum += float(pixel_shift)
                                print(
                                    "[OccMemory][Habitat][Guidance] counterfactual S2 hint; "
                                    f"status={counterfactual_event.get('counterfactual_status')} "
                                    f"base={counterfactual_event.get('counterfactual_base_pixel_goal')} "
                                    f"hinted={counterfactual_event.get('counterfactual_hinted_pixel_goal')} "
                                    f"shift={counterfactual_event.get('counterfactual_pixel_shift')}"
                                )
                            if guidance_cfg.get("shadow_only"):
                                occ_memory_guidance_shadow_skip_count += 1
                                self.occ_memory.record_guidance_event(
                                    action="shadow_skip",
                                    hint=occ_memory_guidance_hint,
                                    reason="shadow_only",
                                    decision=occ_waypoint_decision,
                                    context=guidance_context,
                                )
                            elif pending_occ_memory_guidance_hint:
                                occ_memory_guidance_blocked_count += 1
                                occ_memory_guidance_not_injected_reason = "already_pending"
                                self.occ_memory.record_guidance_event(
                                    action="blocked",
                                    hint=occ_memory_guidance_hint,
                                    reason="already_pending",
                                    decision=occ_waypoint_decision,
                                    context=guidance_context,
                                )
                            else:
                                pending_occ_memory_guidance_hint = occ_memory_guidance_hint
                                occ_memory_guidance_hint_set_count += 1
                                occ_memory_guidance_last_set_step = step_id
                                if occ_memory_guidance_detection_step is None:
                                    occ_memory_guidance_detection_step = step_id
                                occ_memory_guidance_not_injected_reason = "pending_next_s2_query"
                                self.occ_memory.record_guidance_event(
                                    action="queued",
                                    hint=occ_memory_guidance_hint,
                                    reason=guidance_reason,
                                    decision=occ_waypoint_decision,
                                    context=guidance_context,
                                )
                                print(
                                    "[OccMemory][Habitat][Guidance] queue S2 hint; "
                                    f"reason={guidance_reason} "
                                    f"frontier={occ_waypoint_decision.get('frontier_dominant_direction')} "
                                    f"score={occ_waypoint_decision.get('semantic_dead_zone_score')}"
                                )
                                if guidance_cfg.get("requery_on_trigger"):
                                    occ_memory_guidance_requery_count += 1
                                    pixel_goal = None
                                    output_ids = None
                                    traj_latents = None
                                    pix_goal_image = None
                                    pix_goal_depth = None
                                    messages = []
                                    input_images = []
                                    llm_outputs = ""
                                    local_actions = []
                                    action_seq = []
                                    vlmap_recovery_actions = []
                                    pending_vlmap_waypoint_feedback = ""
                                    pending_vlmap_semantic_hint = ""
                                    action = None
                                    forward_action = 0
                                    draw_pixel_goal = False
                                    flag = False
                                    occ_memory_guidance_not_injected_reason = "pending_immediate_requery"
                                    print(
                                        "[OccMemory][Habitat][Guidance] "
                                        "clear current goal and requery S2 with memory hint"
                                    )
                                    if pending_s2_loop_strict_active_execution is not None:
                                        pending_s2_loop_strict_active_execution.update(
                                            {
                                                "event_type": "s2_loop_strict_active_execution",
                                                "reason": "occ_guidance_requery_before_trajectory",
                                                "action_applied": False,
                                                "execution_pending": False,
                                            }
                                        )
                                        self._write_s2_loop_strict_active_event(
                                            pending_s2_loop_strict_active_execution
                                        )
                                        pending_s2_loop_strict_active_execution = None
                                    if pending_s2_loop_path_execution is not None:
                                        pending_s2_loop_path_execution.update(
                                            {
                                                "event_type": "s2_loop_path_reobserve_execution",
                                                "reason": "occ_guidance_requery_before_path_trajectory",
                                                "pixel_action_applied": False,
                                                "execution_pending": False,
                                            }
                                        )
                                        self._write_s2_loop_path_reobserve_event(
                                            pending_s2_loop_path_execution
                                        )
                                        pending_s2_loop_path_execution = None
                                    continue
                        elif dead_zone_candidate:
                            occ_memory_guidance_blocked_count += 1
                            self.occ_memory.record_guidance_event(
                                action="blocked",
                                reason=guidance_reason,
                                decision=occ_waypoint_decision,
                                context=guidance_context,
                            )
                        if vlmap_waypoint_decision.get("waypoint_recovery_required"):
                            recovery_actions = [
                                int(item)
                                for item in vlmap_waypoint_decision.get("waypoint_recovery_actions", [])
                                if int(item) in (action_code.LEFT, action_code.RIGHT)
                            ]
                            if recovery_actions:
                                rejected_goal_grid = self._vlmap_goal_grid_from_decision(vlmap_waypoint_decision)
                                if rejected_goal_grid is not None:
                                    rejected_vlmap_goal_grids.append(rejected_goal_grid)
                                vlmap_recovery_actions = recovery_actions
                                pixel_goal = None
                                output_ids = None
                                messages = []
                                input_images = []
                                llm_outputs = ""
                                local_actions = []
                                action_seq = []
                                action = None
                                forward_action = 0
                                draw_pixel_goal = False
                                flag = False
                                print(
                                    "[VLMapSafety][Habitat][Waypoint] "
                                    f"clear current goal and run VLMap recovery actions {vlmap_recovery_actions}"
                                )
                                if pending_s2_loop_strict_active_execution is not None:
                                    pending_s2_loop_strict_active_execution.update(
                                        {
                                            "event_type": "s2_loop_strict_active_execution",
                                            "reason": "waypoint_recovery_before_trajectory",
                                            "action_applied": False,
                                            "execution_pending": False,
                                        }
                                    )
                                    self._write_s2_loop_strict_active_event(
                                        pending_s2_loop_strict_active_execution
                                    )
                                    pending_s2_loop_strict_active_execution = None
                                if pending_s2_loop_path_execution is not None:
                                    pending_s2_loop_path_execution.update(
                                        {
                                            "event_type": "s2_loop_path_reobserve_execution",
                                            "reason": "waypoint_recovery_before_path_trajectory",
                                            "pixel_action_applied": False,
                                            "execution_pending": False,
                                        }
                                    )
                                    self._write_s2_loop_path_reobserve_event(
                                        pending_s2_loop_path_execution
                                    )
                                    pending_s2_loop_path_execution = None
                                continue

                        if vlmap_waypoint_decision.get("requery_required"):
                            # Drop every state tied to the rejected waypoint. Without
                            # clearing both messages and input_images, the next Qwen-VL
                            # prompt can contain one image token but many stale images.
                            pending_vlmap_waypoint_feedback = self._format_vlmap_waypoint_feedback(
                                pixel_goal, vlmap_waypoint_decision
                            )
                            rejected_goal_grid = self._vlmap_goal_grid_from_decision(vlmap_waypoint_decision)
                            if rejected_goal_grid is not None:
                                rejected_vlmap_goal_grids.append(rejected_goal_grid)
                            pixel_goal = None
                            output_ids = None
                            messages = []
                            input_images = []
                            llm_outputs = ""
                            local_actions = []
                            action_seq = []
                            vlmap_recovery_actions = []
                            action = None
                            forward_action = 0
                            draw_pixel_goal = False
                            flag = False
                            if pending_s2_loop_strict_active_execution is not None:
                                pending_s2_loop_strict_active_execution.update(
                                    {
                                        "event_type": "s2_loop_strict_active_execution",
                                        "reason": "waypoint_safety_requery_before_trajectory",
                                        "action_applied": False,
                                        "execution_pending": False,
                                    }
                                )
                                self._write_s2_loop_strict_active_event(
                                    pending_s2_loop_strict_active_execution
                                )
                                pending_s2_loop_strict_active_execution = None
                            if pending_s2_loop_path_execution is not None:
                                pending_s2_loop_path_execution.update(
                                    {
                                        "event_type": "s2_loop_path_reobserve_execution",
                                        "reason": "waypoint_safety_requery_before_path_trajectory",
                                        "pixel_action_applied": False,
                                        "execution_pending": False,
                                    }
                                )
                                self._write_s2_loop_path_reobserve_event(
                                    pending_s2_loop_path_execution
                                )
                                pending_s2_loop_path_execution = None
                            print("[VLMapSafety][Habitat][Waypoint] clear current goal and requery S2")
                            continue

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        # Preprocess for the active System1 trajectory generator (NextDiT in current DualVLN eval).
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        nextdit_probe_dp_actions = dp_actions.detach().clone() if hasattr(dp_actions, "detach") else None
                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        if (
                            pending_s2_loop_path_execution is not None
                            and self._get_s2_action_loop_cfg().get(
                                "path_reobserve_one_primitive_per_reaudit"
                            )
                        ):
                            local_actions = local_actions[:1]
                        (
                            traj_reject_required,
                            vlmap_traj_decision,
                        ) = self._validate_local_actions_with_vlmap(
                            local_actions,
                            observations,
                            current_depth_m,
                            rgb=rgb,
                            step_id=step_id,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            episode_index=episode_index,
                            episode_count=episode_count,
                            pixel_goal=pixel_goal,
                        )
                        if pending_s2_loop_strict_active_execution is not None:
                            first_action = local_actions[0] if local_actions else None
                            strict_trajectory_pass = bool(
                                not traj_reject_required
                                and first_action is not None
                                and int(first_action) != int(action_code.STOP)
                            )
                            pending_s2_loop_strict_active_execution.update(
                                {
                                    "event_type": "s2_loop_strict_active_execution",
                                    "event_schema_version": "stage21c_strict_active_v1",
                                    "reason": (
                                        "applied"
                                        if strict_trajectory_pass
                                        else "trajectory_preflight_rejected"
                                    ),
                                    "action_applied": strict_trajectory_pass,
                                    "execution_pending": False,
                                    "trajectory_preflight": {
                                        "valid": bool(vlmap_traj_decision.get("valid")),
                                        "safe": bool(vlmap_traj_decision.get("safe")),
                                        "would_reject": bool(
                                            vlmap_traj_decision.get("would_reject")
                                        ),
                                        "reason": vlmap_traj_decision.get("reason"),
                                        "reject_required": bool(traj_reject_required),
                                        "first_action": first_action,
                                        "local_actions": list(local_actions),
                                    },
                                }
                            )
                            self._write_s2_loop_strict_active_event(
                                pending_s2_loop_strict_active_execution
                            )
                            if strict_trajectory_pass:
                                s2_loop_strict_active_applied_count += 1
                                if s2_loop_strict_active_first_step is None:
                                    s2_loop_strict_active_first_step = int(step_id)
                            else:
                                # Never allow the evaluator's legacy STOP->LEFT
                                # fallback to turn a rejected recovery trajectory
                                # into an unreviewed environment action.
                                traj_reject_required = True
                            pending_s2_loop_strict_active_execution = None
                        if pending_s2_loop_path_execution is not None:
                            first_action = local_actions[0] if local_actions else None
                            path_trajectory_pass = bool(
                                not traj_reject_required
                                and first_action is not None
                                and int(first_action) != int(action_code.STOP)
                            )
                            reorient_already_applied = bool(
                                pending_s2_loop_path_execution.get(
                                    "intervention_already_applied"
                                )
                            )
                            pending_s2_loop_path_execution.update(
                                {
                                    "event_type": "s2_loop_path_reobserve_execution",
                                    "event_schema_version": "stage21c_path_reobserve_active_v1",
                                    "reason": (
                                        "path_pixel_applied"
                                        if path_trajectory_pass
                                        else "path_pixel_trajectory_preflight_rejected"
                                    ),
                                    "action_applied": bool(
                                        reorient_already_applied or path_trajectory_pass
                                    ),
                                    "reorient_action_applied": bool(
                                        reorient_already_applied
                                    ),
                                    "pixel_action_applied": bool(path_trajectory_pass),
                                    "execution_pending": False,
                                    "trajectory_preflight": {
                                        "valid": bool(vlmap_traj_decision.get("valid")),
                                        "safe": bool(vlmap_traj_decision.get("safe")),
                                        "would_reject": bool(
                                            vlmap_traj_decision.get("would_reject")
                                        ),
                                        "reason": vlmap_traj_decision.get("reason"),
                                        "reject_required": bool(traj_reject_required),
                                        "first_action": first_action,
                                        "local_actions": list(local_actions),
                                    },
                                }
                            )
                            self._write_s2_loop_path_reobserve_event(
                                pending_s2_loop_path_execution
                            )
                            if path_trajectory_pass:
                                if not reorient_already_applied:
                                    s2_loop_path_reobserve_applied_count += 1
                                    if s2_loop_path_reobserve_first_step is None:
                                        s2_loop_path_reobserve_first_step = int(step_id)
                            else:
                                # A rejected path pixel never falls through to
                                # the evaluator's legacy STOP->LEFT behavior.
                                traj_reject_required = True
                            pending_s2_loop_path_execution = None
                        nextdit_probe_cfg = self._get_nextdit_candidate_probe_cfg()
                        nextdit_query_index = (
                            nextdit_candidate_probe_event_count
                            + nextdit_candidate_probe_skipped_count
                            + nextdit_candidate_active_considered_count
                            + 1
                        )
                        nextdit_probe_event, nextdit_active_status, nextdit_probe_skipped = (
                            self._run_nextdit_candidate_probe_or_active(
                                nextdit_probe_dp_actions,
                                local_actions,
                                vlmap_traj_decision,
                                observations,
                                current_depth_m,
                                rgb,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                episode_index=episode_index,
                                episode_count=episode_count,
                                step_id=step_id,
                                query_index=nextdit_query_index,
                                pixel_goal=pixel_goal,
                                probe_event_count=nextdit_candidate_probe_event_count,
                                active_intervention_count=nextdit_candidate_active_intervention_count,
                            )
                        )
                        if nextdit_probe_skipped:
                            nextdit_candidate_probe_skipped_count += 1
                        if nextdit_probe_cfg.get("enable") and nextdit_probe_event:
                            nextdit_candidate_probe_event_count += 1
                            nextdit_candidate_probe_candidate_sum += int(
                                nextdit_probe_event.get("candidate_count", 0) or 0
                            )
                            nextdit_candidate_probe_unique_action_sum += int(
                                nextdit_probe_event.get("unique_action_sequence_count", 0) or 0
                            )
                            nextdit_candidate_probe_unique_endpoint_sum += int(
                                nextdit_probe_event.get("unique_endpoint_count", 0) or 0
                            )
                            nextdit_candidate_probe_would_reject_candidate_sum += int(
                                nextdit_probe_event.get("would_reject_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_valid_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_valid_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_invalid_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_invalid_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_would_reject_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_would_reject_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_unknown_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_unknown_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_checked_cell_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_checked_cell_count", 0.0) or 0.0
                            )
                            nextdit_candidate_occ_occupied_hit_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_occupied_hit_count", 0.0) or 0.0
                            )
                            nextdit_candidate_occ_unknown_hit_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_unknown_hit_count", 0.0) or 0.0
                            )
                            if nextdit_probe_event.get("current_occ_memory_valid"):
                                nextdit_candidate_occ_current_valid_event_count += 1
                                if nextdit_probe_event.get("current_occ_memory_would_reject"):
                                    nextdit_candidate_occ_current_would_reject_event_count += 1
                                nextdit_candidate_occ_current_occupied_hit_sum += float(
                                    nextdit_probe_event.get("current_occ_memory_occupied_hit_count", 0) or 0.0
                                )
                                nextdit_candidate_occ_current_unknown_hit_sum += float(
                                    nextdit_probe_event.get("current_occ_memory_unknown_hit_count", 0) or 0.0
                                )
                            if int(nextdit_probe_event.get("safer_candidate_count", 0) or 0) > 0:
                                nextdit_candidate_probe_safer_event_count += 1
                            if nextdit_probe_event.get("selected_differs_from_current"):
                                nextdit_candidate_probe_selected_diff_count += 1
                            if nextdit_probe_event.get("current_would_reject"):
                                nextdit_candidate_probe_current_reject_count += 1
                        if nextdit_active_status.get("considered"):
                            nextdit_candidate_active_considered_count += 1
                            if nextdit_active_status.get("applied"):
                                nextdit_candidate_active_intervention_count += 1
                                if nextdit_active_status.get("selected_differs_from_current"):
                                    nextdit_candidate_active_changed_count += 1
                                local_actions = list(nextdit_active_status.get("selected_actions") or local_actions)
                                vlmap_traj_decision = dict(
                                    nextdit_active_status.get("selected_decision") or vlmap_traj_decision
                                )
                                traj_reject_required = False
                            else:
                                nextdit_candidate_active_no_candidate_count += 1
                        if traj_reject_required:
                            pixel_goal = None
                            output_ids = None
                            traj_latents = None
                            pix_goal_image = None
                            pix_goal_depth = None
                            messages = []
                            input_images = []
                            llm_outputs = ""
                            local_actions = []
                            action_seq = []
                            vlmap_recovery_actions = []
                            action = None
                            forward_action = 0
                            draw_pixel_goal = False
                            flag = False
                            continue

                        action = local_actions[0]
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            pre_safety_action = action
                            action = action_code.LEFT
                            self._maybe_save_stuck_snapshot(
                                state=occ_memory_recovery_state,
                                event=occ_memory_recovery_event,
                                rgb=rgb,
                                step_id=step_id,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                instruction=episode_instruction,
                                action=action,
                                pixel_goal=pixel_goal,
                                local_actions=local_actions,
                                action_seq=action_seq,
                                llm_outputs=llm_outputs,
                                action_source="nextdit_stop_fallback_left",
                                pre_safety_action=pre_safety_action,
                                last_s2_query_step=last_s2_query_step,
                                episode_eval_seed=episode_eval_seed,
                                environment_step_applied=True,
                            )
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            self.replay_ledger.record_action(
                                step_id=step_id - 1,
                                action=action,
                                action_source="nextdit_stop_fallback_left",
                                pre_safety_action=pre_safety_action,
                                action_applied=True,
                                safety_decision={},
                                audit_metrics=replay_action_audit_metrics(),
                                next_observation_step_id=step_id,
                            )
                            last_action_applied = True
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                action_source = "fallback_stop"
                if len(vlmap_recovery_actions) != 0:
                    action = vlmap_recovery_actions.pop(0)
                    action_source = "vlmap_recovery_queue"
                    print("vlmap_recovery_action", action, flush=True)
                elif len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                    action_source = "system2_action_queue"
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # Regenerate local actions from the active System1 trajectory generator.
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        try:
                            # A directional recovery rewrites output_ids after the
                            # original S2 latent was consumed. Re-encode that goal
                            # before asking frozen NextDiT for a new local trajectory.
                            if traj_latents is None:
                                if output_ids is None or inputs is None:
                                    raise ValueError("missing_directional_replan_inputs")
                                pixel_values = inputs.pixel_values
                                image_grid_thw = torch.cat(
                                    [thw.unsqueeze(0) for thw in inputs.image_grid_thw],
                                    dim=0,
                                )
                                with torch.no_grad():
                                    traj_latents = self.model.generate_latents(
                                        output_ids, pixel_values, image_grid_thw
                                    )
                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(
                                    traj_latents, images_dp, depths_dp
                                )
                        except Exception as exc:
                            print(
                                "[Stage19][SemanticResilience][ActiveLite] "
                                "suppress local trajectory regeneration; "
                                f"error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            if stage19_active_status.get("applied"):
                                stage19_active_status.update(
                                    {
                                        "post_apply_execution_failure": True,
                                        "post_apply_execution_failure_reason": (
                                            "directional_replan_generation_failed"
                                        ),
                                        "post_apply_execution_error_type": type(exc).__name__,
                                        "post_apply_execution_error": str(exc),
                                    }
                                )
                                self._write_semantic_resilience_active_lite_event(
                                    {
                                        **stage19_active_status,
                                        "event_type": "stage19_semantic_resilience_execution",
                                        "reason": "directional_replan_generation_failed",
                                    }
                                )
                            pixel_goal = None
                            output_ids = None
                            traj_latents = None
                            pix_goal_image = None
                            pix_goal_depth = None
                            local_actions = []
                            action_seq = []
                            action = None
                            forward_action = 0
                            draw_pixel_goal = False
                            flag = False
                            continue

                        nextdit_probe_dp_actions = dp_actions.detach().clone() if hasattr(dp_actions, "detach") else None
                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        if (
                            pending_s2_loop_path_execution is not None
                            and self._get_s2_action_loop_cfg().get(
                                "path_reobserve_one_primitive_per_reaudit"
                            )
                        ):
                            local_actions = local_actions[:1]
                        (
                            traj_reject_required,
                            vlmap_traj_decision,
                        ) = self._validate_local_actions_with_vlmap(
                            local_actions,
                            observations,
                            current_depth_m,
                            rgb=rgb,
                            step_id=step_id,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            episode_index=episode_index,
                            episode_count=episode_count,
                            pixel_goal=pixel_goal,
                        )
                        nextdit_probe_cfg = self._get_nextdit_candidate_probe_cfg()
                        nextdit_query_index = (
                            nextdit_candidate_probe_event_count
                            + nextdit_candidate_probe_skipped_count
                            + nextdit_candidate_active_considered_count
                            + 1
                        )
                        nextdit_probe_event, nextdit_active_status, nextdit_probe_skipped = (
                            self._run_nextdit_candidate_probe_or_active(
                                nextdit_probe_dp_actions,
                                local_actions,
                                vlmap_traj_decision,
                                observations,
                                current_depth_m,
                                rgb,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                episode_index=episode_index,
                                episode_count=episode_count,
                                step_id=step_id,
                                query_index=nextdit_query_index,
                                pixel_goal=pixel_goal,
                                probe_event_count=nextdit_candidate_probe_event_count,
                                active_intervention_count=nextdit_candidate_active_intervention_count,
                            )
                        )
                        if nextdit_probe_skipped:
                            nextdit_candidate_probe_skipped_count += 1
                        if nextdit_probe_cfg.get("enable") and nextdit_probe_event:
                            nextdit_candidate_probe_event_count += 1
                            nextdit_candidate_probe_candidate_sum += int(
                                nextdit_probe_event.get("candidate_count", 0) or 0
                            )
                            nextdit_candidate_probe_unique_action_sum += int(
                                nextdit_probe_event.get("unique_action_sequence_count", 0) or 0
                            )
                            nextdit_candidate_probe_unique_endpoint_sum += int(
                                nextdit_probe_event.get("unique_endpoint_count", 0) or 0
                            )
                            nextdit_candidate_probe_would_reject_candidate_sum += int(
                                nextdit_probe_event.get("would_reject_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_valid_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_valid_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_invalid_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_invalid_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_would_reject_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_would_reject_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_unknown_candidate_sum += int(
                                nextdit_probe_event.get("occ_memory_score_unknown_candidate_count", 0) or 0
                            )
                            nextdit_candidate_occ_checked_cell_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_checked_cell_count", 0.0) or 0.0
                            )
                            nextdit_candidate_occ_occupied_hit_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_occupied_hit_count", 0.0) or 0.0
                            )
                            nextdit_candidate_occ_unknown_hit_sum += float(
                                nextdit_probe_event.get("occ_memory_score_mean_unknown_hit_count", 0.0) or 0.0
                            )
                            if nextdit_probe_event.get("current_occ_memory_valid"):
                                nextdit_candidate_occ_current_valid_event_count += 1
                                if nextdit_probe_event.get("current_occ_memory_would_reject"):
                                    nextdit_candidate_occ_current_would_reject_event_count += 1
                                nextdit_candidate_occ_current_occupied_hit_sum += float(
                                    nextdit_probe_event.get("current_occ_memory_occupied_hit_count", 0) or 0.0
                                )
                                nextdit_candidate_occ_current_unknown_hit_sum += float(
                                    nextdit_probe_event.get("current_occ_memory_unknown_hit_count", 0) or 0.0
                                )
                            if int(nextdit_probe_event.get("safer_candidate_count", 0) or 0) > 0:
                                nextdit_candidate_probe_safer_event_count += 1
                            if nextdit_probe_event.get("selected_differs_from_current"):
                                nextdit_candidate_probe_selected_diff_count += 1
                            if nextdit_probe_event.get("current_would_reject"):
                                nextdit_candidate_probe_current_reject_count += 1
                        if nextdit_active_status.get("considered"):
                            nextdit_candidate_active_considered_count += 1
                            if nextdit_active_status.get("applied"):
                                nextdit_candidate_active_intervention_count += 1
                                if nextdit_active_status.get("selected_differs_from_current"):
                                    nextdit_candidate_active_changed_count += 1
                                local_actions = list(nextdit_active_status.get("selected_actions") or local_actions)
                                vlmap_traj_decision = dict(
                                    nextdit_active_status.get("selected_decision") or vlmap_traj_decision
                                )
                                traj_reject_required = False
                            else:
                                nextdit_candidate_active_no_candidate_count += 1
                        if traj_reject_required:
                            pixel_goal = None
                            output_ids = None
                            traj_latents = None
                            pix_goal_image = None
                            pix_goal_depth = None
                            messages = []
                            input_images = []
                            llm_outputs = ""
                            local_actions = []
                            action_seq = []
                            vlmap_recovery_actions = []
                            action = None
                            forward_action = 0
                            draw_pixel_goal = False
                            flag = False
                            continue
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                        action_source = "nextdit_regenerated_local_queue"
                    else:
                        action = local_actions.pop(0)
                        action_source = "nextdit_local_queue"

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        self._maybe_save_stuck_snapshot(
                            state=occ_memory_recovery_state,
                            event=occ_memory_recovery_event,
                            rgb=rgb,
                            step_id=step_id,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            instruction=episode_instruction,
                            action=action,
                            pixel_goal=pixel_goal,
                            local_actions=local_actions,
                            action_seq=action_seq,
                            llm_outputs=llm_outputs,
                            action_source="nextdit_local_stop_discarded",
                            pre_safety_action=action,
                            last_s2_query_step=last_s2_query_step,
                            episode_eval_seed=episode_eval_seed,
                            environment_step_applied=False,
                        )
                        self.replay_ledger.record_action(
                            step_id=step_id,
                            action=action,
                            action_source="nextdit_local_stop_discarded",
                            pre_safety_action=action,
                            action_applied=False,
                            safety_decision={},
                            audit_metrics=replay_action_audit_metrics(),
                            next_observation_step_id=None,
                        )
                        last_action_applied = False
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                pre_safety_action = action
                action, vlmap_safety_changed, vlmap_safety_decision = self._postprocess_habitat_action_with_vlmap_safety(
                    action,
                    observations,
                    current_depth_m,
                    rgb=rgb,
                    step_id=step_id,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    episode_count=episode_count,
                    pixel_goal=pixel_goal,
                )
                vlmap_safety_replan = bool(
                    vlmap_safety_decision.get("replan_required")
                    or vlmap_safety_decision.get("waypoint_repair_required")
                )
                if vlmap_safety_changed or vlmap_safety_replan:
                    action_seq = []
                    local_actions = []
                    messages = []
                    forward_action = 0
                    recovery_actions = [
                        int(item)
                        for item in vlmap_safety_decision.get("recovery_actions", [])
                        if int(item) in (action_code.LEFT, action_code.RIGHT)
                    ]
                    if recovery_actions:
                        vlmap_recovery_actions = recovery_actions
                        print("[VLMapSafety][Habitat] queue recovery actions", vlmap_recovery_actions)
                    if vlmap_safety_replan:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                    elif pixel_goal is None:
                        output_ids = None

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                self._maybe_save_stuck_snapshot(
                    state=occ_memory_recovery_state,
                    event=occ_memory_recovery_event,
                    rgb=rgb,
                    step_id=step_id,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    instruction=episode_instruction,
                    action=action,
                    pixel_goal=pixel_goal,
                    local_actions=local_actions,
                    action_seq=action_seq,
                    llm_outputs=llm_outputs,
                    action_source=action_source,
                    pre_safety_action=pre_safety_action,
                    vlmap_safety_decision=vlmap_safety_decision,
                    last_s2_query_step=last_s2_query_step,
                    episode_eval_seed=episode_eval_seed,
                    environment_step_applied=True,
                )

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if pixel_goal is not None:
                        if draw_pixel_goal:
                            cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                    next_observation_step_id = int(step_id)
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    next_observation_step_id = int(step_id)
                    messages = []
                    flag = False
                    if action in (action_code.FORWARD, action_code.LEFT, action_code.RIGHT):
                        self._vlmap_last_nav_action = int(action)

                self.replay_ledger.record_action(
                    step_id=step_id if action == action_code.LOOKDOWN else step_id - 1,
                    action=action,
                    action_source=action_source,
                    pre_safety_action=pre_safety_action,
                    action_applied=True,
                    safety_decision=vlmap_safety_decision,
                    audit_metrics=replay_action_audit_metrics(),
                    next_observation_step_id=next_observation_step_id,
                )
                last_action_applied = True

                if (
                    pending_s2_loop_path_reobserve is not None
                    and action_source == "vlmap_recovery_queue"
                ):
                    planned_reorient = [
                        int(item)
                        for item in pending_s2_loop_path_reobserve.get(
                            "reorient_actions"
                        )
                        or []
                    ]
                    applied_reorient = [
                        int(item)
                        for item in pending_s2_loop_path_reobserve.get(
                            "reorient_actions_applied"
                        )
                        or []
                    ]
                    expected_action = (
                        planned_reorient[len(applied_reorient)]
                        if len(applied_reorient) < len(planned_reorient)
                        else None
                    )
                    if expected_action is not None and int(action) == int(
                        expected_action
                    ):
                        applied_reorient.append(int(action))
                        pending_s2_loop_path_reobserve[
                            "reorient_actions_applied"
                        ] = applied_reorient
                        pending_s2_loop_path_reobserve[
                            "last_reorient_environment_step"
                        ] = int(step_id)
                    else:
                        pending_s2_loop_path_reobserve.update(
                            {
                                "event_type": "s2_loop_path_reobserve_execution",
                                "reason": "reorient_action_changed_or_overrun",
                                "action_applied": bool(applied_reorient),
                                "reorient_action_applied": bool(applied_reorient),
                                "actual_action": int(action),
                                "expected_action": expected_action,
                                "execution_pending": False,
                            }
                        )
                        self._write_s2_loop_path_reobserve_event(
                            pending_s2_loop_path_reobserve
                        )
                        pending_s2_loop_path_reobserve = None

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            stuck_snapshot_cfg = dict(getattr(self.model_args, "vlmap_safety", {}) or {})
            force_snapshot_keys = {
                str(item)
                for item in stuck_snapshot_cfg.get(
                    "stuck_snapshot_force_episode_keys", []
                )
            }
            if f"{scene_id}/{episode_id}" in force_snapshot_keys:
                self._maybe_save_stuck_snapshot(
                    state=occ_memory_recovery_state,
                    event=occ_memory_recovery_event,
                    rgb=rgb,
                    step_id=step_id,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    instruction=episode_instruction,
                    action=action,
                    pixel_goal=pixel_goal,
                    local_actions=local_actions,
                    action_seq=action_seq,
                    llm_outputs=llm_outputs,
                    action_source=action_source,
                    pre_safety_action=pre_safety_action,
                    vlmap_safety_decision=vlmap_safety_decision,
                    last_s2_query_step=last_s2_query_step,
                    episode_eval_seed=episode_eval_seed,
                    environment_step_applied=True,
                    force=True,
                )

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()
            safety_summary = self._extract_collision_summary(metrics, steps=step_id)

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            collisions.append(safety_summary["collision_count"])
            collision_free.append(safety_summary["collision_free"])
            cf_sucs.append(safety_summary["cf_success"])
            cf_spls.append(safety_summary["cf_spl"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}, "
                f"collisions: {safety_summary['collision_count']}"
            )
            if pending_vlmap_semantic_hint and semantic_hint_not_injected_reason == "pending_next_s2_query":
                semantic_hint_not_injected_reason = "episode_ended"
            if pending_occ_memory_guidance_hint and occ_memory_guidance_not_injected_reason in (
                "pending_next_s2_query",
                "pending_immediate_requery",
            ):
                occ_memory_guidance_not_injected_reason = "episode_ended"
            if pending_s2_recovery_context is not None:
                pending_s2_recovery_context = None
                s2_recovery_context_expired_count += 1
            if pending_s2_loop_strict_active_execution is not None:
                pending_s2_loop_strict_active_execution.update(
                    {
                        "event_type": "s2_loop_strict_active_execution",
                        "reason": "episode_ended_before_trajectory_preflight",
                        "action_applied": False,
                        "execution_pending": False,
                    }
                )
                self._write_s2_loop_strict_active_event(
                    pending_s2_loop_strict_active_execution
                )
                pending_s2_loop_strict_active_execution = None
            if pending_s2_loop_path_reobserve is not None:
                planned_reorient = list(
                    pending_s2_loop_path_reobserve.get("reorient_actions") or []
                )
                applied_reorient = list(
                    pending_s2_loop_path_reobserve.get(
                        "reorient_actions_applied"
                    )
                    or []
                )
                pending_s2_loop_path_reobserve.update(
                    {
                        "event_type": "s2_loop_path_reobserve_execution",
                        "reason": "episode_ended_before_reobserve_query",
                        "action_applied": bool(applied_reorient),
                        "reorient_action_applied": bool(applied_reorient),
                        "reorient_complete": bool(
                            planned_reorient
                            and applied_reorient == planned_reorient
                        ),
                        "execution_pending": False,
                    }
                )
                self._write_s2_loop_path_reobserve_event(
                    pending_s2_loop_path_reobserve
                )
                pending_s2_loop_path_reobserve = None
            if pending_s2_loop_path_execution is not None:
                pending_s2_loop_path_execution.update(
                    {
                        "event_type": "s2_loop_path_reobserve_execution",
                        "reason": "episode_ended_before_path_trajectory_preflight",
                        "action_applied": False,
                        "pixel_action_applied": False,
                        "execution_pending": False,
                    }
                )
                self._write_s2_loop_path_reobserve_event(
                    pending_s2_loop_path_execution
                )
                pending_s2_loop_path_execution = None
            online_lseg_summary = self.online_lseg_shadow.finish_episode(
                metrics=metrics, steps=step_id, occ_memory=self.occ_memory
            )
            semantic_summary = self.vlmap_semantic.finish_episode(metrics=metrics, steps=step_id)
            occ_memory_summary = self.occ_memory.finish_episode(metrics=metrics, steps=step_id)
            replay_summary = self.replay_ledger.finish_episode(
                success=metrics.get("success"),
                steps=step_id,
                final_metrics={
                    "success": metrics.get("success"),
                    "spl": metrics.get("spl"),
                    "oracle_success": metrics.get("oracle_success"),
                    "distance_to_goal": metrics.get("distance_to_goal"),
                    "collision_count": safety_summary.get("collision_count"),
                    "collision_free": safety_summary.get("collision_free"),
                },
                semantic_summary=semantic_summary,
                occ_summary=occ_memory_summary,
                online_lseg_summary=online_lseg_summary,
            )
            occ_memory_oracle_pose_summary = {}
            if self.occ_memory_oracle_pose is not None:
                occ_memory_oracle_pose_summary = (
                    self.occ_memory_oracle_pose.finish_episode(
                        metrics=metrics, steps=step_id
                    )
                )
            occ_memory_oracle_sensor_summary = {}
            stage23a_pose_comparison = {}
            stage23a_pose_comparison_path = None
            if self.occ_memory_oracle_pose is not None:
                stage23a_pose_comparison = (
                    self.occ_memory.validation_compare_to_reference(
                        self.occ_memory_oracle_pose,
                        tolerance_cells=1,
                    )
                )
                comparison_root = self._get_vlmap_run_dir()
                if comparison_root:
                    comparison_root = os.path.join(
                        comparison_root, "stage23a_pose_occ_comparison"
                    )
                    os.makedirs(comparison_root, exist_ok=True)
                    stage23a_pose_comparison_path = os.path.join(
                        comparison_root,
                        f"{scene_id}_{episode_id}_comparison.json",
                    )
                    with open(
                        stage23a_pose_comparison_path,
                        "w",
                        encoding="utf-8",
                    ) as comparison_file:
                        json.dump(
                            stage23a_pose_comparison,
                            comparison_file,
                            ensure_ascii=False,
                            indent=2,
                        )
            stage23b_navmesh_current = {}
            stage23b_navmesh_oracle_sensor = {}
            stage23b_navmesh_current_clearance = {}
            stage23b_navmesh_oracle_sensor_clearance = {}
            if (
                self._stage23b_navmesh_audit_enabled
                and self.occ_memory_oracle_sensor_pose is not None
            ):
                stage23b_navmesh_current = (
                    self._stage23b_navmesh_traversability_audit(
                        self.occ_memory,
                        self.occ_memory_oracle_sensor_pose,
                        branch_name="current",
                        scene_id=scene_id,
                        episode_id=episode_id,
                    )
                )
                stage23b_navmesh_oracle_sensor = (
                    self._stage23b_navmesh_traversability_audit(
                        self.occ_memory_oracle_sensor_pose,
                        self.occ_memory_oracle_sensor_pose,
                        branch_name="oracle_sensor",
                        scene_id=scene_id,
                        episode_id=episode_id,
                    )
                )
                if self._stage23b_clearance_ablation_enabled:
                    stage23b_navmesh_current_clearance = (
                        self._stage23b_navmesh_traversability_audit(
                            self.occ_memory,
                            self.occ_memory_oracle_sensor_pose,
                            branch_name="current_clearance",
                            scene_id=scene_id,
                            episode_id=episode_id,
                            readout_height_max_m=(
                                self._stage23b_clearance_height_max_m
                            ),
                        )
                    )
                    stage23b_navmesh_oracle_sensor_clearance = (
                        self._stage23b_navmesh_traversability_audit(
                            self.occ_memory_oracle_sensor_pose,
                            self.occ_memory_oracle_sensor_pose,
                            branch_name="oracle_sensor_clearance",
                            scene_id=scene_id,
                            episode_id=episode_id,
                            readout_height_max_m=(
                                self._stage23b_clearance_height_max_m
                            ),
                        )
                    )
            stage23a_sensor_comparison = {}
            stage23a_sensor_comparison_path = None
            if self.occ_memory_oracle_sensor_pose is not None:
                occ_memory_oracle_sensor_summary = (
                    self.occ_memory_oracle_sensor_pose.finish_episode(
                        metrics=metrics, steps=step_id
                    )
                )
                stage23a_sensor_comparison = (
                    self.occ_memory.validation_compare_to_reference(
                        self.occ_memory_oracle_sensor_pose,
                        tolerance_cells=1,
                    )
                )
                comparison_root = self._get_vlmap_run_dir()
                if comparison_root:
                    comparison_root = os.path.join(
                        comparison_root, "stage23a_sensor_occ_comparison"
                    )
                    os.makedirs(comparison_root, exist_ok=True)
                    stage23a_sensor_comparison_path = os.path.join(
                        comparison_root,
                        f"{scene_id}_{episode_id}_comparison.json",
                    )
                    with open(
                        stage23a_sensor_comparison_path,
                        "w",
                        encoding="utf-8",
                    ) as comparison_file:
                        json.dump(
                            stage23a_sensor_comparison,
                            comparison_file,
                            ensure_ascii=False,
                            indent=2,
                        )
            stage23c_semantic_scene_audit = self._stage23c_semantic_scene_audit(
                self.occ_memory
            )
            stage23a_mesh_voxel_gt = self._stage23a_mesh_voxel_gt_summary(
                self.occ_memory_oracle_sensor_pose or self.occ_memory
            )
            occ_memory_recovery_summary = self._summarize_occ_memory_recovery_state(
                occ_memory_recovery_state
            )
            failure_prediction_summary = self._summarize_failure_prediction_state(
                failure_prediction_state
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "episode_index": int(episode_index),
                "episode_count": int(episode_count),
                "rank": int(getattr(self, "rank", 0)),
                "world_size": int(getattr(self, "world_size", 1)),
                "episode_eval_seed": episode_eval_seed,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                "stage23a_oracle_pose_audit_enabled": bool(
                    self.occ_memory_oracle_pose is not None
                ),
                "stage23a_gt_fields_used_for_navigation": [],
                "stage23a_oracle_pose_update_count": int(
                    occ_memory_oracle_pose_summary.get("update_count", 0) or 0
                ),
                "stage23a_oracle_sensor_pose_audit_enabled": bool(
                    self.occ_memory_oracle_sensor_pose is not None
                ),
                "stage23a_oracle_sensor_pose_update_count": int(
                    occ_memory_oracle_sensor_summary.get("update_count", 0) or 0
                ),
                "stage23a_sensor_occ_comparison_path": (
                    stage23a_sensor_comparison_path
                ),
                "stage23a_sensor_occ_comparison": stage23a_sensor_comparison,
                "stage23a_pose_occ_comparison_path": (
                    stage23a_pose_comparison_path
                ),
                "stage23a_pose_occ_comparison": stage23a_pose_comparison,
                "stage23a_gt_relative_height_range_m": (
                    occ_memory_oracle_pose_summary.get(
                        "validation_gt_relative_height_range_m"
                    )
                ),
                "stage23a_mesh_raycast": self._stage23a_mesh_raycast_summary(),
                "stage23a_mesh_voxel_gt": stage23a_mesh_voxel_gt,
                "stage23b_navmesh_traversability_current": (
                    stage23b_navmesh_current
                ),
                "stage23b_navmesh_traversability_oracle_sensor": (
                    stage23b_navmesh_oracle_sensor
                ),
                "stage23b_navmesh_traversability_current_clearance": (
                    stage23b_navmesh_current_clearance
                ),
                "stage23b_navmesh_traversability_oracle_sensor_clearance": (
                    stage23b_navmesh_oracle_sensor_clearance
                ),
                "stage23c_semantic_scene_audit": stage23c_semantic_scene_audit,
                "replay_ledger_enabled": bool(self.replay_ledger.enabled),
                "replay_ledger_observation_count": int(
                    replay_summary.get("observation_count", 0) or 0
                ),
                "replay_ledger_query_count": int(
                    replay_summary.get("query_count", 0) or 0
                ),
                "replay_ledger_action_count": int(
                    replay_summary.get("action_count", 0) or 0
                ),
                "replay_ledger_dir": (
                    str(self.replay_ledger.root / "replay_ledger")
                    if self.replay_ledger.enabled and self.replay_ledger.root is not None
                    else None
                ),
                "online_lseg_shadow_enabled": bool(
                    self.online_lseg_shadow.enabled
                ),
                "online_lseg_shadow_frame_count": int(
                    online_lseg_summary.get("frame_count", 0) or 0
                ),
                "online_lseg_shadow_valid_frame_count": int(
                    online_lseg_summary.get("valid_frame_count", 0) or 0
                ),
                "online_lseg_shadow_error_count": int(
                    online_lseg_summary.get("error_count", 0) or 0
                ),
                "online_lseg_shadow_inference_seconds_mean": (
                    online_lseg_summary.get("inference_seconds_mean")
                ),
                "online_lseg_shadow_surface_sample_count": int(
                    online_lseg_summary.get("stored_surface_sample_count", 0)
                    or 0
                ),
                "online_lseg_shadow_node_count": int(
                    online_lseg_summary.get("node_count", 0) or 0
                ),
                "online_lseg_shadow_multi_view_node_rate": (
                    online_lseg_summary.get("multi_view_node_rate")
                ),
                "online_lseg_shadow_cross_label_conflict_count": int(
                    online_lseg_summary.get("cross_label_conflict_count", 0)
                    or 0
                ),
                "online_lseg_shadow_gt_audit": online_lseg_summary.get(
                    "gt_audit"
                ),
                "online_lseg_shadow_decision_status": online_lseg_summary.get(
                    "decision_status"
                ),
            }
            result.update(safety_summary)
            result["s2_recovery_context_enabled"] = bool(
                self._get_s2_action_loop_cfg().get("recovery_context_enable")
            )
            result["s2_recovery_context_set_count"] = int(
                s2_recovery_context_set_count
            )
            result["s2_recovery_context_injected_count"] = int(
                s2_recovery_context_injected_count
            )
            result["s2_recovery_context_counterfactual_count"] = int(
                s2_recovery_context_counterfactual_count
            )
            result["s2_recovery_context_changed_count"] = int(
                s2_recovery_context_changed_count
            )
            result["s2_recovery_context_expired_count"] = int(
                s2_recovery_context_expired_count
            )
            result["s2_loop_strict_active_enabled"] = bool(
                self._get_s2_action_loop_cfg().get("strict_active_enable")
            )
            result["s2_loop_strict_active_event_count"] = int(
                s2_loop_strict_active_event_count
            )
            result["s2_loop_strict_active_rewrite_count"] = int(
                s2_loop_strict_active_rewrite_count
            )
            result["s2_loop_strict_active_applied_count"] = int(
                s2_loop_strict_active_applied_count
            )
            result["s2_loop_strict_active_first_step"] = s2_loop_strict_active_first_step
            result["s2_loop_projection_bridge_enabled"] = bool(
                self._get_s2_action_loop_cfg().get("projection_bridge_enable")
            )
            result["s2_loop_projection_bridge_event_count"] = int(
                s2_loop_projection_bridge_event_count
            )
            result["s2_loop_projection_bridge_strict_count"] = int(
                s2_loop_projection_bridge_strict_count
            )
            result["s2_loop_projection_bridge_valid_count"] = int(
                s2_loop_projection_bridge_valid_count
            )
            result["s2_loop_path_reobserve_active_enabled"] = bool(
                self._get_s2_action_loop_cfg().get("path_reobserve_active_enable")
            )
            result["s2_loop_path_reobserve_event_count"] = int(
                s2_loop_path_reobserve_event_count
            )
            result["s2_loop_path_reobserve_intervention_count"] = int(
                s2_loop_path_reobserve_intervention_count
            )
            result["s2_loop_path_reobserve_reorient_count"] = int(
                s2_loop_path_reobserve_reorient_count
            )
            result["s2_loop_path_reobserve_post_query_count"] = int(
                s2_loop_path_reobserve_post_query_count
            )
            result["s2_loop_path_reobserve_pixel_rewrite_count"] = int(
                s2_loop_path_reobserve_pixel_rewrite_count
            )
            result["s2_loop_path_reobserve_applied_count"] = int(
                s2_loop_path_reobserve_applied_count
            )
            result["s2_loop_path_reobserve_first_step"] = (
                s2_loop_path_reobserve_first_step
            )
            if self._get_occ_memory_recovery_cfg().get("enable"):
                result["occ_memory_recovery_event_count"] = occ_memory_recovery_summary.get("event_count")
                result["occ_memory_recovery_logged_event_count"] = occ_memory_recovery_summary.get(
                    "logged_event_count"
                )
                result["occ_memory_recovery_trigger_event_count"] = occ_memory_recovery_summary.get(
                    "recovery_trigger_event_count"
                )
                result["occ_memory_recovery_trigger_start_count"] = occ_memory_recovery_summary.get(
                    "recovery_trigger_start_count"
                )
                result["occ_memory_recovery_first_trigger_step"] = occ_memory_recovery_summary.get(
                    "first_recovery_trigger_step"
                )
                result["occ_memory_recovery_map_stagnation_event_count"] = occ_memory_recovery_summary.get(
                    "map_stagnation_event_count"
                )
                result["occ_memory_recovery_map_stagnation_start_count"] = occ_memory_recovery_summary.get(
                    "map_stagnation_start_count"
                )
                result["occ_memory_recovery_first_map_stagnation_step"] = occ_memory_recovery_summary.get(
                    "first_map_stagnation_step"
                )
                result["occ_memory_recovery_total_map_stagnation_event_count"] = (
                    occ_memory_recovery_summary.get("total_map_stagnation_event_count")
                )
                result["occ_memory_recovery_low_displacement_event_count"] = occ_memory_recovery_summary.get(
                    "low_displacement_event_count"
                )
                result["occ_memory_recovery_collision_trigger_event_count"] = occ_memory_recovery_summary.get(
                    "collision_trigger_event_count"
                )
                result["occ_memory_recovery_collision_trigger_start_count"] = occ_memory_recovery_summary.get(
                    "collision_trigger_start_count"
                )
                result["occ_memory_recovery_first_collision_trigger_step"] = occ_memory_recovery_summary.get(
                    "first_collision_trigger_step"
                )
                result["occ_memory_recovery_max_occupied_stagnation_streak"] = (
                    occ_memory_recovery_summary.get("max_occupied_stagnation_streak")
                )
                result["occ_memory_recovery_max_total_stagnation_streak"] = (
                    occ_memory_recovery_summary.get("max_total_stagnation_streak")
                )
                result["occ_memory_recovery_max_collision_delta"] = occ_memory_recovery_summary.get(
                    "max_collision_delta"
                )
                result["occ_memory_recovery_min_pose_window_displacement_m"] = (
                    occ_memory_recovery_summary.get("min_pose_window_displacement_m")
                )
                result["occ_memory_recovery_active_intervention_count"] = (
                    occ_memory_recovery_summary.get("active_intervention_count")
                )
                result["occ_memory_recovery_active_applied_count"] = (
                    occ_memory_recovery_summary.get("active_applied_count")
                )
                result["occ_memory_recovery_active_suppressed_count"] = (
                    occ_memory_recovery_summary.get("active_suppressed_count")
                )
                result["occ_memory_recovery_active_first_step"] = (
                    occ_memory_recovery_summary.get("active_first_step")
                )
                result["occ_memory_recovery_active_reason_counts"] = (
                    occ_memory_recovery_summary.get("active_reason_counts")
                )
            if self._get_semantic_resilience_active_lite_cfg().get("enable"):
                result["stage19_semantic_resilience_active_enabled"] = True
                result["stage19_semantic_resilience_active_shadow_only"] = bool(
                    self._get_semantic_resilience_active_lite_cfg().get("shadow_only")
                )
                result["stage19_semantic_resilience_active_considered_count"] = int(
                    stage19_semantic_resilience_active_considered_count
                )
                result["stage19_semantic_resilience_active_applied_count"] = int(
                    stage19_semantic_resilience_active_applied_count
                )
                result["stage19_semantic_resilience_active_suppressed_count"] = int(
                    stage19_semantic_resilience_active_suppressed_count
                )
                result["stage19_semantic_resilience_active_first_step"] = (
                    stage19_semantic_resilience_active_first_step
                )
                result["stage19_semantic_resilience_active_last_step"] = (
                    stage19_semantic_resilience_active_last_step
                )
                result["stage19_semantic_resilience_active_mean_action_count"] = (
                    float(stage19_semantic_resilience_active_action_sum)
                    / max(1, int(stage19_semantic_resilience_active_applied_count))
                )
                result["stage19_semantic_resilience_active_reason_counts"] = dict(
                    stage19_semantic_resilience_active_reason_counts
                )
                result["stage19_semantic_resilience_failure_type_counts"] = dict(
                    stage19_semantic_resilience_failure_type_counts
                )
                result["stage19_semantic_resilience_recommended_primitive_counts"] = dict(
                    stage19_semantic_resilience_recommended_primitive_counts
                )
                episode_failure_type, episode_recommended_primitive = (
                    self._summarize_stage19_episode_failure_type(
                        metrics,
                        failure_type_counts=stage19_semantic_resilience_failure_type_counts,
                        recommended_primitive_counts=(
                            stage19_semantic_resilience_recommended_primitive_counts
                        ),
                        step_id=step_id,
                        collision_count=metrics.get("collision_count"),
                    )
                )
                result["stage19_semantic_resilience_episode_failure_type"] = episode_failure_type
                result["stage19_semantic_resilience_episode_recommended_primitive"] = (
                    episode_recommended_primitive
                )
            if self._get_failure_prediction_cfg().get("enable"):
                result["failure_prediction_event_count"] = failure_prediction_summary.get("event_count")
                result["failure_prediction_logged_event_count"] = failure_prediction_summary.get(
                    "logged_event_count"
                )
                result["failure_prediction_predicted_event_count"] = failure_prediction_summary.get(
                    "predicted_event_count"
                )
                result["failure_prediction_start_count"] = failure_prediction_summary.get(
                    "prediction_start_count"
                )
                result["failure_prediction_first_step"] = failure_prediction_summary.get(
                    "first_predicted_step"
                )
                result["failure_prediction_max_score"] = failure_prediction_summary.get("max_failure_score")
                result["failure_prediction_max_stagnation_score"] = failure_prediction_summary.get(
                    "max_stagnation_score"
                )
                result["failure_prediction_max_semantic_score"] = failure_prediction_summary.get(
                    "max_semantic_score"
                )
                result["failure_prediction_max_collision_score"] = failure_prediction_summary.get(
                    "max_collision_score"
                )
                result["failure_prediction_max_displacement_score"] = failure_prediction_summary.get(
                    "max_displacement_score"
                )
                result["failure_prediction_max_pg_ecc_score"] = failure_prediction_summary.get(
                    "max_pg_ecc_score"
                )
                result["failure_prediction_max_compass_reversal_score"] = (
                    failure_prediction_summary.get("max_compass_reversal_score")
                )
                result["failure_prediction_max_heading_var_score"] = failure_prediction_summary.get(
                    "max_heading_var_score"
                )
                result["failure_prediction_mode_hint_counts"] = failure_prediction_summary.get(
                    "mode_hint_counts"
                )
            if semantic_summary:
                result["semantic_landmark_count"] = semantic_summary.get("landmark_count")
                result["semantic_seen_count"] = semantic_summary.get("seen_count")
                result["semantic_coverage"] = semantic_summary.get("coverage")
                result["semantic_first_seen_step"] = semantic_summary.get("first_seen_step")
                result["semantic_rank1_coverage"] = semantic_summary.get("rank1_coverage")
                result["semantic_rank1_confident_coverage"] = semantic_summary.get(
                    "rank1_confident_coverage"
                )
                result["semantic_relative_coverage"] = semantic_summary.get("relative_coverage")
                result["semantic_mean_top_score"] = semantic_summary.get("mean_top_score")
                result["semantic_max_top_score"] = semantic_summary.get("max_top_score")
                result["semantic_mean_top_margin"] = semantic_summary.get("mean_top_margin")
                result["semantic_high_conf_seen"] = semantic_summary.get("high_conf_seen")
                result["semantic_high_conf_event_count"] = semantic_summary.get(
                    "high_conf_event_count"
                )
                result["semantic_high_conf_step_fraction"] = semantic_summary.get(
                    "high_conf_step_fraction"
                )
                result["semantic_first_high_conf_step"] = semantic_summary.get(
                    "first_high_conf_step"
                )
                result["semantic_max_low_conf_streak"] = semantic_summary.get(
                    "max_low_conf_streak"
                )
                result["semantic_confidence_would_requery"] = semantic_summary.get(
                    "confidence_would_requery"
                )
                result["semantic_confidence_would_requery_count"] = semantic_summary.get(
                    "confidence_would_requery_count"
                )
                result["semantic_first_confidence_would_requery_step"] = semantic_summary.get(
                    "first_confidence_would_requery_step"
                )
                result["semantic_stagnation_would_requery"] = semantic_summary.get(
                    "stagnation_would_requery"
                )
                result["semantic_stagnation_would_requery_count"] = semantic_summary.get(
                    "stagnation_would_requery_count"
                )
                result["semantic_first_stagnation_would_requery_step"] = semantic_summary.get(
                    "first_stagnation_would_requery_step"
                )
                result["semantic_stagnation_min_recent_unique_count"] = semantic_summary.get(
                    "stagnation_min_recent_unique_count"
                )
                result["semantic_stagnation_low_diversity_window_count"] = semantic_summary.get(
                    "stagnation_low_diversity_window_count"
                )
                result["semantic_stagnation_hint_set_count"] = semantic_hint_set_count
                result["semantic_stagnation_hint_injected_count"] = semantic_hint_injected_count
                result["semantic_first_stagnation_hint_detection_step"] = (
                    semantic_hint_detection_step
                )
                result["semantic_first_stagnation_hint_injection_step"] = (
                    semantic_hint_injection_step
                )
                result["semantic_stagnation_hint_pending_at_end"] = bool(
                    pending_vlmap_semantic_hint
                )
                result["semantic_stagnation_hint_not_injected_reason"] = (
                    semantic_hint_not_injected_reason
                )
                result["semantic_top1_stability"] = semantic_summary.get("top1_stability")
                result["semantic_top1_diversity"] = semantic_summary.get("top1_diversity")
                coverage_by_threshold = semantic_summary.get("coverage_by_threshold") or {}
                for threshold_key, coverage_value in coverage_by_threshold.items():
                    result[f"semantic_coverage_at_{str(threshold_key).replace('.', '_')}"] = (
                        coverage_value
                    )
            result["occ_memory_guidance_trigger_count"] = occ_memory_guidance_trigger_count
            result["occ_memory_guidance_hint_set_count"] = occ_memory_guidance_hint_set_count
            result["occ_memory_guidance_hint_injected_count"] = occ_memory_guidance_hint_injected_count
            result["occ_memory_guidance_requery_count"] = occ_memory_guidance_requery_count
            result["occ_memory_guidance_shadow_skip_count"] = occ_memory_guidance_shadow_skip_count
            result["occ_memory_guidance_blocked_count"] = occ_memory_guidance_blocked_count
            result["occ_memory_guidance_first_detection_step"] = occ_memory_guidance_detection_step
            result["occ_memory_guidance_first_injection_step"] = occ_memory_guidance_injection_step
            result["occ_memory_guidance_pending_at_end"] = bool(pending_occ_memory_guidance_hint)
            result["occ_memory_guidance_not_injected_reason"] = occ_memory_guidance_not_injected_reason
            result["occ_memory_guidance_counterfactual_count"] = occ_memory_guidance_counterfactual_count
            result["occ_memory_guidance_counterfactual_valid_count"] = (
                occ_memory_guidance_counterfactual_valid_count
            )
            result["occ_memory_guidance_counterfactual_changed_count"] = (
                occ_memory_guidance_counterfactual_changed_count
            )
            result["occ_memory_guidance_counterfactual_direction_changed_count"] = (
                occ_memory_guidance_counterfactual_direction_changed_count
            )
            result["occ_memory_guidance_counterfactual_left_right_follow_count"] = (
                occ_memory_guidance_counterfactual_left_right_follow_count
            )
            if occ_memory_guidance_counterfactual_valid_count > 0:
                result["occ_memory_guidance_counterfactual_mean_pixel_shift"] = (
                    occ_memory_guidance_counterfactual_pixel_shift_sum
                    / float(occ_memory_guidance_counterfactual_valid_count)
                )
            else:
                result["occ_memory_guidance_counterfactual_mean_pixel_shift"] = None
            if self._get_som_counterfactual_cfg().get("enable"):
                result["som_counterfactual_count"] = som_counterfactual_count
                result["som_counterfactual_valid_count"] = som_counterfactual_valid_count
                result["som_counterfactual_changed_count"] = som_counterfactual_changed_count
                result["som_counterfactual_direction_changed_count"] = (
                    som_counterfactual_direction_changed_count
                )
                result["som_counterfactual_frontier_follow_count"] = (
                    som_counterfactual_frontier_follow_count
                )
                result["som_counterfactual_unsafe_shift_proxy_count"] = (
                    som_counterfactual_unsafe_shift_proxy_count
                )
                result["som_counterfactual_moved_away_count"] = None
                result["som_counterfactual_moved_away_note"] = (
                    "not computed in Stage14b-v1; use unsafe_shift_proxy only as a "
                    "significant-shift proxy when unsafe signal is present"
                )
                result["som_counterfactual_skipped_count"] = som_counterfactual_skipped_count
                result["som_counterfactual_error_count"] = som_counterfactual_error_count
                result["som_counterfactual_active_applied_count"] = (
                    som_counterfactual_active_applied_count
                )
                if som_counterfactual_valid_count > 0:
                    result["som_counterfactual_mean_pixel_shift"] = (
                        som_counterfactual_pixel_shift_sum
                        / float(som_counterfactual_valid_count)
                    )
                else:
                    result["som_counterfactual_mean_pixel_shift"] = None
            stage15_cfg = self._get_stage15_repair_cfg()
            if stage15_cfg.get("active"):
                result["stage15_repair_active_event_count"] = int(
                    stage15_repair_active_event_count
                )
                result["stage15_repair_active_applied_count"] = int(
                    stage15_repair_active_applied_count
                )
                result["stage15_repair_active_first_step"] = stage15_repair_active_first_step
                result["stage15_repair_active_gate_mode"] = stage15_cfg.get("gate_mode")
                result["stage15_repair_active_gate_min_count"] = int(
                    stage15_cfg.get("gate_min_count", 0) or 0
                )
                result["stage15_repair_active_max_per_episode"] = int(
                    stage15_cfg.get("active_max_per_episode", 0) or 0
                )
                result["stage15_repair_final_consecutive_count"] = int(
                    stage15_repair_consecutive_count
                )
                result["stage15_repair_final_cumulative_count"] = int(
                    stage15_repair_cumulative_count
                )
                for reason_key, reason_count in stage15_repair_active_reason_counts.items():
                    result[f"stage15_repair_active_reason_{reason_key}_count"] = reason_count
            stage_d_cfg = self._get_stage_d_bfs_escape_cfg()
            if stage_d_cfg.get("enable"):
                result["stage_d_bfs_escape_event_count"] = int(stage_d_bfs_escape_event_count)
                result["stage_d_bfs_escape_trigger_count"] = int(stage_d_bfs_escape_trigger_count)
                result["stage_d_bfs_escape_reachable_count"] = int(
                    stage_d_bfs_escape_reachable_count
                )
                result["stage_d_bfs_escape_reachable_rate"] = (
                    float(stage_d_bfs_escape_reachable_count)
                    / max(1, int(stage_d_bfs_escape_trigger_count))
                )
                result["stage_d_bfs_escape_first_trigger_step"] = (
                    stage_d_bfs_escape_first_trigger_step
                )
                result["stage_d_bfs_escape_max_action_steps"] = int(
                    stage_d_cfg.get("max_action_steps", 8) or 8
                )
                result["stage_d_bfs_escape_compass_reversal_threshold"] = float(
                    stage_d_cfg.get("compass_reversal_threshold", 0.07) or 0.07
                )
                result["stage_d_bfs_escape_compass_reversal_metric"] = str(
                    stage_d_cfg.get("compass_reversal_metric", "sign") or "sign"
                )
                result["stage_d_bfs_escape_consecutive_occupied_min"] = int(
                    stage_d_cfg.get("consecutive_occupied_min", 3) or 3
                )
                result["stage_d_bfs_escape_active_enabled"] = bool(
                    stage_d_cfg.get("active")
                    and not bool(stage_d_cfg.get("shadow_only", True))
                )
                result["stage_d_bfs_escape_active_event_count"] = int(
                    stage_d_bfs_escape_active_event_count
                )
                result["stage_d_bfs_escape_active_applied_count"] = int(
                    stage_d_bfs_escape_active_applied_count
                )
                result["stage_d_bfs_escape_active_first_step"] = (
                    stage_d_bfs_escape_active_first_step
                )
                result["stage_d_bfs_escape_active_max_per_episode"] = int(
                    stage_d_cfg.get("active_max_per_episode", 0) or 0
                )
                result["stage_d_bfs_escape_active_require_target_frontier"] = bool(
                    stage_d_cfg.get("active_require_target_frontier", True)
                )
                result["stage_d_bfs_escape_active_path_edge_steps"] = int(
                    stage_d_cfg.get("active_path_edge_steps", 8) or 8
                )
                result["stage_d_bfs_escape_active_pixel_goal_mode"] = str(
                    stage_d_cfg.get("active_pixel_goal_mode", "projection") or "projection"
                )
                for reason_key, reason_count in stage_d_bfs_escape_reason_counts.items():
                    result[f"stage_d_bfs_escape_reason_{reason_key}_count"] = reason_count
                for reason_key, reason_count in stage_d_bfs_escape_trigger_reason_counts.items():
                    result[f"stage_d_bfs_escape_trigger_{reason_key}_count"] = reason_count
                for reason_key, reason_count in stage_d_bfs_escape_active_reason_counts.items():
                    result[f"stage_d_bfs_escape_active_reason_{reason_key}_count"] = reason_count
            candidate_probe_cfg = self._get_occ_memory_candidate_probe_cfg()
            if candidate_probe_cfg.get("enable"):
                result["occ_memory_candidate_probe_event_count"] = (
                    occ_memory_candidate_probe_event_count
                )
                result["occ_memory_candidate_probe_valid_event_count"] = (
                    occ_memory_candidate_probe_valid_event_count
                )
                result["occ_memory_candidate_probe_skipped_count"] = (
                    occ_memory_candidate_probe_skipped_count
                )
                result["occ_memory_candidate_probe_valid_event_ratio"] = (
                    occ_memory_candidate_probe_valid_event_count
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_candidate_count"] = (
                    occ_memory_candidate_probe_candidate_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_geometry_safe_count"] = (
                    occ_memory_candidate_probe_geometry_safe_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_active_gate_safe_count"] = (
                    occ_memory_candidate_probe_active_gate_safe_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_current_aligned_count"] = (
                    occ_memory_candidate_probe_current_aligned_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_next_landmark_relevant_count"] = (
                    occ_memory_candidate_probe_next_landmark_relevant_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_completed_landmark_count"] = (
                    occ_memory_candidate_probe_completed_landmark_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_repeated_semantic_count"] = (
                    occ_memory_candidate_probe_repeated_semantic_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_unknown_target_frontier_bonus_count"] = (
                    occ_memory_candidate_probe_unknown_target_frontier_bonus_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_target_frontier_count"] = (
                    occ_memory_candidate_probe_target_frontier_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_target_frontier_escape_count"] = (
                    occ_memory_candidate_probe_target_frontier_escape_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_target_frontier_intent_safe_count"] = (
                    occ_memory_candidate_probe_target_frontier_intent_safe_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_probe_mean_target_frontier_doorway_like_count"] = (
                    occ_memory_candidate_probe_target_frontier_doorway_like_sum
                    / max(1, occ_memory_candidate_probe_event_count)
                )
                result["occ_memory_candidate_selection_query_count"] = (
                    occ_memory_candidate_selection_query_count
                )
                result["occ_memory_candidate_selection_valid_count"] = (
                    occ_memory_candidate_selection_valid_count
                )
                result["occ_memory_candidate_selection_none_count"] = (
                    occ_memory_candidate_selection_none_count
                )
                result["occ_memory_candidate_selection_error_count"] = (
                    occ_memory_candidate_selection_error_count
                )
                result["occ_memory_candidate_selection_label_count"] = (
                    occ_memory_candidate_selection_label_count
                )
                result["occ_memory_candidate_selection_coordinate_count"] = (
                    occ_memory_candidate_selection_coordinate_count
                )
                result["occ_memory_candidate_selection_direction_count"] = (
                    occ_memory_candidate_selection_direction_count
                )
                result["occ_memory_candidate_selection_valid_ratio"] = (
                    occ_memory_candidate_selection_valid_count
                    / max(1, occ_memory_candidate_selection_query_count)
                )
                result["occ_memory_candidate_selection_active_gate_safe_count"] = (
                    occ_memory_candidate_selection_active_gate_safe_count
                )
                result["occ_memory_candidate_selection_current_aligned_count"] = (
                    occ_memory_candidate_selection_current_aligned_count
                )
                result["occ_memory_candidate_selection_semanticized_count"] = (
                    occ_memory_candidate_selection_semanticized_count
                )
                result["occ_memory_candidate_selection_instruction_relevant_count"] = (
                    occ_memory_candidate_selection_instruction_relevant_count
                )
                result["occ_memory_candidate_selection_next_landmark_relevant_count"] = (
                    occ_memory_candidate_selection_next_landmark_relevant_count
                )
                result["occ_memory_candidate_selection_completed_landmark_count"] = (
                    occ_memory_candidate_selection_completed_landmark_count
                )
                result["occ_memory_candidate_selection_repeated_semantic_count"] = (
                    occ_memory_candidate_selection_repeated_semantic_count
                )
            if occ_memory_summary:
                result["occ_memory_update_count"] = occ_memory_summary.get("update_count")
                result["occ_memory_occupied_voxel_count"] = occ_memory_summary.get("occupied_voxel_count")
                result["occ_memory_free_voxel_count"] = occ_memory_summary.get("free_voxel_count")
                result["occ_memory_occupied_cell_count"] = occ_memory_summary.get("occupied_cell_count")
                result["occ_memory_free_cell_count"] = occ_memory_summary.get("free_cell_count")
                result["occ_memory_frontier_count"] = occ_memory_summary.get("frontier_count")
                result["occ_memory_pose_count"] = occ_memory_summary.get("pose_count")
                result["occ_memory_keyframe_count"] = occ_memory_summary.get("keyframe_count")
                result["occ_memory_semantic_event_count"] = occ_memory_summary.get("semantic_event_count")
                result["occ_memory_waypoint_probe_count"] = occ_memory_summary.get("waypoint_probe_count")
                result["occ_memory_candidate_probe_summary_event_count"] = (
                    occ_memory_summary.get("candidate_probe_event_count")
                )
                result["occ_memory_candidate_probe_summary_valid_event_count"] = (
                    occ_memory_summary.get("candidate_probe_valid_event_count")
                )
                result["occ_memory_candidate_probe_summary_mean_candidate_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_candidate_count")
                )
                result["occ_memory_candidate_probe_summary_mean_semantic_evidence_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_semantic_evidence_count")
                )
                result["occ_memory_candidate_probe_summary_mean_instruction_relevant_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_instruction_relevant_count")
                )
                result["occ_memory_candidate_probe_summary_mean_semanticized_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_semanticized_count")
                )
                result["occ_memory_candidate_probe_summary_mean_next_landmark_relevant_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_next_landmark_relevant_count")
                )
                result["occ_memory_candidate_probe_summary_mean_completed_landmark_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_completed_landmark_count")
                )
                result["occ_memory_candidate_probe_summary_mean_repeated_semantic_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_repeated_semantic_count")
                )
                result[
                    "occ_memory_candidate_probe_summary_mean_unknown_target_frontier_bonus_count"
                ] = occ_memory_summary.get("candidate_probe_mean_unknown_target_frontier_bonus_count")
                result["occ_memory_candidate_probe_summary_mean_target_frontier_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_target_frontier_count")
                )
                result["occ_memory_candidate_probe_summary_mean_target_frontier_escape_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_target_frontier_escape_count")
                )
                result["occ_memory_candidate_probe_summary_mean_target_frontier_intent_safe_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_target_frontier_intent_safe_count")
                )
                result["occ_memory_candidate_probe_summary_mean_target_frontier_doorway_like_count"] = (
                    occ_memory_summary.get("candidate_probe_mean_target_frontier_doorway_like_count")
                )
                result["occ_memory_candidate_selection_summary_event_count"] = (
                    occ_memory_summary.get("candidate_selection_event_count")
                )
                result["occ_memory_candidate_selection_summary_valid_count"] = (
                    occ_memory_summary.get("candidate_selection_valid_count")
                )
                result["occ_memory_candidate_selection_summary_next_landmark_relevant_count"] = (
                    occ_memory_summary.get("candidate_selection_next_landmark_relevant_count")
                )
                result["occ_memory_candidate_selection_summary_completed_landmark_count"] = (
                    occ_memory_summary.get("candidate_selection_completed_landmark_count")
                )
                result["occ_memory_candidate_selection_summary_repeated_semantic_count"] = (
                    occ_memory_summary.get("candidate_selection_repeated_semantic_count")
                )
                candidate_selection_reason_counts = (
                    occ_memory_summary.get("candidate_selection_reason_counts") or {}
                )
                for reason_key, reason_count in candidate_selection_reason_counts.items():
                    result[f"occ_memory_candidate_selection_reason_{reason_key}_count"] = (
                        reason_count
                    )
                result["occ_memory_waypoint_mean_frontier_distance_m"] = (
                    occ_memory_summary.get("waypoint_mean_frontier_distance_m")
                )
                waypoint_state_counts = occ_memory_summary.get("waypoint_goal_state_counts") or {}
                for state_key, state_count in waypoint_state_counts.items():
                    result[f"occ_memory_waypoint_goal_{state_key}_count"] = state_count
                for stage15_key in (
                    "stage15_repair_event_count",
                    "stage15_roundtrip_valid_count",
                    "stage15_roundtrip_error_mean_px",
                    "stage15_roundtrip_error_median_px",
                    "stage15_roundtrip_error_p90_px",
                    "stage15_roundtrip_error_max_px",
                    "stage15_repair_candidate_count",
                    "stage15_repair_free_found_count",
                    "stage15_repair_valid_count",
                    "stage15_repair_no_free_count",
                    "stage15_repair_projection_failed_count",
                    "stage15_repair_pixel_shift_mean",
                    "stage15_repair_pixel_shift_median",
                    "stage15_repair_backtrack_cells_mean",
                    "stage15_repair_backtrack_cells_median",
                ):
                    result[stage15_key] = occ_memory_summary.get(stage15_key)
                stage15_repair_reason_counts = (
                    occ_memory_summary.get("stage15_repair_reason_counts") or {}
                )
                for reason_key, reason_count in stage15_repair_reason_counts.items():
                    result[f"stage15_repair_reason_{reason_key}_count"] = reason_count
                waypoint_direction_counts = occ_memory_summary.get("waypoint_direction_counts") or {}
                for direction_key, direction_count in waypoint_direction_counts.items():
                    result[f"occ_memory_waypoint_direction_{direction_key}_count"] = direction_count
                final_frontier_counts = occ_memory_summary.get("final_frontier_direction_counts") or {}
                for direction_key, direction_count in final_frontier_counts.items():
                    result[f"occ_memory_final_frontier_{direction_key}_count"] = direction_count
                candidate_type_counts = occ_memory_summary.get("candidate_probe_type_counts") or {}
                for type_key, type_count in candidate_type_counts.items():
                    result[f"occ_memory_candidate_type_{type_key}_count"] = type_count
                candidate_direction_counts = occ_memory_summary.get("candidate_probe_direction_counts") or {}
                for direction_key, direction_count in candidate_direction_counts.items():
                    result[f"occ_memory_candidate_direction_{direction_key}_count"] = direction_count
                for current_policy_key in (
                    "current_policy_candidate_valid_count",
                    "current_policy_candidate_valid_rate",
                    "current_policy_candidate_geometry_safe_count",
                    "current_policy_candidate_geometry_safe_rate",
                    "current_policy_candidate_active_gate_safe_count",
                    "current_policy_candidate_active_gate_safe_rate",
                    "current_policy_candidate_revisited_count",
                    "current_policy_candidate_revisited_rate",
                    "current_policy_candidate_dead_zone_count",
                    "current_policy_candidate_dead_zone_rate",
                ):
                    result[f"occ_memory_{current_policy_key}"] = occ_memory_summary.get(
                        current_policy_key
                    )
                for shadow_key in (
                    "progress_ranker_shadow_enabled_count",
                    "progress_ranker_shadow_valid_count",
                    "progress_ranker_shadow_error_count",
                    "progress_ranker_shadow_ranker_change_count",
                    "progress_ranker_shadow_resilience_change_count",
                    "progress_ranker_shadow_resilience_completed_count",
                    "progress_ranker_shadow_resilience_repeated_count",
                    "progress_ranker_shadow_resilience_unsafe_count",
                    "progress_ranker_shadow_resilience_change_ratio",
                    "progress_ranker_shadow_resilience_completed_rate",
                    "progress_ranker_shadow_resilience_repeated_rate",
                    "progress_ranker_shadow_resilience_unsafe_rate",
                    "progress_ranker_shadow_resilience_future_observability_mean",
                    "progress_ranker_shadow_resilience_recoverability_mean",
                    "stage21_multitask_shadow_enabled_count",
                    "stage21_multitask_shadow_valid_count",
                    "stage21_multitask_shadow_error_count",
                    "stage21_multitask_shadow_action_applied_count",
                    "stage21_multitask_shadow_progress_change_count",
                    "stage21_multitask_shadow_intent_change_count",
                    "stage21_multitask_shadow_recovery_candidate_count",
                    "stage21_multitask_shadow_progress_change_rate",
                    "stage21_multitask_shadow_intent_change_rate",
                    "stage21_multitask_shadow_missing_numeric_mean",
                    "stage21_multitask_shadow_latency_mean_ms",
                    "stage21_multitask_shadow_latency_p95_ms",
                ):
                    result[f"occ_memory_{shadow_key}"] = occ_memory_summary.get(shadow_key)
                dead_zone_frontier_counts = (
                    occ_memory_summary.get("semantic_dead_zone_frontier_direction_counts") or {}
                )
                for direction_key, direction_count in dead_zone_frontier_counts.items():
                    result[f"occ_memory_dead_zone_frontier_{direction_key}_count"] = direction_count
                result["occ_memory_semantic_high_conf_event_count"] = (
                    occ_memory_summary.get("semantic_high_conf_event_count")
                )
                result["occ_memory_semantic_high_conf_keyframe_count"] = (
                    occ_memory_summary.get("semantic_high_conf_keyframe_count")
                )
                result["occ_memory_waypoint_frontier_alignment_count"] = (
                    occ_memory_summary.get("waypoint_frontier_alignment_count")
                )
                result["occ_memory_waypoint_frontier_alignment_ratio"] = (
                    occ_memory_summary.get("waypoint_frontier_alignment_ratio")
                )
                result["occ_memory_waypoint_high_conf_alignment_count"] = (
                    occ_memory_summary.get("waypoint_high_conf_alignment_count")
                )
                result["occ_memory_waypoint_high_conf_alignment_ratio"] = (
                    occ_memory_summary.get("waypoint_high_conf_alignment_ratio")
                )
                result["occ_memory_semantic_dead_zone_waypoint_count"] = (
                    occ_memory_summary.get("semantic_dead_zone_waypoint_count")
                )
                result["occ_memory_semantic_dead_zone_waypoint_ratio"] = (
                    occ_memory_summary.get("semantic_dead_zone_waypoint_ratio")
                )
                result["occ_memory_semantic_first_dead_zone_waypoint_step"] = (
                    occ_memory_summary.get("semantic_first_dead_zone_waypoint_step")
                )
                result["occ_memory_semantic_dead_zone_mean_score"] = (
                    occ_memory_summary.get("semantic_dead_zone_mean_score")
                )
                result["occ_memory_semantic_dead_zone_max_score"] = (
                    occ_memory_summary.get("semantic_dead_zone_max_score")
                )
                result["occ_memory_semantic_dead_zone_with_frontier_count"] = (
                    occ_memory_summary.get("semantic_dead_zone_with_frontier_count")
                )
                result["occ_memory_final_frontier_total_count_for_direction"] = (
                    occ_memory_summary.get("final_frontier_total_count_for_direction")
                )
                result["occ_memory_final_frontier_sampled_count_for_direction"] = (
                    occ_memory_summary.get("final_frontier_sampled_count_for_direction")
                )
                result["occ_memory_final_frontier_sample_fraction_for_direction"] = (
                    occ_memory_summary.get("final_frontier_sample_fraction_for_direction")
                )
                result["occ_memory_final_frontier_dominant_direction"] = (
                    occ_memory_summary.get("final_frontier_dominant_direction")
                )
                result["occ_memory_final_frontier_dominant_angle_deg"] = (
                    occ_memory_summary.get("final_frontier_dominant_angle_deg")
                )
                result["occ_memory_final_frontier_direction_entropy"] = (
                    occ_memory_summary.get("final_frontier_direction_entropy")
                )
                result["occ_memory_bev_snapshot_count"] = occ_memory_summary.get("bev_snapshot_count")
                result["occ_memory_candidate_bev_snapshot_count"] = (
                    occ_memory_summary.get("candidate_bev_snapshot_count")
                )
                result["occ_memory_validation_snapshot_count"] = (
                    occ_memory_summary.get("validation_snapshot_count")
                )
                result["occ_memory_validation_final_snapshot_count"] = (
                    occ_memory_summary.get("validation_final_snapshot_count")
                )
            s2_probe_cfg = self._get_s2_candidate_probe_cfg()
            if s2_probe_cfg.get("enable"):
                result["s2_candidate_probe_s2_query_count"] = s2_candidate_probe_s2_query_count
                result["s2_candidate_probe_event_count"] = s2_candidate_probe_event_count
                result["s2_candidate_probe_skipped_query_count"] = (
                    s2_candidate_probe_skipped_query_count
                )
                result["s2_candidate_probe_valid_query_count"] = (
                    s2_candidate_probe_valid_query_count
                )
                result["s2_candidate_probe_diverse_query_count"] = (
                    s2_candidate_probe_diverse_query_count
                )
                result["s2_candidate_probe_valid_query_ratio"] = (
                    s2_candidate_probe_valid_query_count / max(1, s2_candidate_probe_event_count)
                )
                result["s2_candidate_probe_diverse_query_ratio"] = (
                    s2_candidate_probe_diverse_query_count / max(1, s2_candidate_probe_event_count)
                )
                result["s2_candidate_probe_mean_valid_candidate_count"] = (
                    s2_candidate_probe_valid_candidate_sum / max(1, s2_candidate_probe_event_count)
                )
                result["s2_candidate_probe_mean_unique_candidate_count"] = (
                    s2_candidate_probe_unique_candidate_sum / max(1, s2_candidate_probe_event_count)
                )
                result["s2_candidate_probe_mean_pairwise_pixel_distance"] = (
                    s2_candidate_probe_mean_pairwise_distance_sum / max(1, s2_candidate_probe_event_count)
                )
                result["s2_candidate_probe_max_pairwise_pixel_distance"] = (
                    s2_candidate_probe_max_pairwise_distance
                )
            nextdit_probe_cfg = self._get_nextdit_candidate_probe_cfg()
            if nextdit_probe_cfg.get("enable"):
                result["nextdit_candidate_probe_event_count"] = nextdit_candidate_probe_event_count
                result["nextdit_candidate_probe_skipped_count"] = nextdit_candidate_probe_skipped_count
                result["nextdit_candidate_probe_mean_candidate_count"] = (
                    nextdit_candidate_probe_candidate_sum / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_mean_unique_action_count"] = (
                    nextdit_candidate_probe_unique_action_sum / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_mean_unique_endpoint_count"] = (
                    nextdit_candidate_probe_unique_endpoint_sum / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_mean_would_reject_candidate_count"] = (
                    nextdit_candidate_probe_would_reject_candidate_sum
                    / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_safer_event_count"] = (
                    nextdit_candidate_probe_safer_event_count
                )
                result["nextdit_candidate_probe_safer_event_ratio"] = (
                    nextdit_candidate_probe_safer_event_count
                    / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_selected_diff_count"] = (
                    nextdit_candidate_probe_selected_diff_count
                )
                result["nextdit_candidate_probe_selected_diff_ratio"] = (
                    nextdit_candidate_probe_selected_diff_count
                    / max(1, nextdit_candidate_probe_event_count)
                )
                result["nextdit_candidate_probe_current_reject_count"] = (
                    nextdit_candidate_probe_current_reject_count
                )
                if nextdit_probe_cfg.get("occ_memory_score_enable"):
                    result["nextdit_candidate_occ_valid_candidate_count"] = (
                        nextdit_candidate_occ_valid_candidate_sum
                    )
                    result["nextdit_candidate_occ_invalid_candidate_count"] = (
                        nextdit_candidate_occ_invalid_candidate_sum
                    )
                    result["nextdit_candidate_occ_valid_ratio"] = (
                        nextdit_candidate_occ_valid_candidate_sum
                        / max(1, nextdit_candidate_probe_candidate_sum)
                    )
                    result["nextdit_candidate_occ_would_reject_candidate_count"] = (
                        nextdit_candidate_occ_would_reject_candidate_sum
                    )
                    result["nextdit_candidate_occ_would_reject_ratio"] = (
                        nextdit_candidate_occ_would_reject_candidate_sum
                        / max(1, nextdit_candidate_occ_valid_candidate_sum)
                    )
                    result["nextdit_candidate_occ_unknown_candidate_count"] = (
                        nextdit_candidate_occ_unknown_candidate_sum
                    )
                    result["nextdit_candidate_occ_unknown_ratio"] = (
                        nextdit_candidate_occ_unknown_candidate_sum
                        / max(1, nextdit_candidate_occ_valid_candidate_sum)
                    )
                    result["nextdit_candidate_occ_mean_checked_cell_count"] = (
                        nextdit_candidate_occ_checked_cell_sum
                        / max(1, nextdit_candidate_probe_event_count)
                    )
                    result["nextdit_candidate_occ_mean_occupied_hit_count"] = (
                        nextdit_candidate_occ_occupied_hit_sum
                        / max(1, nextdit_candidate_probe_event_count)
                    )
                    result["nextdit_candidate_occ_mean_unknown_hit_count"] = (
                        nextdit_candidate_occ_unknown_hit_sum
                        / max(1, nextdit_candidate_probe_event_count)
                    )
                    result["nextdit_candidate_occ_current_valid_event_count"] = (
                        nextdit_candidate_occ_current_valid_event_count
                    )
                    result["nextdit_candidate_occ_current_would_reject_event_count"] = (
                        nextdit_candidate_occ_current_would_reject_event_count
                    )
                    result["nextdit_candidate_occ_current_would_reject_event_ratio"] = (
                        nextdit_candidate_occ_current_would_reject_event_count
                        / max(1, nextdit_candidate_occ_current_valid_event_count)
                    )
                    result["nextdit_candidate_occ_current_mean_occupied_hit_count"] = (
                        nextdit_candidate_occ_current_occupied_hit_sum
                        / max(1, nextdit_candidate_occ_current_valid_event_count)
                    )
                    result["nextdit_candidate_occ_current_mean_unknown_hit_count"] = (
                        nextdit_candidate_occ_current_unknown_hit_sum
                        / max(1, nextdit_candidate_occ_current_valid_event_count)
                    )
            if nextdit_probe_cfg.get("active_enable"):
                result["nextdit_candidate_active_considered_count"] = (
                    nextdit_candidate_active_considered_count
                )
                result["nextdit_candidate_active_intervention_count"] = (
                    nextdit_candidate_active_intervention_count
                )
                result["nextdit_candidate_active_changed_count"] = (
                    nextdit_candidate_active_changed_count
                )
                result["nextdit_candidate_active_no_candidate_count"] = (
                    nextdit_candidate_active_no_candidate_count
                )
                result["nextdit_candidate_active_intervention_ratio"] = (
                    nextdit_candidate_active_intervention_count
                    / max(1, nextdit_candidate_active_considered_count)
                )
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            # save current progress
            self._write_episode_progress(result)

            # save video
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
            torch.tensor(collisions).to(self.device),
            torch.tensor(collision_free).to(self.device),
            torch.tensor(cf_sucs).to(self.device),
            torch.tensor(cf_spls).to(self.device),
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        (
            sucs,
            spls,
            oss,
            nes,
            ndtw,
            collisions,
            collision_free,
            cf_sucs,
            cf_spls,
        ) = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break
            episode_index = max(0, getattr(self.env, "_current_episode_index", 1) - 1)

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            self._seed_eval_rng_for_episode(episode_index, episode_id, scene_id)
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = self._select_s2_prompt_prefix() + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()
            safety_summary = self._extract_collision_summary(metrics, steps=step_id)

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            collisions.append(safety_summary["collision_count"])
            collision_free.append(safety_summary["collision_free"])
            cf_sucs.append(safety_summary["cf_success"])
            cf_spls.append(safety_summary["cf_spl"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}, "
                f"collisions: {safety_summary['collision_count']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            result.update(safety_summary)
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            self._write_episode_progress(result)
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
            torch.tensor(collisions).to(self.device),
            torch.tensor(collision_free).to(self.device),
            torch.tensor(cf_sucs).to(self.device),
            torch.tensor(cf_spls).to(self.device),
        )

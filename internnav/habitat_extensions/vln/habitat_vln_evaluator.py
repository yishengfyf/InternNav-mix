import argparse
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import cv2
import habitat
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
from PIL import Image
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
from internnav.utils.vlmap_safety import VLMapActionSafety

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

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

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
            f.write(json.dumps(result) + "\n")

        run_dir = self._get_vlmap_run_dir()
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, 'progress.json'), 'a', encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _seed_eval_rng(self, seed: int, label: str = "") -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if label:
            print(f"[HabitatVLN] fixed eval random seed ({label}): {seed}")

    def _seed_eval_rng_for_episode(self, episode_index: int, episode_id: int) -> None:
        base_seed = getattr(self.model_args, "eval_random_seed", None)
        if base_seed is None or not bool(getattr(self.model_args, "eval_seed_per_episode", False)):
            return

        mode = getattr(self.model_args, "eval_episode_seed_mode", "episode_index")
        if mode == "episode_id":
            episode_offset = int(episode_id)
        elif mode == "episode_index":
            episode_offset = int(episode_index)
        else:
            raise ValueError(f"Invalid eval_episode_seed_mode: {mode}")

        rank_offset = int(getattr(self, "rank", 0)) * 100000
        episode_seed = int(base_seed) + episode_offset + rank_offset
        self._seed_eval_rng(episode_seed, f"episode_index={episode_index}, episode_id={episode_id}")

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
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

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
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

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
            self._seed_eval_rng_for_episode(episode_index, episode_id)
            print("episode start", episode_instruction)
            self.vlmap_safety.reset()
            self._vlmap_last_nav_action = None

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
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []
            vlmap_recovery_actions = []
            pending_vlmap_waypoint_feedback = ""
            rejected_vlmap_goal_grids = []

            done = False
            flag = False
            pixel_goal = None

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

                    if pending_vlmap_waypoint_feedback:
                        sources[0]["value"] += f" {pending_vlmap_waypoint_feedback}"
                        print(
                            "[VLMapSafety][Habitat][Waypoint] inject S2 feedback: "
                            f"{pending_vlmap_waypoint_feedback}"
                        )
                        pending_vlmap_waypoint_feedback = ""

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

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
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
                            action = action_code.LEFT
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(vlmap_recovery_actions) != 0:
                    action = vlmap_recovery_actions.pop(0)
                    print("vlmap_recovery_action", action, flush=True)
                elif len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # Regenerate local actions from the active System1 trajectory generator.
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
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
                    else:
                        action = local_actions.pop(0)

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
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

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
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False
                    if action in (action_code.FORWARD, action_code.LEFT, action_code.RIGHT):
                        self._vlmap_last_nav_action = int(action)

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode progress.json entry (still per-rank)
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
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

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
            self._seed_eval_rng_for_episode(episode_index, episode_id)
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

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
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
        )

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
from internnav.utils.sparse_occ_memory import SparseOccSemanticMemory
from internnav.utils.vlmap_safety import VLMapActionSafety
from internnav.utils.vlmap_semantic import VLMapSemanticShadow

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
        self.vlmap_semantic = VLMapSemanticShadow(vlmap_safety_cfg)
        self.vlmap_semantic.set_debug_dir(self._get_vlmap_run_dir())
        self.occ_memory = SparseOccSemanticMemory(
            vlmap_safety_cfg,
            get_intrinsic_matrix(self.sim_sensors_config.depth_sensor),
        )
        self.occ_memory.set_debug_dir(self._get_vlmap_run_dir())

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
            "active_max_interventions_per_episode": int(
                vlmap_safety_cfg.get("nextdit_candidate_active_max_interventions_per_episode", 2)
            ),
            "active_require_current_reject": bool(
                vlmap_safety_cfg.get("nextdit_candidate_active_require_current_reject", True)
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
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "actions": actions,
                    "score": float(self._trajectory_decision_score(decision)),
                    "obstacle_score": float(self._trajectory_obstacle_score(decision)),
                    "decision": decision,
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

    def _nextdit_active_probe_needed(self, current_decision: dict, active_intervention_count: int) -> bool:
        cfg = self._get_nextdit_candidate_probe_cfg()
        if not cfg.get("active_enable"):
            return False
        max_interventions = int(cfg.get("active_max_interventions_per_episode", 2))
        if max_interventions >= 0 and active_intervention_count >= max_interventions:
            return False
        if cfg.get("active_require_current_reject", True) and not bool(
            (current_decision or {}).get("would_reject")
        ):
            return False
        return True

    def _select_nextdit_active_candidate(self, event: dict):
        cfg = self._get_nextdit_candidate_probe_cfg()
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
        if cfg.get("active_require_current_reject", True) and not bool(event.get("current_would_reject")):
            status["reason"] = "current_not_reject"
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
                "selected_candidate_index": None,
                "selected_actions": None,
                "selected_score": None,
                "selected_obstacle_score": None,
                "selected_decision": None,
                "selected_differs_from_current": False,
                "candidate_count": event.get("candidate_count"),
                "would_reject_candidate_count": event.get("would_reject_candidate_count"),
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
            "selected_candidate_index": int(selected.get("candidate_index", -1)),
            "selected_actions": selected_actions,
            "selected_score": selected.get("score"),
            "selected_obstacle_score": selected.get("obstacle_score"),
            "selected_decision": selected.get("decision"),
            "selected_differs_from_current": bool(selected_actions != current_actions),
            "candidate_count": event.get("candidate_count"),
            "would_reject_candidate_count": event.get("would_reject_candidate_count"),
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
            f"score={active_event['current_obstacle_score']}->{active_event['selected_obstacle_score']}"
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
        active_needed = self._nextdit_active_probe_needed(current_decision, active_intervention_count)
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
            pending_vlmap_semantic_hint = ""
            semantic_hint_set_count = 0
            semantic_hint_injected_count = 0
            semantic_hint_detection_step = None
            semantic_hint_injection_step = None
            semantic_hint_not_injected_reason = None
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
            nextdit_candidate_probe_event_count = 0
            nextdit_candidate_probe_skipped_count = 0
            nextdit_candidate_probe_candidate_sum = 0
            nextdit_candidate_probe_unique_action_sum = 0
            nextdit_candidate_probe_unique_endpoint_sum = 0
            nextdit_candidate_probe_safer_event_count = 0
            nextdit_candidate_probe_selected_diff_count = 0
            nextdit_candidate_probe_current_reject_count = 0
            nextdit_candidate_probe_would_reject_candidate_sum = 0
            nextdit_candidate_active_considered_count = 0
            nextdit_candidate_active_intervention_count = 0
            nextdit_candidate_active_changed_count = 0
            nextdit_candidate_active_no_candidate_count = 0

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
                self.occ_memory.update_observation(
                    {
                        "rgb": rgb,
                        "depth": current_depth_m,
                        "gps": observations.get("gps"),
                        "compass": observations.get("compass"),
                    },
                    current_depth_m,
                    rgb=rgb,
                    context={
                        "step_id": step_id,
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "episode_count": episode_count,
                    },
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
                        self.occ_memory.record_semantic(semantic_decision)

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
                            },
                        )
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
                            continue
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

                        nextdit_probe_dp_actions = dp_actions.detach().clone() if hasattr(dp_actions, "detach") else None
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

                        nextdit_probe_dp_actions = dp_actions.detach().clone() if hasattr(dp_actions, "detach") else None
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
            if pending_vlmap_semantic_hint and semantic_hint_not_injected_reason == "pending_next_s2_query":
                semantic_hint_not_injected_reason = "episode_ended"
            semantic_summary = self.vlmap_semantic.finish_episode(metrics=metrics, steps=step_id)
            occ_memory_summary = self.occ_memory.finish_episode(metrics=metrics, steps=step_id)

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
                result["occ_memory_waypoint_mean_frontier_distance_m"] = (
                    occ_memory_summary.get("waypoint_mean_frontier_distance_m")
                )
                waypoint_state_counts = occ_memory_summary.get("waypoint_goal_state_counts") or {}
                for state_key, state_count in waypoint_state_counts.items():
                    result[f"occ_memory_waypoint_goal_{state_key}_count"] = state_count
                result["occ_memory_bev_snapshot_count"] = occ_memory_summary.get("bev_snapshot_count")
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

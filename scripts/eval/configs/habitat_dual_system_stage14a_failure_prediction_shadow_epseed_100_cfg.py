import copy
import importlib.util
from pathlib import Path


def _load_stage13a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage13a_occ_memory_recovery_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage13a_occ_memory_recovery_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage13a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage14a: online failure/stuck prediction shadow.
#
# This experiment does not change InternNav actions. It logs per-step features
# and a conservative rule score for future failure/stuck prediction. Visual
# set-of-marks and active local planning are intentionally disabled here so the
# prediction signal is attributable.
vlmap_cfg["failure_prediction_enable"] = True
vlmap_cfg["failure_prediction_shadow_only"] = True
vlmap_cfg["failure_prediction_version"] = "stage14a_rule_v1"
vlmap_cfg["failure_prediction_min_step"] = 30
vlmap_cfg["failure_prediction_window_steps"] = 20
vlmap_cfg["failure_prediction_threshold"] = 0.65
vlmap_cfg["failure_prediction_stagnation_score_weight"] = 0.40
vlmap_cfg["failure_prediction_semantic_weight"] = 0.30
vlmap_cfg["failure_prediction_collision_weight"] = 0.20
vlmap_cfg["failure_prediction_displacement_weight"] = 0.10
vlmap_cfg["failure_prediction_map_growth_weight"] = 0.15
vlmap_cfg["failure_prediction_unsafe_waypoint_weight"] = 0.15
vlmap_cfg["failure_prediction_stagnation_streak_scale"] = 30.0
vlmap_cfg["failure_prediction_low_map_growth_norm"] = 120.0
vlmap_cfg["failure_prediction_displacement_norm_m"] = 1.25
vlmap_cfg["failure_prediction_collision_norm"] = 2.0
vlmap_cfg["failure_prediction_min_explore_efficiency"] = 20.0
vlmap_cfg["failure_prediction_log_every_step"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14a_failure_prediction_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14a_failure_prediction_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2401"

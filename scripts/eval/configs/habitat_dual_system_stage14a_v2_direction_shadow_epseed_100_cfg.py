import copy
import importlib.util
from pathlib import Path


def _load_stage14a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage14a_failure_prediction_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage14a_failure_prediction_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage14a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage14a-v2: direction-aware failure prediction shadow.
#
# This keeps the experiment observational: no action, waypoint, trajectory, or
# S2 prompt is changed. Direction features are computed from an in-memory cache
# of S2 pixel goals, GPS, and compass values.
vlmap_cfg["failure_prediction_enable"] = True
vlmap_cfg["failure_prediction_shadow_only"] = True
vlmap_cfg["failure_prediction_version"] = "stage14a_v2_direction_rule"
vlmap_cfg["failure_prediction_direction_enable"] = True
vlmap_cfg["failure_prediction_threshold"] = 0.60

# Keep the active score weights normalized to 1.0 for v2.
vlmap_cfg["failure_prediction_stagnation_score_weight"] = 0.35
vlmap_cfg["failure_prediction_semantic_weight"] = 0.20
vlmap_cfg["failure_prediction_collision_weight"] = 0.15
vlmap_cfg["failure_prediction_displacement_weight"] = 0.05
vlmap_cfg["failure_prediction_map_growth_weight"] = 0.0
vlmap_cfg["failure_prediction_unsafe_waypoint_weight"] = 0.0
vlmap_cfg["failure_prediction_pg_ecc_weight"] = 0.10
vlmap_cfg["failure_prediction_compass_reversal_weight"] = 0.10
vlmap_cfg["failure_prediction_heading_var_weight"] = 0.05

vlmap_cfg["failure_prediction_pg_ecc_threshold"] = 0.30
vlmap_cfg["failure_prediction_pg_ecc_norm"] = 0.30
vlmap_cfg["failure_prediction_compass_reversal_max"] = 4.0
vlmap_cfg["failure_prediction_direction_image_width"] = 640.0
vlmap_cfg["failure_prediction_direction_cache_max_events"] = 256

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14a_v2_direction_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14a_v2_direction_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2401"

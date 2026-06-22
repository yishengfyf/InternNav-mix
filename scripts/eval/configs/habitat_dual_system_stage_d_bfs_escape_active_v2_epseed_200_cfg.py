import copy
import importlib.util
from pathlib import Path


def _load_stage_d_active_200_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage_d_bfs_escape_active_epseed_200_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage_d_bfs_escape_active_epseed_200_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage_d_active_200_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage_d_bfs_escape_active_pixel_goal_mode"] = "directional"
vlmap_cfg["stage_d_bfs_escape_active_direction_front_x_ratio"] = 0.50
vlmap_cfg["stage_d_bfs_escape_active_direction_left_x_ratio"] = 0.25
vlmap_cfg["stage_d_bfs_escape_active_direction_right_x_ratio"] = 0.75
vlmap_cfg["stage_d_bfs_escape_active_direction_y_ratio"] = 0.75
vlmap_cfg["stage_d_bfs_escape_active_require_pixel_in_bounds"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_active_v2_epseed_200/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_active_v2_epseed_200"
)
eval_cfg.eval_settings["port"] = "2416"

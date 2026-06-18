import copy
import importlib.util
from pathlib import Path


def _load_stage_d_shadow_200_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage_d_bfs_escape_shadow_epseed_200_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage_d_bfs_escape_shadow_epseed_200_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage_d_shadow_200_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage_d_bfs_escape_shadow_enable"] = True
vlmap_cfg["stage_d_bfs_escape_shadow_only"] = False
vlmap_cfg["stage_d_bfs_escape_active_enable"] = True
vlmap_cfg["stage_d_bfs_escape_active_max_per_episode"] = 2
vlmap_cfg["stage_d_bfs_escape_active_require_target_frontier"] = True
vlmap_cfg["stage_d_bfs_escape_active_path_edge_steps"] = 8
vlmap_cfg["stage_d_bfs_escape_active_goal_world_z"] = 0.0
vlmap_cfg["stage_d_bfs_escape_active_require_pixel_in_bounds"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_active_epseed_200/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_active_epseed_200"
)
eval_cfg.eval_settings["port"] = "2414"

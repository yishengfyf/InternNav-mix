import copy
import importlib.util
from pathlib import Path


def _load_base_cfg():
    base_path = Path(__file__).with_name("habitat_dual_system_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_cfg_base", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_base_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = True
vlmap_cfg["waypoint_shadow_only"] = False
vlmap_cfg["waypoint_requery_enable"] = True
vlmap_cfg["waypoint_requery_on_block"] = True
vlmap_cfg["waypoint_requery_on_high_risk"] = True
vlmap_cfg["waypoint_requery_risk_threshold"] = 0.70
vlmap_cfg["waypoint_requery_min_checked_cells"] = 20
vlmap_cfg["max_waypoint_requeries_per_episode"] = 1
vlmap_cfg["waypoint_requery_cooldown_steps"] = 40
vlmap_cfg["debug_dir"] = "./logs/habitat/compare_vlmap_stage2_100_active_requery_v0/vlmap_safety_debug"
vlmap_cfg["debug_max_snapshots"] = 45
vlmap_cfg["debug_sample_total_snapshots"] = 25
vlmap_cfg["debug_force_max_snapshots"] = 10
vlmap_cfg["debug_force_max_snapshots_per_episode"] = 1
vlmap_cfg["waypoint_save_snapshots"] = True
vlmap_cfg["waypoint_risk_threshold"] = 0.60
vlmap_cfg["waypoint_risk_min_checked_cells"] = 4
vlmap_cfg["waypoint_force_save_on_risk"] = True
vlmap_cfg["waypoint_force_save_on_block"] = True
vlmap_cfg["waypoint_force_max_snapshots"] = 30
vlmap_cfg["verbose"] = True

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_stage2_100_active_requery_v0"
eval_cfg.eval_settings["port"] = "2343"

import copy
import importlib.util
from pathlib import Path


def _load_baseline_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_baseline_fixed_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_baseline_fixed_epseed_100_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_baseline_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = True
vlmap_cfg["traj_validation_shadow_only"] = True
vlmap_cfg["traj_validation_horizon"] = 4
vlmap_cfg["traj_validation_block_threshold"] = 1
vlmap_cfg["traj_validation_max_rejects_per_episode"] = 2
vlmap_cfg["traj_validation_cooldown_steps"] = 20
vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = "./logs/habitat/compare_vlmap_stage4_30_traj_shadow_epseed/vlmap_safety_debug"
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = True

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_stage4_30_traj_shadow_epseed"
eval_cfg.eval_settings["port"] = "2350"

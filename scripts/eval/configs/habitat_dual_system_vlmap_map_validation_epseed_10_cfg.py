import copy
import importlib.util
from pathlib import Path


def _load_baseline_epseed_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_baseline_fixed_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_baseline_fixed_epseed_100_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_baseline_epseed_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage 0 validation: build the online obstacle map but do not change any
# InternNav action. This variant saves PLY/NPZ plus RGB/depth frames so that
# map points can be checked against visible walls, furniture, and door frames.
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = "./logs/habitat/compare_vlmap_map_validation_rgb_ply_epseed/vlmap_safety_debug"
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = True

vlmap_cfg["map_validation_enable"] = True
vlmap_cfg["map_validation_every_updates"] = 2
vlmap_cfg["map_validation_max_snapshots"] = 10
vlmap_cfg["map_validation_depth_sample_rate"] = 10
vlmap_cfg["map_validation_max_points"] = 200000
vlmap_cfg["map_validation_save_npz"] = True
vlmap_cfg["map_validation_save_ply"] = True
vlmap_cfg["map_validation_save_topdown"] = True
vlmap_cfg["map_validation_save_rgb_depth"] = True
# Optional: set this to a large data disk on the server, for example
# "/data/yifeifeng/vlmap_validation/compare_vlmap_map_validation_rgb_ply_epseed".
# If None or absent, files are written under debug_dir/run_*/map_validation.
vlmap_cfg["map_validation_dir"] = "/data/usr_data/yifeifeng/internnav/debug/test_2"

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 2
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_map_validation_rgb_ply_epseed"
eval_cfg.eval_settings["port"] = "2365"

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

# V6a is a shadow-only feasibility probe. It samples extra S2 waypoint
# candidates to measure diversity, then restores RNG state and executes the
# original greedy S2 output so navigation behavior remains baseline-equivalent.
vlmap_cfg["enable"] = False
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["s2_candidate_probe_enable"] = True
vlmap_cfg["s2_candidate_count"] = 3
vlmap_cfg["s2_candidate_temperature"] = 0.7
vlmap_cfg["s2_candidate_top_p"] = 0.9
vlmap_cfg["s2_candidate_min_pixel_distance"] = 50.0
vlmap_cfg["s2_candidate_max_queries_per_episode"] = 12
vlmap_cfg["s2_candidate_max_new_tokens"] = 128

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage6a_30_candidate_probe_epseed/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = True

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage6a_30_candidate_probe_epseed"
)
eval_cfg.eval_settings["port"] = "2364"

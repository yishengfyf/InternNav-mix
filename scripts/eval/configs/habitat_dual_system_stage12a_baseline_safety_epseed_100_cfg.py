import copy
import importlib.util
from pathlib import Path


def _load_baseline_epseed_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_baseline_fixed_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_baseline_fixed_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_baseline_epseed_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["legacy_vlmaps_experiment"] = True
vlmap_cfg["legacy_vlmaps_enable"] = True

# Stage12a baseline safety logging. This keeps InternNav behavior unchanged and
# only records collision/CF metrics plus OccMem shadow waypoint safety probes.
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["nextdit_candidate_probe_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["s2_candidate_probe_enable"] = False
vlmap_cfg["semantic_match_enable"] = True
vlmap_cfg["semantic_match_shadow_only"] = True
vlmap_cfg["semantic_stagnation_policy_enable"] = True
vlmap_cfg["semantic_stagnation_policy_shadow_only"] = True
vlmap_cfg["semantic_match_save_rgb"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["occ_memory_enable"] = True
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["occ_memory_update_every_steps"] = 1
vlmap_cfg["occ_memory_depth_sample_rate"] = 240
vlmap_cfg["occ_memory_raycast_enable"] = True
vlmap_cfg["occ_memory_raycast_stride_cells"] = 2
vlmap_cfg["occ_memory_raycast_max_points_per_update"] = 2200
vlmap_cfg["occ_memory_keyframe_every_steps"] = 10
vlmap_cfg["occ_memory_keyframe_min_distance"] = 0.50
vlmap_cfg["occ_memory_frontier_enable"] = True
vlmap_cfg["occ_memory_waypoint_probe_enable"] = True
vlmap_cfg["occ_memory_waypoint_frontier_sample_limit"] = 2000
vlmap_cfg["occ_memory_attribution_enable"] = True
vlmap_cfg["occ_memory_attribution_frontier_sample_limit"] = 5000
vlmap_cfg["occ_memory_attribution_recent_semantic_window"] = 5
vlmap_cfg["occ_memory_attribution_high_conf_recent_window"] = 5
vlmap_cfg["occ_memory_attribution_stagnation_active_window_steps"] = 20
vlmap_cfg["occ_memory_attribution_dead_zone_min_step"] = 30
vlmap_cfg["occ_memory_attribution_dead_zone_unique_threshold"] = 2
vlmap_cfg["occ_memory_attribution_dead_zone_score_threshold"] = 0.65
vlmap_cfg["occ_memory_attribution_direction_match_degrees"] = 45.0
vlmap_cfg["occ_memory_candidate_probe_enable"] = False
vlmap_cfg["occ_memory_save_bev"] = False
vlmap_cfg["occ_memory_validation_enable"] = False

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage12a_baseline_safety_epseed_100/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = False

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_stage12a_baseline_safety_epseed_100"
eval_cfg.eval_settings["port"] = "2390"

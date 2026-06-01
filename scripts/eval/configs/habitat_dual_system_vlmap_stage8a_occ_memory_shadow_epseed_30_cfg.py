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

# V8a: shadow-only construction validation for OccMem-VLN.
# This run keeps InternNav behavior unchanged and logs sparse 3D occupancy
# memory updates, ray-cast free-space evidence, frontier cells, waypoint memory
# probes, and BEV snapshots for manual inspection.
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["nextdit_candidate_probe_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["s2_candidate_probe_enable"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["semantic_match_enable"] = True
vlmap_cfg["semantic_match_shadow_only"] = True
vlmap_cfg["semantic_stagnation_policy_enable"] = True
vlmap_cfg["semantic_stagnation_policy_shadow_only"] = True
vlmap_cfg["semantic_match_save_rgb"] = False

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
vlmap_cfg["occ_memory_save_bev"] = True
vlmap_cfg["occ_memory_bev_every_updates"] = 15
vlmap_cfg["occ_memory_max_bev_snapshots"] = 10
vlmap_cfg["occ_memory_bev_crop_radius_cells"] = 150
vlmap_cfg["occ_memory_bev_cell_scale"] = 3

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage8a_30_occ_memory_shadow_epseed/"
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
    "./logs/habitat/compare_vlmap_stage8a_30_occ_memory_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2368"

import copy
import importlib.util
from pathlib import Path


def _load_stage8a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage8a_occ_memory_shadow_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage8a_occ_memory_shadow_epseed_30_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage8a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V8b: shadow-only semantic dead-zone attribution for OccMem-VLN.
# It keeps navigation unchanged and records directional frontier, waypoint,
# high-confidence semantic keyframe, and semantic dead-zone diagnostics.
vlmap_cfg["occ_memory_attribution_enable"] = True
vlmap_cfg["occ_memory_attribution_frontier_sample_limit"] = 5000
vlmap_cfg["occ_memory_attribution_recent_semantic_window"] = 5
vlmap_cfg["occ_memory_attribution_high_conf_recent_window"] = 5
vlmap_cfg["occ_memory_attribution_stagnation_active_window_steps"] = 20
vlmap_cfg["occ_memory_attribution_dead_zone_min_step"] = 30
vlmap_cfg["occ_memory_attribution_dead_zone_unique_threshold"] = 2
vlmap_cfg["occ_memory_attribution_dead_zone_score_threshold"] = 0.65
vlmap_cfg["occ_memory_attribution_direction_match_degrees"] = 45.0

# V8a already validated RGB-D geometry with PLY snapshots. Disable the heavy
# validation outputs here so 100-episode attribution runs stay compact.
vlmap_cfg["occ_memory_validation_enable"] = False
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = False
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = False
vlmap_cfg["occ_memory_validation_save_memory_ply"] = False
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = False

# Keep a small number of BEV snapshots for run sanity checks, not full geometry
# validation.
vlmap_cfg["occ_memory_save_bev"] = True
vlmap_cfg["occ_memory_bev_every_updates"] = 25
vlmap_cfg["occ_memory_max_bev_snapshots"] = 4

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage8b_100_occ_memory_attribution_shadow_epseed/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = False

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage8b_100_occ_memory_attribution_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2370"

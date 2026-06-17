import copy
import importlib.util
from pathlib import Path


def _load_stage12a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage12a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage-D BFS semantic escape shadow.
#
# This does not change navigation. It detects high-confidence confusion/stuck
# situations and asks OccMem whether a short BFS path through known free space
# can reach an instruction-relevant frontier.
vlmap_cfg["stage_d_bfs_escape_shadow_enable"] = True
vlmap_cfg["stage_d_bfs_escape_shadow_only"] = True
vlmap_cfg["stage_d_bfs_escape_min_step"] = 30
vlmap_cfg["stage_d_bfs_escape_compass_window_steps"] = 20
vlmap_cfg["stage_d_bfs_escape_compass_reversal_threshold"] = 0.07
# Match the online Stage14a-v2 direction feature: sign changes capture
# left/right oscillation. The event log also records angle-based reversals.
vlmap_cfg["stage_d_bfs_escape_compass_reversal_metric"] = "sign"
vlmap_cfg["stage_d_bfs_escape_consecutive_occupied_min"] = 3
vlmap_cfg["stage_d_bfs_escape_use_compass_reversal"] = True
vlmap_cfg["stage_d_bfs_escape_use_consecutive_occupied"] = True
vlmap_cfg["stage_d_bfs_escape_use_semantic_stagnation"] = True
vlmap_cfg["stage_d_bfs_escape_max_action_steps"] = 8
vlmap_cfg["stage_d_bfs_escape_frontier_sample_limit"] = 5000
vlmap_cfg["stage_d_bfs_escape_require_instruction_relevant"] = True
vlmap_cfg["stage_d_bfs_escape_allow_fallback_target_frontier"] = False
vlmap_cfg["stage_d_bfs_escape_max_events_per_episode"] = -1
vlmap_cfg["stage_d_bfs_escape_log_non_trigger_steps"] = False

# Keep Stage15 in shadow mode only so Stage-D can reuse its sustained occupied
# waypoint counter. No action or pixel-goal is changed.
vlmap_cfg["stage15_repair_shadow_enable"] = True
vlmap_cfg["stage15_repair_active"] = False
vlmap_cfg["stage15_repair_backtrack_max_steps"] = 20

# Enable semantic frontier construction inside OccMem without running the old
# language candidate-selection interface.
vlmap_cfg["occ_memory_candidate_probe_enable"] = False
vlmap_cfg["occ_memory_candidate_probe_semantic_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_goal_progress_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_target_frontier_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_semantic_min_score"] = 0.20
vlmap_cfg["occ_memory_candidate_probe_semantic_high_conf_only"] = False
vlmap_cfg["occ_memory_candidate_probe_semantic_frontier_min_relevance"] = 0.15
vlmap_cfg["occ_memory_candidate_probe_frontier_sample_limit"] = 5000
vlmap_cfg["occ_memory_candidate_probe_min_distance_m"] = 0.50
vlmap_cfg["occ_memory_candidate_probe_max_distance_m"] = 4.0
vlmap_cfg["occ_memory_candidate_probe_save_bev"] = False

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage_d_bfs_escape_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2411"

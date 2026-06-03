import copy
import importlib.util
from pathlib import Path


def _load_stage8b_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage8b_occ_memory_attribution_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage8b_occ_memory_attribution_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage8b_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V10a: shadow-only Sparse 3D Query Candidate Interface.
# OccMem proposes structured A/B/C/D sparse-memory candidates and optionally
# asks S2 to choose one. The selected candidate is logged only; navigation still
# executes the original InternNav/VLMap path.
vlmap_cfg["occ_memory_candidate_probe_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_max_candidates"] = 4
vlmap_cfg["occ_memory_candidate_probe_max_events_per_episode"] = 12
vlmap_cfg["occ_memory_candidate_probe_frontier_sample_limit"] = 5000
vlmap_cfg["occ_memory_candidate_probe_free_sample_limit"] = 5000
vlmap_cfg["occ_memory_candidate_probe_min_distance_m"] = 0.75
vlmap_cfg["occ_memory_candidate_probe_max_distance_m"] = 4.0
vlmap_cfg["occ_memory_candidate_probe_min_separation_m"] = 0.50
vlmap_cfg["occ_memory_candidate_probe_exclude_back_frontier"] = True
vlmap_cfg["occ_memory_candidate_probe_save_bev"] = True
vlmap_cfg["occ_memory_candidate_probe_max_bev_snapshots"] = 12

# Keep this as a bounded shadow probe. It tests whether S2 can solve the
# structured candidate-selection task, without executing the chosen candidate.
vlmap_cfg["occ_memory_candidate_selection_enable"] = True
vlmap_cfg["occ_memory_candidate_selection_max_queries_per_episode"] = 2
vlmap_cfg["occ_memory_candidate_selection_max_new_tokens"] = 32

# Close V9 prompt-hint active path for this experiment.
vlmap_cfg["occ_memory_guidance_enable"] = False
vlmap_cfg["occ_memory_guidance_shadow_only"] = True
vlmap_cfg["occ_memory_guidance_counterfactual_enable"] = False
vlmap_cfg["occ_memory_guidance_requery_on_trigger"] = False

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage10a_100_occ_memory_query_candidates_shadow_epseed/"
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
    "./logs/habitat/compare_vlmap_stage10a_100_occ_memory_query_candidates_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2378"

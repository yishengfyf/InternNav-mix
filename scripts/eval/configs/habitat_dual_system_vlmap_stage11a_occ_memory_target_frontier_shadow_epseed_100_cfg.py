import copy
import importlib.util
from pathlib import Path


def _load_stage10a3_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage10a3_occ_memory_goal_progress_candidates_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage10a3_occ_memory_goal_progress_candidates_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage10a3_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V11a: target-conditioned frontier reasoning shadow.
# It keeps V10a3 goal-progress bookkeeping, but scores frontier/open-space
# candidates by whether their local sparse occupancy pattern looks like a
# useful transition toward the next unseen landmark. Navigation is unchanged.
vlmap_cfg["occ_memory_candidate_probe_target_frontier_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_target_frontier_score_weight"] = 1.25
vlmap_cfg["occ_memory_candidate_probe_target_frontier_cluster_radius_cells"] = 8
vlmap_cfg["occ_memory_candidate_probe_target_frontier_cluster_norm"] = 18
vlmap_cfg["occ_memory_candidate_probe_target_frontier_doorway_threshold"] = 0.35
vlmap_cfg["occ_memory_candidate_probe_target_frontier_candidate_threshold"] = 0.35
vlmap_cfg["occ_memory_candidate_probe_target_frontier_intent_max_deviation_deg"] = 75.0
vlmap_cfg["occ_memory_candidate_probe_target_frontier_intent_penalty_weight"] = 0.50

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage11a_100_occ_memory_target_frontier_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2381"

import copy
import importlib.util
from pathlib import Path


def _load_stage10a2_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage10a2_occ_memory_semantic_query_candidates_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage10a2_occ_memory_semantic_query_candidates_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage10a2_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V10a3: goal-progress aware sparse 3D candidate shadow.
# V10a2 bound candidates to any instruction-overlapping historical semantic
# node. V10a3 additionally estimates the ordered landmark sequence and favors
# candidates tied to the next unseen landmark while penalizing completed or
# recently repeated semantic regions. Navigation remains unchanged.
vlmap_cfg["occ_memory_candidate_probe_goal_progress_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_goal_progress_next_weight"] = 1.35
vlmap_cfg["occ_memory_candidate_probe_goal_progress_completed_penalty"] = 0.95
vlmap_cfg["occ_memory_candidate_probe_goal_progress_repeated_penalty"] = 0.65
vlmap_cfg["occ_memory_candidate_probe_goal_progress_seen_score_threshold"] = 0.25
vlmap_cfg["occ_memory_candidate_probe_goal_progress_high_conf_bonus"] = 0.30
vlmap_cfg["occ_memory_candidate_probe_goal_progress_unknown_target_bonus"] = 0.25

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage10a3_100_occ_memory_goal_progress_candidates_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage10a3_100_occ_memory_goal_progress_candidates_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2380"

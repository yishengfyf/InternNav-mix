import copy
import importlib.util
from pathlib import Path


def _load_stage12b_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage12b_nextdit_occ_memory_traj_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage12b_nextdit_occ_memory_traj_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage12b_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage12c: conservative active rerank over NextDiT raw trajectory samples.
# It only intervenes when the actually executed averaged trajectory hits known
# OccMem occupied cells, and it can only choose an occupied-free raw candidate
# whose endpoint direction stays within 45 degrees of the averaged trajectory.
vlmap_cfg["nextdit_candidate_probe_enable"] = True
vlmap_cfg["nextdit_candidate_max_candidates"] = 32
vlmap_cfg["nextdit_candidate_max_events_per_episode"] = 12
vlmap_cfg["nextdit_candidate_active_enable"] = True
vlmap_cfg["nextdit_candidate_active_strategy"] = "occ_memory_conservative"
vlmap_cfg["nextdit_candidate_active_max_interventions_per_episode"] = 2
vlmap_cfg["nextdit_candidate_active_require_current_reject"] = False
vlmap_cfg["nextdit_candidate_active_occ_current_min_occupied_hits"] = 1
vlmap_cfg["nextdit_candidate_active_occ_max_direction_deviation_deg"] = 45.0
vlmap_cfg["nextdit_candidate_active_occ_unknown_weight"] = 0.15
vlmap_cfg["nextdit_candidate_active_occ_direction_weight"] = 0.30
vlmap_cfg["nextdit_candidate_active_occ_forward_progress_weight"] = 0.05
vlmap_cfg["nextdit_candidate_active_occ_require_action_diff"] = True
vlmap_cfg["nextdit_candidate_active_occ_reject_all_unknown"] = True
vlmap_cfg["nextdit_candidate_active_occ_require_vlmap_nonreject"] = False
vlmap_cfg["nextdit_candidate_occ_memory_score_enable"] = True
vlmap_cfg["nextdit_candidate_occ_memory_score_max_points"] = 33

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage12c_occ_memory_conservative_traj_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage12c_occ_memory_conservative_traj_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2395"

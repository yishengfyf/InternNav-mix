import copy
import importlib.util
from pathlib import Path


def _load_stage12a_100_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage12a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage12b verifies the key engineering premise for trajectory-level safety:
# whether raw NextDiT samples can be intercepted and scored one by one with
# persistent OccMem occupancy. This is still shadow-only and executes the
# original averaged InternNav action sequence.
vlmap_cfg["traj_validation_enable"] = True
vlmap_cfg["traj_validation_shadow_only"] = True
vlmap_cfg["traj_validation_horizon"] = 4
vlmap_cfg["traj_validation_block_threshold"] = 1
vlmap_cfg["traj_validation_max_rejects_per_episode"] = 2
vlmap_cfg["traj_validation_cooldown_steps"] = 20

vlmap_cfg["nextdit_candidate_probe_enable"] = True
vlmap_cfg["nextdit_candidate_max_candidates"] = 32
vlmap_cfg["nextdit_candidate_max_events_per_episode"] = 12
vlmap_cfg["nextdit_candidate_min_endpoint_grid_distance"] = 4.0
vlmap_cfg["nextdit_candidate_action_horizon"] = 4
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["nextdit_candidate_occ_memory_score_enable"] = True
vlmap_cfg["nextdit_candidate_occ_memory_score_max_points"] = 33

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage12b_nextdit_occ_memory_traj_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage12b_nextdit_occ_memory_traj_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2393"

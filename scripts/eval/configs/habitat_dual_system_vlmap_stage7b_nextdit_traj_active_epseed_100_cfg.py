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
vlmap_cfg["legacy_vlmaps_experiment"] = True
vlmap_cfg["legacy_vlmaps_enable"] = True

# V7b-small: conservative active reranking over NextDiT raw trajectory samples.
# The original averaged trajectory is still used unless VLMap trajectory rollout
# marks it as would_reject. In that case, select a non-reject raw sample if one
# exists. This keeps intervention narrow and avoids reranking noise when the
# averaged trajectory is already geometrically valid.
vlmap_cfg["enable"] = True
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["traj_validation_enable"] = True
vlmap_cfg["traj_validation_shadow_only"] = True
vlmap_cfg["traj_validation_horizon"] = 4
vlmap_cfg["traj_validation_block_threshold"] = 1
vlmap_cfg["traj_validation_max_rejects_per_episode"] = 2
vlmap_cfg["traj_validation_cooldown_steps"] = 20

vlmap_cfg["nextdit_candidate_probe_enable"] = False
vlmap_cfg["nextdit_candidate_max_candidates"] = 32
vlmap_cfg["nextdit_candidate_max_events_per_episode"] = 0
vlmap_cfg["nextdit_candidate_min_endpoint_grid_distance"] = 4.0
vlmap_cfg["nextdit_candidate_action_horizon"] = 4
vlmap_cfg["nextdit_candidate_active_enable"] = True
vlmap_cfg["nextdit_candidate_active_max_interventions_per_episode"] = 2
vlmap_cfg["nextdit_candidate_active_require_current_reject"] = True

vlmap_cfg["s2_candidate_probe_enable"] = False
vlmap_cfg["semantic_match_enable"] = False

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage7b_100_nextdit_traj_active_epseed/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = True

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage7b_100_nextdit_traj_active_epseed"
)
eval_cfg.eval_settings["port"] = "2367"

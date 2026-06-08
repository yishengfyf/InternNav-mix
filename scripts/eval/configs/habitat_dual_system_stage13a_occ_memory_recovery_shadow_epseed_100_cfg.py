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

# Stage13a: event-triggered recovery shadow.
# This does not change navigation. It only logs online stuck/collision recovery
# gates based on persistent OccMem map growth, pose displacement, and Habitat
# collision deltas. The goal is to verify whether ep160-like stuck cases are
# detectable without triggering on normal successful episodes.
vlmap_cfg["nextdit_candidate_probe_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = True
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
vlmap_cfg["occ_memory_recovery_min_step"] = 30
vlmap_cfg["occ_memory_recovery_occupied_stagnation_window_steps"] = 20
vlmap_cfg["occ_memory_recovery_total_stagnation_window_steps"] = 20
vlmap_cfg["occ_memory_recovery_displacement_window_steps"] = 20
vlmap_cfg["occ_memory_recovery_low_displacement_threshold_m"] = 0.35
vlmap_cfg["occ_memory_recovery_require_low_displacement_for_map_stagnation"] = True
vlmap_cfg["occ_memory_recovery_collision_trigger_enable"] = True
vlmap_cfg["occ_memory_recovery_total_map_stagnation_trigger_enable"] = False
vlmap_cfg["occ_memory_recovery_log_every_step"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage13a_occ_memory_recovery_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage13a_occ_memory_recovery_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2397"

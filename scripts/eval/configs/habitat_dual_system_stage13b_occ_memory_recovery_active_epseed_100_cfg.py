import copy
import importlib.util
from pathlib import Path


def _load_stage13a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage13a_occ_memory_recovery_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage13a_occ_memory_recovery_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage13a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage13b: event-triggered OccMem recovery active.
# Unlike Stage12c, this is not always-on trajectory reranking. It only queues a
# short deterministic escape when the Stage13 map-stagnation gate fires, while
# suppressing arrival-like free-space stagnation (ep63-style near-goal stop).
vlmap_cfg["occ_memory_recovery_enable"] = True
vlmap_cfg["occ_memory_recovery_shadow_only"] = False
vlmap_cfg["occ_memory_recovery_max_interventions_per_episode"] = 1
vlmap_cfg["occ_memory_recovery_active_use_map_stagnation"] = True
vlmap_cfg["occ_memory_recovery_active_use_collision_trigger"] = False
vlmap_cfg["occ_memory_recovery_arrival_like_protection_enable"] = True
vlmap_cfg["occ_memory_recovery_arrival_like_radius_cells"] = 8
vlmap_cfg["occ_memory_recovery_arrival_like_min_free_ratio"] = 0.35
vlmap_cfg["occ_memory_recovery_arrival_like_max_occupied_ratio"] = 0.04
vlmap_cfg["occ_memory_recovery_escape_probe_distance_m"] = 0.75
vlmap_cfg["occ_memory_recovery_escape_candidate_angles_deg"] = [45.0, -45.0, 60.0, -60.0]
vlmap_cfg["occ_memory_recovery_escape_max_turn_steps"] = 3
vlmap_cfg["occ_memory_recovery_escape_forward_steps"] = 1
vlmap_cfg["occ_memory_recovery_escape_allow_forward_only_if_free"] = True
vlmap_cfg["occ_memory_recovery_escape_clear_goal"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage13b_occ_memory_recovery_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage13b_occ_memory_recovery_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2399"

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

# V9 Level 1: active prompt-only OccMem guidance.
# The policy does not override actions or trajectories. When OccMem detects a
# semantic dead-zone with an available frontier, it clears the current waypoint
# and immediately asks S2 again with a spatial memory hint.
vlmap_cfg["occ_memory_guidance_enable"] = True
vlmap_cfg["occ_memory_guidance_shadow_only"] = False
vlmap_cfg["occ_memory_guidance_min_dead_zone_score"] = 0.65
vlmap_cfg["occ_memory_guidance_require_no_recent_high_conf"] = True
vlmap_cfg["occ_memory_guidance_min_frontier_count"] = 1
vlmap_cfg["occ_memory_guidance_cooldown_steps"] = 24
vlmap_cfg["occ_memory_guidance_max_hints_per_episode"] = 2
vlmap_cfg["occ_memory_guidance_requery_on_trigger"] = True
vlmap_cfg["occ_memory_guidance_prompt_hint"] = (
    "Navigation memory hint: recent observations are semantically repetitive "
    "and no reliable target landmark has been observed. The sparse 3D memory "
    "suggests the most promising unexplored frontier is {direction_text}. "
    "Use this only if it matches the instruction: choose a visible waypoint "
    "toward open floor, a doorway, or an unexplored opening in that direction, "
    "and avoid repeating the same semantic area. Output only the next waypoint "
    "coordinates or STOP."
)

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage9_30_occ_memory_prompt_hint_active_epseed/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = False

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage9_30_occ_memory_prompt_hint_active_epseed"
)
eval_cfg.eval_settings["port"] = "2372"

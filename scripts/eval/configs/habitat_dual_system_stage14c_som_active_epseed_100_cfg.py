import copy
import importlib.util
from pathlib import Path


def _load_stage14b_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage14b_som_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage14b_som_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage14b_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage14c: conservative active SoM replacement.
#
# Reuse Stage14b's overlay and extra S2 forward, but when the overlay goal
# changes substantially under an unsafe OccMem signal, execute the overlay
# pixel goal instead of the base pixel goal.
vlmap_cfg["som_counterfactual_enable"] = True
vlmap_cfg["som_counterfactual_shadow_only"] = False
vlmap_cfg["som_counterfactual_max_queries_per_episode"] = 30
vlmap_cfg["som_counterfactual_overlay_type"] = "frontier_direction_unsafe_goal_v1"
vlmap_cfg["som_counterfactual_pixel_shift_threshold"] = 40.0
vlmap_cfg["som_counterfactual_min_unsafe_signal"] = 0.50
vlmap_cfg["som_counterfactual_active_min_unsafe_signal"] = 0.50
vlmap_cfg["som_counterfactual_active_max_per_episode"] = 3
vlmap_cfg["som_counterfactual_draw_frontier"] = True
vlmap_cfg["som_counterfactual_draw_unsafe_goal"] = True
vlmap_cfg["som_counterfactual_draw_base_goal"] = False
vlmap_cfg["occ_memory_guidance_counterfactual_enable"] = False

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14c_som_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14c_som_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2401"

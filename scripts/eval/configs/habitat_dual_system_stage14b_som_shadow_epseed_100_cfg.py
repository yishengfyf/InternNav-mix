import copy
import importlib.util
from pathlib import Path


def _load_stage14a_v2_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage14a_v2_direction_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage14a_v2_direction_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage14a_v2_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage14b: visual Set-of-Marks counterfactual shadow.
#
# This is independent from failure prediction. It runs an additional S2 forward
# with a visual overlay on the current RGB frame and still executes the original
# base S2 pixel goal.
vlmap_cfg["som_counterfactual_enable"] = True
vlmap_cfg["som_counterfactual_shadow_only"] = True
vlmap_cfg["som_counterfactual_max_queries_per_episode"] = 30
vlmap_cfg["som_counterfactual_overlay_type"] = "frontier_direction_unsafe_goal_v1"
vlmap_cfg["som_counterfactual_pixel_shift_threshold"] = 40.0
vlmap_cfg["som_counterfactual_min_unsafe_signal"] = 0.50
vlmap_cfg["som_counterfactual_draw_frontier"] = True
vlmap_cfg["som_counterfactual_draw_unsafe_goal"] = True
vlmap_cfg["som_counterfactual_draw_base_goal"] = False
vlmap_cfg["som_counterfactual_frontier_alpha"] = 80
vlmap_cfg["som_counterfactual_unsafe_alpha"] = 120
vlmap_cfg["som_counterfactual_max_new_tokens"] = 128

# Keep legacy prompt counterfactual disabled so any S2 change is attributable to
# the visual overlay, not a text hint.
vlmap_cfg["occ_memory_guidance_counterfactual_enable"] = False

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14b_som_counterfactual_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14b_som_counterfactual_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2401"

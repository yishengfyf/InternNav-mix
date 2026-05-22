import copy
import importlib.util
from pathlib import Path


def _load_stage5b_shadow_30_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage5b_semantic_rank_shadow_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage5b_shadow_30_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage5b_shadow_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V5d shadow records semantic stagnation re-observe triggers without changing
# actions. The active version uses the same detector but clears the current S2
# waypoint instead of forcing any turn or local action.
vlmap_cfg["semantic_confidence_policy_enable"] = False
vlmap_cfg["semantic_stagnation_policy_enable"] = True
vlmap_cfg["semantic_stagnation_policy_shadow_only"] = True
vlmap_cfg["semantic_stagnation_window"] = 5
vlmap_cfg["semantic_stagnation_unique_threshold"] = 2
vlmap_cfg["semantic_stagnation_min_step"] = 30
vlmap_cfg["semantic_stagnation_min_events"] = 8
vlmap_cfg["semantic_stagnation_require_no_high_conf"] = True
vlmap_cfg["semantic_stagnation_max_per_episode"] = 1

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage5d_30_stagnation_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage5d_30_stagnation_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2358"

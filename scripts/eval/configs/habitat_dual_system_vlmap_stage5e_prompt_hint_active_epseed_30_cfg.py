import copy
import importlib.util
from pathlib import Path


def _load_stage5d_shadow_30_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage5d_stagnation_shadow_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage5d_shadow_30_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage5d_shadow_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V5e keeps the V5d stagnation detector but replaces hard re-observe with a
# delayed prompt hint injected at the next natural S2 query.
vlmap_cfg["semantic_match_shadow_only"] = False
vlmap_cfg["semantic_stagnation_policy_shadow_only"] = False
vlmap_cfg["semantic_stagnation_intervention"] = "prompt_hint"
vlmap_cfg["semantic_stagnation_min_step"] = 50
vlmap_cfg["semantic_stagnation_prompt_hint"] = (
    "Navigation note: your recent observations look similar. "
    "Re-check the instruction and choose a waypoint that makes progress "
    "toward the next landmark. Output only the next waypoint coordinates or STOP."
)
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage5e_30_prompt_hint_active_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage5e_30_prompt_hint_active_epseed"
)
eval_cfg.eval_settings["port"] = "2362"

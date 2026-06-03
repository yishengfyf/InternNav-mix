import copy
import importlib.util
from pathlib import Path


def _load_stage9_30_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage9_occ_memory_prompt_hint_active_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage9_occ_memory_prompt_hint_active_epseed_30_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage9_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Same active policy as V9 30ep, expanded to 100 episodes to test whether the
# 30ep negative result is sampling noise.
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage9_100_occ_memory_prompt_hint_active_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage9_100_occ_memory_prompt_hint_active_epseed"
)
eval_cfg.eval_settings["port"] = "2373"

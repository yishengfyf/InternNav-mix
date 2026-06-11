import copy
import importlib.util
from pathlib import Path


def _load_stage12a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage12a_baseline_safety_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage12a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# 200-episode baseline with collision/CF metrics.
# Purpose: establish a more stable reference for Stage15b comparison.
# 200ep reduces SR variance from ±0.045 to ±0.032 (SE = sqrt(p(1-p)/n)).
# More importantly, it gives a better estimate of collision_sum distribution
# across diverse scene types (ep160-style stuck cases diluted, other cases visible).
# Note: 200ep still cannot detect a +0.01 SR difference — it is the collision/CF
# axis that benefits most from larger N.
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_baseline_safety_epseed_200/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_baseline_safety_epseed_200"
)
eval_cfg.eval_settings["port"] = "2410"

import copy
import importlib.util
from pathlib import Path


def _load_stage5d_active_30_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage5d_stagnation_active_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage5d_active_30_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage5d_active_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage5d_100_stagnation_active_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage5d_100_stagnation_active_epseed"
)
eval_cfg.eval_settings["port"] = "2361"

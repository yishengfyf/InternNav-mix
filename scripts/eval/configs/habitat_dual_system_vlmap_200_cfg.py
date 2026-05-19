import copy
import importlib.util
from pathlib import Path


def _load_base_cfg():
    base_path = Path(__file__).with_name("habitat_dual_system_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_cfg_base", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_base_cfg())

eval_cfg.agent.model_settings["vlmap_safety"]["enable"] = True
eval_cfg.agent.model_settings["vlmap_safety"]["debug_dir"] = (
    "./logs/habitat/compare_vlmap_200_softguard/vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_200_softguard"

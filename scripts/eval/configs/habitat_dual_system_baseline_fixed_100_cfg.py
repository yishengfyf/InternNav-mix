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

# Fixed-prompt baseline for fair comparison with VLMap stage2 guarded runs.
eval_cfg.agent.model_settings["eval_random_seed"] = 0
eval_cfg.agent.model_settings["s2_prompt_conjunction_index"] = 0

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["enable"] = False
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["debug"] = False
vlmap_cfg["verbose"] = False

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_baseline_100_fixed_prompt"
eval_cfg.eval_settings["port"] = "2344"

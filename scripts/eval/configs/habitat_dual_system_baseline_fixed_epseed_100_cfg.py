import copy
import importlib.util
from pathlib import Path


def _load_fixed_baseline_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_baseline_fixed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_baseline_fixed_100_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_fixed_baseline_cfg())

# Reset RNG at the start of every episode so baseline/VLMap comparisons are not
# affected by one branch consuming extra random numbers in earlier episodes.
eval_cfg.agent.model_settings["eval_seed_per_episode"] = True
eval_cfg.agent.model_settings["eval_episode_seed_mode"] = "episode_index"

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_baseline_100_fixed_prompt_epseed"
eval_cfg.eval_settings["port"] = "2347"

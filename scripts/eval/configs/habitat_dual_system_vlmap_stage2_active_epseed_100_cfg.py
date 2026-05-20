import copy
import importlib.util
from pathlib import Path


def _load_stage2_active_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_vlmap_stage2_active_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_vlmap_stage2_active_100_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage2_active_cfg())

# Keep every episode's prompt/random state comparable with the fixed baseline.
eval_cfg.agent.model_settings["eval_seed_per_episode"] = True
eval_cfg.agent.model_settings["eval_episode_seed_mode"] = "episode_index"

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage2_100_active_requery_v2_guarded_epseed/vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_stage2_100_active_requery_v2_guarded_epseed"
eval_cfg.eval_settings["port"] = "2348"

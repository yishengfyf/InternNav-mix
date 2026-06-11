import copy
import importlib.util
from pathlib import Path


def _load_stage15a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage15a_repair_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage15a_repair_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage15a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage15b broad diagnostic: cumulative occupied count catches more failing
# episodes than strict, with known higher regression risk.
vlmap_cfg["stage15_repair_active"] = True
vlmap_cfg["stage15_repair_gate_mode"] = "cumulative"
vlmap_cfg["stage15_repair_gate_min_count"] = 3
vlmap_cfg["stage15_repair_active_max_per_episode"] = 5

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage15b_broad_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage15b_broad_epseed_100"
)
eval_cfg.eval_settings["port"] = "2409"

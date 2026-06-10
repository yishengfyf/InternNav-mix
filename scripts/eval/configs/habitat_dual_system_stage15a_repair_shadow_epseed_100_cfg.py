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

# Stage15a: OccMem geometric waypoint repair shadow.
#
# No navigation behavior is changed. This only validates the inverse projection
# pixel -> grid -> pixel and logs a deterministic occupied-goal repair candidate
# by backing the waypoint up along the original bearing to the nearest free cell.
vlmap_cfg["stage15_repair_shadow_enable"] = True
vlmap_cfg["stage15_repair_active"] = False
vlmap_cfg["stage15_repair_backtrack_max_steps"] = 20
vlmap_cfg["occ_memory_waypoint_probe_enable"] = True
vlmap_cfg["waypoint_probe_enable"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage15a_repair_shadow_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage15a_repair_shadow_epseed_100"
)
eval_cfg.eval_settings["port"] = "2405"

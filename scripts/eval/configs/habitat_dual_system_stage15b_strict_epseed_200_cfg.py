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

# Stage15b strict: deterministic intent-preserving geometric waypoint repair.
# Gate: consecutive_occupied >= 3 (targets ep160/ep326 class stuck cases, zero
# false positives in Stage15a shadow data). Per-episode cap = 5.
# Target axis: collision_sum / CF-SR. SR/SPL held as non-regression constraint.
#
# !! REQUIRES Stage15b active code in sparse_occ_memory.py and evaluator !!
# (consecutive_count tracking + output_ids token rewrite via Stage14c path)
vlmap_cfg["stage15_repair_active"] = True
vlmap_cfg["stage15_repair_gate_mode"] = "consecutive"   # "consecutive" | "cumulative" | "all"
vlmap_cfg["stage15_repair_gate_min_count"] = 3
vlmap_cfg["stage15_repair_active_max_per_episode"] = 5

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage15b_strict_epseed_200/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage15b_strict_epseed_200"
)
eval_cfg.eval_settings["port"] = "2412"

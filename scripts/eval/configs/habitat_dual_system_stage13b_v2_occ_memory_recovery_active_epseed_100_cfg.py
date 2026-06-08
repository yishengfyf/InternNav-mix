import copy
import importlib.util
from pathlib import Path


def _load_stage13b_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage13b_occ_memory_recovery_active_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage13b_occ_memory_recovery_active_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage13b_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage13b-v2: relaxed escape probe distance + wider angle coverage.
#
# Motivation (from Stage13a shadow analysis):
#   - ep160 is stuck with +45deg direction showing occ=3 free=9 (60% free ratio).
#   - Stage13b uses probe_distance=0.75m -> occ=3 triggers no-forward.
#   - Reducing probe_distance to 0.25m (= 1 Habitat forward step) checks only
#     the immediate 1-step cell; the occ hits may be further away.
#   - Expanding candidate angles [45,-45,90,-90,135,-135] covers more directions,
#     increasing the chance of finding a clear short path.
#   - All other constraints (arrival-like protection, max_interventions=1,
#     collision_trigger disabled) remain unchanged.
vlmap_cfg["occ_memory_recovery_escape_probe_distance_m"] = 0.25
vlmap_cfg["occ_memory_recovery_escape_candidate_angles_deg"] = [
    45.0, -45.0, 90.0, -90.0, 135.0, -135.0
]

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage13b_v2_occ_memory_recovery_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage13b_v2_occ_memory_recovery_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2400"

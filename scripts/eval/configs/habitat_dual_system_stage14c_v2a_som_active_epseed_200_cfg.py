import copy
import importlib.util
from pathlib import Path


def _load_stage14c_v2a_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage14c_v2a_som_active_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage14c_v2a_som_active_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage14c_v2a_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# 200-episode validation of Stage14c-v2a (occupied-only gate SoM active).
# Purpose: confirm that the occupied gate eliminates regressions at larger scale,
# and get a more stable estimate of collision/CF-SR alongside Stage15b comparison.
# Note: a +0.01 SR difference requires ~3000ep for significance; 200ep is used
# to check regression stability and collision-axis effects, not to claim SR gain.
vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14c_v2a_som_active_epseed_200/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 200
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14c_v2a_som_active_epseed_200"
)
eval_cfg.eval_settings["port"] = "2411"

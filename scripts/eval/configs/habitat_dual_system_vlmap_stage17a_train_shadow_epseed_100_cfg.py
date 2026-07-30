import copy
import importlib.util
from pathlib import Path


def _load_stage11a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage11a_occ_memory_target_frontier_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage11a_occ_memory_target_frontier_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage11a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage17a train split shadow collection. It keeps the Stage11a candidate
# generator unchanged and only switches the Habitat dataset split plus log path.
eval_cfg.env.env_settings["config_path"] = "scripts/eval/configs/vln_r2r_train.yaml"
eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 100
eval_cfg.env.env_settings["episode_ids"] = None

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage17a_train_100_occ_memory_target_frontier_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage17a_train_100_occ_memory_target_frontier_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2391"

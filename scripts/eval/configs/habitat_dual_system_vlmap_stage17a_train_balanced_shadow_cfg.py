import copy
import importlib.util
import json
import os
from pathlib import Path


def _load_stage17a_train_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage17a_train_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage17a_train_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def _load_episode_ids():
    path = Path(
        os.environ.get(
            "STAGE17_EPISODE_IDS",
            "data/stage17/train_balanced_200_episode_ids.json",
        )
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Stage17 balanced episode id file not found: {path}. "
            "Run scripts/eval/select_balanced_r2r_episodes.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


eval_cfg = copy.deepcopy(_load_stage17a_train_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
episode_ids = _load_episode_ids()

eval_cfg.env.env_settings["episode_ids"] = episode_ids
eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = None

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage17a_train_balanced_occ_memory_target_frontier_shadow/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage17a_train_balanced_occ_memory_target_frontier_shadow"
)
eval_cfg.eval_settings["port"] = "2392"

"""Stage21 r3 shadow config for deterministic replay of three failure episodes."""

import copy
import importlib.util
from pathlib import Path


def _load_stage21_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage21a_train_recovery_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage21_train_shadow_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage21_cfg())
model_cfg = eval_cfg.agent.model_settings

# Preserve the actual per-rank/per-index seeds logged by the original 40ep run.
# This makes a 1GPU targeted replay comparable to the three original failures.
model_cfg["eval_random_seed"] = 0
model_cfg["eval_seed_per_episode"] = True
model_cfg["eval_episode_seed_mode"] = "episode_index"
model_cfg["eval_episode_seed_overrides"] = {
    "5q7pvUzZiYa/9357": 200001,
    "SN83YJsR3w2/5982": 100006,
    "V2XKFyX4ASd/775": 7,
}
model_cfg["vlmap_safety"]["stuck_snapshot_force_episode_keys"] = list(
    model_cfg["eval_episode_seed_overrides"]
)

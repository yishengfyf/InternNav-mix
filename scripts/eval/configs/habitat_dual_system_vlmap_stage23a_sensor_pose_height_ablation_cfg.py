"""Stage23A current versus oracle-height versus full sensor pose audit."""

import copy
import importlib.util
from pathlib import Path


def _load_sensor_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage23a_sensor_pose_occ_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage23a_sensor_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_sensor_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["occ_memory_validation_oracle_pose_enable"] = True

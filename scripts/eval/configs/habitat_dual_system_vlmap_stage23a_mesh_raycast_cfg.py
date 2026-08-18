"""Stage23A independent Habitat collision-mesh raycast audit."""

import copy
import importlib.util
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23a_sensor_pose_occ_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage23a_sensor_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["occ_memory_validation_mesh_raycast_enable"] = True
vlmap_cfg["occ_memory_validation_mesh_raycast_max_rays"] = 32

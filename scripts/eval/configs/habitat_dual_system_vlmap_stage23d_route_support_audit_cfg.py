"""Stage23D SparseOcc versus route-support connectivity ablation."""

import copy
import importlib.util
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23b_navmesh_traversability_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage23b_navmesh_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["occ_memory_validation_route_support_audit_enable"] = True

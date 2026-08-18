"""Stage23B audit-only SparseOcc traversability versus Habitat navmesh."""

import copy
import importlib.util
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23a_mesh_raycast_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage23a_mesh_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["occ_memory_validation_navmesh_traversability_enable"] = True
vlmap_cfg["occ_memory_validation_navmesh_max_cells"] = 1200
vlmap_cfg["occ_memory_validation_navmesh_max_pairs"] = 12
vlmap_cfg["occ_memory_validation_navmesh_agent_radius_m"] = 0.18
vlmap_cfg["occ_memory_validation_navmesh_clearance_ablation_enable"] = True
vlmap_cfg["occ_memory_validation_navmesh_clearance_height_max_m"] = 1.50

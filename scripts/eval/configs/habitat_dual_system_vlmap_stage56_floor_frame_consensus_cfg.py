"""Stage56 read-only floor-relative independent-frame OCC audit."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23b_navmesh_traversability_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage23b_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["occ_memory_frame_observation_mask_audit_enable"] = True
vlmap_cfg["occ_memory_validation_floor_frame_consensus_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_floor_frame_consensus_audit_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False
eval_cfg.eval_settings["port"] = os.environ.get("STAGE56_EVAL_PORT", "3560")

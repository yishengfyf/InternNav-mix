"""Stage57 read-only local elevation-support graph audit."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage56_floor_frame_consensus_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage56_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["occ_memory_validation_local_elevation_support_enable"] = True
vlmap_cfg["occ_memory_validation_local_elevation_support_min_frames"] = 2
vlmap_cfg["occ_memory_validation_local_elevation_support_max_step_m"] = 0.20
vlmap_cfg["s2_loop_path_reobserve_local_elevation_support_audit_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_stage57_planned_prefix_audit_enable"] = True
vlmap_cfg["s2_loop_stage57_planned_prefix_max_m"] = 1.0
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False
eval_cfg.eval_settings["port"] = os.environ.get("STAGE57_EVAL_PORT", "3561")

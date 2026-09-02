"""Stage70: mainline boundary smoke configuration.

This config is the source-controlled default for the fixed wrapper.  It uses
SparseOcc/recovery diagnostics only; every legacy VLMaps path is explicitly
disabled and remains shadow/audit-only until a separate mainline trajectory
preflight is available.
"""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage66_native_visual_audit_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage66_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["legacy_vlmaps_experiment"] = False
vlmap_cfg["legacy_vlmaps_enable"] = False
vlmap_cfg["legacy_vlmaps_waypoint_enable"] = False
vlmap_cfg["legacy_vlmaps_semantic_enable"] = False
vlmap_cfg["enable"] = False
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["stage65_native_pixel_execution_enable"] = False

run_root = os.environ.get(
    "STAGE70_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage70_mainline_boundary_smoke")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE70_EVAL_PORT", "3700")

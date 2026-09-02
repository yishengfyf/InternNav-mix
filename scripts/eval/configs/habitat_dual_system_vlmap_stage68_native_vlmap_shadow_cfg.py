"""Stage68: native recovery with VLMap waypoint advisor enabled in shadow mode."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage67_native_strict_shadow_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage67_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Exercise the existing VLMap waypoint projection as an advisory shadow.  The
# native recovery gate still requires SparseOcc goal_state=free, complete
# trajectory validation, footprint/headroom checks and no active intervention.
vlmap_cfg["waypoint_check_enable"] = True
vlmap_cfg["waypoint_shadow_only"] = True
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["pixel_translation_active_enable"] = False

run_root = os.environ.get(
    "STAGE68_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage68_native_vlmap_shadow")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE68_EVAL_PORT", "3680")

"""Stage72: strict mainline recovery control with legacy VLMaps disabled.

S2 still receives the native recovery context and may emit pixels, but no
pixel translation is handed to S1 until a SparseOcc-native trajectory
preflight exists.  This is a fail-closed control, not a permissive ablation.
"""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_stage70_mainline_boundary_smoke_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage70_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage65_native_recovery_enable"] = True
vlmap_cfg["stage65_native_recovery_max_queries"] = 5
vlmap_cfg["stage65_native_pixel_execution_enable"] = False
vlmap_cfg["stage71_permissive_s2_ablation_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = True
vlmap_cfg["s2_recovery_context_shadow_only"] = False
vlmap_cfg["s2_recovery_context_max_images"] = 1
vlmap_cfg["s2_action_loop_shadow_only"] = False

run_root = os.environ.get("STAGE72_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs").rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage72_strict_no_vlmap")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE72_EVAL_PORT", "3720")

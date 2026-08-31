"""Stage58.1 read-only support-policy sweep on the physical 0.10m footprint."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage58_geometry_contract_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage58_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage58_geometry_contract_enable"] = False
vlmap_cfg["stage58_support_policy_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False

run_root = os.environ.get("STAGE58_RUN_ROOT", "./logs/habitat").rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage58_support_policy")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE58_EVAL_PORT", "3582")

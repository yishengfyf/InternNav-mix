"""Stage58.0 read-only Habitat geometry contract radius sweep."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage57_local_elevation_support_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage57_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage58_geometry_contract_enable"] = True
vlmap_cfg["stage58_geometry_contract_radii_m"] = [0.10, 0.13, 0.15, 0.18]
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False
eval_cfg.eval_settings["port"] = os.environ.get("STAGE58_EVAL_PORT", "3580")

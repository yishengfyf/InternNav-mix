"""Stage54 exploratory active: a route-only path may guide one turn only."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage49_frozen_reorient_active_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage49_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["s2_loop_path_reobserve_m3_candidate_stage"] = "route_only"
vlmap_cfg["s2_loop_path_reobserve_m3_safety_mode"] = "route_only_turn_only"
vlmap_cfg["s2_loop_path_reobserve_frozen_path_bearing_relaxation"] = True
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False
eval_cfg.eval_settings["port"] = os.environ.get("STAGE54_EVAL_PORT", "3543")

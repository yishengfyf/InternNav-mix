"""Stage55 exploratory active: requery after a collided post-turn S2 forward."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage55_route_occ_audit_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage55_route_occ_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["s2_loop_path_reobserve_post_turn_collision_guard_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_post_turn_collision_guard_max_requeries"] = 8
vlmap_cfg["s2_loop_path_reobserve_post_turn_collision_guard_horizon_steps"] = 400
eval_cfg.eval_settings["port"] = os.environ.get("STAGE55_EVAL_PORT", "3555")

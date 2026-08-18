"""Stage22E: fixed-route evidence aligned in the obstacle height band."""

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage22d_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage22d_fixed_route_evidence_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage22d_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage22d_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["s2_loop_fixed_route_occ_evidence_audit_enable"] = True
vlmap_cfg["s2_loop_fixed_route_height_evidence_audit_enable"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False

eval_cfg.eval_settings["port"] = os.environ.get(
    "STAGE22_FIXED_ROUTE_EVAL_PORT", "2569"
)

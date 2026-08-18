"""Stage22C: fixed Stage22A routes audited on pitch-aware OCC."""

import copy
import importlib.util
import json
import os
from pathlib import Path


def _load_stage22b_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage22b_pitch_aware_occ_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage22b_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage22b_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
manifest_path = (
    Path(__file__).resolve().parents[1]
    / "manifests"
    / "stage22c_stage22a_fixed_route12.json"
)

# Dynamic candidates and triage remain logged, but route-map comparison always
# uses the exact Stage22A anchor/source pair at each of the 12 fixed triggers.
vlmap_cfg["s2_loop_executed_route_occ_audit_enable"] = False
vlmap_cfg["s2_loop_fixed_route_occ_audit_enable"] = True
vlmap_cfg["s2_loop_fixed_route_occ_audit_entries"] = json.loads(
    manifest_path.read_text(encoding="utf-8")
)
vlmap_cfg["occ_memory_camera_pitch_aware_update"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False

eval_cfg.eval_settings["port"] = os.environ.get(
    "STAGE22_FIXED_ROUTE_EVAL_PORT", "2565"
)

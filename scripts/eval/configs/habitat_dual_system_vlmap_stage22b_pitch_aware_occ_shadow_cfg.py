"""Stage22B: pitch-aware sparse OCC projection, fully shadow-only."""

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage22a_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage22a_executed_route_occ_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage22a_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage22a_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# The only mapping change from Stage22A is applying the known camera pitch to
# depth back-projection. Route history, detector, candidates, ranking, S2 and
# all action paths remain frozen/shadow-only.
vlmap_cfg["occ_memory_camera_pitch_aware_update"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False

eval_cfg.eval_settings["port"] = os.environ.get(
    "STAGE22_PITCH_EVAL_PORT", "2563"
)

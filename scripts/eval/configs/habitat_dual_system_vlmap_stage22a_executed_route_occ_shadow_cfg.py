"""Stage22A: strict executed-route versus ray-free OCC shadow audit."""

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage21c_shadow_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage21c_shadow_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage21c_shadow_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# This stage reads pose_trace and OCC state only. It does not promote route
# cells to free space and does not enable any recovery action or S2 re-query.
vlmap_cfg["s2_loop_executed_route_occ_audit_enable"] = True
vlmap_cfg["s2_loop_executed_route_occ_audit_max_edge_m"] = float(
    os.environ.get("STAGE22_ROUTE_MAX_EDGE_M", "0.75")
)
vlmap_cfg["s2_loop_executed_route_occ_audit_sample_spacing_m"] = float(
    os.environ.get("STAGE22_ROUTE_SAMPLE_SPACING_M", "0.05")
)
vlmap_cfg["s2_loop_executed_route_occ_audit_max_path_cells"] = int(
    os.environ.get("STAGE22_ROUTE_MAX_PATH_CELLS", "160")
)
vlmap_cfg["s2_loop_executed_route_occ_audit_max_visited_cells"] = int(
    os.environ.get("STAGE22_ROUTE_MAX_VISITED_CELLS", "20000")
)

vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = True

eval_cfg.eval_settings["port"] = os.environ.get("STAGE22_ROUTE_EVAL_PORT", "2561")

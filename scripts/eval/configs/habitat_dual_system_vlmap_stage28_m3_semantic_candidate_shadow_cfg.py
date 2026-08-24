"""Stage28 E2/E3 conditional LSeg route-reobserve candidate shadow."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage27_candidate_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Use the frozen Stage26 filter for the filtered branch; raw frames remain
# available in parallel and both branches are audit-only.
vlmap_cfg["lseg_online_shadow_component_filter_enable"] = True
vlmap_cfg["lseg_online_shadow_component_filter_min_samples"] = int(
    os.environ.get("STAGE26_COMPONENT_MIN_SAMPLES", "4")
)
vlmap_cfg["lseg_online_shadow_component_filter_radius_m"] = float(
    os.environ.get("STAGE26_COMPONENT_RADIUS_M", "0.20")
)
vlmap_cfg["lseg_online_shadow_component_filter_min_neighbors"] = int(
    os.environ.get("STAGE26_COMPONENT_MIN_NEIGHBORS", "2")
)

candidate_cfg = vlmap_cfg["stage27_candidate_audit_config"]
candidate_cfg.update({
    "semantic_candidate_enable": True,
    "semantic_trigger_min_base_candidates": 1,
    "semantic_candidate_count": 3,
    "semantic_route_neighbors_per_node": 3,
})

# Defense in depth: semantic candidates are serialized but never consumed.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE28_EVAL_PORT", "3295")

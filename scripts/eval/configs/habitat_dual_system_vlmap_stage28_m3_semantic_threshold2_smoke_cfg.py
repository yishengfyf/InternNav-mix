"""Stage28 semantic candidate shadow with a one-candidate fallback trigger."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage28_m3_semantic_candidate_shadow_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage28_semantic_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Permit semantic route-reobserve proposals only when the strict geometric
# pool has fewer than two candidates.  Proposals remain audit-only and must
# pass the unchanged route-OCC, unknown, footprint, and clearance gates.
vlmap_cfg["stage27_candidate_audit_config"]["semantic_trigger_min_base_candidates"] = 2
vlmap_cfg["stage27_candidate_audit_config"]["semantic_route_neighbors_per_node"] = 5
eval_cfg.eval_settings["port"] = os.environ.get("STAGE28_EVAL_PORT", "3595")

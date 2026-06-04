import copy
import importlib.util
from pathlib import Path


def _load_stage10a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage10a_occ_memory_query_candidates_shadow_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage10a_occ_memory_query_candidates_shadow_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage10a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V10a2: semanticized sparse 3D query candidate shadow.
# Compared with V10a, frontier/open-floor candidates are bound to historical
# semantic evidence and instruction relevance. S2 selection can be parsed from
# A/B/C/D labels or from a coordinate on the final BEV candidate map.
vlmap_cfg["occ_memory_candidate_probe_semantic_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_semantic_high_conf_only"] = False
vlmap_cfg["occ_memory_candidate_probe_semantic_min_score"] = 0.20
vlmap_cfg["occ_memory_candidate_probe_semantic_max_candidates"] = 3
vlmap_cfg["occ_memory_candidate_probe_semantic_bind_radius_m"] = 2.50
vlmap_cfg["occ_memory_candidate_probe_semantic_direction_match_degrees"] = 75.0
vlmap_cfg["occ_memory_candidate_probe_semantic_frontier_min_relevance"] = 0.15
vlmap_cfg["occ_memory_candidate_probe_semantic_score_weight"] = 1.10
vlmap_cfg["occ_memory_candidate_probe_semantic_novelty_weight"] = 0.55
vlmap_cfg["occ_memory_candidate_probe_topology_novelty_weight"] = 0.35

vlmap_cfg["occ_memory_candidate_selection_parse_coordinates"] = True
vlmap_cfg["occ_memory_candidate_selection_coordinate_threshold_px"] = 90.0

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage10a2_100_occ_memory_semantic_query_candidates_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage10a2_100_occ_memory_semantic_query_candidates_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2379"

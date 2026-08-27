"""Stage44 natural train-D0 candidate collection; all outputs are shadow-only."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage21a_train_recovery_shadow_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage21a_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage27_candidate_audit_enable"] = True
# Empty manifest is intentional: evaluator hooks only natural D0 loop starts.
vlmap_cfg["stage27_candidate_audit_entries"] = []
vlmap_cfg["stage27_candidate_audit_config"] = {
    "min_distance_m": 0.50, "max_distance_m": 4.0, "min_separation_m": 0.25,
    "near_count": 1, "open_count": 2, "open_rank_floor_safe_first": True,
    "sample_spacing_m": 0.05, "open_radius_m": 0.50,
    "floor_aligned_height_max_m": 1.5, "footprint_radius_m": 0.18,
    "max_occupied_fraction": 0.0, "max_unknown_fraction": 0.0,
    "known_safe_frontier_enable": True, "frontier_search_radius_m": 4.0,
    "frontier_sample_limit": 512, "frontier_path_max_visited_cells": 30000,
    "frontier_standoff_m": 0.25, "frontier_min_route_separation_m": 0.25,
    "frontier_trigger_min_route_candidates": 1,
    "recovery_bev_snapshot_enable": True, "recovery_bev_radius_cells": 24,
}

# No recovery, reobserve, ranker, or action path is enabled.
for key, value in {
    "occ_memory_shadow_only": True, "s2_action_loop_shadow_only": True,
    "s2_loop_strict_active_enable": False, "s2_loop_path_reobserve_active_enable": False,
    "recovery_context_enable": False, "nextdit_candidate_active_enable": False,
    "occ_memory_recovery_enable": False, "occ_memory_recovery_shadow_only": True,
    "occ_memory_candidate_probe_max_events_per_episode": 64,
    "occ_memory_validation_enable": False, "occ_memory_validation_save_rgb_depth": False,
    "occ_memory_validation_save_current_rgb_ply": False,
    "occ_memory_validation_save_memory_ply": False,
    "occ_memory_validation_save_final_memory_ply": False,
    "occ_memory_save_bev": False, "occ_memory_candidate_probe_save_bev": False,
}.items():
    vlmap_cfg[key] = value

eval_cfg.eval_settings["port"] = os.environ.get("STAGE44_EVAL_PORT", "3441")

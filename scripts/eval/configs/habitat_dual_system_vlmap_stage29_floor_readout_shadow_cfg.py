"""Stage29 shadow: conservative local floor-height readout for M3 candidates."""

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

candidate_cfg = vlmap_cfg["stage27_candidate_audit_config"]
candidate_cfg.update({
    # Readout only: the raw SparseOcc state and all safety gates remain fixed.
    "floor_z_estimation_enable": True,
    "floor_z_estimation_radius_m": 0.75,
    "floor_z_estimation_min_m": 0.0,
    "floor_z_estimation_max_m": 0.80,
    "floor_z_estimation_min_support_cells": 8,
    "floor_z_estimation_min_support_ratio": 0.25,
    # A traversed floor can rise only continuously along executed movement.
    # This rejects furniture-height planes without using scene-specific data.
    "floor_z_estimation_max_step_m": 0.20,
})

# Defense in depth: Stage29 remains shadow-only and cannot affect actions.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE29_EVAL_PORT", "3395")

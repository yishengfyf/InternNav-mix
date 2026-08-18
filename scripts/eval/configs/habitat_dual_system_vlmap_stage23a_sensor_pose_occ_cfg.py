"""Stage23A Layer1/2: complete Habitat sensor-pose OCC audit."""

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage21c_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage21c_sensor_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage21c_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["occ_memory_enable"] = True
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["occ_memory_camera_pitch_aware_update"] = True
vlmap_cfg["occ_memory_map_height"] = 6.0
vlmap_cfg["occ_memory_validation_enable"] = True
vlmap_cfg["occ_memory_validation_accumulate_rgb_surface"] = True
vlmap_cfg["occ_memory_validation_oracle_pose_enable"] = False
vlmap_cfg["occ_memory_validation_oracle_sensor_pose_enable"] = True
vlmap_cfg["occ_memory_validation_current_depth_sample_rate"] = 8
vlmap_cfg["occ_memory_validation_max_current_points"] = 120000
vlmap_cfg["occ_memory_validation_max_accumulated_surface_points"] = int(
    os.environ.get("STAGE23A_MAX_SURFACE_POINTS", "250000")
)
vlmap_cfg["occ_memory_validation_projection_size"] = 768
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = True
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = True
vlmap_cfg["occ_memory_validation_save_memory_ply"] = True
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = True

# This branch is audit-only. GT sensor pose is never passed to navigation.
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = True

eval_cfg.eval_settings["port"] = os.environ.get("STAGE23A_EVAL_PORT", "2571")

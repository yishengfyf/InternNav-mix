"""Stage45 offline rejection-truth audit on natural train-D0 candidates."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage44_train_d0_candidate_shadow_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage44_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage45_candidate_rejection_truth_enable"] = True
vlmap_cfg["stage45_candidate_rejection_truth_config"] = {
    "footprint_radius_m": 0.18,
    "floor_aligned_height_max_m": 1.5,
    "max_edge_geodesic_ratio": 2.0,
}

# Oracle sensor pose supplies only local floor height for the offline GT audit.
# It remains an isolated memory branch and is never read by navigation.
vlmap_cfg["occ_memory_validation_enable"] = True
vlmap_cfg["occ_memory_validation_oracle_pose_enable"] = False
vlmap_cfg["occ_memory_validation_oracle_sensor_pose_enable"] = True
vlmap_cfg["occ_memory_validation_accumulate_rgb_surface"] = False
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = False
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = False
vlmap_cfg["occ_memory_validation_save_memory_ply"] = False
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = False

eval_cfg.eval_settings["port"] = os.environ.get("STAGE45_EVAL_PORT", "3451")

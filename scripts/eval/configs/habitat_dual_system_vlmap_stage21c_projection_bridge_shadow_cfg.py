"""Stage21c map-candidate to visible-free-pixel projection bridge shadow."""

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage21c_cfg():
    path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage21c_shadow_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage21c_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# This experiment audits only the execution bridge. Frozen S2/NextDiT and all
# active recovery paths remain unchanged.
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["s2_loop_projection_bridge_enable"] = True
vlmap_cfg["s2_loop_projection_bridge_shadow_only"] = True

# A small deterministic image grid covers central/lateral and near/far-looking
# floor regions. A proposal is valid only when its depth projection lands on a
# known-free OccMem cell and stays within the candidate-bearing tolerance.
vlmap_cfg["s2_loop_projection_bridge_sample_x_ratios"] = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]
vlmap_cfg["s2_loop_projection_bridge_sample_y_ratios"] = [0.55, 0.65, 0.75, 0.85]
vlmap_cfg["s2_loop_projection_bridge_max_angle_error_deg"] = float(
    os.environ.get("STAGE21C_PROJECTION_MAX_ANGLE_DEG", "30.0")
)

# Keep the old fixed directional adapter parameters only as the baseline probe
# in the A/B audit. Its output is never applied in this config.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_execution_mode"] = (
    "directional_pixel_goal"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_y_ratio"] = 0.75
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_front_x_ratio"] = 0.50
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_left_x_ratio"] = 0.25
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_right_x_ratio"] = 0.75

eval_cfg.eval_settings["port"] = os.environ.get(
    "STAGE21C_PROJECTION_EVAL_PORT", "2491"
)

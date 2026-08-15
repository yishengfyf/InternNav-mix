"""Stage21c-r2: strict-only known-free path reorient/reobserve active."""

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

# Only a shared-triage strict_intervention may enter this active state machine.
# The old fixed directional-pixel active path stays disabled.
vlmap_cfg["s2_action_loop_shadow_only"] = False
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_strict_active_allowed_directions"] = ["front", "left", "right"]
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_max_interventions_per_episode"] = 1

# First require a known-free route to the exact resilience_backtrack anchor.
# A visible pixel is executable only when its depth endpoint lies in this path
# corridor and makes monotonic progress along the route.
vlmap_cfg["s2_loop_path_reobserve_max_path_cells"] = int(
    os.environ.get("STAGE21C_PATH_MAX_CELLS", "160")
)
vlmap_cfg["s2_loop_path_reobserve_path_corridor_m"] = float(
    os.environ.get("STAGE21C_PATH_CORRIDOR_M", "0.35")
)
vlmap_cfg["s2_loop_path_reobserve_min_path_progress_m"] = float(
    os.environ.get("STAGE21C_PATH_MIN_PROGRESS_M", "0.25")
)
vlmap_cfg["s2_loop_path_reobserve_max_local_subgoal_m"] = float(
    os.environ.get("STAGE21C_PATH_MAX_LOCAL_M", "3.0")
)
vlmap_cfg["s2_loop_path_reobserve_max_heading_error_deg"] = float(
    os.environ.get("STAGE21C_PATH_MAX_HEADING_ERROR_DEG", "40.0")
)
vlmap_cfg["s2_loop_path_reobserve_lookahead_m"] = float(
    os.environ.get("STAGE21C_PATH_LOOKAHEAD_M", "0.75")
)

# If no path-consistent pixel is visible, rotate toward the first 0.75m of the
# map route. Four 15-degree Habitat turns cap one intervention at 60 degrees.
# When already aligned, one scan opposite the repeated S2 turn breaks the
# mechanical loop and supplies a genuinely new observation.
vlmap_cfg["s2_loop_path_reobserve_max_turn_steps"] = int(
    os.environ.get("STAGE21C_PATH_MAX_TURN_STEPS", "4")
)
vlmap_cfg["s2_loop_path_reobserve_turn_deadband_deg"] = 7.5
vlmap_cfg["s2_loop_path_reobserve_scan_when_aligned"] = True

vlmap_cfg["s2_loop_projection_bridge_enable"] = True
vlmap_cfg["s2_loop_projection_bridge_shadow_only"] = True
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

# Do not mix Stage19/20 active hooks or Stage21d context into the causal test.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = True
vlmap_cfg["s2_recovery_context_enable"] = False

eval_cfg.eval_settings["port"] = os.environ.get(
    "STAGE21C_PATH_ACTIVE_EVAL_PORT", "2501"
)

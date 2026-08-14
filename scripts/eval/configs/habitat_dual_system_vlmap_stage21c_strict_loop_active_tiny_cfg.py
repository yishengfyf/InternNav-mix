"""Stage21c-r1: paired tiny strict S2-loop candidate active treatment."""

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

# This branch is intentionally separate from the legacy Stage19/20 active-lite
# hook.  It acts only on an observed S2-loop event that the shared triage labels
# strict_intervention.  Adapter/abstain events remain frozen-S2 holds.
vlmap_cfg["s2_action_loop_shadow_only"] = False
vlmap_cfg["s2_loop_strict_active_enable"] = True
vlmap_cfg["s2_loop_strict_active_max_interventions_per_episode"] = 1
vlmap_cfg["s2_loop_strict_active_require_active_gate_safe"] = True
vlmap_cfg["s2_loop_strict_active_allowed_directions"] = ["front", "left", "right"]

# Do not mix the old waypoint-time active hook or Stage21d context into this
# treatment.  The multi-task scorer remains inference-only for logging.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = True
vlmap_cfg["s2_recovery_context_enable"] = False

# Reuse the already-tested coarse candidate direction -> visible pixel ->
# NextDiT local replan adapter. Existing OccMem trajectory checks remain on.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_execution_mode"] = (
    "directional_pixel_goal"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_y_ratio"] = 0.75
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_front_x_ratio"] = 0.50
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_left_x_ratio"] = 0.25
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_right_x_ratio"] = 0.75

eval_cfg.eval_settings["port"] = os.environ.get("STAGE21C_ACTIVE_EVAL_PORT", "2481")


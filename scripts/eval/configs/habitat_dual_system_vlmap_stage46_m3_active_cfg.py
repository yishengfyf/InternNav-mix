"""Stage46 tiny active: frozen M3 -> one primitive -> mandatory re-audit."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage27_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# D0 and M3 remain frozen. Only this explicit adapter may bind a final-pool
# candidate to the existing Frozen NextDiT local planner.
vlmap_cfg["s2_action_loop_shadow_only"] = False
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_candidate_source"] = "stage27_frozen_m3"
vlmap_cfg["s2_loop_path_reobserve_one_primitive_per_reaudit"] = True
vlmap_cfg["s2_loop_path_reobserve_iterative_reorient_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_max_interventions_per_episode"] = 1
vlmap_cfg["s2_loop_path_reobserve_max_turn_steps"] = 1
vlmap_cfg["s2_loop_strict_active_allowed_directions"] = ["path"]
vlmap_cfg["s2_loop_path_reobserve_turn_deadband_deg"] = 7.5
vlmap_cfg["s2_loop_path_reobserve_scan_when_aligned"] = True
vlmap_cfg["s2_loop_path_reobserve_max_path_cells"] = 160
# Keep treatment local: v1 showed that a 1.65m observation turn perturbed an
# already-successful route, while the 0.95m candidate resolved a persistent
# failure. This post-v1 gate must be revalidated and is not a learned score.
vlmap_cfg["s2_loop_path_reobserve_max_active_path_m"] = float(
    os.environ.get("STAGE46_MAX_ACTIVE_PATH_M", "1.0")
)
vlmap_cfg["s2_loop_path_reobserve_path_corridor_m"] = 0.35
vlmap_cfg["s2_loop_path_reobserve_min_path_progress_m"] = 0.25
vlmap_cfg["s2_loop_path_reobserve_max_local_subgoal_m"] = 3.0
vlmap_cfg["s2_loop_path_reobserve_max_heading_error_deg"] = 40.0
vlmap_cfg["s2_loop_path_reobserve_lookahead_m"] = 0.75

# No other recovery path is allowed into this causal test.
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True

eval_cfg.eval_settings["port"] = os.environ.get("STAGE46_EVAL_PORT", "3461")

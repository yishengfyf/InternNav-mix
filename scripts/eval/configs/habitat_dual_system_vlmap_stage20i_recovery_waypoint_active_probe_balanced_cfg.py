import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20g_v2_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20g_v2_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20i is an intentionally diagnostic active probe. It reuses the
# Stage-D directional pixel-goal interface, but only after Stage20g-v2 triage
# and only for strict/adapter candidates. Frozen S2/NextDiT replans the local
# trajectory; abstain and non-actionable back-direction candidates remain held.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_enable"] = True
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_require_strict_intervention"
] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_v2_evidence_tiers"] = [
    "strict_intervention",
    "adapter_candidate",
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_direction_buckets"] = [
    "front",
    "left",
    "right",
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = [
    "stuck_collision"
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives"] = [
    "reorient_reobserve",
    "one_safe_forward_reobserve",
]

vlmap_cfg["occ_memory_semantic_resilience_active_lite_execution_mode"] = (
    "directional_pixel_goal"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_y_ratio"] = 0.75
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_front_x_ratio"] = 0.50
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_left_x_ratio"] = 0.25
vlmap_cfg["occ_memory_semantic_resilience_active_lite_direction_right_x_ratio"] = 0.75

vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = 1
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = 45
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = 0.0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = 0.0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = 0.0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = 1.0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = 4.0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = 120
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_active_gate_safe"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20I_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20i_recovery_waypoint_active_probe_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20I_EVAL_PORT", "2416")

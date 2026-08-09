import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20f_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20f_sparse_semantic_recovery_mixed_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20f_sparse_semantic_recovery_mixed_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def _split_env_list(name: str, default: str):
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


eval_cfg = copy.deepcopy(_load_stage20f_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20g: strict shadow gate calibration.
#
# Stage20f showed good failure recall but too many would-apply cases on final
# success episodes.  Stage20g keeps S2/NextDiT frozen and keeps the run
# shadow-only by default, but it evaluates the active gate all the way through
# so the logs contain either shadow_gate_pass or the exact rejection reason.
#
# Default policy: only the strongest stuck/local-trap recovery candidates pass.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE20G_ACTIVE_SHADOW_ONLY", "1") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = (
    os.environ.get("STAGE20G_EVALUATE_GATE_WHEN_SHADOW_ONLY", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = _split_env_list(
    "STAGE20G_ALLOWED_FAILURE_TYPES",
    "stuck_collision",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives"] = _split_env_list(
    "STAGE20G_ALLOWED_PRIMITIVES",
    "reorient_reobserve,one_safe_forward_reobserve",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_any_trigger_reasons"] = _split_env_list(
    "STAGE20G_REQUIRE_ANY_TRIGGER_REASONS",
    "local_trap,semantic_obstacle_near_trap",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_any_context_tags"] = _split_env_list(
    "STAGE20G_REQUIRE_ANY_CONTEXT_TAGS",
    "spatial_constriction,limited_frontier_escape",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_target_frontier_intent_safe"] = (
    os.environ.get("STAGE20G_REQUIRE_TARGET_FRONTIER_INTENT_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_target_frontier_score"] = float(
    os.environ.get("STAGE20G_MIN_TARGET_FRONTIER_SCORE", "0.10")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_completed_landmark_penalty"] = float(
    os.environ.get("STAGE20G_MAX_COMPLETED_LANDMARK_PENALTY", "0.0")
)

vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_step"] = int(
    os.environ.get("STAGE20G_MIN_STEP", "35")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = int(
    os.environ.get("STAGE20G_MAX_INTERVENTIONS", "1")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = int(
    os.environ.get("STAGE20G_COOLDOWN_STEPS", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = float(
    os.environ.get("STAGE20G_UTILITY_THRESHOLD", "0.54")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = float(
    os.environ.get("STAGE20G_LOCAL_TRAP_UTILITY_THRESHOLD", "0.60")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = float(
    os.environ.get("STAGE20G_OPEN_THRESHOLD", "0.70")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = float(
    os.environ.get("STAGE20G_MIN_BACKTRACK_M", "1.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = float(
    os.environ.get("STAGE20G_MAX_BACKTRACK_M", "3.5")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = int(
    os.environ.get("STAGE20G_MAX_STEP_GAP", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = (
    os.environ.get("STAGE20G_REQUIRE_CURRENT_PROBLEM", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = (
    os.environ.get("STAGE20G_REQUIRE_GEOMETRY_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_active_gate_safe"] = (
    os.environ.get("STAGE20G_REQUIRE_ACTIVE_GATE_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = int(
    os.environ.get("STAGE20G_MAX_TURN_STEPS", "2")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = int(
    os.environ.get("STAGE20G_FORWARD_STEPS", "0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = (
    os.environ.get("STAGE20G_ALLOW_FORWARD_TO_BACKTRACK", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_open_threshold"] = float(
    os.environ.get("STAGE20G_FORWARD_OPEN_THRESHOLD", "0.80")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = (
    os.environ.get("STAGE20G_APPEND_REOBSERVE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = (
    os.environ.get("STAGE20G_CLEAR_GOAL", "0") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20G_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20g_sparse_semantic_recovery_gate_calibration_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20G_EVAL_PORT", "2410")

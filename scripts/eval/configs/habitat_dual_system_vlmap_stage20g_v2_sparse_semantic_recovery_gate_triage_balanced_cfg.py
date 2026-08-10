import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20g_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20g_sparse_semantic_recovery_gate_calibration_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20g_sparse_semantic_recovery_gate_calibration_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def _split_env_list(name: str, default: str):
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


eval_cfg = copy.deepcopy(_load_stage20g_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20g-v2: multi-evidence recovery triage.
#
# Stage20g protected successful episodes, but only one event passed.  V2 keeps
# the run shadow-only by default and changes the gate from single-condition
# filtering to a structured triage:
#
#   strict_intervention: S2-policy conflict + obstacle/trap context + safe
#                        escape anchor agree; this is the only would-apply tier.
#   adapter_candidate:  plausible but needs a learned/progress adapter.
#   abstain:            hand control back to frozen S2/NextDiT.
#
# This is meant to reduce dependence on one-off threshold tuning and produce a
# cleaner dataset for the next memory-grounded recovery adapter.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE20GV2_ACTIVE_SHADOW_ONLY", "1") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = (
    os.environ.get("STAGE20GV2_EVALUATE_GATE_WHEN_SHADOW_ONLY", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_enable"] = (
    os.environ.get("STAGE20GV2_EVIDENCE_GATE_ENABLE", "1") != "0"
)
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_require_strict_intervention"
] = (os.environ.get("STAGE20GV2_REQUIRE_STRICT_INTERVENTION", "1") != "0")

vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = _split_env_list(
    "STAGE20GV2_ALLOWED_FAILURE_TYPES",
    "stuck_collision",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives"] = _split_env_list(
    "STAGE20GV2_ALLOWED_PRIMITIVES",
    "reorient_reobserve,one_safe_forward_reobserve",
)

# Let the V2 evidence gate decide trigger/context/target-frontier consistency.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_any_trigger_reasons"] = _split_env_list(
    "STAGE20GV2_REQUIRE_ANY_TRIGGER_REASONS",
    "",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_any_context_tags"] = _split_env_list(
    "STAGE20GV2_REQUIRE_ANY_CONTEXT_TAGS",
    "",
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_target_frontier_intent_safe"] = (
    os.environ.get("STAGE20GV2_REQUIRE_TARGET_FRONTIER_INTENT_SAFE", "0") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_target_frontier_score"] = float(
    os.environ.get("STAGE20GV2_MIN_TARGET_FRONTIER_SCORE", "0.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_completed_landmark_penalty"] = float(
    os.environ.get("STAGE20GV2_MAX_COMPLETED_LANDMARK_PENALTY", "0.0")
)

vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_step"] = int(
    os.environ.get("STAGE20GV2_MIN_STEP", "35")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = int(
    os.environ.get("STAGE20GV2_MAX_INTERVENTIONS", "1")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = int(
    os.environ.get("STAGE20GV2_COOLDOWN_STEPS", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = float(
    os.environ.get("STAGE20GV2_UTILITY_THRESHOLD", "0.54")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = float(
    os.environ.get("STAGE20GV2_LOCAL_TRAP_UTILITY_THRESHOLD", "0.60")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = float(
    os.environ.get("STAGE20GV2_OPEN_THRESHOLD", "0.62")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = float(
    os.environ.get("STAGE20GV2_MIN_BACKTRACK_M", "1.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = float(
    os.environ.get("STAGE20GV2_MAX_BACKTRACK_M", "4.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = int(
    os.environ.get("STAGE20GV2_MAX_STEP_GAP", "120")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = (
    os.environ.get("STAGE20GV2_REQUIRE_CURRENT_PROBLEM", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = (
    os.environ.get("STAGE20GV2_REQUIRE_GEOMETRY_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_active_gate_safe"] = (
    os.environ.get("STAGE20GV2_REQUIRE_ACTIVE_GATE_SAFE", "0") != "0"
)

vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_open_score"] = float(
    os.environ.get("STAGE20GV2_EVIDENCE_MIN_OPEN_SCORE", "0.70")
)
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_doorway_score"
] = float(os.environ.get("STAGE20GV2_EVIDENCE_MIN_DOORWAY_SCORE", "0.60"))
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_target_frontier_score"
] = float(os.environ.get("STAGE20GV2_EVIDENCE_MIN_TARGET_FRONTIER_SCORE", "0.10"))
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_step_gap"] = int(
    os.environ.get("STAGE20GV2_EVIDENCE_MIN_STEP_GAP", "20")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_min_nearby_visits"] = int(
    os.environ.get("STAGE20GV2_EVIDENCE_MIN_NEARBY_VISITS", "3")
)

vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = int(
    os.environ.get("STAGE20GV2_MAX_TURN_STEPS", "2")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = int(
    os.environ.get("STAGE20GV2_FORWARD_STEPS", "0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = (
    os.environ.get("STAGE20GV2_ALLOW_FORWARD_TO_BACKTRACK", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_open_threshold"] = float(
    os.environ.get("STAGE20GV2_FORWARD_OPEN_THRESHOLD", "0.80")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = (
    os.environ.get("STAGE20GV2_APPEND_REOBSERVE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = (
    os.environ.get("STAGE20GV2_CLEAR_GOAL", "0") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20GV2_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20GV2_EVAL_PORT", "2412")

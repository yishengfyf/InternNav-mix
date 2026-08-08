import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20d_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20d_sparse_semantic_recovery_active_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20d_sparse_semantic_recovery_active_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20d_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20f: mixed balanced recovery calibration.
#
# This keeps the sparse semantic OCC stack from Stage20e, but turns the active
# recovery gate into a larger balanced smoke target.  The default remains
# conservative: one intervention per episode, no forward commit, reobserve
# after a safe backtrack.  The run can be kept shadow-only overnight by setting
# STAGE20F_ACTIVE_SHADOW_ONLY=1.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE20F_ACTIVE_SHADOW_ONLY", "1") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = [
    item.strip()
    for item in os.environ.get(
        "STAGE20F_ALLOWED_FAILURE_TYPES",
        "stuck_collision,semantic_stagnation",
    ).split(",")
    if item.strip()
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_step"] = int(
    os.environ.get("STAGE20F_MIN_STEP", "35")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = int(
    os.environ.get("STAGE20F_MAX_INTERVENTIONS", "1")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = int(
    os.environ.get("STAGE20F_COOLDOWN_STEPS", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = float(
    os.environ.get("STAGE20F_UTILITY_THRESHOLD", "0.54")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = float(
    os.environ.get("STAGE20F_LOCAL_TRAP_UTILITY_THRESHOLD", "0.60")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = float(
    os.environ.get("STAGE20F_OPEN_THRESHOLD", "0.62")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = float(
    os.environ.get("STAGE20F_MIN_BACKTRACK_M", "1.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = float(
    os.environ.get("STAGE20F_MAX_BACKTRACK_M", "3.5")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = int(
    os.environ.get("STAGE20F_MAX_STEP_GAP", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = (
    os.environ.get("STAGE20F_REQUIRE_CURRENT_PROBLEM", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = (
    os.environ.get("STAGE20F_REQUIRE_GEOMETRY_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_active_gate_safe"] = (
    os.environ.get("STAGE20F_REQUIRE_ACTIVE_GATE_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = int(
    os.environ.get("STAGE20F_MAX_TURN_STEPS", "2")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = int(
    os.environ.get("STAGE20F_FORWARD_STEPS", "0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = (
    os.environ.get("STAGE20F_ALLOW_FORWARD_TO_BACKTRACK", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_open_threshold"] = float(
    os.environ.get("STAGE20F_FORWARD_OPEN_THRESHOLD", "0.78")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = (
    os.environ.get("STAGE20F_APPEND_REOBSERVE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = (
    os.environ.get("STAGE20F_CLEAR_GOAL", "0") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20F_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20f_sparse_semantic_recovery_mixed_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20F_EVAL_PORT", "2408")

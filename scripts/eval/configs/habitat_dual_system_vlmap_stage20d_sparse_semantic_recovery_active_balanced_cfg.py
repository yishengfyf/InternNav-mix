import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20c_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20c_sparse_semantic_anchor_audit_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20c_sparse_semantic_anchor_audit_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20c_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20d: very-small active recovery smoke.
#
# This keeps the Stage20c sparse semantic anchor audit stack intact, but opens
# the semantic-resilience active-lite gate for only the clearest recovery cases:
# stuck_collision and semantic_stagnation.  The first pass is intentionally
# conservative: reorient + reobserve, no forced forward move by default.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE20D_ACTIVE_SHADOW_ONLY", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = [
    item.strip()
    for item in os.environ.get(
        "STAGE20D_ALLOWED_FAILURE_TYPES",
        "stuck_collision",
    ).split(",")
    if item.strip()
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_step"] = int(
    os.environ.get("STAGE20D_MIN_STEP", "30")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = int(
    os.environ.get("STAGE20D_MAX_INTERVENTIONS", "1")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = int(
    os.environ.get("STAGE20D_COOLDOWN_STEPS", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = float(
    os.environ.get("STAGE20D_UTILITY_THRESHOLD", "0.57")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = float(
    os.environ.get("STAGE20D_LOCAL_TRAP_UTILITY_THRESHOLD", "0.62")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = float(
    os.environ.get("STAGE20D_OPEN_THRESHOLD", "0.65")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = float(
    os.environ.get("STAGE20D_MIN_BACKTRACK_M", "1.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = float(
    os.environ.get("STAGE20D_MAX_BACKTRACK_M", "3.5")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = int(
    os.environ.get("STAGE20D_MAX_STEP_GAP", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = (
    os.environ.get("STAGE20D_REQUIRE_CURRENT_PROBLEM", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = (
    os.environ.get("STAGE20D_REQUIRE_GEOMETRY_SAFE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_active_gate_safe"] = (
    os.environ.get("STAGE20D_REQUIRE_ACTIVE_GATE_SAFE", "1") != "0"
)

# Conservative first smoke: turn/reobserve only.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = int(
    os.environ.get("STAGE20D_MAX_TURN_STEPS", "2")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = int(
    os.environ.get("STAGE20D_FORWARD_STEPS", "0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = (
    os.environ.get("STAGE20D_ALLOW_FORWARD_TO_BACKTRACK", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_open_threshold"] = float(
    os.environ.get("STAGE20D_FORWARD_OPEN_THRESHOLD", "0.80")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = (
    os.environ.get("STAGE20D_APPEND_REOBSERVE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = (
    os.environ.get("STAGE20D_CLEAR_GOAL", "0") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20D_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20C_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE20_BALANCED_RUN_NAME",
            "compare_vlmap_stage20d_sparse_semantic_recovery_active_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20D_EVAL_PORT", "2404")

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage18e_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage18e_semantic_resilience_shadow_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage18e_semantic_resilience_shadow_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage18e_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage19a: semantic-resilience active-lite recovery.
#
# This is intentionally not a new global planner.  Frozen S2/NextDiT remains
# the main policy.  OccMem may briefly intervene only when the Stage18e
# semantic-resilience candidate event passes a conservative recovery gate:
# semantic/local failure context + safe recent backtrack anchor + high open
# score.  The first smoke uses reorient + reobserve by default; forward motion
# can be enabled later with STAGE19_FORWARD_STEPS=1 after the no-forward smoke
# proves that the gate itself is not harmful.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE19_ACTIVE_SHADOW_ONLY", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_step"] = int(
    os.environ.get("STAGE19_MIN_STEP", "30")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = int(
    os.environ.get("STAGE19_MAX_INTERVENTIONS", "1")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_cooldown_steps"] = int(
    os.environ.get("STAGE19_COOLDOWN_STEPS", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_utility_threshold"] = float(
    os.environ.get("STAGE19_UTILITY_THRESHOLD", "0.58")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_local_trap_utility_threshold"] = float(
    os.environ.get("STAGE19_LOCAL_TRAP_UTILITY_THRESHOLD", "0.62")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_open_threshold"] = float(
    os.environ.get("STAGE19_OPEN_THRESHOLD", "0.65")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_min_backtrack_m"] = float(
    os.environ.get("STAGE19_MIN_BACKTRACK_M", "1.0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_backtrack_m"] = float(
    os.environ.get("STAGE19_MAX_BACKTRACK_M", "3.5")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_step_gap"] = int(
    os.environ.get("STAGE19_MAX_STEP_GAP", "45")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_current_problem"] = (
    os.environ.get("STAGE19_REQUIRE_CURRENT_PROBLEM", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_require_geometry_safe"] = (
    os.environ.get("STAGE19_REQUIRE_GEOMETRY_SAFE", "1") != "0"
)

# Execution primitive.  No-forward is the safest first active smoke; it changes
# viewpoint and forces S2 to re-observe instead of committing to a risky
# hand-written backtrack path.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = int(
    os.environ.get("STAGE19_MAX_TURN_STEPS", "4")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = int(
    os.environ.get("STAGE19_FORWARD_STEPS", "0")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = (
    os.environ.get("STAGE19_ALLOW_FORWARD_TO_BACKTRACK", "0") == "1"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_open_threshold"] = float(
    os.environ.get("STAGE19_FORWARD_OPEN_THRESHOLD", "0.80")
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = (
    os.environ.get("STAGE19_APPEND_REOBSERVE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = (
    os.environ.get("STAGE19_CLEAR_GOAL", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE19_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE18_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage19a_semantic_resilience_active_lite_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE19_EVAL_PORT", "2396")

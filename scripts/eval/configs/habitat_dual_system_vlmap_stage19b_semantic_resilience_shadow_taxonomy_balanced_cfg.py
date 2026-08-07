import copy
import importlib.util
import os
from pathlib import Path


def _load_stage19a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage19a_semantic_resilience_active_lite_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage19a_semantic_resilience_active_lite_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage19a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage19b: shadow taxonomy for semantic resilience.
#
# Keep the Stage19 failure profiling logic, but never execute the active
# intervention path. The goal is to collect a clean 500-episode taxonomy of
# what kind of failure was detected and which primitive would have been
# recommended, without changing the trajectory.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get(
        "STAGE19B_ACTIVE_SHADOW_ONLY",
        os.environ.get("STAGE19_ACTIVE_SHADOW_ONLY", "1"),
    )
    == "1"
)
allowed_failure_types = [
    item.strip()
    for item in os.environ.get("STAGE19B_ALLOWED_FAILURE_TYPES", "").split(",")
    if item.strip()
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = (
    allowed_failure_types
)

run_name = os.environ.get(
    "STAGE19B_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE19_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE18_BALANCED_RUN_NAME",
            os.environ.get(
                "STAGE17_BALANCED_RUN_NAME",
                "compare_vlmap_stage19b_semantic_resilience_shadow_taxonomy_balanced",
            ),
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE19B_EVAL_PORT", "2397")

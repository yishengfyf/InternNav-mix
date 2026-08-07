import copy
import importlib.util
import os
from pathlib import Path


def _load_stage19b_shadow_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage19b_semantic_resilience_shadow_taxonomy_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage19b_semantic_resilience_shadow_taxonomy_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage19b_shadow_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage19b active gate is still conservative by default.  It only becomes
# active when STAGE19B_ACTIVE_SHADOW_ONLY=0 is explicitly set.  The default
# allowed failure types are the clearest diagnostic cases from the balanced40
# v2 shadow run.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = (
    os.environ.get("STAGE19B_ACTIVE_SHADOW_ONLY", "1") == "1"
)
allowed_failure_types = [
    item.strip()
    for item in os.environ.get(
        "STAGE19B_ALLOWED_FAILURE_TYPES",
        "stuck_collision,semantic_stagnation",
    ).split(",")
    if item.strip()
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = (
    allowed_failure_types
)

run_name = os.environ.get(
    "STAGE19B_BALANCED_RUN_NAME",
    "compare_vlmap_stage19b_semantic_resilience_active_gate_balanced",
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE19B_EVAL_PORT", "2398")

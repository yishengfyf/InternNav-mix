"""Stage21d temporary recovery-context A/B shadow; no action is changed."""

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

vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = True
vlmap_cfg["s2_recovery_context_shadow_only"] = True
vlmap_cfg["s2_recovery_context_max_images"] = int(
    os.environ.get("STAGE21D_RECOVERY_CONTEXT_MAX_IMAGES", "2")
)
vlmap_cfg["s2_recovery_context_ttl_queries"] = int(
    os.environ.get("STAGE21D_RECOVERY_CONTEXT_TTL_QUERIES", "2")
)
vlmap_cfg["s2_recovery_context_shadow_variants"] = [
    value.strip()
    for value in os.environ.get(
        "STAGE21D_RECOVERY_CONTEXT_VARIANTS", "text_only,text_images"
    ).split(",")
    if value.strip()
]

eval_cfg.eval_settings["port"] = os.environ.get("STAGE21D_EVAL_PORT", "2483")


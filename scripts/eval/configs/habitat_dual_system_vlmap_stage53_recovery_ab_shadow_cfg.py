"""Stage53 same-event look-down and online-context four-arm shadow."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage52_depth_local_search_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage52_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["s2_recovery_context_enable"] = True
vlmap_cfg["s2_recovery_context_shadow_only"] = True
vlmap_cfg["s2_recovery_context_max_images"] = 2
vlmap_cfg["s2_recovery_context_ttl_queries"] = 1
vlmap_cfg["s2_recovery_context_save_images"] = True
# Stage53 owns the four arms; disable the older two-arm query loop.
vlmap_cfg["s2_recovery_context_shadow_variants"] = []
vlmap_cfg["stage53_recovery_ab_enable"] = True
vlmap_cfg["stage53_lookdown_pitch_deg"] = 30.0

for key, value in {
    "occ_memory_shadow_only": True,
    "s2_action_loop_shadow_only": True,
    "s2_loop_strict_active_enable": False,
    "s2_loop_path_reobserve_active_enable": False,
    "nextdit_candidate_active_enable": False,
    "occ_memory_recovery_enable": False,
}.items():
    vlmap_cfg[key] = value

eval_cfg.eval_settings["port"] = os.environ.get("STAGE53_EVAL_PORT", "3455")

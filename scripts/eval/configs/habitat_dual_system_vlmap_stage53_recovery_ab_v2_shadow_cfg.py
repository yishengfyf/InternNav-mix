"""Stage53 v2: identify the already-fresh look-down current view to Frozen S2."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage53_recovery_ab_shadow_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage53_v1_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["stage53_lookdown_view_prompt_enable"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE53_EVAL_PORT", "3475")

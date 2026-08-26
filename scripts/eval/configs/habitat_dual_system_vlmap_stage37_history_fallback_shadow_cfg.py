"""Stage37 all-history safety-first fallback shadow audit."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage27_candidate_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# This branch changes only the shadow candidate pool.  Every historical route
# node is passed through the unchanged OCC, unknown, floor and footprint gates
# before it can be selected.
candidate_cfg = vlmap_cfg["stage27_candidate_audit_config"]
candidate_cfg["history_fallback_enable"] = True
candidate_cfg["history_fallback_count"] = 3
eval_cfg.eval_settings["port"] = os.environ.get("STAGE27_EVAL_PORT", "3895")

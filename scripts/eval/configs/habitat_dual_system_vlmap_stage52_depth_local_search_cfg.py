"""Stage52 bounded local depth-corridor search and rejection diagnostics."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage50_depth_short_lookahead_bev_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage50_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
cfg = vlmap_cfg["stage27_candidate_audit_config"]
cfg["stage52_local_search_enable"] = True
cfg["stage52_local_search_lateral_m"] = 0.20
cfg["stage52_local_search_detour_m"] = 0.25
cfg["stage52_local_search_max_paths"] = 16
eval_cfg.eval_settings["port"] = os.environ.get("STAGE52_EVAL_PORT", "3452")

"""Stage48: bounded per-observation reorientation for one frozen M3 candidate."""

import copy
import importlib.util
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage46_m3_active_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage46_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# One D0 recovery transaction may rotate at most four times. Every primitive
# is followed by a fresh observation and current SparseOcc path re-audit.
vlmap_cfg["s2_loop_path_reobserve_iterative_reorient_enable"] = True
vlmap_cfg["s2_loop_path_reobserve_max_turn_steps"] = 4

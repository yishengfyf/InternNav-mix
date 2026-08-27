"""Stage49 frozen release: one safe reorientation, then Frozen S2 handoff."""

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

# Only bounded reorientation has causal rollout evidence. Pixel-goal
# translation and iterative pursuit remain implemented experiment branches,
# but are not released by this frozen configuration.
vlmap_cfg["s2_loop_path_reobserve_iterative_reorient_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_max_turn_steps"] = 1
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False

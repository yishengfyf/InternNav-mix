"""Stage63.1 shadow: adaptive multi-view re-observation after productive D0."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage59_productive_onset_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage59_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage59_post_turn_counterfactual_enable"] = False
vlmap_cfg["stage63_adaptive_reobserve_enable"] = True
vlmap_cfg["stage63_adaptive_reobserve_primitive_budgets"] = (1, 2, 4)
vlmap_cfg["stage63_adaptive_reobserve_max_turn_steps"] = 12
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False

run_root = os.environ.get("STAGE59_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs").rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage63_adaptive_reobserve")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE59_EVAL_PORT", "3630")


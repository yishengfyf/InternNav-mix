"""Stage74: prompt-v2 continuous recovery active ablation."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_stage73_continuous_recovery_ablation_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage73_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
vlmap_cfg["stage74_recovery_prompt_v2_enable"] = True

run_root = os.environ.get("STAGE74_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs").rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage74_recovery_prompt_v2_ablation")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE74_EVAL_PORT", "3740")

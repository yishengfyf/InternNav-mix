"""Stage66: Stage65-native replay with audit ledger and RGB visualization."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage65_native_recovery_active_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage65_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Read-only recording only. No prompt, safety, memory, or action policy change.
vlmap_cfg["replay_ledger_enable"] = True
vlmap_cfg["replay_ledger_save_rgb"] = True
vlmap_cfg["replay_ledger_save_depth"] = True
vlmap_cfg["replay_ledger_max_observations"] = 0
vlmap_cfg["replay_ledger_max_queries"] = 0
vlmap_cfg["replay_ledger_max_actions"] = 0

eval_cfg.eval_settings["save_video"] = True
eval_cfg.agent.model_settings["vis_debug"] = True
run_root = os.environ.get(
    "STAGE66_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage66_native_visual_audit")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE66_EVAL_PORT", "3660")

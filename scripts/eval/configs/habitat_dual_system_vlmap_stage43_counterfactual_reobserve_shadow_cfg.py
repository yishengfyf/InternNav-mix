"""Stage43 no-action, no-state-change counterfactual depth probe."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name("habitat_dual_system_vlmap_stage41_executor_contract_shadow_cfg.py")
spec = importlib.util.spec_from_file_location("_stage41_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
candidate_cfg = eval_cfg.agent.model_settings["vlmap_safety"]["stage27_candidate_audit_config"]
candidate_cfg.update({
    "stage43_counterfactual_reobserve_enable": True,
    "stage43_center_margin_deg": 10.0,
    "stage43_max_turn_steps": 12,
    "stage43_max_candidate_probes": 3,
    "stage43_zero_history_distance_m": 0.50,
})
eval_cfg.eval_settings["port"] = os.environ.get("STAGE27_EVAL_PORT", "3943")

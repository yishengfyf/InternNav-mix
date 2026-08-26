"""Stage41 real-depth executor contract shadow configuration."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name("habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py")
spec = importlib.util.spec_from_file_location("_stage27_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
candidate_cfg = eval_cfg.agent.model_settings["vlmap_safety"]["stage27_candidate_audit_config"]
candidate_cfg["stage41_executor_contract_enable"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE27_EVAL_PORT", "3911")

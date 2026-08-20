"""Stage25: Frozen-S2 stuck detector GT-contract shadow run."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage24d_online_lseg_shadow_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage24d_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["replay_ledger_rgb_format"] = "png"
vlmap_cfg["replay_ledger_save_rgb"] = True
vlmap_cfg["replay_ledger_save_depth"] = True
vlmap_cfg["replay_ledger_repeat_episode_meta"] = False

# Detector data collection is side-effect-only.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE25_EVAL_PORT", "2795")

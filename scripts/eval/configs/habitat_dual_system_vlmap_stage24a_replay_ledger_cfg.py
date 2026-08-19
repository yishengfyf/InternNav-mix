"""Stage24A: audit-only Replay Ledger on the fixed sensor-pose smoke set."""

import copy
import importlib.util
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23a_sensor_pose_occ_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage23a_sensor_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Replay is strictly side-effect-only. Keep the existing S2, OCC and safety
# shadow settings unchanged so the run can be compared against the baseline.
vlmap_cfg["replay_ledger_enable"] = True
vlmap_cfg["replay_ledger_save_rgb"] = True
vlmap_cfg["replay_ledger_save_depth"] = True
vlmap_cfg["replay_ledger_max_observations"] = 0
vlmap_cfg["replay_ledger_max_queries"] = 0
vlmap_cfg["replay_ledger_max_actions"] = 0
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True

"""Stage25 single-episode repair with the same Frozen-S2 detector contract.

The full holdout stopped while writing the final episode because auxiliary
validation/visualization files exhausted the server filesystem.  This config
keeps the replay ledger and online semantic event data, but disables only
diagnostic copies that are not detector inputs.
"""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name("habitat_dual_system_vlmap_stage25_gt_detector_cfg.py")
spec = importlib.util.spec_from_file_location("_stage25_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Keep all detector inputs: lossless RGB/depth, pose, actions, collisions,
# SparseOcc summaries and query-frame LSeg semantic events.
vlmap_cfg["replay_ledger_rgb_format"] = "png"
vlmap_cfg["replay_ledger_save_rgb"] = True
vlmap_cfg["replay_ledger_save_depth"] = True

# These files are useful for visualization audits but are not consumed by the
# Stage25 analyzer and caused the interrupted holdout to fill /home.
vlmap_cfg["occ_memory_validation_enable"] = False
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = False
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = False
vlmap_cfg["occ_memory_validation_save_memory_ply"] = False
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = False
vlmap_cfg["occ_memory_save_bev"] = False
vlmap_cfg["occ_memory_candidate_probe_save_bev"] = False
vlmap_cfg["lseg_online_shadow_save_overlay"] = False
vlmap_cfg["lseg_online_shadow_save_surface"] = False
vlmap_cfg["lseg_online_shadow_save_visualizations"] = False

# Defense in depth: this remains Frozen-S2 and shadow-only.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True

run_name = os.environ.get(
    "STAGE25_SINGLE_RUN_NAME",
    "compare_vlmap_stage25_gt_detector_holdout96_missing_episode_repair",
)
if not run_name or "/" in run_name or "\\" in run_name:
    raise ValueError("STAGE25_SINGLE_RUN_NAME must be a simple directory name")
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE25_EVAL_PORT", "2797")

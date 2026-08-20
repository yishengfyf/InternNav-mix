"""Stage24D: online same-process LSeg at Frozen-S2 query frames."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage24a_replay_ledger_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage24a_replay_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["lseg_online_shadow_enable"] = True
vlmap_cfg["lseg_online_shadow_repo"] = os.environ.get(
    "STAGE24D_VLMAPS_REPO", "/home/yifeifeng/workspace/vlmaps"
)
vlmap_cfg["lseg_online_shadow_checkpoint"] = os.environ.get(
    "STAGE24D_LSEG_CHECKPOINT",
    "/home/yifeifeng/workspace/vlmaps/vlmaps/lseg/checkpoints/demo_e200.ckpt",
)
vlmap_cfg["lseg_online_shadow_device"] = "same"
vlmap_cfg["lseg_online_shadow_confidence_threshold"] = float(
    os.environ.get("STAGE24D_LSEG_CONFIDENCE", "0.35")
)
vlmap_cfg["lseg_online_shadow_sample_stride"] = int(
    os.environ.get("STAGE24D_LSEG_STRIDE", "8")
)
vlmap_cfg["lseg_online_shadow_merge_radius_m"] = 0.50
vlmap_cfg["lseg_online_shadow_max_surface_samples"] = int(
    os.environ.get("STAGE24D_MAX_SURFACE_SAMPLES", "250000")
)
vlmap_cfg["lseg_online_shadow_save_overlay"] = True
vlmap_cfg["lseg_online_shadow_save_surface"] = True
vlmap_cfg["lseg_online_shadow_save_visualizations"] = True
vlmap_cfg["replay_ledger_rgb_format"] = "png"

# Defense in depth: Stage24D is observation-only.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE24D_EVAL_PORT", "2695")

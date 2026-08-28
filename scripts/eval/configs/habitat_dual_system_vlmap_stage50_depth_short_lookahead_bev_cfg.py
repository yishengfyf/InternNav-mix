"""Stage50/51 natural-D0 depth lookahead and semantic BEV shadow."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage44_train_d0_candidate_shadow_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage44_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]
cfg = vlmap_cfg["stage27_candidate_audit_config"]
cfg["stage50_depth_short_lookahead_enable"] = True
cfg["stage51_semantic_bev_enable"] = True
cfg["semantic_candidate_enable"] = True
cfg["semantic_trigger_min_base_candidates"] = 1

# LSeg remains a separate observation-only branch, using the already frozen
# Stage26 filtered component settings; semantic output cannot affect safety.
vlmap_cfg["lseg_online_shadow_enable"] = True
vlmap_cfg["lseg_online_shadow_repo"] = os.environ.get("STAGE24D_VLMAPS_REPO", "/home/yifeifeng/workspace/vlmaps")
vlmap_cfg["lseg_online_shadow_checkpoint"] = os.environ.get(
    "STAGE24D_LSEG_CHECKPOINT",
    "/home/yifeifeng/workspace/InternNav/results/stage_17/stage24d_lseg_safe_checkpoint_20260820/demo_e200_state_dict.pt",
)
vlmap_cfg["lseg_online_shadow_device"] = "same"
vlmap_cfg["lseg_online_shadow_component_filter_enable"] = True
vlmap_cfg["lseg_online_shadow_component_filter_min_samples"] = int(os.environ.get("STAGE26_COMPONENT_MIN_SAMPLES", "4"))
vlmap_cfg["lseg_online_shadow_component_filter_radius_m"] = float(os.environ.get("STAGE26_COMPONENT_RADIUS_M", "0.20"))
vlmap_cfg["lseg_online_shadow_component_filter_min_neighbors"] = int(os.environ.get("STAGE26_COMPONENT_MIN_NEIGHBORS", "2"))
vlmap_cfg["lseg_online_shadow_save_overlay"] = True
vlmap_cfg["lseg_online_shadow_save_surface"] = True
vlmap_cfg["lseg_online_shadow_save_visualizations"] = True

# Defense in depth: this configuration is shadow-only and has no active path.
for key, value in {
    "occ_memory_shadow_only": True,
    "s2_action_loop_shadow_only": True,
    "s2_loop_strict_active_enable": False,
    "s2_loop_path_reobserve_active_enable": False,
    "nextdit_candidate_active_enable": False,
    "occ_memory_recovery_enable": False,
    "occ_memory_recovery_shadow_only": True,
}.items():
    vlmap_cfg[key] = value
eval_cfg.eval_settings["port"] = os.environ.get("STAGE50_EVAL_PORT", "3450")

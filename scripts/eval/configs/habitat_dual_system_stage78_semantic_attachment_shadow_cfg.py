"""Stage78A: recovery-specific LSeg-to-SparseOcc attachment shadow."""

import copy
import importlib.util
import os
from pathlib import Path


_base = Path(__file__).with_name(
    "habitat_dual_system_stage77_directional_guard_ablation_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage77_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage26-filtered LSeg is causal and audit-only.  Its outputs are deliberately
# not enabled as semantic anchors and are not exposed to the S2 prompt.
vlmap_cfg["lseg_online_shadow_enable"] = True
vlmap_cfg["lseg_online_shadow_repo"] = os.environ.get(
    "STAGE24D_VLMAPS_REPO", "/home/yifeifeng/workspace/vlmaps"
)
vlmap_cfg["lseg_online_shadow_checkpoint"] = os.environ.get(
    "STAGE24D_LSEG_CHECKPOINT",
    (
        "/home/yifeifeng/workspace/InternNav/results/stage_17/"
        "stage24d_lseg_safe_checkpoint_20260820/demo_e200_state_dict.pt"
    ),
)
vlmap_cfg["lseg_online_shadow_device"] = "same"
vlmap_cfg["lseg_online_shadow_confidence_threshold"] = 0.35
vlmap_cfg["lseg_online_shadow_sample_stride"] = 8
vlmap_cfg["lseg_online_shadow_merge_radius_m"] = 0.50
vlmap_cfg["lseg_online_shadow_component_filter_enable"] = True
vlmap_cfg["lseg_online_shadow_component_filter_min_samples"] = 4
vlmap_cfg["lseg_online_shadow_component_filter_radius_m"] = 0.20
vlmap_cfg["lseg_online_shadow_component_filter_min_neighbors"] = 2
vlmap_cfg["lseg_online_shadow_save_overlay"] = True
vlmap_cfg["lseg_online_shadow_save_surface"] = True
vlmap_cfg["lseg_online_shadow_save_visualizations"] = True
vlmap_cfg["stage78_semantic_route_audit_enable"] = True
vlmap_cfg["stage78_semantic_route_max_distance_m"] = 0.75
vlmap_cfg["stage78_semantic_route_min_views"] = 2
vlmap_cfg["stage78_semantic_route_min_confidence"] = 0.35
vlmap_cfg["stage78_recovery_bev_radius_cells"] = 24

# Defense in depth: semantics are not a navigation consumer in Stage78A.
vlmap_cfg["occ_memory_semantic_anchor_enable"] = False
vlmap_cfg["legacy_vlmaps_experiment"] = False
vlmap_cfg["legacy_vlmaps_enable"] = False
vlmap_cfg["legacy_vlmaps_waypoint_enable"] = False
vlmap_cfg["legacy_vlmaps_semantic_enable"] = False

run_root = os.environ.get(
    "STAGE78_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage78_semantic_attachment_shadow")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE78_EVAL_PORT", "3780")

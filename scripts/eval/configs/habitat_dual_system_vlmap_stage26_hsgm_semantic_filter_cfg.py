"""Stage26 HSGM-inspired semantic attachment filter, audit-only."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage25_detector500_lowdisk_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage25_lowdisk_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# This branch consumes the exact raw Q-frame LSeg surface and cannot affect any
# prompt, detector, candidate, safety decision, or action.
vlmap_cfg["lseg_online_shadow_component_filter_enable"] = True
vlmap_cfg["lseg_online_shadow_component_filter_min_samples"] = int(
    os.environ.get("STAGE26_COMPONENT_MIN_SAMPLES", "4")
)
vlmap_cfg["lseg_online_shadow_component_filter_radius_m"] = float(
    os.environ.get("STAGE26_COMPONENT_RADIUS_M", "0.20")
)
vlmap_cfg["lseg_online_shadow_component_filter_min_neighbors"] = int(
    os.environ.get("STAGE26_COMPONENT_MIN_NEIGHBORS", "2")
)

# Keep all established Stage25 safety and detector settings frozen. The
# duplicated assignments make accidental active execution fail code review.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE26_EVAL_PORT", "2995")

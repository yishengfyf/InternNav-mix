"""Stage27 M3 candidate-generation shadow; Frozen S2 and detector unchanged."""

import copy
import importlib.util
import os
import json
from pathlib import Path

path = Path(__file__).with_name("habitat_dual_system_vlmap_stage25_detector500_lowdisk_cfg.py")
spec = importlib.util.spec_from_file_location("_stage25_lowdisk_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage27_candidate_audit_enable"] = True
event_manifest = os.environ.get("STAGE27_EVENT_MANIFEST", "")
vlmap_cfg["stage27_candidate_audit_entries"] = (
    json.loads(Path(event_manifest).read_text(encoding="utf-8"))
    if event_manifest else []
)
vlmap_cfg["stage27_candidate_audit_config"] = {
    "min_distance_m": 0.50,
    "max_distance_m": 4.0,
    "min_separation_m": 0.25,
    "near_count": 1,
    "open_count": 2,
    "sample_spacing_m": 0.05,
    "open_radius_m": 0.50,
    "floor_aligned_height_max_m": 1.5,
    "footprint_radius_m": 0.18,
    "max_occupied_fraction": 0.0,
    "max_unknown_fraction": 0.0,
}
# No candidate output can influence the navigator.  These assignments are
# intentionally duplicated as a review-time guard.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE27_EVAL_PORT", "3095")

"""Stage23E single-label, explicitly pixel-grounded semantic anchor audit."""

import copy
import importlib.util
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage23c_semantic_scene_audit_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage23c_semantic_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Change only the semantic binding rule: one top term at the explicit S2 pixel
# goal.  Generic view samples are disabled for this controlled ablation.
vlmap_cfg["occ_memory_semantic_anchor_max_terms_per_event"] = 1
vlmap_cfg["occ_memory_semantic_anchor_include_threshold_hits"] = False
vlmap_cfg["occ_memory_semantic_anchor_include_pixel_goal"] = True
vlmap_cfg["occ_memory_semantic_anchor_include_view_center"] = False
vlmap_cfg["occ_memory_semantic_anchor_include_view_left"] = False
vlmap_cfg["occ_memory_semantic_anchor_include_view_right"] = False
vlmap_cfg["occ_memory_semantic_anchor_include_view_upper"] = False
vlmap_cfg["occ_memory_semantic_anchor_include_view_lower"] = False

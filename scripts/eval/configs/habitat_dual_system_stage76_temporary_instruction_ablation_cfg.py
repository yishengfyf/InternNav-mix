"""Stage76: bind the SparseOcc route subtask to DualVLN's instruction slot."""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name(
    "habitat_dual_system_stage75_route_guidance_ablation_cfg.py"
)
_spec = importlib.util.spec_from_file_location("_stage75_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

vlmap_cfg["stage76_temporary_instruction_enable"] = True

# Keep the causal arm isolated: no LSeg semantics and no legacy VLMaps.
vlmap_cfg["lseg_online_shadow_enable"] = False
vlmap_cfg["occ_memory_semantic_anchor_enable"] = False
vlmap_cfg["legacy_vlmaps_experiment"] = False
vlmap_cfg["legacy_vlmaps_enable"] = False
vlmap_cfg["legacy_vlmaps_waypoint_enable"] = False
vlmap_cfg["legacy_vlmaps_semantic_enable"] = False

run_root = os.environ.get(
    "STAGE76_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get(
    "STAGE21_RUN_NAME", "stage76_temporary_instruction_ablation"
)
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.agent.model_settings["vis_debug_path"] = f"{output_path}/vis_debug"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE76_EVAL_PORT", "3760")

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20a_sparse_semantic_anchor_shadow_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20a_sparse_semantic_anchor_shadow_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20b: multi-view sparse semantic anchors.
#
# This keeps the Stage20a anchor projection logic but turns on a few additional
# view directions so each semantic event can generate a sparse local fan of
# anchors instead of a single center-biased point.
vlmap_cfg["occ_memory_semantic_anchor_include_view_left"] = True
vlmap_cfg["occ_memory_semantic_anchor_include_view_right"] = True
vlmap_cfg["occ_memory_semantic_anchor_include_view_upper"] = True
vlmap_cfg["occ_memory_semantic_anchor_include_view_lower"] = True

run_name = os.environ.get(
    "STAGE20_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE19B_BALANCED_RUN_NAME",
        "compare_vlmap_stage20b_sparse_semantic_anchor_multiview_shadow_balanced",
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20_EVAL_PORT", "2400")

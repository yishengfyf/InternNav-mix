import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20b_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20b_sparse_semantic_anchor_multiview_shadow_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20b_sparse_semantic_anchor_multiview_shadow_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20b_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20c: audit-only sparse semantic anchor run.
#
# Keep the Stage20b behavior unchanged and only tighten the naming / packaging
# path so the returned logs can be interpreted as a focused audit layer.
run_name = os.environ.get(
    "STAGE20C_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE19B_BALANCED_RUN_NAME",
            "compare_vlmap_stage20c_sparse_semantic_anchor_audit_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20C_EVAL_PORT", os.environ.get("STAGE20_EVAL_PORT", "2403"))

import copy
import importlib.util
import os
from pathlib import Path


def _load_stage17_balanced_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage17a_train_balanced_shadow_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage17a_train_balanced_shadow_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage17_balanced_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage18a is logging/audit only:
#   frozen S2/NextDiT still controls navigation;
#   OccMem still emits its normal candidate set;
#   SparseOccMemory additionally records current_policy_candidate = S2/CURRENT
#   so offline labels can compare keep-S2 vs memory-grounded intervention.
#
# Keep the Stage17 progress-ranker checkpoint out of the default path.  The
# first Stage18a smoke should validate S2/current endpoint logging without
# depending on trained ranker artifacts.
vlmap_cfg["occ_memory_progress_ranker_shadow_enable"] = False

run_name = os.environ.get(
    "STAGE18_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE17_BALANCED_RUN_NAME",
        "compare_vlmap_stage18a_s2_candidate_logging_balanced",
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE18_EVAL_PORT", "2395")

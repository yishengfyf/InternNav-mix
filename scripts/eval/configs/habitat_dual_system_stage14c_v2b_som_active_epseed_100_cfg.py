import copy
import importlib.util
from pathlib import Path


def _load_stage14c_100_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_stage14c_som_active_epseed_100_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_stage14c_som_active_epseed_100_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage14c_100_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage14c-v2b: geometry gate on occupied OR (free AND follows_frontier).
#
# Extends v2a by also allowing replacement when the S2 goal lands on a free
# cell that is directionally aligned with the OccMem frontier ("follows_frontier").
# Post-hoc: adds 3 extra events (17 eps / 19 replacements vs 15/16 in v2a),
# including ep166 (14b=F) and ep262 (14b=S) — marginal vs v2a.
# Regressions 61,62,78,328 are still eliminated (0 occupied or free&follows each).
# Run alongside v2a to test whether free&follows events add independent value.
vlmap_cfg["som_counterfactual_active_goal_state_gate"] = "occupied_or_free_follows"

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14c_v2b_som_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14c_v2b_som_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2404"

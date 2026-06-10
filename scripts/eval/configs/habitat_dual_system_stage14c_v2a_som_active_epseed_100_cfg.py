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

# Stage14c-v2a: geometry gate on OccMem goal_state == "occupied" only.
#
# Post-hoc analysis of Stage14c showed all 4 succ->fail regressions came from
# goal_state=="unknown" replacements (no geometric justification), while the
# 2 recovered fail->succ episodes both had occupied-state events.
# Gating on occupied eliminates all 4 regressions while retaining ep266/ep286.
# ep279 (only unknown events) is lost — that is the known cost.
#
# active_goal_state_gate options:
#   "any"                    -> Stage14c original (no geometry gate)
#   "occupied"               -> v2a: replace only when S2 goal is in occupied cell
#   "occupied_or_free_follows" -> v2b: occupied OR (free AND follows_frontier)
vlmap_cfg["som_counterfactual_active_goal_state_gate"] = "occupied"

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_stage14c_v2a_som_active_epseed_100/"
    "vlmap_safety_debug"
)

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_stage14c_v2a_som_active_epseed_100"
)
eval_cfg.eval_settings["port"] = "2403"

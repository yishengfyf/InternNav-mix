"""Stage67: native recovery with strict trajectory/candidate shadow logging.

The frozen S2 receives the same native recovery context as Stage66.  This
variant only enables read-only trajectory validation and NextDiT candidate
probes so every S2 pixel can be attributed to endpoint/route/footprint
rejections.  No shadow result is allowed to mutate memory or execute actions.
"""

import copy
import importlib.util
import os
from pathlib import Path

_base = Path(__file__).with_name("habitat_dual_system_vlmap_stage66_native_visual_audit_cfg.py")
_spec = importlib.util.spec_from_file_location("_stage66_cfg", _base)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
eval_cfg = copy.deepcopy(_module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# This is a historical VLMaps trajectory audit, intentionally outside the
# mainline recovery stack.
vlmap_cfg["legacy_vlmaps_experiment"] = True
vlmap_cfg["legacy_vlmaps_enable"] = True

# Strict safety remains authoritative; these are diagnostics only.  The
# explicit legacy opt-in above is required solely to run this historical
# trajectory validator and does not redefine the mainline safety authority.
vlmap_cfg["traj_validation_enable"] = True
vlmap_cfg["traj_validation_shadow_only"] = True
vlmap_cfg["nextdit_candidate_probe_enable"] = True
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_pixel_execution_enable"] = False
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["pixel_translation_active_enable"] = False

run_root = os.environ.get(
    "STAGE67_RUN_ROOT", "/data/usr_data/yifeifeng/internnav/stage_results/runs"
).rstrip("/")
run_name = os.environ.get("STAGE21_RUN_NAME", "stage67_native_strict_shadow")
output_path = f"{run_root}/{run_name}"
eval_cfg.eval_settings["output_path"] = output_path
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["port"] = os.environ.get("STAGE67_EVAL_PORT", "3670")

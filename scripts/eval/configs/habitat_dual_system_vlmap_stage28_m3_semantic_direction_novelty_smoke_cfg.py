"""Stage36 semantic route-reobserve with strict direction novelty."""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage28_m3_semantic_threshold2_smoke_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage35_semantic_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# A semantic fallback is useful only when it adds a genuinely different
# route direction.  Candidate eligibility and every SparseOcc hard gate are
# inherited unchanged from Stage35.
candidate_cfg = vlmap_cfg["stage27_candidate_audit_config"]
candidate_cfg["semantic_direction_novelty_enable"] = True
candidate_cfg["semantic_direction_novelty_require_new"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE28_EVAL_PORT", "3795")

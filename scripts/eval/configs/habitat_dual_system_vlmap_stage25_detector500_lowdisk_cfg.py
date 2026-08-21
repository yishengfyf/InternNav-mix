"""Stage25 fresh-500 detector shadow with bounded disk usage.

The online LSeg/SparseOcc path remains enabled at every Frozen-S2 query.  RGB-D,
pose, query/action and semantic summaries remain in the replay ledger; dense
LSeg visualizations are generated later only for selected detector events.
"""

import copy
import importlib.util
import os
from pathlib import Path


path = Path(__file__).with_name(
    "habitat_dual_system_vlmap_stage25_gt_detector_cfg.py"
)
spec = importlib.util.spec_from_file_location("_stage25_detector_cfg", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
eval_cfg = copy.deepcopy(module.eval_cfg)
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Keep the same Frozen-S2, SparseOcc and query-frequency semantics as Stage25.
# JPEG storage preserves raw RGB hashes in the JSONL while avoiding the PNG
# expansion that exhausted the server during the previous 96-episode run.
vlmap_cfg["replay_ledger_rgb_format"] = "jpg"
vlmap_cfg["replay_ledger_save_rgb"] = True
# Online LSeg and SparseOcc still consume metric depth before this audit write.
# Keep source hashes/statistics in JSONL but avoid duplicating every depth array.
vlmap_cfg["replay_ledger_save_depth"] = False
vlmap_cfg["replay_ledger_repeat_episode_meta"] = False

# Keep online semantic inference and compact 3-D semantic summaries, but defer
# event overlays/surface dumps until after detector mining.
vlmap_cfg["lseg_online_shadow_save_overlay"] = False
vlmap_cfg["lseg_online_shadow_save_surface"] = False
vlmap_cfg["lseg_online_shadow_save_visualizations"] = False
vlmap_cfg["lseg_online_shadow_max_surface_samples"] = int(
    os.environ.get("STAGE25_MAX_SURFACE_SAMPLES", "12000")
)

# Defense in depth: detector collection remains shadow-only.
vlmap_cfg["occ_memory_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["s2_loop_path_reobserve_active_enable"] = False
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["nextdit_candidate_active_enable"] = False
vlmap_cfg["occ_memory_recovery_enable"] = False
vlmap_cfg["occ_memory_recovery_shadow_only"] = True
eval_cfg.eval_settings["port"] = os.environ.get("STAGE25_EVAL_PORT", "2895")

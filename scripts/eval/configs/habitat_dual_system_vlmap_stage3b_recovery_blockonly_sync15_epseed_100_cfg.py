import copy
import importlib.util
from pathlib import Path


def _load_stage3_recovery_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_vlmap_stage3_recovery_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage3_recovery_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage3_recovery_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V3b: conservative ablation for V3a.  Only recover from explicitly blocked
# waypoint paths; high-risk-but-free waypoints are logged but not changed.
vlmap_cfg["waypoint_recovery_on_block"] = True
vlmap_cfg["waypoint_recovery_on_high_risk"] = False
vlmap_cfg["max_waypoint_recoveries_per_episode"] = 1
vlmap_cfg["waypoint_recovery_candidate_angles_deg"] = [-15.0, 15.0]

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage3b_100_recovery_blockonly_sync15_epseed/"
    "vlmap_safety_debug"
)
vlmap_cfg["debug_max_snapshots"] = 45
vlmap_cfg["debug_sample_total_snapshots"] = 25
vlmap_cfg["debug_force_max_snapshots"] = 10
vlmap_cfg["debug_force_max_snapshots_per_episode"] = 1

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage3b_100_recovery_blockonly_sync15_epseed"
)
eval_cfg.eval_settings["port"] = "2352"

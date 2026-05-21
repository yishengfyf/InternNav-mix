import copy
import importlib.util
from pathlib import Path


def _load_base_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_vlmap_stage2_active_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage2_epseed_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_base_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V3a: use VLMap as a local waypoint filter/recovery advisor.  When S2 selects
# a blocked or high-risk waypoint, do not ask S2 to repeat itself; first rotate
# toward the safer local fan candidate, then let S2 observe again.
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_requery_feedback_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = True
vlmap_cfg["waypoint_recovery_on_block"] = True
vlmap_cfg["waypoint_recovery_on_high_risk"] = True
vlmap_cfg["waypoint_recovery_risk_threshold"] = 0.70
vlmap_cfg["waypoint_recovery_min_checked_cells"] = 20
vlmap_cfg["max_waypoint_recoveries_per_episode"] = 2
vlmap_cfg["waypoint_recovery_cooldown_steps"] = 20
vlmap_cfg["waypoint_recovery_probe_distance"] = 0.60
vlmap_cfg["waypoint_recovery_require_free_probe"] = True
vlmap_cfg["waypoint_recovery_alignment_weight"] = 0.25
vlmap_cfg["waypoint_recovery_max_turn_steps"] = 1
vlmap_cfg["waypoint_recovery_candidate_angles_deg"] = [-15.0, 15.0]

vlmap_cfg["debug_dir"] = "./logs/habitat/compare_vlmap_stage3_100_recovery_epseed_sync15/vlmap_safety_debug"
vlmap_cfg["debug_max_snapshots"] = 45
vlmap_cfg["debug_sample_total_snapshots"] = 25
vlmap_cfg["debug_force_max_snapshots"] = 10
vlmap_cfg["debug_force_max_snapshots_per_episode"] = 1

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_stage3_100_recovery_epseed_sync15"
eval_cfg.eval_settings["port"] = "2349"

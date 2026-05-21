import copy
import importlib.util
from pathlib import Path


def _load_baseline_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_baseline_fixed_epseed_100_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_baseline_fixed_epseed_100_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_baseline_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V5a is semantic shadow only. Keep all geometry/action interventions disabled
# so behavior remains identical to the fixed-prompt epseed baseline.
vlmap_cfg["enable"] = False
vlmap_cfg["action_safety_enable"] = False
vlmap_cfg["waypoint_check_enable"] = False
vlmap_cfg["waypoint_requery_enable"] = False
vlmap_cfg["waypoint_recovery_enable"] = False
vlmap_cfg["traj_validation_enable"] = False
vlmap_cfg["shadow_only"] = True

vlmap_cfg["semantic_match_enable"] = True
vlmap_cfg["semantic_match_shadow_only"] = True
vlmap_cfg["semantic_match_backend"] = "auto"
vlmap_cfg["semantic_match_device"] = "cpu"
vlmap_cfg["semantic_match_clip_model"] = "ViT-B/32"
vlmap_cfg["semantic_match_model_path"] = "checkpoints/clip-long/longclip-B.pt"
vlmap_cfg["semantic_match_score_threshold"] = 0.20
vlmap_cfg["semantic_match_top_k"] = 3
vlmap_cfg["semantic_match_max_terms"] = 8
vlmap_cfg["semantic_match_use_templates"] = True
vlmap_cfg["semantic_match_save_rgb"] = False

vlmap_cfg["debug"] = True
vlmap_cfg["debug_dir"] = "./logs/habitat/compare_vlmap_stage5_30_semantic_shadow_epseed/vlmap_safety_debug"
vlmap_cfg["debug_log_all_events"] = True
vlmap_cfg["debug_max_snapshots"] = 0
vlmap_cfg["waypoint_save_snapshots"] = False
vlmap_cfg["verbose"] = True

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = "./logs/habitat/compare_vlmap_stage5_30_semantic_shadow_epseed"
eval_cfg.eval_settings["port"] = "2352"

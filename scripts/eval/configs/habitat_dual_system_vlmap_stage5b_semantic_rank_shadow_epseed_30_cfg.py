import copy
import importlib.util
from pathlib import Path


def _load_stage5_shadow_30_cfg():
    cfg_path = Path(__file__).with_name("habitat_dual_system_vlmap_stage5_semantic_shadow_epseed_30_cfg.py")
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage5_shadow_30_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage5_shadow_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V5b keeps navigation shadow-only and evaluates relative semantic ranking
# because V5a showed absolute CLIP thresholds saturate on indoor RGB frames.
vlmap_cfg["semantic_match_device"] = "cuda"
vlmap_cfg["semantic_match_score_threshold"] = 0.20
vlmap_cfg["semantic_match_score_thresholds"] = [0.25, 0.27, 0.29, 0.31]
vlmap_cfg["semantic_match_relative_z_threshold"] = 0.5
vlmap_cfg["semantic_match_margin_threshold"] = 0.01

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage5b_30_semantic_rank_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage5b_30_semantic_rank_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2354"

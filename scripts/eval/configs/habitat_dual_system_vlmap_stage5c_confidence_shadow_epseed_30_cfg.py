import copy
import importlib.util
from pathlib import Path


def _load_stage5b_shadow_30_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage5b_semantic_rank_shadow_epseed_30_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_habitat_dual_system_stage5b_shadow_30_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage5b_shadow_30_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# V5c is still shadow-only. It records when a semantic confidence monitor
# would ask S2 to re-observe/requery, but it does not change actions.
vlmap_cfg["semantic_confidence_policy_enable"] = True
vlmap_cfg["semantic_confidence_policy_shadow_only"] = True
vlmap_cfg["semantic_high_conf_score_threshold"] = 0.31
vlmap_cfg["semantic_low_conf_score_threshold"] = 0.29
vlmap_cfg["semantic_low_conf_streak_threshold"] = 3
vlmap_cfg["semantic_requery_min_step"] = 20
vlmap_cfg["semantic_requery_min_events"] = 3
vlmap_cfg["semantic_requery_max_per_episode"] = 1
vlmap_cfg["semantic_requery_require_no_high_conf"] = True

vlmap_cfg["debug_dir"] = (
    "./logs/habitat/compare_vlmap_stage5c_30_confidence_shadow_epseed/"
    "vlmap_safety_debug"
)

eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = 30
eval_cfg.env.env_settings["episode_ids"] = None

eval_cfg.eval_settings["output_path"] = (
    "./logs/habitat/compare_vlmap_stage5c_30_confidence_shadow_epseed"
)
eval_cfg.eval_settings["port"] = "2356"

"""Stage21a train-split shadow collection for candidate recoverability learning."""

import copy
import importlib.util
import json
import os
from pathlib import Path


# The inherited Stage20g-v2 chain passes through the Stage17 balanced config,
# which validates its legacy manifest during import. Point that compatibility
# variable at the Stage21 manifest before loading the chain; Stage21 then owns
# the final episode selection and run name below.
if os.environ.get("STAGE21_EPISODE_IDS"):
    os.environ.setdefault("STAGE17_EPISODE_IDS", os.environ["STAGE21_EPISODE_IDS"])
if os.environ.get("STAGE21_RUN_NAME"):
    os.environ.setdefault("STAGE17_BALANCED_RUN_NAME", os.environ["STAGE21_RUN_NAME"])


def _load_stage20g_v2_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("_stage20g_v2_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def _load_episode_ids():
    raw_path = os.environ.get("STAGE21_EPISODE_IDS", "").strip()
    if not raw_path:
        raise ValueError("STAGE21_EPISODE_IDS must point to a JSON episode manifest.")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Stage21 episode manifest not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"Stage21 episode manifest must be a non-empty JSON list: {path}")
    return values


def _simple_name(env_name, default):
    value = os.environ.get(env_name, default).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{env_name} must be a non-empty simple directory name, got {value!r}")
    return value


def _debug_run_prefix():
    explicit = os.environ.get("STAGE21_DEBUG_RUN_PREFIX", "").strip()
    if explicit:
        if "/" in explicit or "\\" in explicit:
            raise ValueError("STAGE21_DEBUG_RUN_PREFIX must be a simple file prefix.")
        return explicit
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return "run" if world_size <= 1 else f"rank{int(os.environ.get('RANK', '0'))}_run"


eval_cfg = copy.deepcopy(_load_stage20g_v2_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

run_name = _simple_name(
    "STAGE21_RUN_NAME",
    "compare_vlmap_stage21a_train_recovery_shadow",
)
output_path = f"./logs/habitat/{run_name}"

# Train split is used only to collect frozen-policy shadow trajectories and
# online OccMem features. No model parameter or navigation action is updated.
eval_cfg.env.env_settings["config_path"] = "scripts/eval/configs/vln_r2r_train.yaml"
eval_cfg.env.env_settings["episode_ids"] = _load_episode_ids()
eval_cfg.env.env_settings["episode_start_index"] = 0
eval_cfg.env.env_settings["max_eval_episodes"] = None

vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
vlmap_cfg["debug_run_prefix"] = _debug_run_prefix()

# Stage21a is a data-collection run. Triage is evaluated and logged, while all
# interventions remain suppressed so the frozen S2/NextDiT baseline is intact.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"
] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_enable"] = True
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_require_strict_intervention"
] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = [
    "stuck_collision",
    "s2_turn_loop_obstructed",
    "s2_turn_loop_semantic",
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives"] = [
    "reorient_reobserve",
    "one_safe_forward_reobserve",
    "reobserve",
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_completed_landmark_penalty"] = 1.0

# Save at most one representative RGB + S2 decision JSON per episode when the
# agent exhibits a long repeated-action or low-displacement stagnation window.
vlmap_cfg["stuck_snapshot_enable"] = True
vlmap_cfg["stuck_snapshot_min_step"] = 30
vlmap_cfg["stuck_snapshot_action_window_steps"] = 32
vlmap_cfg["stuck_snapshot_repeat_ratio"] = 0.90

# Detect repeated turn generations across separate frozen-S2 queries. This is
# observation-only: it records a recovery triage event but never changes an
# action queue, pixel goal, or Habitat action.
vlmap_cfg["s2_action_loop_enable"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
vlmap_cfg["s2_action_loop_min_same_turn_generations"] = 5
vlmap_cfg["s2_action_loop_min_cumulative_turn_actions"] = 12
vlmap_cfg["s2_action_loop_min_step_span"] = 6
vlmap_cfg["s2_action_loop_min_episode_step"] = 30
vlmap_cfg["s2_action_loop_max_translation_m"] = 0.35
vlmap_cfg["s2_action_loop_max_snapshots_per_episode"] = 2
vlmap_cfg["occ_memory_candidate_probe_max_events_per_episode"] = 32

# Preserve JSONL evidence but disable high-volume visualization/PLY output for
# overnight runs. These switches materially reduce disk use and serialization.
vlmap_cfg["occ_memory_validation_enable"] = False
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = False
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = False
vlmap_cfg["occ_memory_validation_save_memory_ply"] = False
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = False
vlmap_cfg["occ_memory_save_bev"] = False
vlmap_cfg["occ_memory_candidate_probe_save_bev"] = False

eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE21_EVAL_PORT", "2421")

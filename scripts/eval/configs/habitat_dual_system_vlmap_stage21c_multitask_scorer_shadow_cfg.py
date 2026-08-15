"""Stage21c frozen multi-head scorer online shadow; navigation remains unchanged."""

import copy
import importlib.util
import json
import os
from pathlib import Path


def _load_stage21a_cfg():
    path = Path(__file__).with_name("habitat_dual_system_vlmap_stage21a_train_recovery_shadow_cfg.py")
    spec = importlib.util.spec_from_file_location("_stage21a_shadow_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


checkpoint = os.environ.get("STAGE21C_SCORER_CHECKPOINT", "").strip()
if not checkpoint:
    raise ValueError("STAGE21C_SCORER_CHECKPOINT must point to seed_53/best.pt")
if not Path(checkpoint).is_file():
    raise FileNotFoundError(f"Stage21c scorer checkpoint not found: {checkpoint}")

eval_cfg = copy.deepcopy(_load_stage21a_cfg())
vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Optional explicit replay identity.  The normal Stage21 runs leave this unset
# and retain the rank/local-index seed formula.  A paired replay manifest may
# carry ``episode_eval_seed`` so control and treatment use the same random
# stream even when 4-GPU sharding changes the local episode index.
seed_replay_manifest = os.environ.get("STAGE21_EPISODE_SEED_REPLAY_MANIFEST", "").strip()
if seed_replay_manifest:
    replay_path = Path(seed_replay_manifest)
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"STAGE21_EPISODE_SEED_REPLAY_MANIFEST not found: {replay_path}"
        )
    replay_rows = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(replay_rows, list) or not replay_rows:
        raise ValueError("STAGE21 episode seed replay manifest must be a non-empty list")
    overrides = {}
    for row in replay_rows:
        if not isinstance(row, dict):
            raise ValueError("STAGE21 episode seed replay rows must be objects")
        scene_id = row.get("scene_id")
        episode_id = row.get("episode_id")
        seed = row.get("episode_eval_seed")
        if scene_id is None or episode_id is None or seed is None:
            raise ValueError(
                "STAGE21 replay rows require scene_id, episode_id and episode_eval_seed"
            )
        overrides[f"{scene_id}/{int(episode_id)}"] = int(seed)
    eval_cfg.agent.model_settings["eval_episode_seed_overrides"] = overrides

# The old Stage17 scorer has a different schema and must remain disabled.
vlmap_cfg["occ_memory_progress_ranker_shadow_enable"] = False
vlmap_cfg["occ_memory_stage21_multitask_shadow_enable"] = True
vlmap_cfg["occ_memory_stage21_multitask_shadow_checkpoint"] = checkpoint
vlmap_cfg["occ_memory_stage21_multitask_shadow_device"] = os.environ.get(
    "STAGE21C_SCORER_DEVICE", "cpu"
)

# Defense in depth: Stage21c can score and log, but no active branch can apply.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = True
vlmap_cfg["s2_action_loop_shadow_only"] = True
# Keep Stage21c as the frozen, reproducible control.  Recovery-conditioned
# re-query belongs to the independent Stage21d config.
vlmap_cfg["s2_recovery_context_enable"] = False
vlmap_cfg["s2_loop_strict_active_enable"] = False
vlmap_cfg["stuck_snapshot_enable"] = True
vlmap_cfg["s2_action_loop_max_snapshots_per_episode"] = int(
    os.environ.get("STAGE21C_LOOP_SNAPSHOTS_PER_EPISODE", "3")
)
vlmap_cfg["occ_memory_candidate_probe_max_events_per_episode"] = int(
    os.environ.get("STAGE21C_MAX_EVENTS_PER_EPISODE", "64")
)

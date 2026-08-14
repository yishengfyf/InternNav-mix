"""Stage21c frozen multi-head scorer online shadow; navigation remains unchanged."""

import copy
import importlib.util
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
vlmap_cfg["stuck_snapshot_enable"] = True
vlmap_cfg["s2_action_loop_max_snapshots_per_episode"] = int(
    os.environ.get("STAGE21C_LOOP_SNAPSHOTS_PER_EPISODE", "3")
)
vlmap_cfg["occ_memory_candidate_probe_max_events_per_episode"] = int(
    os.environ.get("STAGE21C_MAX_EVENTS_PER_EPISODE", "64")
)

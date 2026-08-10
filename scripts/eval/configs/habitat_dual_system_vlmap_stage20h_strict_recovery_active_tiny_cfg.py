import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20g_v2_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20g_v2_sparse_semantic_recovery_gate_triage_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20g_v2_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20h is a paired tiny-active validation of the Stage20g-v2 strict tier.
# These safeguards are intentionally not configurable through environment
# variables: adapter candidates and abstain events must remain frozen-S2 holds.
vlmap_cfg["occ_memory_semantic_resilience_active_lite_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_shadow_only"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_evaluate_gate_when_shadow_only"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_v2_evidence_gate_enable"] = True
vlmap_cfg[
    "occ_memory_semantic_resilience_active_lite_v2_evidence_gate_require_strict_intervention"
] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_failure_types"] = [
    "stuck_collision"
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allowed_recommended_primitives"] = [
    "reorient_reobserve",
    "one_safe_forward_reobserve",
]
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_interventions_per_episode"] = 1
vlmap_cfg["occ_memory_semantic_resilience_active_lite_max_turn_steps"] = 2
vlmap_cfg["occ_memory_semantic_resilience_active_lite_forward_steps"] = 0
vlmap_cfg["occ_memory_semantic_resilience_active_lite_allow_forward_to_backtrack"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_append_reobserve_action"] = True
vlmap_cfg["occ_memory_semantic_resilience_active_lite_clear_goal"] = False
vlmap_cfg["occ_memory_semantic_resilience_active_lite_log_all_considered"] = True

run_name = os.environ.get(
    "STAGE20H_TINY_RUN_NAME",
    "compare_vlmap_stage20h_strict_recovery_active_tiny",
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20H_EVAL_PORT", "2414")

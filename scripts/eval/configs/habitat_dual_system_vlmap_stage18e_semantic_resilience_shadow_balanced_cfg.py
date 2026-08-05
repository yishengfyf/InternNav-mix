import copy
import importlib.util
import os
from pathlib import Path


def _load_stage18a_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage18a_s2_candidate_logging_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage18a_s2_candidate_logging_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage18a_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage18e: semantic resilience shadow probe.
#
# Frozen S2 still controls navigation.  OccMem only records whether the current
# sparse metric-semantic memory suggests a broad recovery context: semantic
# stagnation, revisit-loop risk, policy-memory conflict, spatial constriction,
# or limited frontier escape.  A dead-end is only one possible subtype, not the
# definition of the module.  This is the first code step toward "resilient
# navigation": do not replace S2; first verify that the memory can explain and
# propose short recoveries in the right moments.
vlmap_cfg["occ_memory_semantic_resilience_shadow_enable"] = True
vlmap_cfg["occ_memory_semantic_resilience_local_radius_cells"] = int(
    os.environ.get("STAGE18E_RESILIENCE_RADIUS_CELLS", "18")
)
vlmap_cfg["occ_memory_semantic_resilience_min_observed_cells"] = int(
    os.environ.get("STAGE18E_RESILIENCE_MIN_OBSERVED_CELLS", "24")
)
vlmap_cfg["occ_memory_semantic_resilience_occupied_ratio_threshold"] = float(
    os.environ.get("STAGE18E_RESILIENCE_OCC_RATIO", "0.28")
)
vlmap_cfg["occ_memory_semantic_resilience_blocked_bucket_threshold"] = float(
    os.environ.get("STAGE18E_RESILIENCE_BLOCKED_BUCKET", "0.45")
)
vlmap_cfg["occ_memory_semantic_resilience_frontier_escape_threshold"] = int(
    os.environ.get("STAGE18E_RESILIENCE_FRONTIER_ESCAPE", "4")
)
vlmap_cfg["occ_memory_semantic_resilience_min_backtrack_distance_m"] = float(
    os.environ.get("STAGE18E_RESILIENCE_MIN_BACKTRACK_M", "0.75")
)
vlmap_cfg["occ_memory_semantic_resilience_max_backtrack_distance_m"] = float(
    os.environ.get("STAGE18E_RESILIENCE_MAX_BACKTRACK_M", "4.0")
)
vlmap_cfg["occ_memory_semantic_resilience_backtrack_min_step_gap"] = int(
    os.environ.get("STAGE18E_RESILIENCE_BACKTRACK_STEP_GAP", "6")
)
vlmap_cfg["occ_memory_semantic_resilience_backtrack_score_weight"] = float(
    os.environ.get("STAGE18E_RESILIENCE_SCORE_WEIGHT", "1.35")
)
vlmap_cfg["occ_memory_semantic_resilience_candidate_source_score"] = float(
    os.environ.get("STAGE18E_RESILIENCE_SOURCE_SCORE", "2.40")
)

# Give the shadow candidate one stable slot, but keep the event compact enough
# for 4-GPU collection and later GT label audits.
vlmap_cfg["occ_memory_candidate_probe_max_candidates"] = int(
    os.environ.get("STAGE18E_MAX_CANDIDATES", "5")
)

# Stage18e does not need S2 to answer the candidate-selection prompt; disabling
# it keeps the run closer to pure logging and avoids extra LLM call latency.
vlmap_cfg["occ_memory_candidate_selection_enable"] = False

run_name = os.environ.get(
    "STAGE18E_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE18_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage18e_semantic_resilience_shadow_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE18E_EVAL_PORT", "2395")

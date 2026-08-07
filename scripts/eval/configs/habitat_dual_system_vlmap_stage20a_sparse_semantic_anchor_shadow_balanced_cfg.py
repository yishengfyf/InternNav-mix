import copy
import importlib.util
import os
from pathlib import Path


def _load_stage19b_shadow_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage19b_semantic_resilience_shadow_taxonomy_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage19b_semantic_resilience_shadow_taxonomy_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage19b_shadow_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20a: sparse 3D semantic anchors.
#
# This keeps frozen S2/NextDiT in control and preserves the Stage19b failure
# taxonomy shadow.  The new part is only memory-side logging: high-confidence
# image-level semantic terms are projected from a few sparse RGB-D sources
# (S2 pixel_goal + view center by default) into the existing sparse 3D
# occupancy memory.  This is the first verifiable step toward online sparse
# semantic OCC without paying the cost of dense segmentation or dense VLMaps.
vlmap_cfg["occ_memory_semantic_anchor_enable"] = (
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_ENABLE", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_anchor_min_score"] = float(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_MIN_SCORE", "0.20")
)
vlmap_cfg["occ_memory_semantic_anchor_max_terms_per_event"] = int(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_MAX_TERMS", "3")
)
vlmap_cfg["occ_memory_semantic_anchor_include_threshold_hits"] = (
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_THRESHOLD_HITS", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_anchor_include_pixel_goal"] = (
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_PIXEL_GOAL", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_anchor_include_view_center"] = (
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_VIEW_CENTER", "1") != "0"
)
vlmap_cfg["occ_memory_semantic_anchor_view_center_x"] = float(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_VIEW_CENTER_X", "0.50")
)
vlmap_cfg["occ_memory_semantic_anchor_view_center_y"] = float(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_VIEW_CENTER_Y", "0.56")
)
vlmap_cfg["occ_memory_semantic_anchor_merge_radius_cells"] = int(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_MERGE_RADIUS_CELLS", "6")
)
vlmap_cfg["occ_memory_semantic_anchor_local_radius_cells"] = int(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_LOCAL_RADIUS_CELLS", "6")
)
vlmap_cfg["occ_memory_semantic_anchor_max_anchors_per_episode"] = int(
    os.environ.get("STAGE20_SEMANTIC_ANCHOR_MAX_EPISODE", "256")
)

# Keep candidate semantic retrieval on so anchors can enter the same semantic
# node interface as earlier pose/keyframe semantic events.  Navigation remains
# unchanged because this config is shadow-only.
vlmap_cfg["occ_memory_candidate_probe_semantic_enable"] = True
vlmap_cfg["occ_memory_candidate_probe_semantic_high_conf_only"] = False
vlmap_cfg["occ_memory_candidate_probe_semantic_min_score"] = float(
    os.environ.get("STAGE20_CANDIDATE_SEMANTIC_MIN_SCORE", "0.20")
)

visual_validation = os.environ.get("STAGE20_VIS_VALIDATION", "0") == "1"
vlmap_cfg["occ_memory_validation_enable"] = visual_validation
vlmap_cfg["occ_memory_validation_every_updates"] = int(
    os.environ.get("STAGE20_VIS_VALIDATION_EVERY", "20")
)
vlmap_cfg["occ_memory_validation_max_snapshots"] = int(
    os.environ.get("STAGE20_VIS_VALIDATION_MAX", "3")
)
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = visual_validation
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = visual_validation
vlmap_cfg["occ_memory_validation_save_memory_ply"] = visual_validation
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = visual_validation

run_name = os.environ.get(
    "STAGE20_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE19B_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20a_sparse_semantic_anchor_shadow_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20_EVAL_PORT", "2399")

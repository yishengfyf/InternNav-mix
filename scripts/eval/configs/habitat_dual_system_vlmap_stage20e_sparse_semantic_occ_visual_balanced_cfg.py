import copy
import importlib.util
import os
from pathlib import Path


def _load_stage20c_cfg():
    cfg_path = Path(__file__).with_name(
        "habitat_dual_system_vlmap_stage20c_sparse_semantic_anchor_audit_balanced_cfg.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_habitat_dual_system_vlmap_stage20c_sparse_semantic_anchor_audit_balanced_cfg",
        cfg_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


eval_cfg = copy.deepcopy(_load_stage20c_cfg())

vlmap_cfg = eval_cfg.agent.model_settings["vlmap_safety"]

# Stage20e: visual-only sparse 3D semantic occupancy export.
#
# Keep the Stage20c behavior unchanged and enable validation snapshots so a
# small selected episode set can return real PLY point clouds:
# occupied/free/frontier/pose/keyframe/semantic-anchor points share one memory
# cloud, while RGB/depth snapshots make the projection auditable.
vlmap_cfg["occ_memory_validation_enable"] = True
vlmap_cfg["occ_memory_validation_every_updates"] = int(os.environ.get("STAGE20E_VALIDATION_EVERY_UPDATES", "40"))
vlmap_cfg["occ_memory_validation_max_snapshots"] = int(os.environ.get("STAGE20E_VALIDATION_MAX_SNAPSHOTS", "2"))
vlmap_cfg["occ_memory_validation_save_rgb_depth"] = True
vlmap_cfg["occ_memory_validation_save_current_rgb_ply"] = True
vlmap_cfg["occ_memory_validation_save_memory_ply"] = True
vlmap_cfg["occ_memory_validation_save_final_memory_ply"] = True
vlmap_cfg["occ_memory_validation_current_depth_sample_rate"] = int(
    os.environ.get("STAGE20E_CURRENT_DEPTH_SAMPLE_RATE", "10")
)
vlmap_cfg["occ_memory_validation_max_current_points"] = int(
    os.environ.get("STAGE20E_MAX_CURRENT_POINTS", "80000")
)
vlmap_cfg["occ_memory_validation_max_memory_points"] = int(
    os.environ.get("STAGE20E_MAX_MEMORY_POINTS", "120000")
)
vlmap_cfg["occ_memory_validation_max_occupied_points"] = int(
    os.environ.get("STAGE20E_MAX_OCCUPIED_POINTS", "50000")
)
vlmap_cfg["occ_memory_validation_max_free_points"] = int(
    os.environ.get("STAGE20E_MAX_FREE_POINTS", "40000")
)
vlmap_cfg["occ_memory_validation_max_frontier_points"] = int(
    os.environ.get("STAGE20E_MAX_FRONTIER_POINTS", "16000")
)

vlmap_cfg["occ_memory_max_bev_snapshots"] = int(os.environ.get("STAGE20E_MAX_BEV_SNAPSHOTS", "6"))
vlmap_cfg["occ_memory_candidate_probe_max_bev_snapshots"] = int(
    os.environ.get("STAGE20E_MAX_CANDIDATE_BEV_SNAPSHOTS", "6")
)

run_name = os.environ.get(
    "STAGE20E_BALANCED_RUN_NAME",
    os.environ.get(
        "STAGE20_BALANCED_RUN_NAME",
        os.environ.get(
            "STAGE17_BALANCED_RUN_NAME",
            "compare_vlmap_stage20e_sparse_semantic_occ_visual_balanced",
        ),
    ),
)
output_path = f"./logs/habitat/{run_name}"
vlmap_cfg["debug_dir"] = f"{output_path}/vlmap_safety_debug"
eval_cfg.eval_settings["output_path"] = output_path
eval_cfg.eval_settings["port"] = os.environ.get("STAGE20E_EVAL_PORT", "2406")

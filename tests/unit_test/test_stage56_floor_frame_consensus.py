import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory_module = _load("sparse_occ_memory_stage56", "internnav/utils/sparse_occ_memory.py")
audit_module = _load(
    "stage56_floor_relative_occ_audit",
    "internnav/utils/stage56_floor_relative_occ_audit.py",
)
analyzer_module = _load(
    "analyze_stage56_floor_frame_consensus",
    "scripts/eval/analyze_stage56_floor_frame_consensus.py",
)


def _memory():
    memory = memory_module.SparseOccSemanticMemory(
        {
            "occ_memory_enable": True,
            "occ_memory_cell_size": 0.05,
            "occ_memory_map_height": 3.0,
            "occ_memory_frame_observation_mask_audit_enable": True,
        }
    )
    memory.observation_count = 4
    memory.frame_observation_metadata = {
        1: {"view_source": "forward", "requested_camera_pitch_deg": 0.0},
        2: {"view_source": "forward", "requested_camera_pitch_deg": 0.0},
        3: {"view_source": "lookdown", "requested_camera_pitch_deg": 30.0},
        4: {"view_source": "forward", "requested_camera_pitch_deg": 0.0},
    }
    return memory


def test_floor_relative_frame_consensus_deduplicates_frames_across_heights():
    memory = _memory()
    memory.occ_counts[(10, 10, 4)] = 8
    memory.occ_counts[(10, 10, 5)] = 6
    memory.occ3d_frame_masks[(10, 10, 4)] = (1 << 1) | (1 << 2)
    memory.occ3d_frame_masks[(10, 10, 5)] = (1 << 2)
    evidence = memory.floor_relative_frame_cell_evidence(
        10, 10, 0.0, height_max_m=1.5
    )
    assert evidence["occupied_frame_count"] == 2
    assert evidence["state"] == "blocked"
    assert evidence["last_occupied_observation"] == 2


def test_floor_relative_frame_consensus_requires_recent_multiframe_free():
    memory = _memory()
    memory.occ3d_frame_masks[(10, 10, 4)] = 1 << 1
    memory.free3d_frame_masks[(10, 10, 4)] = (1 << 2) | (1 << 3)
    evidence = memory.floor_relative_frame_cell_evidence(
        10, 10, 0.0, height_max_m=1.5
    )
    assert evidence["state"] == "free"
    assert evidence["free_frame_count"] == 2
    assert evidence["last_free_metadata"]["view_source"] == "lookdown"


def test_height_bin_consensus_does_not_merge_single_frames_from_different_heights():
    memory = _memory()
    memory.occ3d_frame_masks[(10, 10, 4)] = 1 << 1
    memory.occ3d_frame_masks[(10, 10, 5)] = 1 << 2
    evidence = memory.floor_relative_height_bin_frame_cell_evidence(
        10, 10, 0.0, height_max_m=1.5
    )
    assert evidence["occupied_frame_count"] == 2
    assert evidence["blocked_height_bin_count"] == 0
    assert evidence["state"] == "unknown"


def test_candidate_audit_is_read_only_and_uses_same_height_scope():
    memory = _memory()
    for col in (10, 11, 12):
        key = (10, col, 4)
        memory.free_counts[key] = 4
        memory.free3d_frame_masks[key] = (1 << 2) | (1 << 3)
    candidate = {
        "candidate_id": "route:1",
        "path_cells": [[10, 10], [10, 11], [10, 12]],
        "trigger_grid": [10, 10],
        "floor_z_m": 0.0,
        "floor_aligned_height_max_m": 1.5,
        "footprint_radius_m": 0.01,
    }
    report = audit_module.audit_candidate_floor_relative_frames(memory, candidate)
    assert report["decision_applied"] is False
    assert report["unknown_is_free"] is False
    assert report["frame_consensus_scope"] == "same_floor_relative_height_band"
    assert report["corridor_summary_is_complete"] is True
    assert report["complete_corridor_frame_consensus_free"] is True


def test_candidate_audit_keeps_unobserved_side_footprint_unknown():
    memory = _memory()
    for col in (10, 11, 12):
        key = (10, col, 4)
        memory.free_counts[key] = 4
        memory.free3d_frame_masks[key] = (1 << 2) | (1 << 3)
    report = audit_module.audit_candidate_floor_relative_frames(
        memory,
        {
            "candidate_id": "route:side-unknown",
            "path_cells": [[10, 10], [10, 11], [10, 12]],
            "trigger_grid": [10, 10],
            "floor_z_m": 0.0,
            "floor_aligned_height_max_m": 1.5,
            "footprint_radius_m": 0.05,
        },
    )
    assert report["complete_corridor_frame_consensus_free"] is False
    assert report["frame_consensus_state_counts"]["unknown"] > 0


def test_analyzer_requires_same_seed_and_reports_vertical_gain(tmp_path):
    manifest = [
        {
            "scene_id": "flat",
            "episode_id": 1,
            "episode_eval_seed": 11,
            "audit_role": "flat_control_0m",
        },
        {
            "scene_id": "stairs",
            "episode_id": 2,
            "episode_eval_seed": 12,
            "audit_role": "stairs_height_change",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    progress_dir = tmp_path / "vlmap_safety_debug" / "rank0_run_001"
    progress_dir.mkdir(parents=True)

    def branch(recall, route_recall, false_free=0.01):
        return {
            "valid": True,
            "readout_mode": "legacy_any_hit",
            "decision_applied": False,
            "frame_masks_available": True,
            "free_metrics_observed_domain": {
                "precision": 0.95,
                "recall": recall,
            },
            "false_free_rate": false_free,
            "unknown_coverage": 0.1,
            "executed_route_predicted_free_recall": route_recall,
            "historical_edge_strict_free_recall": route_recall,
        }

    rows = []
    for item in manifest:
        legacy = branch(0.7, 0.7)
        consensus = branch(0.8, 0.8)
        consensus["readout_mode"] = "floor_frame_consensus"
        rows.append(
            {
                **item,
                "stage23a_gt_fields_used_for_navigation": [],
                "stage23b_navmesh_traversability_current_clearance": legacy,
                "stage56_navmesh_current_floor_frame_consensus": consensus,
            }
        )
    (progress_dir / "progress.json").write_text(
        "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = analyzer_module.analyze(tmp_path, manifest_path)
    assert report["integrity_passed"] is True
    assert report["gate"]["stage57_shadow_ready"] is True

import json
from pathlib import Path

from scripts.eval.analyze_stage30_base_confirmation import (
    confirm,
    detector_audit,
    geometry_audit,
    semantic_audit,
)


def test_geometry_reports_vertical_pose_limitation():
    payload = {
        "audit_name": "stage23a_pose_occ",
        "integrity_passed": True,
        "episodes": [
            {
                "audit_role": "stairs_height_change",
                "comparison": {
                    "occupied_tolerance": {"precision": 0.9, "recall": 0.8},
                    "free_exact": {"precision": 0.99, "recall": 0.7},
                    "unknown_coverage": 0.2,
                    "false_free_rate": 0.1,
                },
                "navmesh_traversability_current_clearance": {
                    "sampled_cell_count": 10,
                    "predicted_unknown_cell_count": 2,
                    "historical_edge_strict_free_recall": 0.3,
                    "executed_route_predicted_free_recall": 0.4,
                    "false_free_rate": 0.2,
                    "free_metrics_observed_domain": {"precision": 0.8, "recall": 0.4},
                },
                "navmesh_traversability_oracle_sensor_clearance": {
                    "sampled_cell_count": 10,
                    "predicted_unknown_cell_count": 1,
                    "historical_edge_strict_free_recall": 0.8,
                    "executed_route_predicted_free_recall": 0.8,
                    "false_free_rate": 0.1,
                    "free_metrics_observed_domain": {"precision": 0.9, "recall": 0.8},
                },
            }
        ],
    }
    report = geometry_audit(payload)
    assert report["vertical_episode_count"] == 1
    assert report["vertical_geometry_status"] == "evidence_limited_current_2d_pose"
    assert report["vertical_pose_oracle_gap"]["historical_edge_recall_delta_current_minus_oracle"] < 0


def test_semantic_shadow_acceptance_and_detector_freeze():
    semantic = semantic_audit({
        "audit_name": "stage26_hsgm_semantic_filter",
        "integrity_passed": True,
        "episode_count": 48,
        "all_trajectories_exact_match": True,
        "all_raw_semantics_exact_match": True,
        "evidence_gate": {"passed": True},
        "delta": {"severe_conflict_reduction_rate": 0.28},
    })
    detector = detector_audit({
        "contract_passed": True,
        "future_used_by_detector": False,
        "outcome_is_event_gt": False,
        "manifest_expected_count": 500,
        "manifest_verified_count": 500,
    })
    result = confirm(
        {"online_geometry_backend_changed": False},
        {"online_semantic_backend_changed": False},
        detector,
    )
    assert semantic["filtered_shadow_accepted"] is True
    assert result["detector_revalidation_required"] is False
    assert result["detector_frozen_confirmation"] is True
    assert result["unknown_is_free"] is False

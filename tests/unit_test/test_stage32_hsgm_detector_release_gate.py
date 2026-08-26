from scripts.eval.analyze_stage32_hsgm_detector_release_gate import (
    build_report,
    stage31_contract,
)


def _stage31(*, readable_depth: int) -> dict:
    return {
        "schema_version": "stage31_hsgm_yoloe_replay_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "online_geometry_backend_changed": False,
        "online_semantic_backend_changed": False,
        "sparse_occ_is_only_safety_authority": True,
        "hsgm_surface_is_traversability_truth": False,
        "detector_revalidation_required": False,
        "online_gt_fields_used": [],
        "audit_gt_fields_used": [
            "semantic_scene_gt",
            "coordinate_transforms.map_to_habitat_world",
        ],
        "input": {
            "episode_count": 2,
            "query_frame_count": 6,
            "readable_depth_frame_count": readable_depth,
            "checkpoint_sha256": "a" * 64,
            "mobileclip_sha256": "b" * 64,
            "model_confidence": 0.6,
            "model_iou": 0.3,
            "query_frames_only": True,
        },
        "integrity": {
            "manifest_count_matches_query_frame_count": True,
            "depth_hash_mismatch_count": 0,
        },
        "variants": {},
        "paired_lseg": {},
    }


def _stage30() -> dict:
    return {
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "confirmation": {
            "detector_revalidation_required": False,
            "detector_frozen_confirmation": True,
        },
    }


def _stage25() -> dict:
    return {
        "contract_passed": True,
        "manifest_expected_count": 500,
        "manifest_verified_count": 500,
        "future_used_by_detector": False,
        "outcome_is_event_gt": False,
    }


def _event_gt() -> dict:
    row = {
        "event_precision_on_adjudicated": 0.9,
        "combined_confirmed_recall": 0.8,
    }
    return {
        "future_used_by_detector": False,
        "outcome_is_event_gt": False,
        "objective_manifest_count": 161,
        "detectors": {"D2_ER_selected": {"all": row, "holdout": row}},
    }


def test_stage31_lossless_requires_all_depth_frames():
    assert stage31_contract(_stage31(readable_depth=6), lossless_depth=True)["passed"]
    report = stage31_contract(_stage31(readable_depth=5), lossless_depth=True)
    assert not report["passed"]
    assert "lossless_depth_complete" in report["failed_checks"]


def test_release_freezes_detector_when_all_shadow_contracts_pass():
    report = build_report(
        _stage30(), _stage31(readable_depth=6), _stage31(readable_depth=0),
        _stage25(), _event_gt(),
    )
    assert report["gates"]["release"]["passed"]
    assert report["stage2_decision"]["detector_revalidation_required"] is False
    assert report["stage2_decision"]["detector_frozen_confirmation"] is True
    assert report["stage2_decision"]["rerun_500_navigation_episodes"] is False


def test_online_backend_change_blocks_release():
    changed = _stage31(readable_depth=6)
    changed["online_geometry_backend_changed"] = True
    report = build_report(
        _stage30(), changed, _stage31(readable_depth=0), _stage25(), _event_gt(),
    )
    assert not report["gates"]["release"]["passed"]
    assert report["stage2_decision"]["detector_revalidation_required"] is True

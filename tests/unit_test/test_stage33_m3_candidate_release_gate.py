from scripts.eval.analyze_stage33_m3_candidate_release_gate import build_report


SAFE_STAGE = "route_occ_clearance_frontier"


def _candidate(event_count: int, candidate_count: int) -> dict:
    final = {
        "event_count": event_count,
        "event_coverage": 0.5,
        "candidate_count": candidate_count,
        "route_occ_conflict_count": 0,
        "unknown_fraction_max": 0.0,
        "occupied_fraction_mean": 0.0,
        "clearance_failure_count": 0,
        "action_applied_count": 0,
        "gt_fields_used_union": [],
        "floor_aligned_known_free_count": candidate_count,
    }
    return {
        "task": "stage27_m3_candidate_generation_shadow_audit",
        "event_count": event_count,
        "integrity_passed": True,
        "unknown_is_free": False,
        "active_recovery_enabled": False,
        "ranker_trained": False,
        "success_is_event_gt": False,
        "reports": {SAFE_STAGE: final},
        "integrity_checks": {
            "schema_v5_or_v6": True,
            "route_pool_contract": True,
            "frontier_pool_contract": True,
            "frontier_geodesic_standoff_contract": True,
            "semantic_proposal_safety_contract": True,
            "shadow_only": True,
            "no_action": True,
            "no_gt_fields": True,
            "candidate_records_shadow_only": True,
        },
        "manifest_candidate_coverage": {
            "unique_expected_event_count": event_count,
            "observed_exact_event_count": event_count,
            "missing_expected_event_count": 0,
            "unexpected_event_count": 0,
            "all": {
                "emitted_event_recall": 1.0,
                "reports": {
                    SAFE_STAGE: {
                        "candidate_count": candidate_count,
                        "event_coverage": 0.5,
                        "multi_candidate_event_count": 3,
                    }
                },
            },
            "by_gt_split": {},
        },
    }


def _stage32() -> dict:
    return {
        "action_applied_count": 0,
        "unknown_is_free": False,
        "online_gt_fields_used": [],
        "gates": {"release": {"passed": True, "failed_checks": []}},
        "stage1_decision": {
            "sparse_occ_online_geometry": "unchanged_frozen",
            "semantic_backend": "retain_stage26_filtered_lseg",
        },
        "stage2_decision": {
            "detector_revalidation_required": False,
            "detector_frozen_confirmation": True,
        },
    }


def _stratified() -> dict:
    payload = _candidate(62, 36)
    payload["semantic_metrics"] = {
        branch: {
            "semantic_triggered_event_count": 28,
            "proposed_candidate_count": 81,
            "safe_proposed_candidate_count": 0,
            "increment_candidate_count": 0,
        }
        for branch in ("raw", "filtered")
    }
    reports = payload["manifest_candidate_coverage"]["all"]["reports"]
    for branch in ("raw", "filtered"):
        reports[f"{SAFE_STAGE}_semantic_{branch}"] = {
            "candidate_count": 36,
            "event_coverage": 0.5,
        }
    return payload


def _m4(holdout_multi: int = 3) -> dict:
    return {
        "integrity_passed": True,
        "event_manifest_count": 119,
        "observed_event_count": 119,
        "missing_manifest_event_count": 0,
        "unsafe_candidate_record_count": 0,
        "executor_called": False,
        "active_recovery_enabled": False,
        "ranker_trained": False,
        "multi_candidate_event_count": 25,
        "by_gt_split": {"holdout": {"composite": {"event_count": holdout_multi}}},
    }


def test_release_passes_only_with_exact_current_head_smoke():
    report = build_report(
        _stage32(), _candidate(2, 3), _candidate(119, 75), _stratified(), _m4()
    )
    assert report["gates"]["release"]["passed"]
    assert report["decision"]["m3_candidate_generation_frozen"] is True
    assert report["decision"]["rerun_all119_full500"] is False
    assert report["decision"]["ranker_training"] is False
    assert report["decision"]["next_stage"] == "stage29_candidate_executor_contract"


def test_smoke_manifest_mismatch_blocks_release():
    smoke = _candidate(2, 3)
    smoke["manifest_candidate_coverage"]["observed_exact_event_count"] = 1
    smoke["manifest_candidate_coverage"]["missing_expected_event_count"] = 1
    report = build_report(
        _stage32(), smoke, _candidate(119, 75), _stratified(), _m4()
    )
    assert not report["gates"]["release"]["passed"]
    assert report["decision"]["m3_candidate_generation_frozen"] is False
    assert report["decision"]["next_stage"] == "m3_release_blocked"


def test_semantic_safe_increment_blocks_release():
    stratified = _stratified()
    stratified["semantic_metrics"]["filtered"]["safe_proposed_candidate_count"] = 1
    stratified["semantic_metrics"]["filtered"]["increment_candidate_count"] = 1
    report = build_report(
        _stage32(), _candidate(2, 3), _candidate(119, 75), stratified, _m4()
    )
    assert not report["gates"]["release"]["passed"]
    assert "filtered_safe_increment_zero" in report["gates"]["stratified_semantic"]["failed_checks"]


def test_ranker_gate_reopens_at_five_holdout_multi_events():
    report = build_report(
        _stage32(), _candidate(2, 3), _candidate(119, 75), _stratified(), _m4(5)
    )
    assert not report["gates"]["release"]["passed"]
    assert report["gates"]["m4"]["ranker_decision"] == "reopen_ranker_review"

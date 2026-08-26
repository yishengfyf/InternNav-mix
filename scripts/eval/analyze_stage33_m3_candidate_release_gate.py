#!/usr/bin/env python3
"""Close the frozen M3 candidate-generation validation gate.

This analyzer only reads existing structured reports. It does not import the
evaluator, regenerate candidates, train a ranker, or emit navigation actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFE_STAGE = "route_occ_clearance_frontier"
SEMANTIC_STAGES = (
    "route_occ_clearance_frontier_semantic_raw",
    "route_occ_clearance_frontier_semantic_filtered",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _check(name: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed}


def _finish(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row["name"] for row in checks if not row["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}


def candidate_contract(
    payload: dict[str, Any], *, expected_event_count: int
) -> dict[str, Any]:
    coverage = payload.get("manifest_candidate_coverage") or {}
    all_coverage = coverage.get("all") or {}
    final_report = (payload.get("reports") or {}).get(SAFE_STAGE) or {}
    integrity = payload.get("integrity_checks") or {}
    candidate_count = int(final_report.get("candidate_count", -1) or 0)
    checks = [
        _check(
            "candidate_audit_task",
            payload.get("task") == "stage27_m3_candidate_generation_shadow_audit",
            payload.get("task"),
        ),
        _check("integrity_passed", payload.get("integrity_passed") is True, payload.get("integrity_passed")),
        _check("unknown_is_not_free", payload.get("unknown_is_free") is False, payload.get("unknown_is_free")),
        _check("active_recovery_disabled", payload.get("active_recovery_enabled") is False, payload.get("active_recovery_enabled")),
        _check("ranker_not_trained", payload.get("ranker_trained") is False, payload.get("ranker_trained")),
        _check("success_is_not_event_gt", payload.get("success_is_event_gt") is False, payload.get("success_is_event_gt")),
        _check("event_count", payload.get("event_count") == expected_event_count, payload.get("event_count")),
        _check("manifest_event_count", coverage.get("unique_expected_event_count") == expected_event_count, coverage.get("unique_expected_event_count")),
        _check("manifest_observed_exact", coverage.get("observed_exact_event_count") == expected_event_count, coverage.get("observed_exact_event_count")),
        _check("manifest_missing_zero", coverage.get("missing_expected_event_count") == 0, coverage.get("missing_expected_event_count")),
        _check("manifest_unexpected_zero", coverage.get("unexpected_event_count") == 0, coverage.get("unexpected_event_count")),
        _check("emitted_event_recall_complete", all_coverage.get("emitted_event_recall") == 1.0, all_coverage.get("emitted_event_recall")),
        _check("schema_contract", integrity.get("schema_v5_or_v6") is True, integrity.get("schema_v5_or_v6")),
        _check("route_pool_contract", integrity.get("route_pool_contract") is True, integrity.get("route_pool_contract")),
        _check("frontier_pool_contract", integrity.get("frontier_pool_contract") is True, integrity.get("frontier_pool_contract")),
        _check("frontier_geodesic_contract", integrity.get("frontier_geodesic_standoff_contract") is True, integrity.get("frontier_geodesic_standoff_contract")),
        _check("semantic_safety_contract", integrity.get("semantic_proposal_safety_contract") is True, integrity.get("semantic_proposal_safety_contract")),
        _check("shadow_only", integrity.get("shadow_only") is True, integrity.get("shadow_only")),
        _check("no_action", integrity.get("no_action") is True, integrity.get("no_action")),
        _check("no_gt_fields", integrity.get("no_gt_fields") is True, integrity.get("no_gt_fields")),
        _check("candidate_records_shadow_only", integrity.get("candidate_records_shadow_only") is True, integrity.get("candidate_records_shadow_only")),
        _check("final_route_occ_conflict_zero", final_report.get("route_occ_conflict_count") == 0, final_report.get("route_occ_conflict_count")),
        _check("final_unknown_zero", final_report.get("unknown_fraction_max") == 0.0, final_report.get("unknown_fraction_max")),
        _check("final_occupied_zero", final_report.get("occupied_fraction_mean") == 0.0, final_report.get("occupied_fraction_mean")),
        _check("final_clearance_failure_zero", final_report.get("clearance_failure_count") == 0, final_report.get("clearance_failure_count")),
        _check("final_action_zero", final_report.get("action_applied_count") == 0, final_report.get("action_applied_count")),
        _check("final_gt_fields_empty", final_report.get("gt_fields_used_union") == [], final_report.get("gt_fields_used_union")),
        _check("all_candidates_floor_aligned_known_free", final_report.get("floor_aligned_known_free_count") == candidate_count, {"known_free": final_report.get("floor_aligned_known_free_count"), "candidate_count": candidate_count}),
    ]
    return {
        **_finish(checks),
        "event_count": payload.get("event_count"),
        "candidate_count": candidate_count,
        "event_coverage": final_report.get("event_coverage"),
        "multi_candidate_event_count": ((all_coverage.get("reports") or {}).get(SAFE_STAGE) or {}).get("multi_candidate_event_count"),
        "by_gt_split": coverage.get("by_gt_split") or {},
    }


def semantic_contract(payload: dict[str, Any]) -> dict[str, Any]:
    coverage = payload.get("manifest_candidate_coverage") or {}
    reports = ((coverage.get("all") or {}).get("reports") or {})
    base = reports.get(SAFE_STAGE) or {}
    semantic_metrics = payload.get("semantic_metrics") or {}
    checks = [
        _check("stratified_integrity", payload.get("integrity_passed") is True, payload.get("integrity_passed")),
        _check("stratified_exact_62", coverage.get("observed_exact_event_count") == 62 and coverage.get("unique_expected_event_count") == 62, {"expected": coverage.get("unique_expected_event_count"), "observed": coverage.get("observed_exact_event_count")}),
        _check("stratified_missing_zero", coverage.get("missing_expected_event_count") == 0, coverage.get("missing_expected_event_count")),
        _check("stratified_unexpected_zero", coverage.get("unexpected_event_count") == 0, coverage.get("unexpected_event_count")),
        _check("stratified_unknown_is_not_free", payload.get("unknown_is_free") is False, payload.get("unknown_is_free")),
        _check("stratified_no_active", payload.get("active_recovery_enabled") is False, payload.get("active_recovery_enabled")),
        _check("stratified_no_ranker", payload.get("ranker_trained") is False, payload.get("ranker_trained")),
    ]
    for branch, stage in zip(("raw", "filtered"), SEMANTIC_STAGES):
        metrics = semantic_metrics.get(branch) or {}
        stage_report = reports.get(stage) or {}
        checks.extend([
            _check(f"{branch}_semantic_triggered", int(metrics.get("semantic_triggered_event_count", 0) or 0) > 0, metrics.get("semantic_triggered_event_count")),
            _check(f"{branch}_semantic_proposed", int(metrics.get("proposed_candidate_count", 0) or 0) > 0, metrics.get("proposed_candidate_count")),
            _check(f"{branch}_safe_increment_zero", metrics.get("safe_proposed_candidate_count") == 0 and metrics.get("increment_candidate_count") == 0, {"safe": metrics.get("safe_proposed_candidate_count"), "increment": metrics.get("increment_candidate_count")}),
            _check(f"{branch}_does_not_change_safe_pool", stage_report.get("candidate_count") == base.get("candidate_count") and stage_report.get("event_coverage") == base.get("event_coverage"), {"base": {"candidates": base.get("candidate_count"), "coverage": base.get("event_coverage")}, "semantic": {"candidates": stage_report.get("candidate_count"), "coverage": stage_report.get("event_coverage")}}),
        ])
    return {**_finish(checks), "semantic_metrics": semantic_metrics}


def m4_contract(payload: dict[str, Any]) -> dict[str, Any]:
    holdout = payload.get("by_gt_split") or {}
    holdout_rules = holdout.get("holdout") or {}
    holdout_multi = int((holdout_rules.get("composite") or {}).get("event_count", 0) or 0)
    checks = [
        _check("m4_integrity", payload.get("integrity_passed") is True, payload.get("integrity_passed")),
        _check("m4_manifest_119", payload.get("event_manifest_count") == 119 and payload.get("observed_event_count") == 119, {"manifest": payload.get("event_manifest_count"), "observed": payload.get("observed_event_count")}),
        _check("m4_missing_zero", payload.get("missing_manifest_event_count") == 0, payload.get("missing_manifest_event_count")),
        _check("m4_unsafe_zero", payload.get("unsafe_candidate_record_count") == 0, payload.get("unsafe_candidate_record_count")),
        _check("m4_executor_not_called", payload.get("executor_called") is False, payload.get("executor_called")),
        _check("m4_active_disabled", payload.get("active_recovery_enabled") is False, payload.get("active_recovery_enabled")),
        _check("m4_ranker_not_trained", payload.get("ranker_trained") is False, payload.get("ranker_trained")),
        _check("ranker_holdout_gate_not_met", holdout_multi < 5, holdout_multi),
    ]
    return {
        **_finish(checks),
        "multi_candidate_event_count": payload.get("multi_candidate_event_count"),
        "holdout_multi_candidate_event_count": holdout_multi,
        "ranker_decision": "no_go_insufficient_holdout_multi_events" if holdout_multi < 5 else "reopen_ranker_review",
    }


def stage32_contract(payload: dict[str, Any]) -> dict[str, Any]:
    release = (payload.get("gates") or {}).get("release") or {}
    stage1 = payload.get("stage1_decision") or {}
    stage2 = payload.get("stage2_decision") or {}
    checks = [
        _check("stage32_release_passed", release.get("passed") is True, release.get("failed_checks")),
        _check("stage32_unknown_is_not_free", payload.get("unknown_is_free") is False, payload.get("unknown_is_free")),
        _check("stage32_no_action", payload.get("action_applied_count") == 0, payload.get("action_applied_count")),
        _check("stage32_online_gt_empty", payload.get("online_gt_fields_used") == [], payload.get("online_gt_fields_used")),
        _check("stage32_sparse_occ_frozen", stage1.get("sparse_occ_online_geometry") == "unchanged_frozen", stage1.get("sparse_occ_online_geometry")),
        _check("stage32_filtered_lseg_retained", stage1.get("semantic_backend") == "retain_stage26_filtered_lseg", stage1.get("semantic_backend")),
        _check("stage32_detector_frozen", stage2.get("detector_revalidation_required") is False and stage2.get("detector_frozen_confirmation") is True, {"revalidation": stage2.get("detector_revalidation_required"), "frozen": stage2.get("detector_frozen_confirmation")}),
    ]
    return _finish(checks)


def build_report(
    stage32: dict[str, Any],
    smoke: dict[str, Any],
    all119: dict[str, Any],
    stratified: dict[str, Any],
    m4: dict[str, Any],
) -> dict[str, Any]:
    base_gate = stage32_contract(stage32)
    smoke_gate = candidate_contract(smoke, expected_event_count=2)
    all119_gate = candidate_contract(all119, expected_event_count=119)
    semantic_gate = semantic_contract(stratified)
    ranking_gate = m4_contract(m4)
    release = _finish([
        _check("stage32_base_release", base_gate["passed"], base_gate["failed_checks"]),
        _check("current_head_smoke", smoke_gate["passed"], smoke_gate["failed_checks"]),
        _check("frozen_all119", all119_gate["passed"], all119_gate["failed_checks"]),
        _check("stratified_semantic", semantic_gate["passed"], semantic_gate["failed_checks"]),
        _check("m4_no_go_contract", ranking_gate["passed"], ranking_gate["failed_checks"]),
    ])
    return {
        "audit_name": "stage33_m3_candidate_final_release_gate",
        "schema_version": "stage33_m3_candidate_final_release_gate_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "online_gt_fields_used": [],
        "gates": {
            "stage32_base": base_gate,
            "current_head_smoke": smoke_gate,
            "frozen_all119": all119_gate,
            "stratified_semantic": semantic_gate,
            "m4": ranking_gate,
            "release": release,
        },
        "decision": {
            "m3_candidate_generation_frozen": release["passed"],
            "rerun_all119_full500": False if release["passed"] else None,
            "semantic_backend": "stage26_filtered_lseg_confirmation_only",
            "semantic_safe_candidate_increment": 0,
            "ranker_training": False,
            "ranker_reason": ranking_gate["ranker_decision"],
            "next_stage": "stage29_candidate_executor_contract" if release["passed"] else "m3_release_blocked",
            "active_recovery_enabled": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage32-release", type=Path, required=True)
    parser.add_argument("--current-head-smoke", type=Path, required=True)
    parser.add_argument("--frozen-all119", type=Path, required=True)
    parser.add_argument("--stratified-semantic", type=Path, required=True)
    parser.add_argument("--m4-ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(
        _load(args.stage32_release),
        _load(args.current_head_smoke),
        _load(args.frozen_all119),
        _load(args.stratified_semantic),
        _load(args.m4_ranking),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gates"]["release"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

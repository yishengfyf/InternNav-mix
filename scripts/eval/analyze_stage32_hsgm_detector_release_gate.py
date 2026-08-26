#!/usr/bin/env python3
"""Close the Stage31 shadow ablation and detector-freeze release gate.

This analyzer only reads existing reports. It does not import the evaluator,
run a model, alter detector thresholds, or emit navigation actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_AUDIT_GT_FIELDS = {
    "semantic_scene_gt",
    "coordinate_transforms.map_to_habitat_world",
}


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


def stage31_contract(payload: dict[str, Any], *, lossless_depth: bool) -> dict[str, Any]:
    source = payload.get("input") or {}
    integrity = payload.get("integrity") or {}
    query_count = int(source.get("query_frame_count", 0) or 0)
    readable_depth = int(source.get("readable_depth_frame_count", 0) or 0)
    checkpoint_sha = source.get("checkpoint_sha256")
    mobileclip_sha = source.get("mobileclip_sha256")
    checks = [
        _check("schema", payload.get("schema_version") == "stage31_hsgm_yoloe_replay_v1", payload.get("schema_version")),
        _check("shadow_only", payload.get("shadow_only") is True, payload.get("shadow_only")),
        _check("no_action_applied", payload.get("action_applied_count") == 0, payload.get("action_applied_count")),
        _check("unknown_is_not_free", payload.get("unknown_is_free") is False, payload.get("unknown_is_free")),
        _check("sparse_occ_is_only_safety_authority", payload.get("sparse_occ_is_only_safety_authority") is True, payload.get("sparse_occ_is_only_safety_authority")),
        _check("hsgm_surface_is_not_traversability_truth", payload.get("hsgm_surface_is_traversability_truth") is False, payload.get("hsgm_surface_is_traversability_truth")),
        _check("online_geometry_unchanged", payload.get("online_geometry_backend_changed") is False, payload.get("online_geometry_backend_changed")),
        _check("online_semantic_unchanged", payload.get("online_semantic_backend_changed") is False, payload.get("online_semantic_backend_changed")),
        _check("report_does_not_require_detector_revalidation", payload.get("detector_revalidation_required") is False, payload.get("detector_revalidation_required")),
        _check("online_gt_fields_empty", payload.get("online_gt_fields_used") == [], payload.get("online_gt_fields_used")),
        _check("audit_gt_fields_scoped", set(payload.get("audit_gt_fields_used") or []) == EXPECTED_AUDIT_GT_FIELDS, payload.get("audit_gt_fields_used")),
        _check("manifest_integrity", integrity.get("manifest_count_matches_query_frame_count") is True, integrity.get("manifest_count_matches_query_frame_count")),
        _check("depth_hash_integrity", integrity.get("depth_hash_mismatch_count") == 0, integrity.get("depth_hash_mismatch_count")),
        _check("query_frames_present", query_count > 0, query_count),
        _check("checkpoint_hash_present", isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64, checkpoint_sha),
        _check("mobileclip_hash_present", isinstance(mobileclip_sha, str) and len(mobileclip_sha) == 64, mobileclip_sha),
        _check("fixed_confidence", source.get("model_confidence") == 0.6, source.get("model_confidence")),
        _check("fixed_iou", source.get("model_iou") == 0.3, source.get("model_iou")),
        _check("front_query_frames_only", source.get("query_frames_only") is True, source.get("query_frames_only")),
    ]
    if lossless_depth:
        checks.append(_check("lossless_depth_complete", readable_depth == query_count, {"readable": readable_depth, "query": query_count}))
    else:
        checks.append(_check("rgb_only_depth_absent_as_declared", readable_depth == 0, readable_depth))
    return {
        **_finish(checks),
        "episode_count": source.get("episode_count"),
        "query_frame_count": query_count,
        "readable_depth_frame_count": readable_depth,
        "checkpoint_sha256": checkpoint_sha,
        "mobileclip_sha256": mobileclip_sha,
        "raw": (payload.get("variants") or {}).get("raw") or {},
        "hsgm_center_filtered": (payload.get("variants") or {}).get("hsgm_center_filtered") or {},
        "paired_lseg": payload.get("paired_lseg") or {},
        "center_filter_retention_rate": payload.get("hsgm_center_filter_retention_rate"),
    }


def stage30_contract(payload: dict[str, Any]) -> dict[str, Any]:
    confirmation = payload.get("confirmation") or {}
    checks = [
        _check("stage30_shadow_only", payload.get("shadow_only") is True, payload.get("shadow_only")),
        _check("stage30_no_action", payload.get("action_applied_count") == 0, payload.get("action_applied_count")),
        _check("stage30_unknown_is_not_free", payload.get("unknown_is_free") is False, payload.get("unknown_is_free")),
        _check("stage30_online_gt_empty", payload.get("gt_fields_used") == [], payload.get("gt_fields_used")),
        _check("stage30_detector_revalidation_not_required", confirmation.get("detector_revalidation_required") is False, confirmation.get("detector_revalidation_required")),
        _check("stage30_detector_frozen", confirmation.get("detector_frozen_confirmation") is True, confirmation.get("detector_frozen_confirmation")),
    ]
    return _finish(checks)


def detector_contract(contract: dict[str, Any], event_gt: dict[str, Any]) -> dict[str, Any]:
    selected = ((event_gt.get("detectors") or {}).get("D2_ER_selected") or {})
    all_metrics = selected.get("all") or {}
    holdout_metrics = selected.get("holdout") or {}
    checks = [
        _check("stage25_contract_passed", contract.get("contract_passed") is True, contract.get("contract_passed")),
        _check("stage25_manifest_500_expected", contract.get("manifest_expected_count") == 500, contract.get("manifest_expected_count")),
        _check("stage25_manifest_500_verified", contract.get("manifest_verified_count") == 500, contract.get("manifest_verified_count")),
        _check("stage25_detector_did_not_use_future", contract.get("future_used_by_detector") is False, contract.get("future_used_by_detector")),
        _check("stage25_outcome_not_event_gt", contract.get("outcome_is_event_gt") is False, contract.get("outcome_is_event_gt")),
        _check("event_gt_detector_did_not_use_future", event_gt.get("future_used_by_detector") is False, event_gt.get("future_used_by_detector")),
        _check("event_gt_outcome_not_event_gt", event_gt.get("outcome_is_event_gt") is False, event_gt.get("outcome_is_event_gt")),
        _check("event_gt_objective_manifest_present", int(event_gt.get("objective_manifest_count", 0) or 0) > 0, event_gt.get("objective_manifest_count")),
    ]
    return {
        **_finish(checks),
        "formal_detector": "D0+D1+D2 confirmed",
        "development_only_extension": "D2_ER_selected",
        "development_reference_metrics": {
            "all_precision": all_metrics.get("event_precision_on_adjudicated"),
            "all_combined_recall": all_metrics.get("combined_confirmed_recall"),
            "holdout_precision": holdout_metrics.get("event_precision_on_adjudicated"),
            "holdout_combined_recall": holdout_metrics.get("combined_confirmed_recall"),
        },
    }


def build_report(
    stage30: dict[str, Any],
    lossless: dict[str, Any],
    rgb_expansion: dict[str, Any],
    stage25: dict[str, Any],
    event_gt: dict[str, Any],
) -> dict[str, Any]:
    base_gate = stage30_contract(stage30)
    lossless_gate = stage31_contract(lossless, lossless_depth=True)
    rgb_gate = stage31_contract(rgb_expansion, lossless_depth=False)
    detector_gate = detector_contract(stage25, event_gt)
    asset_hashes_match = (
        lossless_gate["checkpoint_sha256"] == rgb_gate["checkpoint_sha256"]
        and lossless_gate["mobileclip_sha256"] == rgb_gate["mobileclip_sha256"]
    )
    online_backends_unchanged = all(
        payload.get("online_geometry_backend_changed") is False
        and payload.get("online_semantic_backend_changed") is False
        for payload in (lossless, rgb_expansion)
    )
    release_checks = [
        _check("stage30_base_gate", base_gate["passed"], base_gate["failed_checks"]),
        _check("stage31_lossless_gate", lossless_gate["passed"], lossless_gate["failed_checks"]),
        _check("stage31_rgb_expansion_gate", rgb_gate["passed"], rgb_gate["failed_checks"]),
        _check("stage31_assets_identical", asset_hashes_match, {
            "lossless_checkpoint": lossless_gate["checkpoint_sha256"],
            "rgb_checkpoint": rgb_gate["checkpoint_sha256"],
            "lossless_mobileclip": lossless_gate["mobileclip_sha256"],
            "rgb_mobileclip": rgb_gate["mobileclip_sha256"],
        }),
        _check("online_backends_unchanged", online_backends_unchanged, online_backends_unchanged),
        _check("stage25_detector_contract", detector_gate["passed"], detector_gate["failed_checks"]),
    ]
    release = _finish(release_checks)
    detector_revalidation_required = not (
        release["passed"] and online_backends_unchanged
    )
    return {
        "audit_name": "stage32_hsgm_detector_release_gate",
        "schema_version": "stage32_hsgm_detector_release_gate_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "online_gt_fields_used": [],
        "gates": {
            "stage30_base": base_gate,
            "stage31_lossless": lossless_gate,
            "stage31_rgb_expansion": rgb_gate,
            "stage25_detector": detector_gate,
            "release": release,
        },
        "stage1_decision": {
            "sparse_occ_online_geometry": "unchanged_frozen",
            "hsgm_height_surface": "audit_only_not_free_unknown_or_clearance_truth",
            "semantic_backend": "retain_stage26_filtered_lseg",
            "yoloe_raw": "optional_paper_ablation_only",
            "hsgm_center_filtered_yoloe": "rejected_for_single_front_view_mainline",
            "expand_stage31_to_fresh48": False,
            "tune_yoloe_thresholds": False,
        },
        "stage2_decision": {
            "detector_revalidation_required": detector_revalidation_required,
            "detector_frozen_confirmation": release["passed"] and not detector_revalidation_required,
            "stage2_confirmation": (
                "freeze_detector_no_online_base_change"
                if release["passed"] and not detector_revalidation_required
                else "release_gate_failed_revalidation_status_unresolved"
            ),
            "formal_detector": "D0+D1+D2 confirmed",
            "development_only_extension": "D2_ER_selected",
            "rerun_500_navigation_episodes": False if release["passed"] else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage30-base", type=Path, required=True)
    parser.add_argument("--stage31-lossless", type=Path, required=True)
    parser.add_argument("--stage31-rgb-expansion", type=Path, required=True)
    parser.add_argument("--stage25-contract", type=Path, required=True)
    parser.add_argument("--stage25-event-gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(
        _load(args.stage30_base),
        _load(args.stage31_lossless),
        _load(args.stage31_rgb_expansion),
        _load(args.stage25_contract),
        _load(args.stage25_event_gt),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gates"]["release"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

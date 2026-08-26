#!/usr/bin/env python3
"""Combine geometry, semantic-attachment, and detector-contract audits.

This is a read-only release gate.  It does not modify SparseOcc, semantic
nodes, detector thresholds, S2 actions, or recovery behavior.  The report
distinguishes an audit-only improvement from a change that invalidates the
frozen detector/candidate baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VERTICAL_TOKENS = ("stairs", "mixed_height", "height_change", "layer")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _metric(episode: dict[str, Any], branch: str, metric: str) -> float | None:
    row = (episode.get("navmesh_traversability_" + branch) or {})
    metrics = row.get("free_metrics_observed_domain") or {}
    value = metrics.get(metric)
    return float(value) if value is not None else None


def _navmesh_summary(episodes: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    row = {
        "episode_count": len(episodes),
        "free_precision_mean": _mean(_metric(e, branch, "precision") for e in episodes),
        "free_recall_mean": _mean(_metric(e, branch, "recall") for e in episodes),
        "false_free_rate_mean": _mean(
            ((e.get("navmesh_traversability_" + branch) or {}).get("false_free_rate"))
            for e in episodes
        ),
        "unknown_coverage_mean": _mean(
            ((e.get("navmesh_traversability_" + branch) or {}).get("predicted_unknown_cell_count"))
            / max(1, ((e.get("navmesh_traversability_" + branch) or {}).get("sampled_cell_count", 1)))
            for e in episodes
        ),
        "historical_edge_strict_free_recall_mean": _mean(
            ((e.get("navmesh_traversability_" + branch) or {}).get("historical_edge_strict_free_recall"))
            for e in episodes
        ),
        "executed_route_predicted_free_recall_mean": _mean(
            ((e.get("navmesh_traversability_" + branch) or {}).get("executed_route_predicted_free_recall"))
            for e in episodes
        ),
    }
    return row


def geometry_audit(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = [item for item in payload.get("episodes", []) if isinstance(item, dict)]
    vertical = [
        item for item in episodes
        if any(token in str(item.get("audit_role", "")).lower() for token in VERTICAL_TOKENS)
    ]
    flat = [item for item in episodes if item not in vertical]
    return {
        "source_audit_name": payload.get("audit_name"),
        "integrity_passed": bool(payload.get("integrity_passed")),
        "episode_count": len(episodes),
        "flat_episode_count": len(flat),
        "vertical_episode_count": len(vertical),
        "vertical_roles": sorted({str(item.get("audit_role")) for item in vertical}),
        "occupied_precision_mean": _mean(
            ((item.get("comparison") or {}).get("occupied_tolerance") or {}).get("precision")
            for item in episodes
        ),
        "occupied_recall_mean": _mean(
            ((item.get("comparison") or {}).get("occupied_tolerance") or {}).get("recall")
            for item in episodes
        ),
        "free_precision_mean": _mean(
            ((item.get("comparison") or {}).get("free_exact") or {}).get("precision")
            for item in episodes
        ),
        "free_recall_mean": _mean(
            ((item.get("comparison") or {}).get("free_exact") or {}).get("recall")
            for item in episodes
        ),
        "unknown_coverage_mean": _mean(
            ((item.get("comparison") or {}).get("unknown_coverage")) for item in episodes
        ),
        "false_free_rate_mean": _mean(
            ((item.get("comparison") or {}).get("false_free_rate")) for item in episodes
        ),
        "flat_navmesh_current_clearance": _navmesh_summary(flat, "current_clearance"),
        "vertical_navmesh_current_clearance": _navmesh_summary(vertical, "current_clearance"),
        "vertical_navmesh_oracle_sensor_clearance": _navmesh_summary(vertical, "oracle_sensor_clearance"),
        "vertical_pose_oracle_gap": {
            "free_recall_delta_current_minus_oracle": (
                _navmesh_summary(vertical, "current_clearance")["free_recall_mean"]
                - _navmesh_summary(vertical, "oracle_sensor_clearance")["free_recall_mean"]
                if _navmesh_summary(vertical, "current_clearance")["free_recall_mean"] is not None
                and _navmesh_summary(vertical, "oracle_sensor_clearance")["free_recall_mean"] is not None
                else None
            ),
            "historical_edge_recall_delta_current_minus_oracle": (
                _navmesh_summary(vertical, "current_clearance")["historical_edge_strict_free_recall_mean"]
                - _navmesh_summary(vertical, "oracle_sensor_clearance")["historical_edge_strict_free_recall_mean"]
                if _navmesh_summary(vertical, "current_clearance")["historical_edge_strict_free_recall_mean"] is not None
                and _navmesh_summary(vertical, "oracle_sensor_clearance")["historical_edge_strict_free_recall_mean"] is not None
                else None
            ),
        },
        "online_geometry_backend_changed": False,
        "vertical_geometry_status": (
            "evidence_limited_current_2d_pose"
            if vertical else "not_tested_in_this_audit"
        ),
    }


def semantic_audit(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence_gate") or {}
    delta = payload.get("delta") or {}
    return {
        "source_audit_name": payload.get("audit_name"),
        "integrity_passed": bool(payload.get("integrity_passed")),
        "episode_count": int(payload.get("episode_count", 0) or 0),
        "all_trajectories_exact_match": bool(payload.get("all_trajectories_exact_match")),
        "all_raw_semantics_exact_match": bool(payload.get("all_raw_semantics_exact_match")),
        "raw": payload.get("raw") or {},
        "filtered": payload.get("filtered") or {},
        "delta": delta,
        "evidence_gate": evidence,
        "filtered_shadow_accepted": bool(evidence.get("passed")) and bool(
            payload.get("integrity_passed")
        ),
        "online_semantic_backend_changed": False,
        "yoloe_hsgm_backend_integrated": False,
        "semantic_status": "filtered_lseg_shadow_only",
    }


def detector_audit(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "contract_passed": None}
    return {
        "available": True,
        "contract_passed": bool(payload.get("contract_passed")),
        "future_used_by_detector": bool(payload.get("future_used_by_detector")),
        "outcome_is_event_gt": bool(payload.get("outcome_is_event_gt")),
        "manifest_expected_count": payload.get("manifest_expected_count"),
        "manifest_verified_count": payload.get("manifest_verified_count"),
        "detector_options": payload.get("detector_options") or {},
    }


def confirm(
    geometry: dict[str, Any],
    semantic: dict[str, Any],
    detector: dict[str, Any],
) -> dict[str, Any]:
    geometry_changed = bool(geometry.get("online_geometry_backend_changed"))
    semantic_changed = bool(semantic.get("online_semantic_backend_changed"))
    detector_revalidation_required = geometry_changed or semantic_changed
    return {
        "geometry_online_branch_unchanged": not geometry_changed,
        "semantic_online_branch_unchanged": not semantic_changed,
        "detector_revalidation_required": detector_revalidation_required,
        "detector_frozen_confirmation": not detector_revalidation_required,
        "stage2_confirmation": (
            "freeze_detector_no_online_base_change"
            if not detector_revalidation_required else "rerun_detector_on_same_manifest"
        ),
        "candidate_and_recovery_status": "not_started_by_this_audit",
        "gt_or_future_used_for_online_decision": False,
        "unknown_is_free": False,
        "action_applied_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--detector-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = geometry_audit(_load(args.geometry_audit))
    semantic = semantic_audit(_load(args.semantic_audit))
    detector = detector_audit(_load(args.detector_audit) if args.detector_audit else None)
    result = {
        "audit_name": "stage30_geometry_semantic_detector_base_confirmation",
        "schema_version": "stage30_base_confirmation_v1",
        "shadow_only": True,
        "action_applied_count": 0,
        "unknown_is_free": False,
        "gt_fields_used": [],
        "geometry": geometry,
        "semantic": semantic,
        "detector": detector,
        "confirmation": confirm(geometry, semantic, detector),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

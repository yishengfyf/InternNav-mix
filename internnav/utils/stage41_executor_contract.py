"""Stage41 pre-active executor contract, without action execution."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "stage41_executor_contract_v2"


def validate_executor_contract(
    *,
    sensor: Mapping[str, Any],
    edge_audits: Iterable[Mapping[str, Any]],
    candidate_safety: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate prerequisites for a future executor; never emits an action."""
    edges = list(edge_audits)
    hfov = sensor.get("hfov_deg")
    depth_readable = bool(sensor.get("depth_readable"))
    safety_ok = (
        bool(candidate_safety.get("floor_aligned_known_free"))
        and float(candidate_safety.get("unknown_fraction", 0.0) or 0.0) == 0.0
        and float(candidate_safety.get("occupied_fraction", 0.0) or 0.0) == 0.0
        and not bool(candidate_safety.get("route_occ_conflict"))
    )
    edge_ok = bool(edges) and all(
        bool(edge.get("sparseocc_safe"))
        and bool(edge.get("depth_occlusion_checked"))
        and bool(edge.get("depth_readable"))
        and not bool(edge.get("unknown"))
        and not bool(edge.get("occupied"))
        for edge in edges
    )
    hfov_ok = hfov is not None and 0.5 <= float(hfov) <= 180.0
    return {
        "schema_version": SCHEMA_VERSION,
        "sensor_hfov_deg": float(hfov) if hfov is not None else None,
        "sensor_hfov_source": sensor.get("hfov_source"),
        "depth_readable": depth_readable,
        "edge_count": len(edges),
        "candidate_safety_reaudited": safety_ok,
        "all_edges_depth_checked": edge_ok,
        "executor_eligible": bool(hfov_ok and depth_readable and safety_ok and edge_ok),
        "abstain_reason": None if (hfov_ok and depth_readable and safety_ok and edge_ok) else "contract_not_satisfied",
        "action_emitted": False,
        "action_applied": False,
        "shadow_only": True,
        "unknown_is_free": False,
        "gt_fields_used": [],
    }


def executor_contract_ok(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("schema_version") == SCHEMA_VERSION
        and not report.get("action_emitted")
        and not report.get("action_applied")
        and report.get("shadow_only")
        and report.get("unknown_is_free") is False
        and not report.get("gt_fields_used")
        and (report.get("executor_eligible") or report.get("abstain_reason"))
    )

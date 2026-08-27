"""Stage43 bounded counterfactual re-observation contracts.

The planner only quantizes an in-place viewing direction. It never emits a
Habitat action and never grants safety to a historical route edge.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = "stage43_counterfactual_reobserve_v1"


def normalize_angle_deg(value: float) -> float:
    """Normalize an angle to (-180, 180]."""
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if result <= -180.0 else result


def plan_bounded_reorientation(
    relative_bearing_deg: float,
    *,
    hfov_deg: float,
    turn_angle_deg: float,
    center_margin_deg: float = 10.0,
    max_turn_steps: int = 12,
) -> dict[str, Any]:
    """Plan the minimum quantized yaw change that exposes a target bearing."""
    bearing = normalize_angle_deg(relative_bearing_deg)
    hfov = float(hfov_deg)
    turn_angle = abs(float(turn_angle_deg))
    margin = max(0.0, float(center_margin_deg))
    half_visible = hfov / 2.0 - margin
    valid_sensor = 0.0 < hfov <= 180.0 and turn_angle > 0.0 and half_visible > 0.0
    if not valid_sensor:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "reason": "invalid_sensor_or_margin",
            "relative_bearing_deg": bearing,
            "hfov_deg": hfov,
            "center_margin_deg": margin,
            "action_emitted": False,
            "action_applied": False,
            "shadow_only": True,
        }

    required = max(0.0, abs(bearing) - half_visible)
    requested_steps = int(math.ceil(required / turn_angle - 1e-12))
    limit = max(0, int(max_turn_steps))
    steps = min(requested_steps, limit)
    direction_sign = 1.0 if bearing >= 0.0 else -1.0
    delta = direction_sign * float(steps) * turn_angle
    residual = normalize_angle_deg(bearing - delta)
    visible_after = abs(residual) <= half_visible + 1e-9
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": bool(visible_after),
        "reason": "target_in_probe_fov" if visible_after else "turn_budget_insufficient",
        "relative_bearing_deg": bearing,
        "hfov_deg": hfov,
        "center_margin_deg": margin,
        "half_visible_deg": half_visible,
        "turn_angle_deg": turn_angle,
        "max_turn_steps": limit,
        "requested_turn_steps": requested_steps,
        "turn_steps": steps,
        "turn_direction": "left" if delta > 0.0 else ("right" if delta < 0.0 else "none"),
        "planned_yaw_delta_deg": delta,
        "residual_bearing_deg": residual,
        "target_visible_before": abs(bearing) <= half_visible + 1e-9,
        "target_visible_after": bool(visible_after),
        "action_emitted": False,
        "action_applied": False,
        "shadow_only": True,
        "unknown_is_free": False,
        "gt_fields_used": [],
    }


def counterfactual_contract_ok(record: Mapping[str, Any]) -> bool:
    """Check non-intervention and safety-authority invariants."""
    return bool(
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("shadow_only")
        and not record.get("action_emitted")
        and not record.get("action_applied")
        and record.get("unknown_is_free") is False
        and not record.get("gt_fields_used")
        and record.get("sim_pose_restored")
        and record.get("official_memory_mutated") is False
        and record.get("safety_authority") == "temporary_current_sparseocc_reaudit"
    )

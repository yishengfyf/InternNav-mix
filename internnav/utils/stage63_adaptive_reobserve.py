"""Pure Stage63 counterfactual view-sweep planning contracts."""

from __future__ import annotations

import math
from typing import Any, Iterable

from internnav.utils.stage43_counterfactual_reobserve import normalize_angle_deg


SCHEMA_VERSION = "stage63_adaptive_reobserve_v1"


def _probe(
    arm: str,
    bearing_deg: float,
    step_count: int,
    turn_angle_deg: float,
) -> dict[str, Any]:
    sign = 1.0 if bearing_deg >= 0.0 else -1.0
    yaw_delta = sign * max(0, int(step_count)) * turn_angle_deg
    residual = normalize_angle_deg(bearing_deg - yaw_delta)
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": str(arm),
        "valid": True,
        "reason": "counterfactual_view_probe",
        "relative_bearing_deg": float(bearing_deg),
        "turn_steps": max(0, int(step_count)),
        "turn_angle_deg": float(turn_angle_deg),
        "planned_yaw_delta_deg": float(yaw_delta),
        "residual_bearing_deg": float(residual),
        "action_emitted": False,
        "action_applied": False,
        "shadow_only": True,
        "unknown_is_free": False,
        "gt_fields_used": [],
    }


def plan_adaptive_view_sweep(
    relative_bearing_deg: float,
    *,
    hfov_deg: float,
    turn_angle_deg: float,
    primitive_budgets: Iterable[int] = (1, 2, 4),
    center_margin_deg: float = 10.0,
    max_turn_steps: int = 12,
    overscan_steps: int = 1,
) -> list[dict[str, Any]]:
    """Return unique cumulative-yaw probes without emitting an action."""
    bearing = normalize_angle_deg(relative_bearing_deg)
    hfov = float(hfov_deg)
    turn_angle = abs(float(turn_angle_deg))
    margin = max(0.0, float(center_margin_deg))
    max_steps = max(0, int(max_turn_steps))
    half_visible = hfov / 2.0 - margin
    if not (0.0 < hfov <= 180.0 and turn_angle > 0.0 and half_visible > 0.0):
        return []

    center_steps = min(
        max_steps,
        int(math.floor(abs(bearing) / turn_angle + 0.5)),
    )
    entry_steps = min(
        max_steps,
        int(math.ceil(max(0.0, abs(bearing) - half_visible) / turn_angle - 1e-12)),
    )
    requested = [
        (f"budget_{int(value)}", min(max_steps, max(0, int(value))))
        for value in primitive_budgets
    ]
    requested.extend(
        [
            ("fov_entry", entry_steps),
            ("path_center", center_steps),
            (
                "path_center_overscan",
                min(max_steps, center_steps + max(0, int(overscan_steps))),
            ),
        ]
    )

    probes = []
    by_steps: dict[int, dict[str, Any]] = {}
    for arm, steps in requested:
        if steps in by_steps:
            by_steps[steps]["arm_aliases"].append(str(arm))
            continue
        probe = _probe(arm, bearing, steps, turn_angle)
        probe.update(
            {
                "arm_aliases": [str(arm)],
                "hfov_deg": hfov,
                "center_margin_deg": margin,
                "half_visible_deg": half_visible,
                "max_turn_steps": max_steps,
                "target_visible_after": bool(
                    abs(float(probe["residual_bearing_deg"]))
                    <= half_visible + 1e-9
                ),
            }
        )
        probes.append(probe)
        by_steps[steps] = probe
    return probes

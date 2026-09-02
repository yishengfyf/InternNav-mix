"""DualVLN-native language helpers for SparseOcc recovery guidance.

The geometry remains controller evidence.  These helpers only translate a
read-only, current-pose route audit into coarse natural language that the
frozen System 2 can use while viewing the current and reference RGB frames.
"""
from __future__ import annotations

import math
from typing import Any, Dict


SCHEMA_VERSION = "stage75_route_guidance_v1"


def normalize_bearing_deg(value: float) -> float:
    """Normalize a relative bearing to [-180, 180]."""
    return (float(value) + 180.0) % 360.0 - 180.0


def natural_route_direction(angle_deg: float) -> str:
    """Return an eight-way egocentric direction phrase.

    SparseOcc uses positive relative bearings for left and negative bearings
    for right.  Keeping the sign convention here avoids prompt-side inversion.
    """
    angle = normalize_bearing_deg(angle_deg)
    magnitude = abs(angle)
    if magnitude <= 22.5:
        return "ahead"
    if magnitude <= 67.5:
        return "ahead-left" if angle > 0.0 else "ahead-right"
    if magnitude <= 112.5:
        return "left" if angle > 0.0 else "right"
    if magnitude <= 157.5:
        return "behind-left" if angle > 0.0 else "behind-right"
    return "behind"


def quantize_bearing_deg(angle_deg: float, quantum_deg: float = 15.0) -> int:
    quantum = max(1.0, float(quantum_deg))
    angle = normalize_bearing_deg(angle_deg)
    quantized = int(round(abs(angle) / quantum) * quantum)
    return min(180, max(0, quantized))


def quantize_distance_m(distance_m: float, quantum_m: float = 0.25) -> float:
    quantum = max(0.05, float(quantum_m))
    distance = max(0.0, float(distance_m))
    return float(round(distance / quantum) * quantum)


def route_guidance_from_bridge(
    bridge: Dict[str, Any],
    *,
    arrival_distance_m: float = 0.15,
) -> Dict[str, Any]:
    """Reduce a SparseOcc path bridge to prompt-safe route fields."""
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "arrived": False,
        "reason": None,
        "path_reachable": bool((bridge or {}).get("path_reachable")),
        "start_grid": (bridge or {}).get("start_grid"),
        "anchor_grid": (bridge or {}).get("anchor_grid"),
        "path_cell_count": int((bridge or {}).get("path_cell_count", 0) or 0),
        "path_distance_m": (bridge or {}).get("path_m"),
        "relative_bearing_deg": (bridge or {}).get("initial_direction_angle_deg"),
        "direction_bucket": (bridge or {}).get("initial_direction_bucket"),
        "natural_direction": None,
        "quantized_bearing_deg": None,
        "quantized_distance_m": None,
    }
    if not result["path_reachable"]:
        result["reason"] = str((bridge or {}).get("reason") or "path_not_reachable")
        return result
    try:
        distance = float(result["path_distance_m"])
    except (TypeError, ValueError):
        result["reason"] = "missing_path_distance"
        return result
    if not math.isfinite(distance):
        result["reason"] = "nonfinite_path_distance"
        return result
    result["arrived"] = bool(
        distance <= max(0.0, float(arrival_distance_m)) + 1e-9
    )
    if result["arrived"]:
        result.update(
            {
                "valid": True,
                "reason": "anchor_reached",
                "natural_direction": "here",
                "quantized_bearing_deg": 0,
                "quantized_distance_m": quantize_distance_m(distance),
            }
        )
        return result
    try:
        bearing = float(result["relative_bearing_deg"])
    except (TypeError, ValueError):
        result["reason"] = "missing_route_bearing"
        return result
    if not math.isfinite(bearing):
        result["reason"] = "nonfinite_route_bearing"
        return result
    result.update(
        {
            "valid": True,
            "reason": "ok",
            "relative_bearing_deg": normalize_bearing_deg(bearing),
            "natural_direction": natural_route_direction(bearing),
            "quantized_bearing_deg": quantize_bearing_deg(bearing),
            "quantized_distance_m": quantize_distance_m(distance),
        }
    )
    return result


def build_dualvln_route_recovery_card(guidance: Dict[str, Any]) -> str:
    """Build a compact addendum while preserving the native output protocol."""
    if not guidance or not guidance.get("valid") or guidance.get("arrived"):
        return (
            "Before continuing the original task, first return toward the previously "
            "visited place shown in the recovery reference observation. Compare that "
            "reference with the current view and your historical observations. If the "
            "place is outside the current view, output only one TURN LEFT (←), TURN "
            "RIGHT (→), or look-down (↓) observation action, then re-observe. When it "
            "is visible, output the next waypoint coordinates in the current image. "
            "Output STOP only when the original task is complete."
        )

    direction = str(guidance.get("natural_direction") or "ahead")
    bearing = int(guidance.get("quantized_bearing_deg", 0) or 0)
    distance = float(guidance.get("quantized_distance_m", 0.0) or 0.0)
    return (
        "Before continuing the original task, first return toward the previously "
        "visited place shown in the recovery reference observation. From your current "
        f"pose, the previously travelled route begins {direction} (about {bearing} "
        f"degrees), and the reference place is about {distance:.2f} meters away along "
        "that route. Use this as a viewing direction, not as an image coordinate. A "
        "recent repeated action did not make local progress, so do not repeat a long "
        "turn sequence. If the reference place is outside the current view, output "
        "only one TURN LEFT (←), TURN RIGHT (→), or look-down (↓) observation action, "
        "then re-observe. When it is visible, output the next waypoint coordinates in "
        "the current image. Output STOP only when the original task is complete."
    )


def build_dualvln_route_recovery_instruction(guidance: Dict[str, Any]) -> str:
    """Create a temporary task that fits directly in DualVLN's instruction slot."""
    destination = (
        "return to the previously visited place shown in the recovery reference "
        "observation"
    )
    if not guidance or not guidance.get("valid") or guidance.get("arrived"):
        return destination
    direction = str(guidance.get("natural_direction") or "ahead")
    bearing = int(guidance.get("quantized_bearing_deg", 0) or 0)
    distance = float(guidance.get("quantized_distance_m", 0.0) or 0.0)
    if direction == "ahead":
        orientation = "facing the route ahead"
    elif direction == "behind":
        orientation = "turning around about 180 degrees"
    elif "left" in direction:
        orientation = f"turning left about {bearing} degrees"
    elif "right" in direction:
        orientation = f"turning right about {bearing} degrees"
    else:
        orientation = f"turning toward the route about {bearing} degrees"
    return (
        f"{destination} by first {orientation} and then following the previously "
        f"travelled route for about {distance:.2f} meters"
    )


def build_dualvln_temporary_reference_card() -> str:
    """Bind the extra image without restating the original episode task."""
    return (
        "The image marked recovery reference observation shows the destination of "
        "this temporary navigation task. Compare it with the current view and your "
        "historical observations. In case the destination is out of view, use only "
        "one TURN LEFT (←), TURN RIGHT (→), or look-down (↓) observation action, "
        "then re-observe. When it is visible, output the next waypoint coordinates "
        "in the current image. Output STOP when you have reached this temporary "
        "destination."
    )


def bind_dualvln_temporary_instruction(
    prompt_template: str, temporary_instruction: str
) -> tuple[str, bool]:
    """Replace exactly the native DualVLN instruction slot in a prompt copy."""
    marker = "<instruction>"
    if str(prompt_template).count(marker) != 1 or not str(
        temporary_instruction
    ).strip():
        return str(prompt_template), False
    return (
        str(prompt_template).replace(
            marker, str(temporary_instruction).strip(), 1
        ),
        True,
    )

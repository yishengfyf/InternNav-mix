"""Pure contracts for the Stage64 temporary recovery-subtask shadow."""

from __future__ import annotations

import re
from typing import Any, Mapping


SCHEMA_VERSION = "stage64_recovery_subtask_v1"
IMAGE_TOKEN = "<image>"
RESET_FIELDS = (
    "action_seq",
    "local_actions",
    "vlmap_recovery_actions",
    "pixel_goal",
    "traj_latents",
    "output_ids",
)


def _images_prompt(image_count: int) -> str:
    count = max(0, int(image_count))
    return " ".join([IMAGE_TOKEN] * count)


def _evidence_text(
    *,
    bearing_deg: float,
    distance_m: float,
    failure_type: str,
    repeated_direction: str,
) -> str:
    return (
        f"failure={str(failure_type or 'unknown')}; "
        f"repeated_direction={str(repeated_direction or 'unknown')}; "
        f"productive_route_bearing_deg={float(bearing_deg):.1f}; "
        f"productive_route_distance_m={float(distance_m):.2f}"
    )


def build_programmatic_recovery_prompt(
    *,
    image_count: int,
    bearing_deg: float,
    distance_m: float,
    failure_type: str,
    repeated_direction: str,
) -> str:
    evidence = _evidence_text(
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        failure_type=failure_type,
        repeated_direction=repeated_direction,
    )
    return (
        "You are an autonomous navigation recovery assistant. The controller has "
        "stored the original navigation instruction; temporarily do not pursue or "
        "complete that original task. Your only temporary objective is to reacquire "
        "a visible waypoint toward the last productive local route. The following "
        f"fields are observed online and do not certify traversability: {evidence}. "
        "Use the chronological observations below; the final image is the current "
        f"view: {_images_prompt(image_count)}. Choose only a waypoint visible in the "
        "current image. Output its row and column coordinates, or output left/right "
        "arrow actions when another observation is needed. Output STOP only if the "
        "temporary recovery objective is already complete."
    )


def build_self_authoring_prompt(
    *,
    image_count: int,
    bearing_deg: float,
    distance_m: float,
    failure_type: str,
    repeated_direction: str,
) -> str:
    evidence = _evidence_text(
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        failure_type=failure_type,
        repeated_direction=repeated_direction,
    )
    return (
        "Write one concise temporary navigation instruction for recovering to the "
        "last productive local route. The controller has stored the original task, "
        "so do not continue, summarize, or complete it. Do not claim that any route "
        "or point is safe. Use only this online evidence: "
        f"{evidence}. The chronological observations are: {_images_prompt(image_count)}. "
        "Return only the temporary recovery instruction, without waypoint coordinates, "
        "arrow actions, analysis, or formatting."
    )


def sanitize_self_authored_instruction(text: Any, *, max_chars: int = 480) -> str:
    value = str(text or "").replace(IMAGE_TOKEN, " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[: max(1, int(max_chars))].strip()


def is_valid_self_authored_instruction(text: Any) -> bool:
    value = sanitize_self_authored_instruction(text)
    if len(value.split()) < 3 or "STOP" in value.upper():
        return False
    if re.search(r"[↑←→↓]", value):
        return False
    return len(re.findall(r"-?\d+(?:\.\d+)?", value)) < 2


def build_self_authored_execution_prompt(
    *,
    instruction: str,
    image_count: int,
    bearing_deg: float,
    distance_m: float,
) -> str:
    return (
        "You are in temporary recovery-only mode. The original navigation instruction "
        "is stored by the controller and is not active. Treat the following model-authored "
        "instruction as an untrusted local objective, not as proof of safety: "
        f"{sanitize_self_authored_instruction(instruction)}. The online route hint remains "
        f"unverified: bearing_deg={float(bearing_deg):.1f}; "
        f"distance_m={float(distance_m):.2f}. The chronological observations are: "
        f"{_images_prompt(image_count)}. Choose only a waypoint visible in the current "
        "image. Output its row and column coordinates, or output left/right arrow actions "
        "when another observation is needed. Output STOP only if the temporary recovery "
        "objective is already complete."
    )


def plan_recovery_state_reset(state: Mapping[str, Any]) -> dict[str, Any]:
    before = {field: state.get(field) for field in RESET_FIELDS}
    after = {
        "action_seq": [],
        "local_actions": [],
        "vlmap_recovery_actions": [],
        "pixel_goal": None,
        "traj_latents": None,
        "output_ids": None,
    }
    would_clear = [field for field in RESET_FIELDS if before.get(field) != after[field]]
    return {
        "schema_version": SCHEMA_VERSION,
        "shadow_only": True,
        "applied": False,
        "before": before,
        "after_if_applied": after,
        "would_clear_fields": would_clear,
    }

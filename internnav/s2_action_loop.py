"""Pure cross-query action-loop detection for frozen System2 outputs."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional


DEFAULT_S2_ACTION_LOOP_CONFIG = {
    "enable": False,
    "shadow_only": True,
    "min_same_turn_generations": 5,
    "min_cumulative_turn_actions": 12,
    "min_step_span": 6,
    "min_episode_step": 30,
    "max_translation_m": 0.35,
}


def init_s2_action_loop_state() -> dict[str, Any]:
    return {
        "direction": None,
        "generation_streak": 0,
        "cumulative_turn_actions": 0,
        "start_step": None,
        "start_gps": None,
        "start_heading_rad": None,
        "last_step": None,
        "emitted": False,
        "loop_index": 0,
    }


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _gps_xy(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) < 2:
        return None
    x_value = _finite_float(values[0])
    y_value = _finite_float(values[1])
    if x_value is None or y_value is None:
        return None
    return [x_value, y_value]


def _heading(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        values = list(value)
    except TypeError:
        values = [value]
    if not values:
        return None
    return _finite_float(values[0])


def _angle_distance_degrees(first: Optional[float], second: Optional[float]) -> Optional[float]:
    if first is None or second is None:
        return None
    delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def normalize_direct_turn_output(output: Any) -> Optional[dict[str, Any]]:
    text = str(output or "").strip()
    if not text or re.search(r"\d", text):
        return None
    arrows = re.findall(r"[←→]", text)
    residue = re.sub(r"[←→\s,.;:!?，。；：！？]", "", text)
    if not arrows or residue or len(set(arrows)) != 1:
        return None
    direction = "left" if arrows[0] == "←" else "right"
    return {
        "direction": direction,
        "turn_action_count": len(arrows),
        "normalized_output": arrows[0] * len(arrows),
    }


def observe_s2_action_query(
    state: dict[str, Any],
    *,
    output: Any,
    step_id: int,
    gps: Any,
    compass: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    cfg = dict(DEFAULT_S2_ACTION_LOOP_CONFIG)
    cfg.update(dict(config or {}))
    if not bool(cfg.get("enable", False)):
        return None

    turn = normalize_direct_turn_output(output)
    current_gps = _gps_xy(gps)
    current_heading = _heading(compass)
    if turn is None:
        previous_direction = state.get("direction")
        previous_streak = int(state.get("generation_streak", 0) or 0)
        was_emitted = bool(state.get("emitted"))
        loop_index = int(state.get("loop_index", 0) or 0)
        state.clear()
        state.update(init_s2_action_loop_state())
        state["loop_index"] = loop_index
        if was_emitted:
            return {
                "transition": "end",
                "reason": "non_turn_s2_output",
                "previous_direction": previous_direction,
                "previous_generation_streak": previous_streak,
                "step_id": int(step_id),
            }
        return None

    direction = str(turn["direction"])
    if direction != state.get("direction"):
        loop_index = int(state.get("loop_index", 0) or 0)
        state.clear()
        state.update(init_s2_action_loop_state())
        state["loop_index"] = loop_index
        state["direction"] = direction
        state["start_step"] = int(step_id)
        state["start_gps"] = current_gps
        state["start_heading_rad"] = current_heading

    state["generation_streak"] = int(state.get("generation_streak", 0) or 0) + 1
    state["cumulative_turn_actions"] = int(
        state.get("cumulative_turn_actions", 0) or 0
    ) + int(turn["turn_action_count"])
    state["last_step"] = int(step_id)

    start_step = int(state.get("start_step", step_id) or step_id)
    step_span = max(0, int(step_id) - start_step)
    start_gps = _gps_xy(state.get("start_gps"))
    displacement = None
    if start_gps is not None and current_gps is not None:
        displacement = math.hypot(
            current_gps[0] - start_gps[0], current_gps[1] - start_gps[1]
        )
    low_translation = bool(
        displacement is not None
        and displacement <= float(cfg["max_translation_m"])
    )
    threshold_met = bool(
        int(state["generation_streak"]) >= int(cfg["min_same_turn_generations"])
        and int(state["cumulative_turn_actions"])
        >= int(cfg["min_cumulative_turn_actions"])
        and step_span >= int(cfg["min_step_span"])
        and int(step_id) >= int(cfg["min_episode_step"])
        and low_translation
    )
    if not threshold_met or bool(state.get("emitted")):
        return None

    state["emitted"] = True
    state["loop_index"] = int(state.get("loop_index", 0) or 0) + 1
    return {
        "transition": "start",
        "loop_index": int(state["loop_index"]),
        "step_id": int(step_id),
        "start_step": start_step,
        "step_span": step_span,
        "turn_direction": direction,
        "same_turn_generation_streak": int(state["generation_streak"]),
        "cumulative_turn_actions": int(state["cumulative_turn_actions"]),
        "current_output_turn_actions": int(turn["turn_action_count"]),
        "normalized_s2_output": str(turn["normalized_output"]),
        "translation_m": displacement,
        "max_translation_m": float(cfg["max_translation_m"]),
        "low_translation": low_translation,
        "start_heading_rad": state.get("start_heading_rad"),
        "current_heading_rad": current_heading,
        "heading_cycle_error_deg": _angle_distance_degrees(
            _finite_float(state.get("start_heading_rad")), current_heading
        ),
    }

"""Stage59 productive-onset route-anchor shadow audit.

This module only selects descriptive anchors from the authoritative executed
pose trace.  It never changes SparseOcc, creates a candidate, or emits an
action.  Habitat truth is attached by the evaluator as an offline field.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "stage59_productive_onset_v1"


def _distance(first: Mapping[str, Any], second: Mapping[str, Any], cell_size_m: float) -> float:
    try:
        x0, y0 = float(first["x"]), float(first["y"])
        x1, y1 = float(second["x"]), float(second["y"])
        value = math.hypot(x1 - x0, y1 - y0)
        if math.isfinite(value):
            return value
    except (KeyError, TypeError, ValueError):
        pass
    return math.hypot(
        int(second["row"]) - int(first["row"]),
        int(second["col"]) - int(first["col"]),
    ) * float(cell_size_m)


def _translation_nodes(pose_trace: Sequence[Mapping[str, Any]], cell_size_m: float) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for raw in pose_trace:
        try:
            node = {
                "step_id": int(raw.get("step_id")),
                "row": int(raw["row"]),
                "col": int(raw["col"]),
                "x": raw.get("x"),
                "y": raw.get("y"),
                "z": float(raw.get("z", 0.0) or 0.0),
                "yaw": float(raw.get("yaw", 0.0) or 0.0),
                "camera_pitch_deg": float(raw.get("camera_pitch_deg", 0.0) or 0.0),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if not nodes or _distance(nodes[-1], node, cell_size_m) > 1e-4:
            nodes.append(node)
    return nodes


def _path_cells(
    nodes: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
    rasterize_edge: Callable[..., Sequence[Sequence[int]]],
    cell_size_m: float,
) -> list[tuple[int, int]]:
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    cells: list[tuple[int, int]] = []
    for first, second in zip(nodes[start_index:end_index], nodes[start_index + 1 : end_index + 1]):
        edge = rasterize_edge(
            (int(first["row"]), int(first["col"])),
            (int(second["row"]), int(second["col"])),
            edge_length_m=_distance(first, second, cell_size_m),
            sample_spacing_m=max(1e-3, float(cell_size_m)),
        )
        for cell in edge:
            value = (int(cell[0]), int(cell[1]))
            if not cells or value != cells[-1]:
                cells.append(value)
    return cells


def _anchor_record(name: str, node: Mapping[str, Any] | None, *, reason: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "anchor": name,
        "valid": node is not None,
        "reason": reason,
        "step_id": None,
        "grid": None,
        "z_m": None,
        "node_index": None,
    }
    if node is not None:
        record.update(
            {
                "step_id": int(node["step_id"]),
                "grid": [int(node["row"]), int(node["col"])],
                "z_m": float(node.get("z", 0.0) or 0.0),
                "node_index": int(node["_index"]),
            }
        )
    return record


def audit_productive_onset_anchors(
    pose_trace: Sequence[Mapping[str, Any]],
    *,
    trigger_step: int | None,
    onset_step: int | None,
    state_fn: Callable[[int, int], str],
    rasterize_edge: Callable[..., Sequence[Sequence[int]]],
    cell_size_m: float = 0.05,
    min_productive_edge_m: float = 0.05,
    max_route_nodes: int = 128,
) -> dict[str, Any]:
    """Compare raw trigger, estimated onset and last productive pre-loop anchors."""
    nodes = _translation_nodes(pose_trace, cell_size_m)
    for index, node in enumerate(nodes):
        node["_index"] = index
    if not nodes:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "shadow_only": True,
            "decision_applied": False,
            "action_applied": False,
            "pixel_translation_allowed": False,
            "unknown_is_free": False,
            "gt_used_for_navigation": False,
            "reason": "empty_pose_trace",
            "anchors": [],
        }

    trigger = int(trigger_step) if trigger_step is not None else int(nodes[-1]["step_id"])
    onset = int(onset_step) if onset_step is not None else trigger
    pre_nodes = [node for node in nodes if int(node["step_id"]) <= onset]
    if not pre_nodes:
        pre_nodes = [nodes[0]]
    onset_node = pre_nodes[-1]
    raw_source = next(
        (node for node in reversed(nodes) if int(node["step_id"]) <= trigger),
        nodes[-1],
    )
    raw_node = dict(raw_source)
    raw_node["step_id"] = trigger

    productive: Mapping[str, Any] | None = None
    for index in range(len(pre_nodes) - 2, -1, -1):
        node = pre_nodes[index]
        nxt = pre_nodes[index + 1]
        if _distance(node, nxt, cell_size_m) < float(min_productive_edge_m):
            continue
        # Reject the middle node of an immediate A-B-A revisit.
        if index > 0 and index + 2 < len(pre_nodes):
            before = pre_nodes[index - 1]
            after = pre_nodes[index + 2]
            if (int(before["row"]), int(before["col"])) == (int(after["row"]), int(after["col"])):
                continue
        productive = node
        break

    anchors = [
        _anchor_record("raw_trigger", raw_node),
        _anchor_record("estimated_loop_onset", onset_node),
        _anchor_record(
            "last_productive_pre_loop",
            productive,
            reason="no_pre_loop_translation_edge" if productive is None else None,
        ),
    ]
    current_index = int(raw_node["_index"])
    for record in anchors:
        if not record["valid"]:
            continue
        anchor_index = int(record["node_index"])
        if anchor_index >= current_index:
            record.update(
                {
                    "route_edge_count": 0,
                    "route_length_m": 0.0,
                    "current_to_anchor_path_cells": [],
                    "first_edge": {"state": "same_node", "safe_0p25m_prefix": False},
                }
            )
            continue
        if current_index - anchor_index > int(max_route_nodes):
            anchor_index = current_index - int(max_route_nodes)
        path_anchor_to_current = _path_cells(
            nodes, anchor_index, current_index, rasterize_edge, cell_size_m
        )
        path_current_to_anchor = list(reversed(path_anchor_to_current))
        first_edge = path_current_to_anchor[1:] if len(path_current_to_anchor) > 1 else []
        first_states = [state_fn(int(cell[0]), int(cell[1])) for cell in first_edge]
        # Exclude the robot's current cell. Endpoint returns can mark the self
        # cell occupied; that is audited separately and must not erase an
        # otherwise connected retreat prefix.
        route_states = list(first_states)
        first_prefix_count = max(1, int(round(0.25 / max(float(cell_size_m), 1e-6))))
        prefix_states = first_states[:first_prefix_count]
        record.update(
            {
                "route_edge_count": max(0, current_index - anchor_index),
                "route_length_m": float(sum(_distance(a, b, cell_size_m) for a, b in zip(nodes[anchor_index:current_index], nodes[anchor_index + 1:current_index + 1]))),
                "current_to_anchor_path_cells": [[int(r), int(c)] for r, c in path_current_to_anchor],
                "route_state_counts": {state: route_states.count(state) for state in ("free", "unknown", "occupied")},
                "first_edge": {
                    "cell": list(first_edge[0]) if first_edge else None,
                    "state": first_states[0] if first_states else "missing",
                    "state_counts": {state: first_states.count(state) for state in ("free", "unknown", "occupied")},
                    "safe_0p25m_prefix": bool(prefix_states) and all(state == "free" for state in prefix_states),
                    "prefix_state_counts": {state: prefix_states.count(state) for state in ("free", "unknown", "occupied")},
                },
                "route_contains_occupied": "occupied" in route_states,
                "route_contains_unknown": "unknown" in route_states,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "shadow_only": True,
        "audit_only": True,
        "decision_applied": False,
        "action_applied": False,
        "pixel_translation_allowed": False,
        "unknown_is_free": False,
        "gt_used_for_navigation": False,
        "trigger_step": trigger,
        "estimated_loop_onset_step": onset,
        "translation_node_count": len(nodes),
        "anchors": anchors,
        "anchor_order_contract": "raw_trigger_then_estimated_loop_onset_then_last_productive_pre_loop",
        "productive_definition": "pre_onset_translation_edge_not_immediate_ABA",
    }

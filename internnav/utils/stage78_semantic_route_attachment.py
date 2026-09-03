"""Read-only semantic-node attachment to a SparseOcc recovery route.

The helper deliberately does not infer traversability from semantic labels.
It only reports where an already observed semantic node falls relative to the
current known-free route and which SparseOcc state owns its centroid cell.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage78_semantic_route_attachment_v1"
STRUCTURAL_LABELS = {"door", "stairs", "wall", "floor", "window"}


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def attach_semantic_nodes_to_route(
    nodes: Sequence[Mapping[str, Any]],
    path_cells: Sequence[Sequence[int]],
    *,
    cell_size_m: float,
    max_route_distance_m: float = 0.75,
    min_multiview_observations: int = 2,
    min_mean_confidence: float = 0.35,
    max_nodes: int = 48,
) -> dict[str, Any]:
    """Summarize observed semantic nodes near one current recovery path.

    Callers must provide each node's map ``grid`` and
    ``occ_state_at_centroid`` from the current SparseOcc memory.  No semantic
    value is allowed to change that state.
    """
    size = max(1e-6, float(cell_size_m))
    route = [item for item in (_cell(value) for value in path_cells) if item]
    route_bound = []
    all_records = []
    state_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    stable_label_counts: Counter[str] = Counter()
    invalid_grid_count = 0

    for raw in nodes:
        node = dict(raw)
        grid = _cell(node.get("grid"))
        if grid is None:
            invalid_grid_count += 1
            continue
        label = str(node.get("label") or "other").strip().lower()
        confidence = _finite_float(node.get("mean_confidence"))
        source_observations = list(node.get("source_observations") or [])
        source_steps = list(node.get("source_steps") or [])
        observation_support = len(set(str(value) for value in source_observations))
        step_support = len(set(str(value) for value in source_steps))
        multiview_stable = bool(
            observation_support >= max(1, int(min_multiview_observations))
            and confidence is not None
            and confidence >= float(min_mean_confidence)
        )
        nearest_index = None
        nearest_distance_m = None
        if route:
            distances = [math.hypot(grid[0] - cell[0], grid[1] - cell[1]) for cell in route]
            nearest_index = min(range(len(distances)), key=distances.__getitem__)
            nearest_distance_m = float(distances[nearest_index] * size)
        near_route = bool(
            nearest_distance_m is not None
            and nearest_distance_m <= float(max_route_distance_m) + 1e-9
        )
        centroid = list(node.get("centroid") or [])
        height_m = _finite_float(centroid[2]) if len(centroid) >= 3 else None
        state = str(node.get("occ_state_at_centroid") or "unknown")
        if state not in {"free", "occupied", "unknown"}:
            state = "unknown"
        record = {
            "node_id": node.get("node_id"),
            "label": label,
            "grid": [grid[0], grid[1]],
            "centroid": centroid[:3],
            "height_m": height_m,
            "occ_state_at_centroid": state,
            "point_count": int(node.get("point_count", 0) or 0),
            "mean_confidence": confidence,
            "evidence_tier": str(node.get("evidence_tier") or "unknown"),
            "observation_support": observation_support,
            "step_support": step_support,
            "multiview_stable": multiview_stable,
            "structural_label": label in STRUCTURAL_LABELS,
            "nearest_route_index": nearest_index,
            "nearest_route_distance_m": nearest_distance_m,
            "distance_along_route_m": (
                None if nearest_index is None else float(nearest_index * size)
            ),
            "near_route": near_route,
        }
        all_records.append(record)
        state_counts[state] += 1
        label_counts[label] += 1
        if multiview_stable:
            stable_label_counts[label] += 1
        if near_route:
            route_bound.append(record)

    def ranking(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            not bool(record.get("multiview_stable")),
            not bool(record.get("structural_label")),
            float(record.get("nearest_route_distance_m") or 0.0),
            -int(record.get("observation_support", 0) or 0),
            -float(record.get("mean_confidence") or 0.0),
            str(record.get("node_id") or ""),
        )

    route_bound.sort(key=ranking)
    stable_bound = [record for record in route_bound if record["multiview_stable"]]
    structural_bound = [record for record in route_bound if record["structural_label"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": bool(route),
        "reason": "ok" if route else "missing_route_path",
        "path_cell_count": len(route),
        "semantic_node_count": len(all_records),
        "invalid_grid_count": invalid_grid_count,
        "route_bound_node_count": len(route_bound),
        "stable_route_bound_node_count": len(stable_bound),
        "structural_route_bound_node_count": len(structural_bound),
        "label_counts": dict(sorted(label_counts.items())),
        "stable_label_counts": dict(sorted(stable_label_counts.items())),
        "occ_state_at_centroid_counts": dict(sorted(state_counts.items())),
        "route_bound_nodes": route_bound[: max(1, int(max_nodes))],
        "cell_size_m": size,
        "max_route_distance_m": float(max_route_distance_m),
        "min_multiview_observations": int(min_multiview_observations),
        "min_mean_confidence": float(min_mean_confidence),
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "prompt_injected": False,
        "action_applied": False,
        "shadow_only": True,
        "gt_fields_used": [],
    }

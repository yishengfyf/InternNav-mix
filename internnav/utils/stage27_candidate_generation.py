"""Stage27/M3 shadow candidate generation.

The generator is deliberately a data-only adapter.  It exposes the executed
route as strong historical support, but never treats that support as a
replacement for current SparseOcc evidence.  In particular, ``unknown`` is
never converted to ``free`` and an OCC conflict is retained in the record.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _grid_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


_SEMANTIC_INSTRUCTION_TERMS = {
    "door": {"door", "doorway", "entrance", "entry"},
    "chair": {"chair", "chairs", "seat"},
    "table": {"table", "tables", "desk"},
    "stairs": {"stairs", "stair", "staircase", "steps"},
    "sofa": {"sofa", "couch"},
    "bed": {"bed", "bedroom"},
    "cabinet": {"cabinet", "cabinets", "drawer", "drawers", "chest"},
    "window": {"window", "windows"},
    "wall": {"wall", "walls"},
    "floor": {"floor", "ground"},
    "shelving": {"shelf", "shelves", "shelving"},
    "closet": {"closet", "wardrobe"},
    "painting": {"painting", "picture", "artwork"},
}


def _estimate_local_floor_z_from_occ(
    occ_counts: Mapping[Tuple[int, int, int], int],
    row: int,
    col: int,
    *,
    cell_size_m: float,
    radius_m: float = 0.75,
    min_z_m: float = 0.0,
    max_z_m: float = 0.80,
    min_support_cells: int = 8,
    min_support_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Conservatively estimate a local floor height from observed surfaces.

    This is a readout-only aid for maps whose pose z is unavailable.  It uses
    only occupied endpoint evidence, requires a spatially broad low surface,
    and returns ``accepted=False`` when the map does not support a stable
    estimate.  It never changes the underlying occupied/free/unknown sets.
    """
    cs = max(1e-6, float(cell_size_m))
    radius_cells = max(1, int(math.ceil(float(radius_m) / cs)))
    min_height = int(math.floor(float(min_z_m) / cs))
    max_height = int(math.ceil(float(max_z_m) / cs))
    by_height: Dict[int, set[Tuple[int, int]]] = {}
    for key in occ_counts:
        try:
            cell_row, cell_col, height = (int(key[0]), int(key[1]), int(key[2]))
        except (TypeError, ValueError, IndexError):
            continue
        if abs(cell_row - int(row)) > radius_cells or abs(cell_col - int(col)) > radius_cells:
            continue
        if (cell_row - int(row)) ** 2 + (cell_col - int(col)) ** 2 > radius_cells ** 2:
            continue
        if height < min_height or height > max_height:
            continue
        by_height.setdefault(height, set()).add((cell_row, cell_col))
    if not by_height:
        return {
            "accepted": False,
            "floor_z_m": 0.0,
            "support_cells": 0,
            "support_ratio": 0.0,
            "height_bin": None,
            "source": "gps_compass_2d_fallback",
        }

    # Merge adjacent 5cm bins so quantization does not split one floor plane.
    candidates: List[Tuple[int, int, set[Tuple[int, int]]]] = []
    for height in sorted(by_height):
        support = set()
        for neighbor in (height - 1, height, height + 1):
            support.update(by_height.get(neighbor, set()))
        candidates.append((len(support), height, support))
    support_cells, height, support = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    all_cells = set().union(*(value for value in by_height.values()))
    support_ratio = float(support_cells / max(1, len(all_cells)))
    if (
        support_cells < max(1, int(min_support_cells))
        or support_ratio < float(min_support_ratio)
    ):
        return {
            "accepted": False,
            "floor_z_m": 0.0,
            "support_cells": int(support_cells),
            "support_ratio": support_ratio,
            "height_bin": int(height),
            "source": "gps_compass_2d_fallback_insufficient_surface",
        }
    return {
        "accepted": True,
        "floor_z_m": float(height * cs),
        "support_cells": int(support_cells),
        "support_ratio": support_ratio,
        "height_bin": int(height),
        "source": "local_occupied_floor_surface",
    }


def _instruction_semantic_labels(instruction: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(instruction or "").lower()))
    return {
        label for label, terms in _SEMANTIC_INSTRUCTION_TERMS.items()
        if tokens.intersection(terms)
    }


def _semantic_route_reobserve_candidates(
    *,
    semantic_nodes: Sequence[Mapping[str, Any]],
    route_candidates: Sequence[Mapping[str, Any]],
    instruction: str,
    branch: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    relevant_labels = _instruction_semantic_labels(instruction)
    relevant_nodes = [
        dict(node) for node in semantic_nodes
        if str(node.get("label") or "").lower() in relevant_labels
        and list(node.get("source_steps") or [])
    ]
    neighbors_per_node = max(
        1, int(config.get("semantic_route_neighbors_per_node", 3))
    )
    proposals: Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}
    for node in relevant_nodes:
        source_steps = []
        for value in node.get("source_steps") or []:
            try:
                source_steps.append(int(value))
            except (TypeError, ValueError):
                continue
        if not source_steps:
            continue
        ranked_route = sorted(
            route_candidates,
            key=lambda candidate: min(
                abs(int(candidate.get("source_step", -10**9)) - step)
                for step in source_steps
            ),
        )[:neighbors_per_node]
        for candidate in ranked_route:
            step_distance = min(
                abs(int(candidate.get("source_step", -10**9)) - step)
                for step in source_steps
            )
            priority = (
                0 if node.get("evidence_tier") == "strong" else 1,
                -float(node.get("mean_confidence", 0.0) or 0.0),
                -int(node.get("point_count", 0) or 0),
                int(step_distance),
                -max(source_steps),
            )
            item = dict(candidate)
            item["source_type"] = f"S-route-reobserve-{branch}"
            item["source_families"] = [item["source_type"]]
            item["semantic_role"] = "proposal_only_not_safety_evidence"
            item["semantic_evidence"] = {
                "branch": str(branch),
                "node_id": node.get("node_id"),
                "label": node.get("label"),
                "evidence_tier": node.get("evidence_tier"),
                "mean_confidence": node.get("mean_confidence"),
                "point_count": node.get("point_count"),
                "source_steps": source_steps,
                "route_step_distance": int(step_distance),
                "instruction_relevant": True,
                "safety_vote": False,
            }
            candidate_id = str(item.get("candidate_id"))
            previous = proposals.get(candidate_id)
            if previous is None or priority < previous[0]:
                proposals[candidate_id] = (priority, item)
    ordered = [item for _, item in sorted(proposals.values(), key=lambda value: value[0])]
    safe = [item for item in ordered if _passes(item, config, "route_occ_clearance")]
    selected = safe[:max(1, int(config.get("semantic_candidate_count", 3)))]
    return {
        "instruction_relevant_labels": sorted(relevant_labels),
        "semantic_node_count": len(semantic_nodes),
        "relevant_semantic_node_count": len(relevant_nodes),
        "proposed_candidate_count": len(ordered),
        "safe_proposed_candidate_count": len(safe),
        "selected_candidates": selected,
    }


def _dedupe_translation_nodes(nodes: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in nodes:
        try:
            grid = [int(item["grid"][0]), int(item["grid"][1])]
            step = int(item.get("step_id", len(result)))
            xy = [float(item["xy"][0]), float(item["xy"][1])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        z = float(item.get("z", 0.0) or 0.0)
        z_source = str(item.get("z_source") or "unspecified")
        if result and _distance(result[-1]["xy"], xy) <= 1e-4:
            result[-1] = {
                **result[-1], "step_id": step, "grid": grid, "xy": xy,
                "z": z, "z_source": z_source,
            }
            continue
        result.append({
            "step_id": step, "grid": grid, "xy": xy,
            "z": z, "z_source": z_source,
        })
    return result


def _path_cells_for_node(
    node_index: int,
    nodes: Sequence[Mapping[str, Any]],
    *,
    rasterize_edge,
    sample_spacing_m: float,
) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for first, second in zip(nodes[node_index:], nodes[node_index + 1 :]):
        length = _distance(first["xy"], second["xy"])
        edge = rasterize_edge(
            first["grid"],
            second["grid"],
            edge_length_m=length,
            sample_spacing_m=sample_spacing_m,
        )
        for cell in edge:
            pair = (int(cell[0]), int(cell[1]))
            if not cells or pair != cells[-1]:
                cells.append(pair)
    return cells


def _call_state_fn(state_fn, row: int, col: int, floor_z_m: Optional[float] = None):
    """Call a 2-D state reader, or its optional floor-aware form."""
    if floor_z_m is None:
        return state_fn(int(row), int(col))
    try:
        return state_fn(int(row), int(col), float(floor_z_m))
    except TypeError:
        return state_fn(int(row), int(col))


def _state_counts(
    cells: Iterable[Tuple[int, int]], state_fn, *, floor_z_m: Optional[float] = None
) -> Counter:
    counts = Counter()
    for row, col in cells:
        counts[str(_call_state_fn(state_fn, row, col, floor_z_m))] += 1
    for name in ("free", "occupied", "unknown"):
        counts.setdefault(name, 0)
    return counts


def _local_state_counts(
    grid: Sequence[int], state_fn, *, radius_cells: int
) -> Counter:
    cells = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if math.hypot(dr, dc) <= radius_cells:
                cells.append((int(grid[0]) + dr, int(grid[1]) + dc))
    return _state_counts(cells, state_fn)


def _candidate_record(
    *,
    source_type: str,
    node: Mapping[str, Any],
    node_index: int,
    trigger_grid: Sequence[int],
    path_cells: Sequence[Tuple[int, int]],
    state_counts: Mapping[str, int],
    floor_state_counts: Optional[Mapping[str, int]],
    floor_z_m: float,
    floor_z_source: str,
    local_state_counts: Mapping[str, int],
    path_length_m: float,
    floor_aligned_height_max_m: float,
    footprint_radius_m: float,
    route_node_count: int,
    semantic: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    total = max(1, sum(int(value) for value in state_counts.values()))
    local_total = max(1, sum(int(value) for value in local_state_counts.values()))
    semantic = dict(semantic or {})
    return {
        "candidate_id": f"{source_type}:{node.get('step_id', node_index)}",
        "source_type": source_type,
        "route_node_index": int(node_index),
        "source_step": node.get("step_id"),
        "grid": [int(node["grid"][0]), int(node["grid"][1])],
        "xy": [float(node["xy"][0]), float(node["xy"][1])],
        "trigger_grid": [int(trigger_grid[0]), int(trigger_grid[1])],
        "route_node_count": int(route_node_count),
        "path_cells": [[int(row), int(col)] for row, col in path_cells],
        "path_length_m": float(path_length_m),
        "route_support": "executed_transition_chain",
        "route_support_edge_count": int(max(0, route_node_count - node_index - 1)),
        "state_counts": {name: int(state_counts.get(name, 0)) for name in ("free", "occupied", "unknown")},
        "free_fraction": float(state_counts.get("free", 0) / total),
        "occupied_fraction": float(state_counts.get("occupied", 0) / total),
        "unknown_fraction": float(state_counts.get("unknown", 0) / total),
        "route_occ_conflict": bool(state_counts.get("occupied", 0)),
        "floor_aligned_height_max_m": float(floor_aligned_height_max_m),
        "floor_z_m": float(floor_z_m),
        "floor_z_source": str(floor_z_source),
        "floor_z_estimate_support_cells": int(
            node.get("floor_z_estimate_support_cells", 0) or 0
        ),
        "floor_z_estimate_support_ratio": float(
            node.get("floor_z_estimate_support_ratio", 0.0) or 0.0
        ),
        "footprint_radius_m": float(footprint_radius_m),
        "floor_aligned_known_free": bool(
            (floor_state_counts or state_counts).get("free", 0) > 0
            and (floor_state_counts or state_counts).get("occupied", 0) == 0
            and (floor_state_counts or state_counts).get("unknown", 0) == 0
        ),
        "floor_aligned_state_counts": {
            name: int((floor_state_counts or state_counts).get(name, 0))
            for name in ("free", "occupied", "unknown")
        },
        "local_state_counts": {
            name: int(local_state_counts.get(name, 0))
            for name in ("free", "occupied", "unknown")
        },
        "local_free_fraction": float(local_state_counts.get("free", 0) / local_total),
        "local_occupied_fraction": float(local_state_counts.get("occupied", 0) / local_total),
        "local_unknown_fraction": float(local_state_counts.get("unknown", 0) / local_total),
        "semantic_evidence": semantic,
        "gt_fields_used": [],
        "shadow_only": True,
        "action_applied": False,
    }


def _passes(candidate: Mapping[str, Any], config: Mapping[str, Any], stage: str) -> bool:
    if float(candidate.get("path_length_m", 0.0)) < float(config.get("min_distance_m", 0.50)):
        return False
    if float(candidate.get("path_length_m", 0.0)) > float(config.get("max_distance_m", 4.0)):
        return False
    if stage in {"route_occ", "route_occ_clearance"}:
        if float(candidate.get("occupied_fraction", 0.0)) > float(config.get("max_occupied_fraction", 0.0)):
            return False
        if float(candidate.get("unknown_fraction", 0.0)) > float(config.get("max_unknown_fraction", 0.0)):
            return False
    if stage == "route_occ_clearance":
        if not bool(candidate.get("floor_aligned_known_free")):
            return False
    return True


def _dedupe(candidates: Sequence[Mapping[str, Any]], min_separation_m: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in candidates:
        if any(_distance(item["xy"], other["xy"]) < float(min_separation_m) for other in selected):
            continue
        selected.append(dict(item))
    return selected


def _sample_evenly(items: Sequence[Any], limit: int) -> List[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    return [
        items[int(round(index * (len(items) - 1) / (limit - 1)))]
        for index in range(limit)
    ]


def _known_free_geodesic_paths(
    start: Sequence[int],
    targets: Sequence[Sequence[int]],
    *,
    free_cells: Iterable[Tuple[int, int]],
    neighbors_fn,
    cell_size_m: float,
    max_distance_m: float,
    max_visited_cells: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Find bounded known-free paths without diagonal corner cutting."""
    start_cell = (int(start[0]), int(start[1]))
    remaining = {(int(cell[0]), int(cell[1])) for cell in targets}
    free = {(int(cell[0]), int(cell[1])) for cell in free_cells}
    distance = {start_cell: 0.0}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
    queue = [(0.0, start_cell)]
    reached: Dict[Tuple[int, int], Dict[str, Any]] = {}
    visited = 0
    while queue and remaining and visited < max(1, int(max_visited_cells)):
        path_m, cell = heapq.heappop(queue)
        if path_m > distance.get(cell, float("inf")) + 1e-9:
            continue
        if path_m > float(max_distance_m) + 1e-9:
            break
        visited += 1
        if cell in remaining:
            path = []
            cursor: Optional[Tuple[int, int]] = cell
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            reached[cell] = {
                "path_cells": path,
                "path_length_m": float(path_m),
                "visited_cell_count": int(visited),
            }
            remaining.remove(cell)
        for raw_neighbor in neighbors_fn(cell[0], cell[1]):
            neighbor = (int(raw_neighbor[0]), int(raw_neighbor[1]))
            if neighbor not in free:
                continue
            dr = int(neighbor[0] - cell[0])
            dc = int(neighbor[1] - cell[1])
            if dr and dc:
                if (cell[0] + dr, cell[1]) not in free:
                    continue
                if (cell[0], cell[1] + dc) not in free:
                    continue
            next_m = float(path_m + math.hypot(dr, dc) * float(cell_size_m))
            if next_m > float(max_distance_m) + 1e-9:
                continue
            if next_m + 1e-9 >= distance.get(neighbor, float("inf")):
                continue
            distance[neighbor] = next_m
            parent[neighbor] = cell
            heapq.heappush(queue, (next_m, neighbor))
    return reached


def _frontier_standoff_path(
    path_cells: Sequence[Sequence[int]], *, cell_size_m: float, standoff_m: float
) -> Dict[str, Any]:
    cells = [(int(cell[0]), int(cell[1])) for cell in path_cells]
    if not cells:
        return {"path_cells": [], "path_length_m": 0.0, "standoff_m": 0.0}
    index = len(cells) - 1
    backed_off_m = 0.0
    while index > 0 and backed_off_m + 1e-9 < float(standoff_m):
        first, second = cells[index - 1], cells[index]
        backed_off_m += math.hypot(
            second[0] - first[0], second[1] - first[1]
        ) * float(cell_size_m)
        index -= 1
    candidate_path = cells[: index + 1]
    path_length_m = sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        * float(cell_size_m)
        for first, second in zip(candidate_path, candidate_path[1:])
    )
    return {
        "path_cells": candidate_path,
        "path_length_m": float(path_length_m),
        "standoff_m": float(backed_off_m),
    }


def generate_stage27_candidates(
    *,
    route_nodes: Sequence[Mapping[str, Any]],
    trigger_grid: Sequence[int],
    state_fn,
    rasterize_edge,
    floor_state_fn=None,
    semantic_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    semantic_raw_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    semantic_filtered_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    instruction: str = "",
    frontier_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate M3 candidate families and cumulative safety ablations.

    ``route_nodes`` must contain only poses observed before the trigger.  The
    function is suitable for live shadow calls and replay audits alike.
    """
    cfg = dict(config or {})
    semantic_candidate_enabled = bool(cfg.get("semantic_candidate_enable", False))
    nodes = _dedupe_translation_nodes(route_nodes)
    trigger = [int(trigger_grid[0]), int(trigger_grid[1])]
    result: Dict[str, Any] = {
        "event_schema_version": (
            "stage27_m3_candidate_generation_v6"
            if semantic_candidate_enabled else "stage27_m3_candidate_generation_v5"
        ),
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
        "trigger_grid": trigger,
        "route_node_count": len(nodes),
        "families": {},
        "ablation": {},
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "candidate_direction_count": 0,
        "unknown_is_free": False,
        "candidate_pool_contract": "R-route-near_union_R-route-open",
        "frontier_pool_contract": "F-local-known-safe-frontier",
        "semantic_candidate_enabled": semantic_candidate_enabled,
        "semantic_pool_contract": "instruction_relevant_LSeg_to_executed_route_reobserve",
        "semantic_safety_contract": "proposal_only_then_route_OCC_strict_unknown_1.5m_full_footprint",
    }
    if len(nodes) < 2:
        result["reason"] = "insufficient_executed_route_nodes"
        return result

    # Candidate source points are strictly before the current pose and are
    # ordered nearest-first to preserve the route-near baseline.
    eligible_indices = [
        index for index, node in enumerate(nodes[:-1])
        if float(_grid_distance(node["grid"], trigger)) * float(cfg.get("cell_size_m", 0.05))
        >= float(cfg.get("min_distance_m", 0.50))
    ]
    eligible_indices.sort(key=lambda index: _grid_distance(nodes[index]["grid"], trigger))
    if not eligible_indices:
        result["reason"] = "no_distance_eligible_route_node"
        return result

    route_candidates: List[Dict[str, Any]] = []
    for index in eligible_indices:
        path_cells = _path_cells_for_node(
            index, nodes,
            rasterize_edge=rasterize_edge,
            sample_spacing_m=float(cfg.get("sample_spacing_m", 0.05)),
        )
        states = _state_counts(path_cells, state_fn)
        floor_z_m = float(nodes[index].get("z", 0.0) or 0.0)
        floor_z_source = str(nodes[index].get("z_source") or "unspecified")
        local_states = _local_state_counts(
            nodes[index]["grid"], state_fn,
            radius_cells=max(1, int(math.ceil(
                float(cfg.get("open_radius_m", 0.50))
                / float(cfg.get("cell_size_m", 0.05))
            ))),
        )
        floor_states = (
            _state_counts(path_cells, floor_state_fn, floor_z_m=floor_z_m)
            if floor_state_fn is not None else states
        )
        semantic = {}
        for item in semantic_nodes or ():
            if item.get("grid") and _grid_distance(item["grid"], nodes[index]["grid"]) <= float(cfg.get("semantic_bind_radius_cells", 50)):
                semantic = dict(item)
                break
        route_candidates.append(_candidate_record(
            source_type="route_node",
            node=nodes[index], node_index=index, trigger_grid=trigger,
            path_cells=path_cells, state_counts=states,
            floor_state_counts=floor_states,
            floor_z_m=floor_z_m,
            floor_z_source=floor_z_source,
            local_state_counts=local_states,
            path_length_m=sum(
                _distance(first["xy"], second["xy"])
                for first, second in zip(nodes[index:], nodes[index + 1 :])
            ),
            floor_aligned_height_max_m=float(cfg.get("floor_aligned_height_max_m", 1.5)),
            footprint_radius_m=float(cfg.get("footprint_radius_m", 0.18)),
            route_node_count=len(nodes), semantic=semantic,
        ))

    min_path_m = float(cfg.get("min_distance_m", 0.50))
    max_path_m = float(cfg.get("max_distance_m", 4.0))
    path_eligible_candidates = [
        item for item in route_candidates
        if min_path_m <= float(item.get("path_length_m", 0.0)) <= max_path_m
    ]
    # Route-near means the shortest authoritative retreat along the executed
    # transition chain.  Spatial proximity alone is ambiguous after a loop.
    path_eligible_candidates.sort(
        key=lambda item: (
            float(item.get("path_length_m", 0.0)),
            -int(item.get("source_step", -1) or -1),
        )
    )
    near = [
        {**item, "source_type": "R-route-near", "source_families": ["R-route-near"]}
        for item in path_eligible_candidates[: max(1, int(cfg.get("near_count", 1)))]
    ]
    # Open is the best historical node by observed free fraction, then by
    # clearance and shorter path.  It remains a generator result, not a ranker.
    open_candidates = sorted(
        path_eligible_candidates,
        key=lambda item: (
            float(item.get("local_free_fraction", 0.0)),
            -float(item.get("local_unknown_fraction", 0.0)),
            -float(item.get("local_occupied_fraction", 0.0)),
            -float(item.get("path_length_m", 0.0)),
        ), reverse=True,
    )[: max(1, int(cfg.get("open_count", 1)))]
    open_candidates = [
        {**item, "source_type": "R-route-open", "source_families": ["R-route-open"]}
        for item in open_candidates
    ]
    families = {
        "R-route-near": _dedupe(near, float(cfg.get("min_separation_m", 0.25))),
        "R-route-open": _dedupe(open_candidates, float(cfg.get("min_separation_m", 0.25))),
    }
    result["families"] = families
    all_candidates = _dedupe(
        families["R-route-near"] + families["R-route-open"],
        float(cfg.get("min_separation_m", 0.25)),
    )
    for item in all_candidates:
        matching_families = [
            name for name, family in families.items()
            if any(candidate["candidate_id"] == item["candidate_id"] for candidate in family)
        ]
        item["source_families"] = matching_families
        item["source_type"] = "+".join(matching_families)
    result["route_candidate_universe_count"] = len(route_candidates)
    result["route_path_eligible_candidate_count"] = len(path_eligible_candidates)
    result["candidate_count"] = len(all_candidates)
    result["candidate_direction_count"] = len({
        round(math.degrees(math.atan2(
            float(item["xy"][1]) - float(nodes[-1]["xy"][1]),
            float(item["xy"][0]) - float(nodes[-1]["xy"][0]),
        )) / 45.0) for item in all_candidates
    })
    for stage in ("route_only", "route_occ", "route_occ_clearance"):
        pool = [item for item in all_candidates if _passes(item, cfg, stage)]
        result["ablation"][stage] = {
            "candidates": pool,
            "candidate_count": len(pool),
            "event_has_candidate": bool(pool),
            "safe_candidate_count": sum(
                int(item.get("occupied_fraction", 0.0) == 0.0 and item.get("unknown_fraction", 0.0) == 0.0)
                for item in pool
            ),
        }
    route_clearance_pool = result["ablation"]["route_occ_clearance"]["candidates"]
    frontier_triggered = (
        len(route_clearance_pool)
        < max(1, int(cfg.get("frontier_trigger_min_route_candidates", 1)))
    )
    frontier_candidates: List[Dict[str, Any]] = []
    if frontier_triggered:
        for frontier_index, node in enumerate(frontier_nodes or ()):
            try:
                grid = [int(node["grid"][0]), int(node["grid"][1])]
                xy = [float(node["xy"][0]), float(node["xy"][1])]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if any(
                _distance(xy, route_node["xy"])
                < float(cfg.get("frontier_min_route_separation_m", 0.25))
                for route_node in nodes
            ):
                continue
            path_length = float(
                node.get("path_length_m", _distance(nodes[-1]["xy"], xy))
            )
            if not (min_path_m <= path_length <= max_path_m):
                continue
            if node.get("path_cells"):
                path_cells = list(dict.fromkeys(
                    (int(cell[0]), int(cell[1]))
                    for cell in node.get("path_cells", [])
                ))
                path_geometry = str(node.get("path_geometry") or "known_free_geodesic")
            else:
                path_cells = list(dict.fromkeys(
                    (int(cell[0]), int(cell[1]))
                    for cell in rasterize_edge(
                        trigger, grid, edge_length_m=path_length,
                        sample_spacing_m=float(cfg.get("sample_spacing_m", 0.05)),
                    )
                ))
                path_geometry = "straight_line_fallback"
            states = _state_counts(path_cells, state_fn)
            floor_z_m = float(node.get("z", nodes[-1].get("z", 0.0)) or 0.0)
            floor_z_source = str(
                node.get("z_source") or nodes[-1].get("z_source") or "unspecified"
            )
            floor_states = (
                _state_counts(path_cells, floor_state_fn, floor_z_m=floor_z_m)
                if floor_state_fn is not None else states
            )
            local_states = _local_state_counts(
                grid, state_fn,
                radius_cells=max(1, int(math.ceil(
                    float(cfg.get("open_radius_m", 0.50))
                    / float(cfg.get("cell_size_m", 0.05))
                ))),
            )
            frontier_candidate = _candidate_record(
                source_type="F-local-known-safe-frontier",
                node={"step_id": node.get("step_id", frontier_index), "grid": grid, "xy": xy},
                node_index=-1, trigger_grid=trigger,
                path_cells=path_cells, state_counts=states,
                floor_state_counts=floor_states, floor_z_m=floor_z_m,
                floor_z_source=floor_z_source, local_state_counts=local_states,
                path_length_m=path_length,
                floor_aligned_height_max_m=float(cfg.get("floor_aligned_height_max_m", 1.5)),
                footprint_radius_m=float(cfg.get("footprint_radius_m", 0.18)),
                route_node_count=len(nodes),
            )
            frontier_candidate["route_support"] = "local_known_safe_frontier"
            frontier_candidate["route_support_edge_count"] = 0
            frontier_candidate["path_geometry"] = path_geometry
            frontier_candidate["frontier_boundary_grid"] = [
                int(value) for value in node.get("frontier_boundary_grid", grid)
            ]
            frontier_candidate["frontier_standoff_m"] = float(
                node.get("frontier_standoff_m", 0.0) or 0.0
            )
            frontier_candidates.append(frontier_candidate)
        frontier_candidates = _dedupe(
            frontier_candidates, float(cfg.get("min_separation_m", 0.25))
        )
    frontier_safe = [
        item for item in frontier_candidates
        if _passes(item, cfg, "route_occ_clearance")
    ]
    cumulative = _dedupe(
        route_clearance_pool + frontier_safe,
        float(cfg.get("min_separation_m", 0.25)),
    )
    result["frontier_triggered"] = bool(frontier_triggered)
    result["frontier_candidates"] = frontier_candidates
    result["frontier_candidate_count"] = len(frontier_candidates)
    result["frontier_safe_candidate_count"] = len(frontier_safe)
    result["ablation"]["route_occ_clearance_frontier"] = {
        "candidates": cumulative, "candidate_count": len(cumulative),
        "event_has_candidate": bool(cumulative), "safe_candidate_count": len(cumulative),
        "frontier_increment_count": max(0, len(cumulative) - len(route_clearance_pool)),
    }
    if semantic_candidate_enabled:
        semantic_triggered = len(cumulative) < max(
            1, int(cfg.get("semantic_trigger_min_base_candidates", 1))
        )
        semantic_reports = {}
        for branch, branch_nodes in (
            ("raw", list(semantic_raw_nodes or [])),
            ("filtered", list(semantic_filtered_nodes or [])),
        ):
            report = {
                "instruction_relevant_labels": sorted(
                    _instruction_semantic_labels(instruction)
                ),
                "semantic_node_count": len(branch_nodes),
                "relevant_semantic_node_count": 0,
                "proposed_candidate_count": 0,
                "safe_proposed_candidate_count": 0,
                "selected_candidates": [],
            }
            if semantic_triggered:
                report = _semantic_route_reobserve_candidates(
                    semantic_nodes=branch_nodes,
                    route_candidates=path_eligible_candidates,
                    instruction=instruction,
                    branch=branch,
                    config=cfg,
                )
            semantic_candidates = list(report.pop("selected_candidates"))
            semantic_cumulative = _dedupe(
                cumulative + semantic_candidates,
                float(cfg.get("min_separation_m", 0.25)),
            )
            stage = f"route_occ_clearance_frontier_semantic_{branch}"
            result["ablation"][stage] = {
                "candidates": semantic_cumulative,
                "candidate_count": len(semantic_cumulative),
                "event_has_candidate": bool(semantic_cumulative),
                "safe_candidate_count": len(semantic_cumulative),
                "semantic_increment_count": max(
                    0, len(semantic_cumulative) - len(cumulative)
                ),
            }
            semantic_reports[branch] = {
                **report,
                "selected_candidate_count": len(semantic_candidates),
                "increment_candidate_count": max(
                    0, len(semantic_cumulative) - len(cumulative)
                ),
            }
        result["semantic_triggered"] = bool(semantic_triggered)
        result["semantic_reports"] = semantic_reports
    result["eligible_candidate_count"] = int(result["ablation"]["route_occ_clearance"]["candidate_count"])
    result["reason"] = "ok"
    return result


def generate_from_sparse_memory(
    memory, *, trigger_grid: Sequence[int], config: Optional[Mapping[str, Any]] = None,
    semantic_raw_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    semantic_filtered_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    instruction: str = "",
) -> Dict[str, Any]:
    """Build a Stage27 event from a live ``SparseOccSemanticMemory`` object."""
    cfg = {"cell_size_m": float(getattr(memory, "cs", 0.05)), **dict(config or {})}
    raw_nodes = list(getattr(memory, "pose_trace", []) or [])
    nodes = [
        {
            "step_id": item.get("step_id"),
            "grid": [item.get("row"), item.get("col")],
            "xy": [item.get("x"), item.get("y")],
            "z": item.get("z", 0.0),
            "z_source": item.get("pose_height_source", "unspecified"),
        }
        for item in raw_nodes
        if item.get("row") is not None and item.get("col") is not None
    ]
    floor_estimation_enabled = bool(cfg.get("floor_z_estimation_enable", False))
    floor_estimation_accepted = 0
    if floor_estimation_enabled:
        adjusted_nodes = []
        for node in nodes:
            adjusted = dict(node)
            estimate = _estimate_local_floor_z_from_occ(
                getattr(memory, "occ_counts", {}),
                int(adjusted["grid"][0]),
                int(adjusted["grid"][1]),
                cell_size_m=float(getattr(memory, "cs", 0.05)),
                radius_m=float(cfg.get("floor_z_estimation_radius_m", 0.75)),
                min_z_m=float(cfg.get("floor_z_estimation_min_m", 0.0)),
                max_z_m=float(cfg.get("floor_z_estimation_max_m", 0.80)),
                min_support_cells=int(cfg.get("floor_z_estimation_min_support_cells", 8)),
                min_support_ratio=float(cfg.get("floor_z_estimation_min_support_ratio", 0.25)),
            )
            adjusted["floor_z_estimate_support_cells"] = int(
                estimate["support_cells"]
            )
            adjusted["floor_z_estimate_support_ratio"] = float(
                estimate["support_ratio"]
            )
            adjusted["floor_z_estimate_source"] = str(estimate["source"])
            if estimate["accepted"] and str(adjusted.get("z_source")) == "gps_compass_2d":
                adjusted["z"] = float(estimate["floor_z_m"])
                adjusted["z_source"] = str(estimate["source"])
                floor_estimation_accepted += 1
            adjusted_nodes.append(adjusted)
        nodes = adjusted_nodes
    radius_cells = max(
        0,
        int(math.ceil(float(cfg.get("footprint_radius_m", 0.18)) / float(getattr(memory, "cs", 0.05)))),
    )
    def floor_aligned_state(row, col, floor_z_m=None):
        # A historical route node can be on a different level from the
        # trigger (e.g. stairs). Read its footprint at that node's floor.
        floor_z = float(floor_z_m or 0.0)
        center = memory.validation_floor_aligned_cell_evidence(
            int(row), int(col), floor_z,
            height_max_m=float(cfg.get("floor_aligned_height_max_m", 1.5)),
        )
        all_free = center["state"] == "free"
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if math.hypot(dr, dc) > radius_cells:
                    continue
                evidence = memory.validation_floor_aligned_cell_evidence(
                    int(row) + dr, int(col) + dc, floor_z,
                    height_max_m=float(cfg.get("floor_aligned_height_max_m", 1.5)),
                )
                if evidence["state"] == "blocked":
                    return "occupied"
                all_free = all_free and evidence["state"] == "free"
        return "free" if all_free else "unknown"
    frontier_nodes = []
    frontier_local_cell_count = 0
    frontier_sampled_cell_count = 0
    if bool(cfg.get("known_safe_frontier_enable", False)) and hasattr(memory, "get_frontier_cells"):
        max_frontier_distance_m = float(cfg.get("frontier_search_radius_m", 4.0))
        local_frontiers = [
            cell for cell in memory.get_frontier_cells(sample_limit=0)
            if _grid_distance(cell, trigger_grid) * float(memory.cs)
            <= max_frontier_distance_m
        ]
        frontier_local_cell_count = len(local_frontiers)
        sampled_frontiers = _sample_evenly(
            local_frontiers, int(cfg.get("frontier_sample_limit", 512))
        )
        frontier_sampled_cell_count = len(sampled_frontiers)
        free_cells = set(getattr(memory, "free2d_counts", {}).keys()) - set(
            getattr(memory, "occ2d_counts", {}).keys()
        )
        paths = _known_free_geodesic_paths(
            trigger_grid, sampled_frontiers,
            free_cells=free_cells,
            neighbors_fn=memory._neighbors2d,
            cell_size_m=float(memory.cs),
            max_distance_m=max_frontier_distance_m,
            max_visited_cells=int(cfg.get("frontier_path_max_visited_cells", 30000)),
        )
        for index, cell in enumerate(sampled_frontiers):
            path = paths.get((int(cell[0]), int(cell[1])))
            if path is None:
                continue
            standoff = _frontier_standoff_path(
                path["path_cells"], cell_size_m=float(memory.cs),
                standoff_m=float(cfg.get("frontier_standoff_m", 0.25)),
            )
            if not standoff["path_cells"]:
                continue
            candidate_cell = standoff["path_cells"][-1]
            xy = memory._grid_to_xy(candidate_cell)
            nearest = min(
                nodes, key=lambda item: _grid_distance(item["grid"], candidate_cell), default=None
            )
            frontier_nodes.append({
                "step_id": f"frontier_{index}",
                "grid": [int(candidate_cell[0]), int(candidate_cell[1])],
                "xy": [float(xy[0]), float(xy[1])],
                "z": float((nearest or {}).get("z", 0.0) or 0.0),
                "z_source": str((nearest or {}).get("z_source", "unspecified")),
                "path_cells": standoff["path_cells"],
                "path_length_m": float(standoff["path_length_m"]),
                "path_geometry": "known_free_geodesic",
                "frontier_boundary_grid": [int(cell[0]), int(cell[1])],
                "frontier_standoff_m": float(standoff["standoff_m"]),
            })
    result = generate_stage27_candidates(
        route_nodes=nodes,
        trigger_grid=trigger_grid,
        state_fn=memory._cell_state,
        rasterize_edge=memory._rasterize_executed_route_edge,
        floor_state_fn=floor_aligned_state,
        semantic_nodes=getattr(memory, "semantic_anchors", None),
        semantic_raw_nodes=semantic_raw_nodes,
        semantic_filtered_nodes=semantic_filtered_nodes,
        instruction=instruction,
        frontier_nodes=frontier_nodes,
        config=cfg,
    )
    result["frontier_path_mode"] = "known_free_geodesic"
    result["frontier_local_cell_count"] = int(frontier_local_cell_count)
    result["frontier_sampled_cell_count"] = int(frontier_sampled_cell_count)
    result["frontier_geodesic_reachable_count"] = int(len(frontier_nodes))
    result["floor_z_estimation"] = {
        "enabled": floor_estimation_enabled,
        "accepted_node_count": int(floor_estimation_accepted),
        "node_count": int(len(nodes)),
        "uses_gt": False,
        "fallback": "gps_compass_2d",
    }
    return result

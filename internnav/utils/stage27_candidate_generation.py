"""Stage27/M3 shadow candidate generation.

The generator is deliberately a data-only adapter.  It exposes the executed
route as strong historical support, but never treats that support as a
replacement for current SparseOcc evidence.  In particular, ``unknown`` is
never converted to ``free`` and an OCC conflict is retained in the record.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _grid_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


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


def generate_stage27_candidates(
    *,
    route_nodes: Sequence[Mapping[str, Any]],
    trigger_grid: Sequence[int],
    state_fn,
    rasterize_edge,
    floor_state_fn=None,
    semantic_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    frontier_nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate M3 candidate families and cumulative safety ablations.

    ``route_nodes`` must contain only poses observed before the trigger.  The
    function is suitable for live shadow calls and replay audits alike.
    """
    cfg = dict(config or {})
    nodes = _dedupe_translation_nodes(route_nodes)
    trigger = [int(trigger_grid[0]), int(trigger_grid[1])]
    result: Dict[str, Any] = {
        "event_schema_version": "stage27_m3_candidate_generation_v4",
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
            path_length = _distance(nodes[-1]["xy"], xy)
            if not (min_path_m <= path_length <= max_path_m):
                continue
            path_cells = list(dict.fromkeys(
                (int(cell[0]), int(cell[1]))
                for cell in rasterize_edge(
                    trigger, grid, edge_length_m=path_length,
                    sample_spacing_m=float(cfg.get("sample_spacing_m", 0.05)),
                )
            ))
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
    result["eligible_candidate_count"] = int(result["ablation"]["route_occ_clearance"]["candidate_count"])
    result["reason"] = "ok"
    return result


def generate_from_sparse_memory(memory, *, trigger_grid: Sequence[int], config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
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
        any_free = center["state"] == "free"
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
                any_free = any_free or evidence["state"] == "free"
        return "free" if any_free else "unknown"
    frontier_nodes = []
    if bool(cfg.get("known_safe_frontier_enable", False)) and hasattr(memory, "get_frontier_cells"):
        max_frontier_distance_m = float(cfg.get("frontier_search_radius_m", 4.0))
        for index, cell in enumerate(memory.get_frontier_cells(
            sample_limit=int(cfg.get("frontier_sample_limit", 512))
        )):
            if _grid_distance(cell, trigger_grid) * float(memory.cs) > max_frontier_distance_m:
                continue
            xy = memory._grid_to_xy([int(cell[0]), int(cell[1])])
            nearest = min(
                nodes, key=lambda item: _grid_distance(item["grid"], cell), default=None
            )
            frontier_nodes.append({
                "step_id": f"frontier_{index}",
                "grid": [int(cell[0]), int(cell[1])],
                "xy": [float(xy[0]), float(xy[1])],
                "z": float((nearest or {}).get("z", 0.0) or 0.0),
                "z_source": str((nearest or {}).get("z_source", "unspecified")),
            })
    return generate_stage27_candidates(
        route_nodes=nodes,
        trigger_grid=trigger_grid,
        state_fn=memory._cell_state,
        rasterize_edge=memory._rasterize_executed_route_edge,
        floor_state_fn=floor_aligned_state,
        semantic_nodes=getattr(memory, "semantic_anchors", None),
        frontier_nodes=frontier_nodes,
        config=cfg,
    )

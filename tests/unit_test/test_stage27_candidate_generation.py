import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "internnav"
    / "utils"
    / "stage27_candidate_generation.py"
)
SPEC = importlib.util.spec_from_file_location("stage27_candidate_generation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_stage27_candidates = MODULE.generate_stage27_candidates
known_free_geodesic_paths = MODULE._known_free_geodesic_paths
frontier_standoff_path = MODULE._frontier_standoff_path


def _nodes():
    return [
        {"step_id": 0, "grid": [0, 0], "xy": [0.0, 0.0]},
        {"step_id": 1, "grid": [0, 5], "xy": [0.0, 0.25]},
        {"step_id": 2, "grid": [0, 10], "xy": [0.0, 0.50]},
        {"step_id": 3, "grid": [0, 15], "xy": [0.0, 0.75]},
    ]


def _rasterize(start, end, **_):
    lo, hi = sorted((int(start[1]), int(end[1])))
    return [(int(start[0]), col) for col in range(lo, hi + 1)]


def test_unknown_is_never_promoted_to_free():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if col == 12 else "free",
        rasterize_edge=_rasterize,
        config={"cell_size_m": 0.05, "min_distance_m": 0.25, "max_distance_m": 4.0},
    )
    assert result["reason"] == "ok"
    assert result["unknown_is_free"] is False
    assert result["ablation"]["route_only"]["event_has_candidate"] is True
    assert result["ablation"]["route_occ"]["event_has_candidate"] is False
    assert any(item["unknown_fraction"] > 0 for item in result["ablation"]["route_only"]["candidates"])


def test_occupied_conflict_is_explicit_and_filtered():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "occupied" if col == 12 else "free",
        rasterize_edge=_rasterize,
        config={"cell_size_m": 0.05, "min_distance_m": 0.25, "max_distance_m": 4.0},
    )
    route_only = result["ablation"]["route_only"]["candidates"]
    assert route_only
    assert all(item["route_occ_conflict"] for item in route_only)
    assert result["ablation"]["route_occ"]["event_has_candidate"] is False


def test_clearance_is_a_separate_hard_filter():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "free",
        rasterize_edge=_rasterize,
        floor_state_fn=lambda row, col: "unknown" if col == 12 else "free",
        config={"cell_size_m": 0.05, "min_distance_m": 0.25, "max_distance_m": 4.0},
    )
    assert result["ablation"]["route_occ"]["event_has_candidate"] is True
    assert result["ablation"]["route_occ_clearance"]["event_has_candidate"] is False


def test_route_near_and_open_families_remain_distinct_when_possible():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "free",
        rasterize_edge=_rasterize,
        config={
            "cell_size_m": 0.05,
            "min_distance_m": 0.25,
            "max_distance_m": 4.0,
            "near_count": 1,
            "open_count": 1,
        },
    )
    near = result["families"]["R-route-near"][0]
    opened = result["families"]["R-route-open"][0]
    assert near["source_step"] == 2
    assert opened["source_step"] in {0, 1, 2}
    assert near["shadow_only"] and not near["action_applied"]
    assert near["gt_fields_used"] == []
    assert result["route_candidate_universe_count"] == 3
    assert result["candidate_count"] <= 2
    assert all(
        candidate["source_families"]
        for candidate in result["ablation"]["route_only"]["candidates"]
    )


def test_floor_readout_uses_source_route_node_height():
    nodes = [dict(item, z=0.0 if item["step_id"] < 2 else 1.0) for item in _nodes()]
    seen_heights = []

    def floor_state(row, col, floor_z_m):
        seen_heights.append(float(floor_z_m))
        return "free" if float(floor_z_m) == 0.0 else "occupied"

    result = generate_stage27_candidates(
        route_nodes=nodes,
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "free",
        rasterize_edge=_rasterize,
        floor_state_fn=floor_state,
        config={"cell_size_m": 0.05, "min_distance_m": 0.25, "max_distance_m": 4.0},
    )
    assert seen_heights
    assert result["families"]["R-route-near"][0]["floor_z_m"] == 1.0
    assert result["families"]["R-route-near"][0]["floor_aligned_known_free"] is False


def test_route_near_uses_executed_path_distance_after_loop():
    nodes = [
        {"step_id": 0, "grid": [0, 12], "xy": [0.0, 0.6]},
        {"step_id": 1, "grid": [0, 40], "xy": [0.0, 2.0]},
        {"step_id": 2, "grid": [0, 60], "xy": [0.0, 3.0]},
        {"step_id": 3, "grid": [0, 20], "xy": [0.0, 1.0]},
        {"step_id": 4, "grid": [0, 0], "xy": [0.0, 0.0]},
    ]
    result = generate_stage27_candidates(
        route_nodes=nodes,
        trigger_grid=[0, 0],
        state_fn=lambda row, col: "free",
        rasterize_edge=_rasterize,
        config={
            "cell_size_m": 0.05,
            "min_distance_m": 0.50,
            "max_distance_m": 4.0,
            "near_count": 1,
            "open_count": 1,
        },
    )
    near = result["families"]["R-route-near"][0]
    assert near["source_step"] == 3
    assert near["path_length_m"] == 1.0
    assert result["route_candidate_universe_count"] == 4
    assert result["route_path_eligible_candidate_count"] == 3
    assert result["ablation"]["route_only"]["event_has_candidate"] is True


def _frontier_nodes():
    return [
        {"step_id": "frontier_0", "grid": [0, 25], "xy": [0.0, 1.25], "z": 0.0},
        {"step_id": "frontier_near_route", "grid": [0, 16], "xy": [0.0, 0.8], "z": 0.0},
    ]


def _frontier_config():
    return {
        "cell_size_m": 0.05,
        "min_distance_m": 0.25,
        "max_distance_m": 4.0,
        "frontier_trigger_min_route_candidates": 1,
        "frontier_min_route_separation_m": 0.25,
    }


def test_known_safe_frontier_is_only_fallback_after_route_clearance_empty():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if (row, col) == (0, 12) else "free",
        rasterize_edge=_rasterize,
        frontier_nodes=_frontier_nodes(),
        config=_frontier_config(),
    )
    assert result["frontier_triggered"] is True
    assert result["ablation"]["route_occ_clearance"]["event_has_candidate"] is False
    frontier = result["ablation"]["route_occ_clearance_frontier"]
    assert frontier["event_has_candidate"] is True
    assert frontier["frontier_increment_count"] >= 1
    assert all(item["source_type"].startswith("F-local-known-safe-frontier") for item in frontier["candidates"])


def test_frontier_path_unknown_is_rejected():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if (row, col) in {(0, 12), (0, 20)} else "free",
        rasterize_edge=_rasterize,
        frontier_nodes=_frontier_nodes()[:1],
        config=_frontier_config(),
    )
    assert result["frontier_triggered"] is True
    assert result["frontier_candidate_count"] == 1
    assert result["frontier_safe_candidate_count"] == 0
    assert result["ablation"]["route_occ_clearance_frontier"]["event_has_candidate"] is False


def test_frontier_path_occupied_is_rejected():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "occupied" if (row, col) in {(0, 12), (0, 20)} else "free",
        rasterize_edge=_rasterize,
        frontier_nodes=_frontier_nodes()[:1],
        config=_frontier_config(),
    )
    assert result["frontier_triggered"] is True
    assert result["frontier_safe_candidate_count"] == 0


def test_frontier_not_added_when_route_clearance_candidate_exists():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "free",
        rasterize_edge=_rasterize,
        frontier_nodes=_frontier_nodes(),
        config=_frontier_config(),
    )
    assert result["frontier_triggered"] is False
    assert result["frontier_candidate_count"] == 0
    assert result["ablation"]["route_occ_clearance_frontier"]["candidate_count"] == result["ablation"]["route_occ_clearance"]["candidate_count"]


def test_frontier_near_executed_route_is_separated():
    result = generate_stage27_candidates(
        route_nodes=_nodes(),
        trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if (row, col) == (0, 12) else "free",
        rasterize_edge=_rasterize,
        frontier_nodes=_frontier_nodes(),
        config=_frontier_config(),
    )
    assert result["frontier_triggered"] is True
    assert all(item["source_step"] != "frontier_near_route" for item in result["frontier_candidates"])
    assert result["event_schema_version"] == "stage27_m3_candidate_generation_v5"
    assert result["shadow_only"] is True and result["action_applied"] is False


def _neighbors8(row, col):
    return [
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def test_known_free_geodesic_routes_around_blocked_straight_line():
    free = {(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)}
    paths = known_free_geodesic_paths(
        [0, 0], [[0, 2]],
        free_cells=free, neighbors_fn=_neighbors8,
        cell_size_m=0.1, max_distance_m=1.0, max_visited_cells=100,
    )
    path = paths[(0, 2)]
    assert (0, 1) not in path["path_cells"]
    assert path["path_cells"][0] == (0, 0)
    assert path["path_cells"][-1] == (0, 2)
    assert path["path_length_m"] == 0.4


def test_known_free_geodesic_does_not_cut_unknown_diagonal_corner():
    paths = known_free_geodesic_paths(
        [0, 0], [[1, 1]],
        free_cells={(0, 0), (1, 1)}, neighbors_fn=_neighbors8,
        cell_size_m=0.1, max_distance_m=1.0, max_visited_cells=100,
    )
    assert paths == {}


def test_frontier_standoff_moves_candidate_inside_known_path():
    standoff = frontier_standoff_path(
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        cell_size_m=0.05, standoff_m=0.18,
    )
    assert standoff["path_cells"] == [(0, 0), (0, 1)]
    assert standoff["path_length_m"] == 0.05
    assert round(standoff["standoff_m"], 6) == 0.2


def test_frontier_uses_precomputed_known_free_geodesic_path():
    frontier = [{
        "step_id": "frontier_detour", "grid": [2, 25], "xy": [0.1, 1.25],
        "path_cells": [(0, 15), (1, 15), (2, 15), (2, 20), (2, 25)],
        "path_length_m": 0.6, "path_geometry": "known_free_geodesic",
    }]
    result = generate_stage27_candidates(
        route_nodes=_nodes(), trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if (row, col) == (0, 12) else "free",
        rasterize_edge=_rasterize, frontier_nodes=frontier,
        config=_frontier_config(),
    )
    candidate = result["ablation"]["route_occ_clearance_frontier"]["candidates"][0]
    assert candidate["source_type"] == "F-local-known-safe-frontier"
    assert candidate["path_geometry"] == "known_free_geodesic"
    assert candidate["path_cells"] == [[0, 15], [1, 15], [2, 15], [2, 20], [2, 25]]


def _semantic_nodes(label="bed"):
    return [{
        "node_id": "SN00001", "label": label, "source_steps": [0],
        "evidence_tier": "strong", "mean_confidence": 0.72,
        "point_count": 48,
    }]


def test_semantic_route_reobserve_adds_only_hard_safe_executed_route_node():
    nodes = [
        {**item, "z": 0.0 if item["step_id"] == 0 else 1.0}
        for item in _nodes()
    ]

    def floor_state(_row, _col, floor_z_m):
        return "free" if float(floor_z_m) == 0.0 else "occupied"

    result = generate_stage27_candidates(
        route_nodes=nodes, trigger_grid=[0, 15],
        state_fn=lambda row, col: "free", rasterize_edge=_rasterize,
        floor_state_fn=floor_state,
        semantic_raw_nodes=_semantic_nodes(),
        semantic_filtered_nodes=_semantic_nodes(),
        instruction="Leave the bedroom and turn toward the hall.",
        config={
            "cell_size_m": 0.05, "min_distance_m": 0.25,
            "max_distance_m": 4.0, "near_count": 1, "open_count": 1,
            "semantic_candidate_enable": True,
            "semantic_route_neighbors_per_node": 1,
            "semantic_candidate_count": 1,
        },
    )
    assert result["event_schema_version"] == "stage27_m3_candidate_generation_v6"
    assert not result["ablation"]["route_occ_clearance_frontier"]["event_has_candidate"]
    for branch in ("raw", "filtered"):
        stage = f"route_occ_clearance_frontier_semantic_{branch}"
        candidate = result["ablation"][stage]["candidates"][0]
        assert candidate["source_step"] == 0
        assert candidate["route_support"] == "executed_transition_chain"
        assert candidate["floor_aligned_known_free"] is True
        assert candidate["semantic_evidence"]["safety_vote"] is False
        assert candidate["semantic_role"] == "proposal_only_not_safety_evidence"


def test_semantic_candidate_requires_instruction_relevance():
    result = generate_stage27_candidates(
        route_nodes=_nodes(), trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown", rasterize_edge=_rasterize,
        semantic_raw_nodes=_semantic_nodes("chair"),
        semantic_filtered_nodes=_semantic_nodes("chair"),
        instruction="Walk down the hallway.",
        config={
            "cell_size_m": 0.05, "min_distance_m": 0.25,
            "max_distance_m": 4.0, "semantic_candidate_enable": True,
        },
    )
    for branch in ("raw", "filtered"):
        report = result["semantic_reports"][branch]
        assert report["relevant_semantic_node_count"] == 0
        assert report["selected_candidate_count"] == 0


def test_semantic_evidence_never_overrides_unknown_path():
    result = generate_stage27_candidates(
        route_nodes=_nodes(), trigger_grid=[0, 15],
        state_fn=lambda row, col: "unknown" if col <= 10 else "free",
        rasterize_edge=_rasterize,
        semantic_raw_nodes=_semantic_nodes(),
        semantic_filtered_nodes=_semantic_nodes(),
        instruction="Return toward the bed.",
        config={
            "cell_size_m": 0.05, "min_distance_m": 0.25,
            "max_distance_m": 4.0, "semantic_candidate_enable": True,
            "semantic_route_neighbors_per_node": 1,
        },
    )
    for branch in ("raw", "filtered"):
        stage = f"route_occ_clearance_frontier_semantic_{branch}"
        assert result["semantic_reports"][branch]["safe_proposed_candidate_count"] == 0
        assert not result["ablation"][stage]["event_has_candidate"]

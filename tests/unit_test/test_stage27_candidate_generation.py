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

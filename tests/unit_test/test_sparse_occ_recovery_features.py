from internnav.utils.sparse_occ_memory import SparseOccSemanticMemory


def _memory():
    return SparseOccSemanticMemory(
        None,
        enable=True,
        frontier_enable=False,
        grid_size=64,
        cell_size=0.25,
        semantic_resilience_anchor_feature_radius_cells=4,
        semantic_resilience_cycle_radius_cells=1,
        semantic_resilience_cycle_window_steps=16,
    )


def test_open_multi_sector_anchor_has_high_branch_count():
    memory = _memory()
    center = (32, 32)
    for row, col in ((31, 32), (30, 32), (33, 32), (34, 32), (32, 31), (32, 30), (32, 33), (32, 34)):
        memory.free2d_counts[(row, col)] = 1

    info = memory._anchor_spatial_information(center)

    assert info["branch_count"] == 4
    assert info["direction_entropy"] > 0.95


def test_a_b_a_trace_sets_short_cycle_risk():
    memory = _memory()
    memory.pose_trace = [
        {"row": 32, "col": 32, "step_id": 1},
        {"row": 32, "col": 36, "step_id": 2},
        {"row": 32, "col": 32, "step_id": 3},
    ]

    info = memory._anchor_trace_information((32, 32), latest_step=3)

    assert info["return_count"] == 2
    assert info["recent_cycle_count"] == 1
    assert info["short_cycle_risk"] > 0.0

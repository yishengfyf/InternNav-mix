import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage57_local_elevation_support.py"
_spec = importlib.util.spec_from_file_location("stage57_local_elevation_support", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


class _Memory:
    cs = 0.05

    def __init__(self):
        self.occ_counts = {
            (10, 10, 0): 4,
            (10, 11, 1): 4,
            (10, 12, 2): 4,
            (10, 13, 3): 4,
        }
        self.occ3d_frame_counts = {key: 2 for key in self.occ_counts}


def test_local_support_graph_accepts_continuous_step_sequence_without_mutation():
    memory = _Memory()
    before = dict(memory.occ_counts)
    report = _module.audit_local_elevation_support(
        memory,
        [[10, 10], [10, 11], [10, 12], [10, 13]],
        footprint_radius_m=0.0,
        max_step_up_m=0.06,
        max_step_down_m=0.06,
        support_search_radius_m=0.0,
    )
    assert report["continuous_support_centerline"] is True
    assert report["full_footprint_support"] is True
    assert report["full_footprint_safe_corridor"] is False
    assert report["eligible_corridor"] is False
    assert memory.occ_counts == before


def test_local_support_graph_rejects_unknown_corridor_and_large_step():
    memory = _Memory()
    del memory.occ_counts[(10, 12, 2)]
    report = _module.audit_local_elevation_support(
        memory,
        [[10, 10], [10, 11], [10, 12], [10, 13]],
        footprint_radius_m=0.0,
        max_step_up_m=0.06,
        max_step_down_m=0.06,
        support_search_radius_m=0.0,
    )
    assert report["centerline_support_coverage"] < 1.0
    assert report["continuous_support_centerline"] is False
    assert report["full_footprint_safe_corridor"] is False


def test_local_support_graph_rasterizes_sparse_edges_and_bridges_depth_sampling():
    memory = _Memory()
    memory.occ_counts[(10, 14, 4)] = 4
    memory.occ3d_frame_counts[(10, 14, 4)] = 2
    report = _module.audit_local_elevation_support(
        memory,
        [[10, 10], [10, 14]],
        footprint_radius_m=0.0,
        max_step_up_m=0.06,
        max_step_down_m=0.06,
        support_search_radius_m=0.05,
    )
    assert report["sparse_path_cell_count"] == 2
    assert report["path_cell_count"] == 5
    assert report["centerline_support_coverage"] == 1.0
    assert report["longest_full_footprint_safe_segment_m"] == 0.2


def test_local_support_graph_requires_minimum_safe_segment_length():
    memory = _Memory()
    for col in range(14, 17):
        key = (10, col, col - 10)
        memory.occ_counts[key] = 4
        memory.occ3d_frame_counts[key] = 2
    report = _module.audit_local_elevation_support(
        memory,
        [[10, 10], [10, 16]],
        footprint_radius_m=0.0,
        max_step_up_m=0.06,
        max_step_down_m=0.06,
        support_search_radius_m=0.0,
    )
    assert abs(report["longest_full_footprint_safe_segment_m"] - 0.3) < 1e-9
    assert report["full_footprint_safe_corridor"] is True

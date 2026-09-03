import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[2] / "internnav/utils/stage78_semantic_route_attachment.py"
_SPEC = importlib.util.spec_from_file_location("stage78_semantic_route_attachment", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
attach_semantic_nodes_to_route = _MODULE.attach_semantic_nodes_to_route


def test_stage78_binds_stable_structural_node_without_overriding_occ():
    report = attach_semantic_nodes_to_route(
        [
            {
                "node_id": "SN0",
                "label": "door",
                "grid": [10, 12],
                "centroid": [1.0, 2.0, 0.8],
                "occ_state_at_centroid": "occupied",
                "point_count": 30,
                "mean_confidence": 0.52,
                "source_observations": [1, 2, 2],
                "source_steps": [4, 8],
                "evidence_tier": "strong",
            }
        ],
        [[10, 10], [10, 11]],
        cell_size_m=0.05,
    )
    assert report["valid"] is True
    assert report["stable_route_bound_node_count"] == 1
    node = report["route_bound_nodes"][0]
    assert node["label"] == "door"
    assert node["occ_state_at_centroid"] == "occupied"
    assert node["nearest_route_distance_m"] == 0.05
    assert report["unknown_is_free"] is False
    assert report["semantic_can_override_safety"] is False
    assert report["prompt_injected"] is False


def test_stage78_rejects_single_view_as_stable_and_handles_missing_path():
    node = {
        "node_id": "SN1",
        "label": "stairs",
        "grid": [2, 3],
        "centroid": [0.0, 0.0, -0.2],
        "occ_state_at_centroid": "free",
        "mean_confidence": 0.8,
        "source_observations": [9],
        "source_steps": [9],
    }
    report = attach_semantic_nodes_to_route([node], [], cell_size_m=0.05)
    assert report["valid"] is False
    assert report["reason"] == "missing_route_path"
    assert report["stable_route_bound_node_count"] == 0
    assert report["semantic_node_count"] == 1

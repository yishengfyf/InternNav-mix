import importlib.util
from pathlib import Path

from PIL import Image


_path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "visualize_stage38_recovery_bev.py"
_spec = importlib.util.spec_from_file_location("stage38_recovery_bev_visualization", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_unknown_is_explicit_and_semantic_is_outline(tmp_path):
    anchor = {"anchor_id": "a", "capture": {"path_cells": [[0, 0], [0, 1]]}}
    digest = {
        "channels": {
            "known_free": [[0, 0]],
            "occupied": [],
            "unknown": [[0, 1]],
            "semantic": [[0, 1]],
        },
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
    }
    meta = _module.render_recovery_bev(anchor, digest, tmp_path / "bev.png", scale=8)
    assert meta["unknown_is_free"] is False
    assert meta["semantic_can_override_safety"] is False
    assert meta["unknown_cells_drawn"] == 1
    assert meta["semantic_overlay_mode"] == "outline_diagnostic_only"
    image = Image.open(tmp_path / "bev.png").convert("RGB")
    colors = set(image.getdata())
    assert _module.COLORS["unknown"] in colors
    assert _module.COLORS["free"] in colors


def test_stage27_event_requires_and_renders_spatial_channels(tmp_path):
    event = {
        "scene_id": "s", "episode_id": 1, "step_id": 2,
        "recovery_bev_spatial": {
            "center_grid": [1, 1], "executed_route": [[1, 0], [1, 1]],
            "channels": {"known_free": [[1, 1]], "occupied": [[0, 1]],
                         "unknown": [[0, 0]], "semantic": [[0, 0]]},
        },
        "ablation": {"route_occ_clearance_frontier_semantic_filtered": {"candidates": []}},
    }
    meta = _module.render_stage27_event(event, tmp_path / "event.png")
    assert meta["unknown_cells_drawn"] == 1
    assert meta["candidate_count"] == 0


def test_robot_forward_up_bev_includes_pose_hfov_depth_and_semantic_metadata(tmp_path):
    anchor = {"anchor_id": "rich", "capture": {"pose": [2, 2]}}
    digest = {
        "center_grid": [2, 2],
        "current_pose": {"grid": [2, 2], "yaw": 0.0},
        "hfov_deg": 79.0,
        "channels": {
            "known_free": [[2, 2], [1, 2]], "occupied": [[0, 2]],
            "unknown": [[2, 1]], "semantic": [],
            "semantic_nodes": [{"grid": [1, 2], "label": "chair"}],
        },
        "depth_endpoints": [
            {"surface_grid": [0, 2], "lookahead_grid": [1, 2]},
        ],
    }
    meta = _module.render_recovery_bev(anchor, digest, tmp_path / "rich.png", scale=8)
    assert meta["coordinate_frame"] == "robot_forward_up"
    assert meta["hfov_deg"] == 79.0
    assert meta["depth_endpoint_count"] == 1
    assert meta["semantic_node_count"] == 1
    assert (tmp_path / "rich.png").is_file()


def test_stage27_event_uses_last_pose_mapping_without_slice_index_error(tmp_path):
    event = {
        "scene_id": "s", "episode_id": 1, "step_id": 2,
        "recovery_bev_spatial": {
            "center_grid": [1, 1],
            "current_pose": {"grid": [1, 1], "yaw": 0.0},
            "channels": {"known_free": [[1, 1]], "occupied": [], "unknown": [[0, 0]]},
            "executed_route": [[1, 1]],
        },
        "ablation": {"route_occ_clearance_frontier_semantic_filtered": {"candidates": []}},
    }
    meta = _module.render_stage27_event(event, tmp_path / "pose.png")
    assert meta["coordinate_frame"] == "robot_forward_up"

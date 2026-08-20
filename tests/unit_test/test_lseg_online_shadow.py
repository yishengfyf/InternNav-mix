from pathlib import Path
import hashlib

import numpy as np

from internnav.utils.lseg_online_shadow import OnlineLSegSemanticShadow


class _OccMemory:
    gs = 64
    cs = 0.25

    def __init__(self):
        self.occ2d_counts = {}
        self.free2d_counts = {}
        self.pose_trace = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.5, "y": 0.0, "z": 0.0},
        ]


def _shadow(tmp_path: Path, enabled: bool = True):
    shadow = OnlineLSegSemanticShadow(
        {
            "lseg_online_shadow_enable": enabled,
            "lseg_online_shadow_sample_stride": 1,
            "lseg_online_shadow_confidence_threshold": 0.35,
            "lseg_online_shadow_save_visualizations": True,
        },
        np.asarray([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]),
        "cpu",
    )
    shadow.set_root(str(tmp_path))
    shadow.reset_episode(
        scene_id="scene", episode_id=1, rank=0,
        semantic_scene_gt={"objects": [], "regions": []},
        coordinate_transforms={"map_to_habitat_world": np.eye(4).tolist()},
    )
    return shadow


def test_disabled_shadow_never_loads_model(tmp_path):
    shadow = _shadow(tmp_path, enabled=False)
    event = shadow.process_query_frame(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_m=np.ones((4, 4), dtype=np.float32),
        camera_pose_map=np.eye(4),
        step_id=0, query_id=0, observation_index=0, occ_memory=_OccMemory(),
    )
    assert event["reason"] == "disabled"
    assert shadow.model is None


def test_homogeneous_camera_intrinsic_is_normalized(tmp_path):
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = 388.0
    intrinsic[1, 1] = 388.0
    intrinsic[0, 2] = 319.5
    intrinsic[1, 2] = 239.5
    shadow = OnlineLSegSemanticShadow(
        {"lseg_online_shadow_enable": False}, intrinsic, "cpu"
    )

    assert shadow.camera_intrinsic.shape == (3, 3)
    assert np.array_equal(shadow.camera_intrinsic, intrinsic[:3, :3])


def test_projection_keeps_unknown_distinct_from_free(tmp_path):
    shadow = _shadow(tmp_path)
    pred = np.zeros((4, 4), dtype=np.int16)
    confidence = np.ones((4, 4), dtype=np.float32)
    depth = np.ones((4, 4), dtype=np.float32)
    memory = _OccMemory()

    samples = shadow._project(
        pred, confidence, depth, np.eye(4, dtype=np.float32), memory
    )

    assert samples["map_xyz"].shape == (16, 3)
    assert np.all(samples["occ_state"] == 2)


def test_query_event_hashes_exact_inputs(tmp_path):
    shadow = _shadow(tmp_path)
    shadow._load_model = lambda: None
    shadow._infer_logits = lambda image: np.stack([
        np.ones(image.shape[:2], dtype=np.float32),
        np.zeros(image.shape[:2], dtype=np.float32),
    ] + [np.full(image.shape[:2], -10.0, dtype=np.float32)] * 12)
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    depth = np.ones((4, 4), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)

    event = shadow.process_query_frame(
        rgb=rgb, depth_m=depth, camera_pose_map=pose, step_id=0, query_id=0,
        observation_index=0, occ_memory=_OccMemory(),
    )

    assert event["valid"]
    assert event["rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert event["depth_sha256"] == hashlib.sha256(depth.tobytes()).hexdigest()
    assert event["camera_pose_sha256"] == hashlib.sha256(pose.tobytes()).hexdigest()


def test_node_merge_and_visualizations_are_audit_only(tmp_path):
    shadow = _shadow(tmp_path)
    shadow.surface_frames = [
        {
            "map_xyz": np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            "class_id": np.asarray([0, 0], dtype=np.int16),
            "confidence": np.asarray([0.8, 0.9], dtype=np.float32),
            "occ_state": np.asarray([2, 2], dtype=np.int8),
            "observation_index": np.asarray([0, 1], dtype=np.int32),
            "step_id": np.asarray([0, 4], dtype=np.int32),
        }
    ]
    shadow._stored_surface_count = 2

    summary = shadow.finish_episode(metrics={"success": 1}, steps=4, occ_memory=_OccMemory())

    assert summary["node_count"] == 1
    assert summary["multi_view_node_rate"] == 1.0
    assert summary["decision_status"] == "audit_only_not_navigation_ready"
    assert summary["action_applied_count"] == 0
    assert len(summary["visualizations"]) == 5
    for relative in summary["visualizations"].values():
        assert (shadow.episode_dir / relative).is_file()


def test_gt_audit_reports_exact_hit_counts_and_per_label(tmp_path):
    shadow = _shadow(tmp_path)
    shadow.episode_meta["semantic_scene_gt"] = {
        "objects": [{
            "category": "door", "center": [0.0, 0.0, 1.0],
            "lower": [-0.1, -0.1, 0.9], "upper": [0.1, 0.1, 1.1],
        }],
        "regions": [],
    }
    nodes = [
        {"label": "door", "centroid": [0.0, 0.0, 1.0]},
        {"label": "door", "centroid": [1.0, 0.0, 1.0]},
    ]

    audit = shadow._audit_nodes_with_gt(nodes)

    assert audit["compatible_node_count"] == 2
    assert audit["surface_distance_le_050m_count"] == 1
    assert audit["surface_distance_le_050m_rate"] == 0.5
    assert audit["per_label"]["door"]["surface_distance_le_050m_count"] == 1

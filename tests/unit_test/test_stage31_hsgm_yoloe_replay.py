import numpy as np

from scripts.eval.analyze_stage31_hsgm_yoloe_replay import (
    _depth_hash_matches,
    _scene_split,
    associate_instances,
    conflict_audit,
    gt_surface_audit,
    hsgm_center_filter,
    hsgm_height_band_counts,
    instruction_coverage,
    project_mask_depth,
)


def test_center_filter_matches_hsgm_outer_twenty_percent_contract():
    assert hsgm_center_filter([200, 150, 300, 250], 640, 480)
    assert not hsgm_center_filter([0, 150, 100, 250], 640, 480)
    assert _scene_split("fixed_scene") == _scene_split("fixed_scene")


def test_metric_z_depth_projection_and_height_bands():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    intrinsic = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 1.25
    projected = project_mask_depth(mask, depth, intrinsic, pose, sample_stride=1)
    assert projected["valid_depth_ratio"] == 1.0
    assert np.allclose(projected["map_xyz"], [[0.0, 0.0, 3.25]])
    bands = hsgm_height_band_counts(np.asarray([[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]]), 0.0)
    assert bands["floor_band_count"] == 1
    assert bands["stair_height_candidate_count"] == 1
    assert bands["obstacle_band_count"] == 2
    assert _depth_hash_matches(depth, {"depth_sha256": "bad"}) is False


def test_causal_association_conflict_and_instruction_coverage():
    instances = [
        {"label": "door", "centroid_map": [0, 0, 0], "observation_index": 1,
         "detection_index": 0, "step_id": 1, "sampled_point_count": 5, "confidence": 0.8},
        {"label": "door", "centroid_map": [0.2, 0, 0], "observation_index": 2,
         "detection_index": 0, "step_id": 2, "sampled_point_count": 5, "confidence": 0.9},
        {"label": "chair", "centroid_map": [0.1, 0, 0], "observation_index": 2,
         "detection_index": 1, "step_id": 2, "sampled_point_count": 5, "confidence": 0.7},
    ]
    nodes = associate_instances(instances, merge_radius_m=0.5)
    assert len(nodes) == 2
    assert nodes[0]["multi_view"] is True
    assert conflict_audit(nodes, radius_m=0.25)["count"] == 1
    coverage = instruction_coverage(["door", "chair"], ["doorway", "kitchen"])
    assert coverage["matched_terms"] == ["doorway"]


def test_gt_audit_is_explicitly_map_to_world():
    nodes = [{"label": "door", "centroid_map": [1.0, 2.0, 3.0]}]
    meta = {
        "coordinate_transforms": {"map_to_habitat_world": np.eye(4).tolist()},
        "semantic_scene_gt": {"objects": [{
            "category": "doorway", "center": [1, 2, 3],
            "lower": [0.9, 1.9, 2.9], "upper": [1.1, 2.1, 3.1],
        }]},
    }
    audit = gt_surface_audit(nodes, meta)
    assert audit["compatible_node_count"] == 1
    assert audit["surface_distance_le_050m_rate"] == 1.0

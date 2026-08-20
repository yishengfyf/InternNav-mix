import numpy as np

from internnav.utils.lseg_multiview_fusion import (
    aggregate_frame_voxels, fuse_voxel_evidence, isolated_voxel_rate,
)


def _frame(
    points, probabilities, depth, observation, embeddings=None, center=(0.0, 0.0, 0.0),
):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return aggregate_frame_voxels(
        np.asarray(points, dtype=np.float32), probabilities,
        np.log(np.maximum(probabilities, 1e-6)),
        None if embeddings is None else np.asarray(embeddings, dtype=np.float32),
        np.asarray(depth, dtype=np.float32), np.asarray(center, dtype=np.float32),
        observation, 0.05,
    )


def test_frame_aggregation_prevents_pixel_density_from_counting_as_views():
    frame = _frame(
        [[0.011, 0.0, 1.0], [0.019, 0.0, 1.0], [0.021, 0.0, 1.0]],
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], [1.0, 1.0, 1.0], 7,
    )

    assert len(frame.keys) == 1
    assert np.allclose(frame.probabilities[0], [0.8, 0.2])
    fused = fuse_voxel_evidence([frame], "prob")
    assert fused["view_count"].tolist() == [1]


def test_probability_fusion_can_override_hard_majority_with_strong_evidence():
    weak_wrong_a = _frame([[0.01, 0.0, 1.0]], [[0.49, 0.51]], [1.0], 0)
    weak_wrong_b = _frame([[0.01, 0.0, 1.0]], [[0.49, 0.51]], [1.0], 1)
    strong_right = _frame([[0.01, 0.0, 1.0]], [[0.99, 0.01]], [1.0], 2)

    hard = fuse_voxel_evidence([weak_wrong_a, weak_wrong_b, strong_right], "hard")
    probability = fuse_voxel_evidence(
        [weak_wrong_a, weak_wrong_b, strong_right], "prob"
    )

    assert hard["class_id"].tolist() == [1]
    assert probability["class_id"].tolist() == [0]
    assert probability["view_count"].tolist() == [3]
    assert probability["conflict"].tolist() == [True]


def test_distance_weight_matches_vlmaps_near_view_preference():
    near = _frame([[0.01, 0.0, 1.0]], [[0.95, 0.05]], [0.5], 0)
    far = _frame([[0.01, 0.0, 1.0]], [[0.05, 0.95]], [4.0], 1)

    hard = fuse_voxel_evidence([near, far], "hard")
    probability = fuse_voxel_evidence([near, far], "prob")

    assert np.isclose(hard["evidence"][0, 0], 0.5)
    assert probability["class_id"].tolist() == [0]


def test_embedding_fusion_queries_text_after_averaging():
    first = _frame(
        [[0.01, 0.0, 1.0]], [[0.8, 0.2]], [1.0], 0, embeddings=[[0.9, 0.1]],
    )
    second = _frame(
        [[0.01, 0.0, 1.0]], [[0.6, 0.4]], [1.0], 1, embeddings=[[0.7, 0.3]],
    )
    text = np.eye(2, dtype=np.float32)

    fused = fuse_voxel_evidence([first, second], "embedding", text_features=text)

    assert fused["class_id"].tolist() == [0]
    assert fused["view_count"].tolist() == [2]


def test_isolated_voxel_rate_uses_same_class_six_neighborhood():
    keys = np.asarray([[0, 0, 0], [1, 0, 0], [4, 0, 0]], dtype=np.int32)
    labels = np.asarray([0, 0, 1], dtype=np.int16)

    assert np.isclose(isolated_voxel_rate(keys, labels), 1.0 / 3.0)

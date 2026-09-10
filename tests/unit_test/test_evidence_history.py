import numpy as np
import pytest

from internnav.model.basemodel.internvla_n1.evidence_history import (
    build_causal_history_metadata,
    planar_pose_from_sim_observation,
    planar_poses_from_transforms,
    relative_pose_from_planar,
    select_causal_history_ids,
)


def test_history_selection_is_causal_uniform_and_bounded():
    assert select_causal_history_ids(0, 8) == []
    assert select_causal_history_ids(3, 8) == [0, 1, 2]
    assert select_causal_history_ids(10, 4) == [0, 3, 6, 9]
    assert max(select_causal_history_ids(10, 4)) < 10


def test_relative_pose_is_expressed_in_current_heading_frame():
    result = relative_pose_from_planar(
        history_positions=np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        history_yaws=np.asarray([0.0, np.pi / 2]),
        current_position=np.asarray([0.0, 0.0]),
        current_yaw=np.pi / 2,
    )

    np.testing.assert_allclose(result[0], [0.0, -1.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1], [1.0, -1.0, 0.0, 1.0], atol=1e-6)


def test_metadata_rejects_current_or_future_frames():
    positions = np.zeros((5, 2), dtype=np.float32)
    yaws = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="strict causal prefix"):
        build_causal_history_metadata([0, 4], positions, yaws, current_frame_id=4)


def test_metadata_has_frame_age_quality_and_relative_pose():
    positions = np.asarray([[0, 0], [1, 0], [2, 0], [2, 1]], dtype=np.float32)
    yaws = np.zeros(4, dtype=np.float32)
    metadata = build_causal_history_metadata([0, 2], positions, yaws, current_frame_id=3)

    assert metadata.frame_ids.tolist() == [0, 2]
    assert metadata.ages[:, 0].tolist() == [3.0, 1.0]
    np.testing.assert_allclose(metadata.relative_poses[:, :2], [[-2, -1], [0, -1]])
    np.testing.assert_allclose(metadata.qualities, np.ones((2, 2)))


def test_sim_quaternion_and_homogeneous_transforms_use_xy_plane():
    position, yaw = planar_pose_from_sim_observation([2.0, 3.0, 0.5], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(position, [2.0, 3.0])
    assert yaw == pytest.approx(0.0)

    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    transforms[1, :2, 3] = [2.0, 3.0]
    positions, yaws = planar_poses_from_transforms(transforms)
    np.testing.assert_allclose(positions, [[0.0, 0.0], [2.0, 3.0]])
    np.testing.assert_allclose(yaws, [0.0, 0.0])

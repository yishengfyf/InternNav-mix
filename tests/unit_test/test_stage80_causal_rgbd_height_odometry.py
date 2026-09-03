import numpy as np

from scripts.eval.analyze_stage80_causal_rgbd_height_odometry import (
    estimate_pair_delta,
    project_depth,
)


def test_pair_delta_recovers_vertical_translation_without_labels():
    x, y = np.meshgrid(np.linspace(-0.8, 0.8, 30), np.linspace(-0.8, 0.8, 30))
    z = 0.4 + 0.25 * x - 0.10 * y
    previous = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    expected_delta = 0.14
    current = previous.copy()
    current[:, 2] -= expected_delta
    result = estimate_pair_delta(previous, current)
    assert result["valid"] is True
    assert abs(result["estimated_delta_m"] - expected_delta) < 1e-6
    assert result["peak_inlier_count"] >= 800


def test_pair_delta_abstains_without_xy_overlap():
    previous = np.column_stack((np.linspace(0, 1, 100), np.zeros(100), np.zeros(100)))
    current = previous.copy()
    current[:, 0] += 4.0
    result = estimate_pair_delta(previous, current)
    assert result["valid"] is False
    assert result["reason"] == "insufficient_xy_overlap"


def test_projection_uses_known_camera_pitch_without_height_gt():
    depth = np.ones((32, 32), dtype=np.float32)
    intrinsic = np.asarray(((20.0, 0.0, 15.5), (0.0, 20.0, 15.5), (0.0, 0.0, 1.0)))
    level = project_depth(depth, intrinsic, np.eye(4), 0.0, stride=16)
    down = project_depth(depth, intrinsic, np.eye(4), 30.0, stride=16)
    assert level.shape == down.shape == (4, 3)
    assert not np.allclose(level, down)
    assert np.allclose(level[:, 2], [2.025, 2.025, 1.225, 1.225])

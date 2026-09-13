import numpy as np

from internnav.habitat_extensions.vln.measures import dtw


def test_dtw_fallback_matches_zero_and_known_path_costs():
    assert dtw([[0.0, 0.0]], [[0.0, 0.0]])[0] == 0.0
    cost = dtw([[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]], dist=lambda a, b: np.linalg.norm(np.asarray(a) - b))[0]
    assert cost == 1.0

import json

import numpy as np

from scripts.eval.analyze_freeocc_mapping_audit import (
    _relative_rotation_errors_degrees,
    analyze,
)


def _matrix_trajectory(points):
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None], len(points), axis=0)
    matrices[:, :3, 3] = np.asarray(points, dtype=np.float32)
    return matrices


def test_analyzer_detects_filter_collapse(tmp_path):
    run_dir = tmp_path / "run"
    audit_dir = run_dir / "audit"
    mesh_dir = run_dir / "mesh"
    audit_dir.mkdir(parents=True)
    mesh_dir.mkdir()
    rows = [
        {"frame_id": i, "raw_valid": 10000, "after_multiview": 2, "final_valid": 1}
        for i in range(3)
    ]
    audit = {
        "video_frames": 3,
        "camera_uids": [0, 1, 2],
        "gaussian_frame_uids": [0, 1, 2],
        "filter_calls": [{"frames": rows}],
        "mapping_calls": [
            {
                "window_size": None,
                "total_gaussians": 3,
                "frames": [
                    {"frame_id": i, "combined_valid": 1, "finite_valid": 1}
                    for i in range(3)
                ],
            }
        ],
    }
    (audit_dir / "freeocc_mapping_audit.json").write_text(json.dumps(audit))
    gt = _matrix_trajectory([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    estimated = _matrix_trajectory([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    np.savez_compressed(
        audit_dir / "trajectories.npz",
        estimated_c2w=estimated,
        gt_c2w=gt,
        timestamps=np.arange(3),
    )
    (mesh_dir / "final_mono.ply").write_text("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n")

    summary, _, aligned, target = analyze(run_dir, expected_input_frames=3)

    assert summary["diagnosis"] == "filter_collapse"
    assert summary["ply_aligned"]["vertices"] == 3
    assert summary["trajectory"]["sim3_scale"] == 0.5
    assert np.allclose(aligned, target)


def test_relative_rotation_error_is_global_frame_invariant():
    gt = _matrix_trajectory([[0, 0, 0], [1, 0, 0]])
    estimated = gt.copy()
    global_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    estimated[:, :3, :3] = np.einsum("ij,njk->nik", global_rotation, gt[:, :3, :3])
    assert np.allclose(_relative_rotation_errors_degrees(estimated, gt), 0.0)

    estimated[1, :3, :3] = estimated[1, :3, :3] @ global_rotation
    errors = _relative_rotation_errors_degrees(estimated, gt)
    assert np.isclose(errors[0], 0.0)
    assert np.isclose(errors[1], 90.0)

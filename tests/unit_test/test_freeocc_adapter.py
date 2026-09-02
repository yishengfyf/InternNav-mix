import json

import numpy as np

from internnav.utils.freeocc_adapter import FreeOccConfig, FreeOccWorldMemory


def test_freeocc_rgb_ledger_and_query(tmp_path):
    memory = FreeOccWorldMemory(FreeOccConfig(output_dir=str(tmp_path), semantic_names=["wall", "chair"]))
    memory.reset("scene0", "episode0")
    row = memory.update(np.zeros((4, 5, 3), dtype=np.uint8), 7)
    assert row["step_id"] == 7
    ledger = json.loads((tmp_path / "freeocc_frame_ledger.json").read_text())
    assert ledger["rgb_only_external_input"] is True
    assert ledger["frames"][0]["pose_present"] is False

    labels = np.zeros((3, 3, 2), dtype=np.int16)
    labels[1, 1, 0] = 1
    npz = tmp_path / "occ.npz"
    np.savez_compressed(npz, pred=labels, valid_mask=np.ones_like(labels, bool), voxel_origin=np.zeros(3), voxel_size=.08)
    memory.load_occupancy(npz)
    result = memory.query_semantics("wall")
    assert result["valid"] is True
    assert result["count"] == 1


def test_freeocc_run_requires_explicit_command(tmp_path):
    memory = FreeOccWorldMemory(FreeOccConfig(output_dir=str(tmp_path)))
    memory.reset("scene0")
    memory.update(np.zeros((2, 2, 3), dtype=np.uint8), 0)
    try:
        memory.run_external()
    except RuntimeError as exc:
        assert "command" in str(exc)
    else:
        raise AssertionError("run_external must fail closed without a command")

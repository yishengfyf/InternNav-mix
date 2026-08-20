import hashlib
import json

import numpy as np
from PIL import Image

from internnav.utils.replay_ledger import ReplayLedger


def test_replay_ledger_saves_lossless_rgb_with_hash(tmp_path):
    ledger = ReplayLedger({
        "replay_ledger_enable": True,
        "replay_ledger_rgb_format": "png",
    })
    ledger.set_root(str(tmp_path))
    ledger.reset_episode(scene_id="scene", episode_id=1, rank=0)
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    depth = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    record = ledger.record_observation(
        step_id=0, observation_index=0, rgb=rgb, depth=depth,
    )
    stored = np.asarray(Image.open(ledger.episode_dir / record["rgb_path"]).convert("RGB"))

    assert record["rgb_path"].endswith(".png")
    assert np.array_equal(stored, rgb)
    assert record["rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert record["depth_sha256"] == hashlib.sha256(depth.tobytes()).hexdigest()


def test_replay_ledger_keeps_legacy_jpg_default(tmp_path):
    ledger = ReplayLedger({"replay_ledger_enable": True})
    ledger.set_root(str(tmp_path))
    ledger.reset_episode(scene_id="scene", episode_id=2, rank=0)
    record = ledger.record_observation(
        step_id=0, observation_index=0,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=None,
    )

    assert record["rgb_storage_format"] == "jpg"
    assert record["rgb_path"].endswith(".jpg")

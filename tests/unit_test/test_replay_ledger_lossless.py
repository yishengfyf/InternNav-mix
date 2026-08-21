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
        audit_metrics={"collision_count": 1.0, "collision_delta": 1.0},
    )
    stored = np.asarray(Image.open(ledger.episode_dir / record["rgb_path"]).convert("RGB"))

    assert record["rgb_path"].endswith(".png")
    assert record["rgb_saved"] is True
    assert record["depth_saved"] is True
    assert np.array_equal(stored, rgb)
    assert record["rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert record["depth_sha256"] == hashlib.sha256(depth.tobytes()).hexdigest()
    assert record["audit_metrics"]["collision_delta"] == 1.0
    ledger.record_action(
        step_id=0, action=1, action_source="test", pre_safety_action=1,
        action_applied=True,
        audit_metrics={"collision_count": 1.0, "collision_delta": 1.0},
    )
    action = json.loads(
        (ledger.episode_dir / "actions.jsonl").read_text().splitlines()[0]
    )
    assert action["audit_metrics"]["collision_delta"] == 1.0


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


def test_replay_ledger_hashes_depth_without_storing_array(tmp_path):
    ledger = ReplayLedger({
        "replay_ledger_enable": True,
        "replay_ledger_save_depth": False,
    })
    ledger.set_root(str(tmp_path))
    ledger.reset_episode(scene_id="scene", episode_id=4, rank=0)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    record = ledger.record_observation(
        step_id=0, observation_index=0,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8), depth=depth,
    )

    assert record["depth_saved"] is False
    assert "depth_path" not in record
    assert record["depth_sha256"] == hashlib.sha256(depth.tobytes()).hexdigest()
    assert not list((ledger.episode_dir / "depth").iterdir())


def test_replay_ledger_can_avoid_repeating_large_episode_metadata(tmp_path):
    ledger = ReplayLedger({
        "replay_ledger_enable": True,
        "replay_ledger_repeat_episode_meta": False,
    })
    ledger.set_root(str(tmp_path))
    ledger.reset_episode(
        scene_id="scene", episode_id=3, rank=0,
        semantic_scene_gt={"objects": [{"category": "wall"}]},
    )
    record = ledger.record_observation(
        step_id=0, observation_index=0,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=None,
    )

    assert record["scene_id"] == "scene"
    assert "semantic_scene_gt" not in record
    meta = json.loads((ledger.episode_dir / "episode_meta.json").read_text())
    assert meta["semantic_scene_gt"]["objects"][0]["category"] == "wall"

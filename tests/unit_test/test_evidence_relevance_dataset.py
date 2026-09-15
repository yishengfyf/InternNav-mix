import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from internnav.dataset.internvla_n1_lerobot_dataset import (
    evidence_relevance_for_sample,
    load_evidence_relevance_manifest,
)


SCRIPT = Path(__file__).parents[2] / "scripts" / "dualvln_mainline" / "real_overfit.py"
SPEC = importlib.util.spec_from_file_location("real_overfit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_manifest(path, frame_ids=(8, 2, 5), targets=(0, 1, 1)):
    row = {
        "schema_version": 1,
        "annotation_id": "r2r_train_ep000003_t0010_deadbeef",
        "episode_id": "3",
        "candidate_frame_ids": list(frame_ids),
        "targets": list(targets),
        "supervised": True,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_relevance_manifest_sorts_frames_and_targets_together(tmp_path):
    records = load_evidence_relevance_manifest(write_manifest(tmp_path / "relevance.jsonl"))

    assert records[(3, 10)]["candidate_frame_ids"] == [2, 5, 8]
    assert records[(3, 10)]["targets"] == [1.0, 1.0, 0.0]


def test_relevance_manifest_rejects_noncausal_frame(tmp_path):
    with pytest.raises(ValueError, match="strict causal"):
        load_evidence_relevance_manifest(
            write_manifest(tmp_path / "relevance.jsonl", frame_ids=(2, 10), targets=(1, 0))
        )


def test_scene_aware_relevance_disambiguates_reused_episode_ids(tmp_path):
    rows = []
    for scene, target in (("scene_a", 1), ("scene_b", 0)):
        rows.append(
            {
                "schema_version": 1,
                "annotation_id": f"r2r_train_{scene}_ep000003_t0010_deadbeef",
                "scene_id": scene,
                "episode_id": "3",
                "candidate_frame_ids": [2, 5, 8],
                "targets": [target, 1 - target, 0],
                "supervised": True,
            }
        )
    path = tmp_path / "relevance.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    records = load_evidence_relevance_manifest(path)
    samples = []
    for scene in ("scene_a", "scene_b"):
        entry = [None] * 11
        entry[0] = 3
        entry[2] = f"/data/r2r/{scene}/videos/chunk-000"
        entry[7] = (10, 12)
        samples.append(tuple(entry))

    assert evidence_relevance_for_sample(records, samples[0])["targets"] == [1.0, 0.0, 0.0]
    assert evidence_relevance_for_sample(records, samples[1])["targets"] == [0.0, 1.0, 0.0]


def test_relevance_sample_selection_balances_states_and_episodes():
    entries, relevance = [], {}
    for episode_id in range(8):
        frame_id = 10 + episode_id
        entry = [None] * 11
        entry[0] = episode_id
        entry[7] = (frame_id, frame_id + 2)
        entry[9] = [[0.0, 0.0]]
        entries.append(tuple(entry))
        relevance[(episode_id, frame_id)] = {
            "supervised": True,
            "targets": [1.0, 0.0, 0.0] if episode_id < 4 else [0.0, 0.0, 0.0],
        }
    dataset = SimpleNamespace(list_data_dict=entries, evidence_relevance=relevance)

    selected = MODULE.select_relevance_samples(dataset, samples=8, seed=23)

    states = [
        any(target > 0.5 for target in relevance[(entries[index][0], entries[index][7][0])]["targets"])
        for index in selected
    ]
    assert len(selected) == 8
    assert len({entries[index][0] for index in selected}) == 8
    assert sum(states) == 4

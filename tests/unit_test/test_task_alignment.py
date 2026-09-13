import json

import pytest

from internnav.dataset.internvla_n1_lerobot_dataset import (
    NavPixelGoalDataset,
    load_task_alignment_manifest,
)
from scripts.dualvln_mainline.task_state_probe import mismatch_instructions_by_group


def write_manifest(tmp_path, **updates):
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "raw_sha256": "a" * 64,
        "tasks_sha256": "b" * 64,
        "converted_instructions": 4,
        "matched_instructions": 4,
        "unique_trajectories": 2,
        "instructions": [
            {
                "instruction": " Go to the kitchen. ",
                "trajectory_id": 10,
                "paraphrases": ["Go to the kitchen.", " Walk   into the kitchen. "],
            },
            {
                "instruction": "Walk into the kitchen.",
                "trajectory_id": 10,
                "paraphrases": ["Go to the kitchen.", "Walk into the kitchen."],
            },
            {
                "instruction": "Find the sofa.",
                "trajectory_id": 20,
                "paraphrases": ["Find the sofa.", "Stop beside the sofa."],
            },
            {
                "instruction": "Stop beside the sofa.",
                "trajectory_id": 20,
                "paraphrases": ["Find the sofa.", "Stop beside the sofa."],
            },
        ],
    }
    manifest.update(updates)
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_dataset(alignment):
    dataset = object.__new__(NavPixelGoalDataset)
    dataset.task_alignment = alignment
    dataset.task_instructions = {
        ("r2r", 10): ["Go to the kitchen.", "Walk into the kitchen."],
        ("r2r", 20): ["Find the sofa.", "Stop beside the sofa."],
    }
    dataset.task_keys = [("r2r", 10), ("r2r", 20)]
    dataset.task_key_indices = {key: index for index, key in enumerate(dataset.task_keys)}
    return dataset


def test_alignment_normalizes_and_selects_true_route_pairs(tmp_path):
    alignment = load_task_alignment_manifest(write_manifest(tmp_path))
    dataset = make_dataset(alignment)

    assert dataset.task_key("r2r", 999, " Go  to the kitchen. ") == ("r2r", 10)
    positive, negative, valid = dataset.task_instruction_pair("r2r", 999, " Go  to the kitchen. ")
    assert positive == "Walk into the kitchen."
    assert negative == "Find the sofa."
    assert valid


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "failed"},
        {"schema_version": 2},
        {"instructions": []},
    ],
)
def test_alignment_rejects_unvalidated_or_empty_manifest(tmp_path, updates):
    with pytest.raises(ValueError):
        load_task_alignment_manifest(write_manifest(tmp_path, **updates))


def test_alignment_rejects_duplicate_normalized_instruction(tmp_path):
    path = write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["instructions"][1]["instruction"] = "Go   to the kitchen."
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate instruction"):
        load_task_alignment_manifest(path)


def test_grouped_mismatch_never_uses_same_route():
    instructions = ["a0", "a1", "b0", "b1", "c0", "c1"]
    groups = ["a", "a", "b", "b", "c", "c"]
    replacements = mismatch_instructions_by_group(instructions, groups, seed=23)
    route_by_instruction = dict(zip(instructions, groups))
    assert all(route_by_instruction[source] != route_by_instruction[target] for source, target in zip(instructions, replacements))

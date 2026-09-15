import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "dualvln_mainline" / "probe_annotation_features.py"
SPEC = importlib.util.spec_from_file_location("probe_annotation_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_data():
    annotation_ids, episode_ids, targets, visual, text, metadata = [], [], [], [], [], []
    for episode in range(8):
        for card in range(2):
            annotation_id = f"ep{episode}_card{card}"
            for candidate in range(3):
                target = int(candidate == card)
                annotation_ids.append(annotation_id)
                episode_ids.append(str(episode))
                targets.append(target)
                current = np.array([episode, card, 1.0, 0.5], dtype=np.float32)
                history = current + np.array([target * 2 - 1, 0.1, candidate, 0.0], dtype=np.float32)
                visual.append([current, history])
                text.append(np.array([card, 1.0, 0.0, 0.5], dtype=np.float32))
                metadata.append([candidate + 1, candidate, 0.0, candidate, 0.0, 1.0])
    return {
        "annotation_ids": np.asarray(annotation_ids),
        "episode_ids": np.asarray(episode_ids),
        "targets": np.asarray(targets),
        "visual": np.asarray(visual),
        "text": np.asarray(text),
        "metadata": np.asarray(metadata),
    }


def test_feature_variants_keep_candidate_rows_aligned():
    data = fake_data()
    assert MODULE.feature_matrix(data, "pose_age").shape == (48, 6)
    assert MODULE.feature_matrix(data, "rgb_text").shape == (48, 12)
    assert MODULE.feature_matrix(data, "rgb_text_pose").shape == (48, 18)


def test_grouped_probe_evaluates_every_card_without_episode_leakage():
    result = MODULE.evaluate(fake_data(), "rgb_text", seed=23, splits=4)
    assert result["evaluated_cards"] == 16
    assert result["top1_hit"] > 0.9
    for fold in result["folds"]:
        assert fold["train_episodes"] == 6
        assert fold["test_episodes"] == 2

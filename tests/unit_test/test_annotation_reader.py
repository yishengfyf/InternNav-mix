import importlib.util
import math
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).parents[2] / "scripts" / "dualvln_mainline" / "train_annotation_reader.py"
SPEC = importlib.util.spec_from_file_location("train_annotation_reader", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_cards():
    cards = []
    for episode in range(4):
        for card_index in range(2):
            rng = np.random.default_rng(episode * 10 + card_index)
            target = np.array([1.0, 0.0, 0.0]) if card_index == 0 else np.zeros(3, dtype=np.float32)
            cards.append(
                {
                    "annotation_id": f"ep{episode}_card{card_index}",
                    "episode_id": str(episode),
                    "history": rng.normal(size=(3, 8)).astype(np.float32),
                    "current": rng.normal(size=8).astype(np.float32),
                    "text": rng.normal(size=8).astype(np.float32),
                    "poses": rng.normal(size=(3, 4)).astype(np.float32),
                    "ages": np.ones((3, 1), dtype=np.float32),
                    "qualities": np.ones((3, 2), dtype=np.float32),
                    "targets": target,
                }
            )
    return cards


def test_reader_only_smoke_is_finite_for_all_ablations():
    cards = fake_cards()
    for variant in ("content", "spatial", "task_spatial"):
        result = MODULE.run_one(
            cards, list(range(6)), [6, 7], variant, seed=23, steps=3,
            learning_rate=1e-3, device=torch.device("cpu"),
        )
        assert math.isfinite(result["final_train"]["loss"])
        assert math.isfinite(result["final_val"]["loss"])
        assert result["trainable_parameters"] > 0


def test_balanced_episode_folds_keep_groups_disjoint_and_balance_labels():
    cards = fake_cards()
    manifest = MODULE.balanced_episode_folds(cards, 2, trials=100)
    first, second = [set(item["episodes"]) for item in manifest["folds"]]
    assert first.isdisjoint(second)
    assert first | second == {"0", "1", "2", "3"}
    assert all(item["yes_no_uncertain"] == [2, 2, 0] for item in manifest["folds"])

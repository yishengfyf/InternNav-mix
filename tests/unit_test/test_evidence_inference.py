from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from internnav.model.basemodel.internvla_n1.evidence_inference import (
    configure_task_spatial_inference,
    load_evidence_adapter,
    planar_pose_from_habitat_observation,
    prepare_evidence_inputs,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import EVIDENCE_TOKEN_INDEX, IMAGE_TOKEN_INDEX


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.task_state_estimator = nn.Linear(2, 2)
        self.model.evidence_memory = nn.Linear(2, 2)
        self.model.cond_projector = nn.Linear(2, 2)
        self.model.latent_queries = nn.Parameter(torch.zeros(1, 2, 2))


def test_configure_and_strictly_load_selected_adapter(tmp_path):
    config = SimpleNamespace()
    configure_task_spatial_inference(config)
    assert config.use_evidence_memory
    assert config.evidence_ablation == "task_spatial"
    assert config.evidence_gradient_bypass_mode == "cross_attention"

    source = AdapterModel()
    state = {name: torch.ones_like(value) for name, value in source.state_dict().items()}
    adapter_path = tmp_path / "adapter_state.pt"
    torch.save(state, adapter_path)
    target = AdapterModel()
    manifest = load_evidence_adapter(target, adapter_path)
    assert manifest["parameter_tensors"] == len(state)
    assert all(torch.equal(value, torch.ones_like(value)) for value in target.state_dict().values())


def test_adapter_loader_rejects_partial_state(tmp_path):
    model = AdapterModel()
    adapter_path = tmp_path / "adapter_state.pt"
    torch.save({"model.latent_queries": model.model.latent_queries.detach()}, adapter_path)
    with pytest.raises(ValueError, match="does not exactly match"):
        load_evidence_adapter(model, adapter_path)


def test_habitat_pose_and_empty_history_are_causal():
    pose = planar_pose_from_habitat_observation(
        {"gps": np.asarray([1.5, -2.0]), "compass": np.asarray([0.25])}
    )
    assert np.allclose(pose, [1.5, -2.0, 0.25])

    model = SimpleNamespace(
        config=SimpleNamespace(
            use_evidence_memory=True,
            image_token_id=IMAGE_TOKEN_INDEX,
            num_evidence_tokens=4,
        )
    )
    inputs = AttrDict(
        input_ids=torch.tensor([[10, 151652, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 151653, 11]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
    )
    inputs, kwargs = prepare_evidence_inputs(model, inputs, [], [pose], 1, "cpu")
    assert int(inputs.input_ids.eq(EVIDENCE_TOKEN_INDEX).sum()) == 4
    assert tuple(kwargs["evidence_relative_poses"].shape) == (1, 0, 4)
    assert kwargs["evidence_history_counts"].tolist() == [0]


def test_habitat_pose_rejects_missing_or_nonfinite_observations():
    with pytest.raises(ValueError, match="gps and compass"):
        planar_pose_from_habitat_observation({"gps": [0, 0]})
    with pytest.raises(ValueError, match="finite"):
        planar_pose_from_habitat_observation({"gps": [np.nan, 0], "compass": [0]})

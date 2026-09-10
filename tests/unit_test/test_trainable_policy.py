import pytest
import torch
import torch.nn as nn

from internnav.model.basemodel.internvla_n1.trainable import configure_trainable_parameters


class TinyDualSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4)
        self.language = nn.Linear(4, 4)
        self.evidence_memory = nn.Linear(4, 4)
        self.cond_projector = nn.Linear(4, 4)
        self.traj_dit = nn.Linear(4, 4)
        self.latent_queries = nn.Parameter(torch.randn(1, 2, 4))


def test_explicit_allowlist_freezes_everything_else():
    model = TinyDualSystem()
    summary = configure_trainable_parameters(
        model,
        (
            "evidence_memory.*",
            "cond_projector.*",
            "latent_queries",
        ),
    )

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "cond_projector.bias",
        "cond_projector.weight",
        "evidence_memory.bias",
        "evidence_memory.weight",
        "latent_queries",
    }
    assert set(summary.matched_parameters) == trainable
    assert summary.trainable_parameters + summary.frozen_parameters == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_unmatched_pattern_fails_before_mutating_flags():
    model = TinyDualSystem()
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}

    with pytest.raises(ValueError, match="matched nothing"):
        configure_trainable_parameters(model, ("missing_adapter.*",))

    after = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert before == after

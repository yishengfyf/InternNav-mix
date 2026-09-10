from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Sequence

import torch.nn as nn


@dataclass(frozen=True)
class TrainableParameterSummary:
    trainable_parameters: int
    frozen_parameters: int
    matched_parameters: tuple[str, ...]


def configure_trainable_parameters(
    model: nn.Module,
    allowlist: Sequence[str],
) -> TrainableParameterSummary:
    """Freeze a model and enable only parameters matching explicit glob patterns."""
    if not allowlist:
        raise ValueError("allowlist must contain at least one parameter pattern")

    named_parameters = list(model.named_parameters())
    matched_by_pattern = {
        pattern: [name for name, _ in named_parameters if fnmatchcase(name, pattern)] for pattern in allowlist
    }
    unmatched_patterns = [pattern for pattern, names in matched_by_pattern.items() if not names]
    if unmatched_patterns:
        joined = ", ".join(unmatched_patterns)
        raise ValueError(f"trainable parameter patterns matched nothing: {joined}")

    matched_names = {name for names in matched_by_pattern.values() for name in names}
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in named_parameters:
        parameter.requires_grad = name in matched_names
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
        else:
            frozen_parameters += parameter.numel()

    return TrainableParameterSummary(
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        matched_parameters=tuple(sorted(matched_names)),
    )

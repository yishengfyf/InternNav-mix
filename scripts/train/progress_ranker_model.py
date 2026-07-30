"""Torch-only model definition for the offline Stage17 progress ranker."""

import torch
from torch import nn


class ProgressCandidateRanker(nn.Module):
    """Score candidates independently and train them with a listwise loss."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.scorer(candidate_features).squeeze(-1)

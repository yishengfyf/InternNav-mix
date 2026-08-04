"""Small set-aware Stage18b S2-aware intervention adapter."""

from __future__ import annotations

import torch
from torch import nn


class S2AwareInterventionAdapter(nn.Module):
    """Score safe recovery candidates and classify keep/intervene/abstain.

    Inputs are compact online feature vectors only.  S2/NextDiT image models
    remain frozen and are not imported by this module.
    """

    def __init__(
        self,
        pair_feature_dim: int,
        event_context_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_feature_dim),
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_head = nn.Linear(hidden_dim, 1)
        self.decision_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + event_context_dim),
            nn.Linear(2 * hidden_dim + event_context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        candidate_pair_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        event_context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate scores and keep/intervene/abstain logits."""

        embeddings = self.pair_encoder(candidate_pair_features)
        candidate_scores = self.candidate_head(embeddings).squeeze(-1)
        mask_float = candidate_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        valid_count = mask_float.sum(dim=1).clamp_min(1.0)
        pooled_mean = (embeddings * mask_float).sum(dim=1) / valid_count
        pooled_max = embeddings.masked_fill(
            ~candidate_mask.unsqueeze(-1),
            float("-inf"),
        ).max(dim=1).values
        has_candidate = candidate_mask.any(dim=1, keepdim=True)
        pooled_max = torch.where(has_candidate, pooled_max, torch.zeros_like(pooled_max))
        decision_features = torch.cat(
            [pooled_mean, pooled_max, event_context_features],
            dim=-1,
        )
        decision_logits = self.decision_head(decision_features)
        return candidate_scores, decision_logits

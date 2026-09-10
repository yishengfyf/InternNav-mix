from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EvidenceMemoryOutput:
    tokens: torch.Tensor
    read_weights: torch.Tensor
    null_weights: torch.Tensor
    stage_logits: torch.Tensor
    no_evidence: torch.Tensor


class TaskConditionedEvidenceMemory(nn.Module):
    """Encode causal spatial evidence and retrieve it with a task-state query."""

    def __init__(
        self,
        feature_dim: int,
        task_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        pose_dim: int = 4,
        quality_dim: int = 2,
        num_evidence_tokens: int = 4,
        num_heads: int = 8,
        num_stages: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        output_dim = hidden_dim if output_dim is None else output_dim
        if min(feature_dim, task_dim, hidden_dim, output_dim, pose_dim, num_evidence_tokens, num_stages) <= 0:
            raise ValueError("feature, task, hidden, pose, token, and stage dimensions must be positive")
        if quality_dim < 0:
            raise ValueError("quality_dim must be non-negative")

        self.feature_dim = feature_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pose_dim = pose_dim
        self.quality_dim = quality_dim
        self.num_evidence_tokens = num_evidence_tokens

        metadata_dim = pose_dim + 1 + quality_dim
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.write_norm = nn.LayerNorm(hidden_dim)

        self.evidence_queries = nn.Parameter(torch.empty(1, num_evidence_tokens, hidden_dim))
        self.null_evidence = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.task_projection = nn.Linear(task_dim, hidden_dim)
        self.reader = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.read_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.bottleneck_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.stage_head = nn.Linear(task_dim, num_stages)

        nn.init.normal_(self.evidence_queries, std=0.02)
        nn.init.normal_(self.null_evidence, std=0.02)

    def _validate_inputs(
        self,
        history_features: torch.Tensor,
        relative_poses: torch.Tensor,
        ages: torch.Tensor,
        qualities: torch.Tensor,
        valid_mask: torch.Tensor,
        task_state: torch.Tensor,
    ) -> None:
        if history_features.ndim != 3:
            raise ValueError("history_features must have shape [batch, history, feature_dim]")

        batch_size, history_size, feature_dim = history_features.shape
        expected = {
            "relative_poses": (batch_size, history_size, self.pose_dim),
            "ages": (batch_size, history_size, 1),
            "qualities": (batch_size, history_size, self.quality_dim),
            "valid_mask": (batch_size, history_size),
            "task_state": (batch_size, self.task_dim),
        }
        actual = {
            "relative_poses": tuple(relative_poses.shape),
            "ages": tuple(ages.shape),
            "qualities": tuple(qualities.shape),
            "valid_mask": tuple(valid_mask.shape),
            "task_state": tuple(task_state.shape),
        }
        if feature_dim != self.feature_dim:
            raise ValueError(f"history feature dimension must be {self.feature_dim}, got {feature_dim}")
        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {actual[name]}")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean tensor")
        age_valid_mask = valid_mask.to(device=ages.device)
        if torch.any(ages.squeeze(-1)[age_valid_mask] < 0):
            raise ValueError("valid causal evidence cannot have a negative age")

    def write(
        self,
        history_features: torch.Tensor,
        relative_poses: torch.Tensor,
        ages: torch.Tensor,
        qualities: torch.Tensor,
    ) -> torch.Tensor:
        """Write task-independent observations into a shared evidence space."""
        device = self.feature_projection.weight.device
        dtype = self.feature_projection.weight.dtype
        history_features = history_features.to(device=device, dtype=dtype)
        stable_ages = torch.log1p(ages.clamp_min(0))
        metadata = torch.cat((relative_poses, stable_ages, qualities), dim=-1).to(device=device, dtype=dtype)
        return self.write_norm(self.feature_projection(history_features) + self.metadata_projection(metadata))

    def forward(
        self,
        history_features: torch.Tensor,
        relative_poses: torch.Tensor,
        ages: torch.Tensor,
        qualities: torch.Tensor,
        valid_mask: torch.Tensor,
        task_state: torch.Tensor,
    ) -> EvidenceMemoryOutput:
        self._validate_inputs(history_features, relative_poses, ages, qualities, valid_mask, task_state)

        evidence = self.write(history_features, relative_poses, ages, qualities)
        batch_size = evidence.shape[0]
        null_evidence = self.null_evidence.expand(batch_size, -1, -1)
        reader_memory = torch.cat((evidence, null_evidence), dim=1)

        valid_mask = valid_mask.to(device=evidence.device)
        null_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=evidence.device)
        reader_valid_mask = torch.cat((valid_mask, null_valid), dim=1)
        key_padding_mask = ~reader_valid_mask

        task_state = task_state.to(device=evidence.device, dtype=evidence.dtype)
        task_query = self.task_projection(task_state).unsqueeze(1)
        queries = self.evidence_queries.expand(batch_size, -1, -1) + task_query
        retrieved, per_head_weights = self.reader(
            queries,
            reader_memory,
            reader_memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = self.read_norm(queries + retrieved)
        bottleneck_tokens = self.bottleneck_norm(hidden + self.read_mlp(hidden))
        tokens = self.output_norm(self.output_projection(bottleneck_tokens))

        weights = per_head_weights.mean(dim=1)
        read_weights = weights[..., :-1]
        null_weights = weights[..., -1]
        read_weights = read_weights.masked_fill(~valid_mask.unsqueeze(1), 0.0)

        return EvidenceMemoryOutput(
            tokens=tokens,
            read_weights=read_weights,
            null_weights=null_weights,
            stage_logits=self.stage_head(task_state),
            no_evidence=~valid_mask.any(dim=1),
        )

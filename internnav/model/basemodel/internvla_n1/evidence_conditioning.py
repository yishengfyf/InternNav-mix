from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VisualEvidenceBatch:
    history_features: torch.Tensor
    current_features: torch.Tensor


def pool_visual_features(
    image_embeddings: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Pool flattened Qwen visual tokens into one feature per input image."""
    if image_embeddings.ndim != 2 or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("expected image embeddings [tokens, hidden] and image_grid_thw [images, 3]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    token_counts = image_grid_thw.prod(dim=1).div(spatial_merge_size**2, rounding_mode="floor").tolist()
    if sum(token_counts) != image_embeddings.shape[0]:
        raise ValueError(
            f"visual token count mismatch: grid describes {sum(token_counts)}, got {image_embeddings.shape[0]}"
        )
    return torch.stack([chunk.mean(dim=0) for chunk in image_embeddings.split(token_counts)], dim=0)


def gather_visual_evidence(
    pooled_images: torch.Tensor,
    image_counts: torch.Tensor,
    history_counts: torch.Tensor,
    max_history: int,
) -> VisualEvidenceBatch:
    """Recover per-sample history/current features from the flattened image order."""
    if pooled_images.ndim != 2:
        raise ValueError("pooled_images must have shape [images, hidden]")
    if image_counts.ndim != 1 or history_counts.shape != image_counts.shape:
        raise ValueError("image_counts and history_counts must be one-dimensional and aligned")
    if int(image_counts.sum().item()) != pooled_images.shape[0]:
        raise ValueError("image_counts do not cover all pooled images")
    if max_history < 0 or torch.any(history_counts < 0) or torch.any(history_counts > max_history):
        raise ValueError("history counts are outside the padded history size")
    if torch.any(image_counts <= history_counts):
        raise ValueError("each sample must contain a current image after its history images")

    hidden_size = pooled_images.shape[1]
    history = pooled_images.new_zeros((len(image_counts), max_history, hidden_size))
    current = []
    offset = 0
    for sample_idx, (image_count, history_count) in enumerate(zip(image_counts.tolist(), history_counts.tolist())):
        if history_count:
            history[sample_idx, :history_count] = pooled_images[offset : offset + history_count]
        current.append(pooled_images[offset + history_count])
        offset += image_count
    return VisualEvidenceBatch(history_features=history, current_features=torch.stack(current))


class ObservableTaskStateEstimator(nn.Module):
    """Estimate task state from the current observation and causally available prompt tokens."""

    def __init__(self, input_dim: int, task_dim: int):
        super().__init__()
        self.visual_projection = nn.Linear(input_dim, task_dim)
        self.text_projection = nn.Linear(input_dim, task_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(task_dim * 2),
            nn.Linear(task_dim * 2, task_dim),
            nn.GELU(),
            nn.LayerNorm(task_dim),
        )

    def forward(
        self,
        current_features: torch.Tensor,
        token_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.ndim != 3 or prompt_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("token embeddings and prompt mask must have aligned [batch, sequence] dimensions")
        if current_features.shape != (token_embeddings.shape[0], token_embeddings.shape[2]):
            raise ValueError("current visual features must match token batch and hidden dimensions")
        prompt_mask = prompt_mask.to(device=token_embeddings.device, dtype=torch.bool)
        denominator = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1).to(token_embeddings.dtype)
        prompt_features = (token_embeddings * prompt_mask.unsqueeze(-1)).sum(dim=1) / denominator
        fused = torch.cat(
            (self.visual_projection(current_features), self.text_projection(prompt_features)), dim=-1
        )
        return self.fusion(fused)

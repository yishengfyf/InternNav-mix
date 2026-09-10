from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class ConditionedSequenceBatch:
    input_ids: tuple[torch.Tensor, ...]
    labels: tuple[torch.Tensor, ...]
    evidence_positions: tuple[int, ...]
    trajectory_positions: tuple[int, ...]


def prepare_conditioned_sequences(
    input_ids: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    evidence_insert_positions: Sequence[int],
    evidence_token_id: int,
    trajectory_token_id: int,
    num_evidence_tokens: int,
    num_trajectory_tokens: int,
    ignore_index: int = -100,
    max_length: Optional[int] = None,
) -> ConditionedSequenceBatch:
    """Insert evidence placeholders and append trajectory tokens without silent misalignment."""
    batch_size = len(input_ids)
    if len(labels) != batch_size or len(evidence_insert_positions) != batch_size:
        raise ValueError("input_ids, labels, and evidence_insert_positions must have equal batch sizes")
    if num_evidence_tokens <= 0 or num_trajectory_tokens < 0:
        raise ValueError("evidence token count must be positive and trajectory token count cannot be negative")
    if max_length is not None and max_length < num_evidence_tokens + num_trajectory_tokens:
        raise ValueError("max_length cannot hold the requested evidence and trajectory tokens")

    output_ids = []
    output_labels = []
    evidence_positions = []
    trajectory_positions = []
    reserved_tokens = num_evidence_tokens + num_trajectory_tokens

    for sample_ids, sample_labels, insert_position in zip(input_ids, labels, evidence_insert_positions):
        if sample_ids.ndim != 1 or sample_labels.ndim != 1:
            raise ValueError("each input_ids and labels sample must be one-dimensional")
        if sample_ids.shape != sample_labels.shape:
            raise ValueError("each input_ids sample must have the same shape as its labels")

        retained_length = sample_ids.numel()
        if max_length is not None:
            retained_length = min(retained_length, max_length - reserved_tokens)
        if insert_position < 0 or insert_position > retained_length:
            raise ValueError(
                f"evidence insert position {insert_position} falls outside retained prefix length {retained_length}"
            )

        retained_ids = sample_ids[:retained_length]
        retained_labels = sample_labels[:retained_length]
        evidence_ids = torch.full(
            (num_evidence_tokens,), evidence_token_id, dtype=sample_ids.dtype, device=sample_ids.device
        )
        evidence_labels = torch.full(
            (num_evidence_tokens,), ignore_index, dtype=sample_labels.dtype, device=sample_labels.device
        )
        trajectory_ids = torch.full(
            (num_trajectory_tokens,), trajectory_token_id, dtype=sample_ids.dtype, device=sample_ids.device
        )
        trajectory_labels = torch.full(
            (num_trajectory_tokens,), ignore_index, dtype=sample_labels.dtype, device=sample_labels.device
        )

        conditioned_ids = torch.cat(
            (retained_ids[:insert_position], evidence_ids, retained_ids[insert_position:], trajectory_ids)
        )
        conditioned_labels = torch.cat(
            (
                retained_labels[:insert_position],
                evidence_labels,
                retained_labels[insert_position:],
                trajectory_labels,
            )
        )
        if max_length is not None and conditioned_ids.numel() > max_length:
            raise AssertionError("conditioned sequence exceeds max_length")

        output_ids.append(conditioned_ids)
        output_labels.append(conditioned_labels)
        evidence_positions.append(insert_position)
        trajectory_positions.append(retained_length + num_evidence_tokens)

    return ConditionedSequenceBatch(
        input_ids=tuple(output_ids),
        labels=tuple(output_labels),
        evidence_positions=tuple(evidence_positions),
        trajectory_positions=tuple(trajectory_positions),
    )


def replace_evidence_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    evidence_tokens: torch.Tensor,
    evidence_token_id: int,
) -> torch.Tensor:
    """Replace registered evidence placeholders with one fixed token block per sample."""
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3 or evidence_tokens.ndim != 3:
        raise ValueError("expected input_ids [B,L], inputs_embeds [B,L,H], and evidence_tokens [B,K,H]")
    if input_ids.shape != inputs_embeds.shape[:2]:
        raise ValueError("input_ids and inputs_embeds sequence shapes do not match")
    if inputs_embeds.shape[0] != evidence_tokens.shape[0] or inputs_embeds.shape[2] != evidence_tokens.shape[2]:
        raise ValueError("evidence token batch and hidden dimensions must match inputs_embeds")

    evidence_mask = input_ids.eq(evidence_token_id)
    expected_per_sample = evidence_tokens.shape[1]
    actual_per_sample = evidence_mask.sum(dim=1)
    if not torch.all(actual_per_sample.eq(expected_per_sample)):
        raise ValueError(
            f"each sample must contain {expected_per_sample} evidence placeholders; got {actual_per_sample.tolist()}"
        )

    output = inputs_embeds.clone()
    replacement = evidence_tokens.to(device=output.device, dtype=output.dtype).reshape(-1, output.shape[-1])
    output[evidence_mask] = replacement
    return output

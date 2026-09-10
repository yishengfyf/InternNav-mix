import pytest
import torch

from internnav.model.basemodel.internvla_n1.evidence_sequence import (
    prepare_conditioned_sequences,
    replace_evidence_embeddings,
)


def test_prepare_sequences_keeps_positions_and_ignores_adapter_tokens():
    result = prepare_conditioned_sequences(
        input_ids=(torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21])),
        labels=(torch.tensor([-100, 11, 12, 13]), torch.tensor([-100, 21])),
        evidence_insert_positions=(2, 0),
        evidence_token_id=90,
        trajectory_token_id=91,
        num_evidence_tokens=2,
        num_trajectory_tokens=3,
    )

    assert result.input_ids[0].tolist() == [10, 11, 90, 90, 12, 13, 91, 91, 91]
    assert result.labels[0].tolist() == [-100, 11, -100, -100, 12, 13, -100, -100, -100]
    assert result.input_ids[1].tolist() == [90, 90, 20, 21, 91, 91, 91]
    assert result.evidence_positions == (2, 0)
    assert result.trajectory_positions == (6, 4)


def test_truncation_reserves_both_token_blocks():
    result = prepare_conditioned_sequences(
        input_ids=(torch.arange(10),),
        labels=(torch.arange(10),),
        evidence_insert_positions=(3,),
        evidence_token_id=90,
        trajectory_token_id=91,
        num_evidence_tokens=2,
        num_trajectory_tokens=2,
        max_length=8,
    )

    assert result.input_ids[0].tolist() == [0, 1, 2, 90, 90, 3, 91, 91]
    assert result.trajectory_positions == (6,)
    assert len(result.input_ids[0]) == 8


def test_truncation_rejects_lost_insertion_anchor():
    with pytest.raises(ValueError, match="outside retained prefix"):
        prepare_conditioned_sequences(
            input_ids=(torch.arange(10),),
            labels=(torch.arange(10),),
            evidence_insert_positions=(7,),
            evidence_token_id=90,
            trajectory_token_id=91,
            num_evidence_tokens=2,
            num_trajectory_tokens=2,
            max_length=8,
        )


def test_replace_embeddings_is_per_sample_and_preserves_other_tokens():
    input_ids = torch.tensor([[1, 90, 90, 2], [90, 3, 90, 4]])
    inputs_embeds = torch.zeros(2, 4, 3)
    evidence_tokens = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    result = replace_evidence_embeddings(input_ids, inputs_embeds, evidence_tokens, evidence_token_id=90)

    assert torch.equal(result[0, 1:3], evidence_tokens[0])
    assert torch.equal(result[1, [0, 2]], evidence_tokens[1])
    assert torch.count_nonzero(result[input_ids.ne(90)]) == 0
    assert torch.count_nonzero(inputs_embeds) == 0


def test_replace_embeddings_rejects_placeholder_count_mismatch():
    with pytest.raises(ValueError, match="each sample must contain 2"):
        replace_evidence_embeddings(
            input_ids=torch.tensor([[1, 90, 2], [90, 90, 3]]),
            inputs_embeds=torch.zeros(2, 3, 4),
            evidence_tokens=torch.zeros(2, 2, 4),
            evidence_token_id=90,
        )

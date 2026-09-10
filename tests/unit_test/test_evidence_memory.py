import pytest
import torch

from internnav.model.basemodel.internvla_n1.evidence_memory import TaskConditionedEvidenceMemory


def make_inputs(batch_size=2, history_size=5):
    return {
        "history_features": torch.randn(batch_size, history_size, 6),
        "relative_poses": torch.randn(batch_size, history_size, 4),
        "ages": torch.arange(history_size, dtype=torch.float32).view(1, history_size, 1).expand(batch_size, -1, -1),
        "qualities": torch.rand(batch_size, history_size, 2),
        "valid_mask": torch.tensor([[True, True, True, False, False], [False] * history_size]),
        "task_state": torch.randn(batch_size, 7),
    }


def make_memory(output_dim=8):
    return TaskConditionedEvidenceMemory(
        feature_dim=6,
        task_dim=7,
        hidden_dim=8,
        output_dim=output_dim,
        pose_dim=4,
        quality_dim=2,
        num_evidence_tokens=3,
        num_heads=2,
        num_stages=4,
    )


def test_fixed_shapes_masks_and_empty_history_are_stable():
    memory = make_memory().eval()
    inputs = make_inputs()

    output = memory(**inputs)

    assert output.tokens.shape == (2, 3, 8)
    assert output.read_weights.shape == (2, 3, 5)
    assert output.null_weights.shape == (2, 3)
    assert output.stage_logits.shape == (2, 4)
    assert output.no_evidence.tolist() == [False, True]
    assert torch.count_nonzero(output.read_weights[0, :, 3:]) == 0
    assert torch.count_nonzero(output.read_weights[1]) == 0
    assert torch.allclose(output.null_weights[1], torch.ones_like(output.null_weights[1]))
    assert torch.isfinite(output.tokens).all()


def test_bottleneck_projects_to_s2_hidden_space_without_parameter_explosion():
    memory = TaskConditionedEvidenceMemory(
        feature_dim=3584,
        task_dim=512,
        hidden_dim=512,
        output_dim=3584,
        num_evidence_tokens=4,
        num_heads=8,
    )
    trainable = sum(parameter.numel() for parameter in memory.parameters())

    assert memory.output_projection.weight.shape == (3584, 512)
    assert 7_000_000 < trainable < 8_000_000

    small_memory = make_memory(output_dim=11)
    output = small_memory(**make_inputs())
    assert output.tokens.shape == (2, 3, 11)


def test_padding_values_do_not_change_retrieval():
    torch.manual_seed(7)
    memory = make_memory().eval()
    inputs = make_inputs(batch_size=2)
    reference = memory(**inputs)

    changed = {name: value.clone() for name, value in inputs.items()}
    invalid = ~changed["valid_mask"]
    changed["history_features"][invalid] = 1000
    changed["relative_poses"][invalid] = -1000
    changed["ages"][invalid] = -1
    changed["qualities"][invalid] = 500
    result = memory(**changed)

    assert torch.allclose(reference.tokens, result.tokens)
    assert torch.allclose(reference.read_weights, result.read_weights)


def test_task_state_conditions_reading_but_not_writing():
    torch.manual_seed(11)
    memory = make_memory().eval()
    inputs = make_inputs(batch_size=2)
    written = memory.write(inputs["history_features"], inputs["relative_poses"], inputs["ages"], inputs["qualities"])

    changed = {name: value.clone() for name, value in inputs.items()}
    changed["task_state"] += 2.0
    changed_written = memory.write(
        changed["history_features"], changed["relative_poses"], changed["ages"], changed["qualities"]
    )

    assert torch.equal(written, changed_written)
    assert not torch.allclose(memory(**inputs).tokens, memory(**changed).tokens)


def test_gradients_reach_evidence_and_task_paths():
    memory = make_memory()
    inputs = make_inputs(batch_size=2)
    inputs["history_features"].requires_grad_()
    inputs["task_state"].requires_grad_()

    output = memory(**inputs)
    loss = output.tokens[..., 0].sum() + output.stage_logits.square().mean()
    loss.backward()

    assert inputs["history_features"].grad is not None
    assert inputs["history_features"].grad[inputs["valid_mask"]].abs().sum() > 0
    assert inputs["task_state"].grad is not None
    assert inputs["task_state"].grad.abs().sum() > 0
    assert memory.feature_projection.weight.grad is not None
    assert memory.task_projection.weight.grad is not None


def test_fp32_metadata_is_cast_for_bfloat16_model():
    memory = make_memory().to(dtype=torch.bfloat16).eval()
    inputs = make_inputs(batch_size=2)
    inputs["history_features"] = inputs["history_features"].to(dtype=torch.bfloat16)
    inputs["task_state"] = inputs["task_state"].to(dtype=torch.bfloat16)

    output = memory(**inputs)

    assert output.tokens.dtype == torch.bfloat16
    assert torch.isfinite(output.tokens).all()


def test_negative_age_is_rejected_only_for_valid_evidence():
    memory = make_memory()
    inputs = make_inputs()
    inputs["ages"][0, 0, 0] = -1

    with pytest.raises(ValueError, match="negative age"):
        memory(**inputs)

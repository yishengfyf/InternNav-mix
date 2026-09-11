import torch
import torch.nn as nn

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    EVIDENCE_TOKEN_INDEX,
    IMAGE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)


class FakeVisual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.hidden_size = hidden_size

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, pixel_values, grid_thw):
        count = int(grid_thw.prod(dim=1).div(4, rounding_mode="floor").sum())
        values = torch.arange(count * self.hidden_size, dtype=self.dtype, device=pixel_values.device)
        return values.view(count, self.hidden_size) / 100.0


def tiny_model():
    config = InternVLAN1ModelConfig(
        vocab_size=151700,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
            "out_hidden_size": 32,
        },
        image_token_id=IMAGE_TOKEN_INDEX,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        use_evidence_memory=True,
        evidence_task_dim=16,
        evidence_bottleneck_dim=16,
        num_evidence_tokens=2,
        evidence_num_heads=4,
        evidence_num_stages=4,
        evidence_dropout=0.0,
    )
    model = InternVLAN1ForCausalLM(config)
    model.visual = FakeVisual(config.hidden_size)
    return model


def test_tiny_production_forward_injects_evidence_and_splits_s2_loss():
    model = tiny_model()
    input_ids = torch.tensor(
        [[10, 151652, IMAGE_TOKEN_INDEX, 151653, 11, 151652, IMAGE_TOKEN_INDEX, 151653, EVIDENCE_TOKEN_INDEX, EVIDENCE_TOKEN_INDEX, 12, 13]]
    )
    labels = torch.tensor([[-100] * 10 + [12, 13]])
    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.zeros(2, 3),
        image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        evidence_relative_poses=torch.zeros(1, 1, 4),
        evidence_ages=torch.ones(1, 1, 1),
        evidence_qualities=torch.ones(1, 1, 2),
        evidence_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        evidence_history_counts=torch.tensor([1]),
        evidence_image_counts=torch.tensor([2]),
        return_dict=True,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.s2_loss is not None and torch.equal(output.loss, output.s2_loss)
    assert output.trajectory_loss is None
    output.loss.backward()
    assert model.model.evidence_memory.output_projection.weight.grad is not None


def test_late_evidence_residual_is_position_specific_and_differentiable():
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    evidence = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]], requires_grad=True)

    residual = InternVLAN1ForCausalLM._late_evidence_residual(hidden, evidence)

    assert residual.shape == hidden.shape
    assert not torch.allclose(residual[:, 0], residual[:, 1])
    residual.square().sum().backward()
    assert evidence.grad is not None and evidence.grad.abs().sum() > 0

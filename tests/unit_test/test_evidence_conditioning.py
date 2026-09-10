import pytest
import torch

from internnav.model.basemodel.internvla_n1.evidence_conditioning import (
    ObservableTaskStateEstimator,
    gather_visual_evidence,
    pool_visual_features,
)


def test_pooling_and_sample_gather_preserve_image_ownership():
    grids = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]])
    image_embeddings = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    pooled = pool_visual_features(image_embeddings, grids, spatial_merge_size=2)
    result = gather_visual_evidence(
        pooled,
        image_counts=torch.tensor([3, 2]),
        history_counts=torch.tensor([2, 0]),
        max_history=2,
    )

    assert torch.equal(result.history_features[0], pooled[:2])
    assert torch.count_nonzero(result.history_features[1]) == 0
    assert torch.equal(result.current_features, torch.stack((pooled[2], pooled[3])))


def test_pooling_rejects_grid_token_mismatch():
    with pytest.raises(ValueError, match="visual token count mismatch"):
        pool_visual_features(torch.zeros(3, 4), torch.tensor([[1, 2, 2]]), spatial_merge_size=2)


def test_task_estimator_ignores_masked_target_tokens_and_has_gradients():
    torch.manual_seed(3)
    estimator = ObservableTaskStateEstimator(input_dim=6, task_dim=4)
    current = torch.randn(2, 6, requires_grad=True)
    tokens = torch.randn(2, 5, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False], [True, False, True, False, False]])
    reference = estimator(current, tokens, mask)
    changed = tokens.detach().clone()
    changed[~mask] = 1000

    assert torch.allclose(reference, estimator(current, changed, mask))
    reference.square().mean().backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert tokens.grad is not None and tokens.grad[mask].abs().sum() > 0

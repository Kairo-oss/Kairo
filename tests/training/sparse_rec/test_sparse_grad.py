"""Tests for SparseRec sparse gradient autograd Function."""

from __future__ import annotations

import torch

from kairo.training.sparse_rec.sparse_grad import SparseEmbeddingLookup


class TestSparseEmbeddingLookupForward:
    """Tests for SparseEmbeddingLookup.forward."""

    def test_forward_without_mask(self) -> None:
        weight = torch.randn(10, 4, requires_grad=True)
        ids = torch.tensor([0, 2, 5])
        active_ids = torch.arange(10)
        result = SparseEmbeddingLookup.apply(weight, ids, active_ids, None)
        expected = weight[ids].detach()
        assert torch.allclose(result, expected)

    def test_forward_with_mask(self) -> None:
        weight = torch.randn(10, 4, requires_grad=True)
        ids = torch.tensor([1, 3])
        active_ids = torch.arange(10)
        dense_mask = torch.ones(10, 4)
        dense_mask[1, 2] = 0.0  # zero out position (1, 2)
        result = SparseEmbeddingLookup.apply(
            weight, ids, active_ids, dense_mask,
        )
        expected = weight[ids].detach() * dense_mask[ids]
        assert torch.allclose(result, expected)

    def test_forward_shape(self) -> None:
        weight = torch.randn(20, 8, requires_grad=True)
        ids = torch.tensor([0, 5, 10, 15])
        active_ids = torch.arange(20)
        result = SparseEmbeddingLookup.apply(weight, ids, active_ids, None)
        assert result.shape == (4, 8)


class TestSparseEmbeddingLookupBackward:
    """Tests for SparseEmbeddingLookup.backward — sparse gradient."""

    def test_grad_only_for_active_and_batch_ids(self) -> None:
        """Gradients should only exist for IDs in both batch and active set."""
        weight = torch.randn(10, 4, requires_grad=True)
        ids = torch.tensor([0, 2, 5, 7])
        active_ids = torch.tensor([0, 5, 9])  # intersection = {0, 5}
        result = SparseEmbeddingLookup.apply(weight, ids, active_ids, None)
        loss = result.sum()
        loss.backward()

        grad = weight.grad
        assert grad is not None
        # IDs 0 and 5 should have nonzero gradient
        assert grad[0].abs().sum() > 0
        assert grad[5].abs().sum() > 0
        # IDs 2 and 7 are in batch but NOT in active_ids
        assert grad[2].abs().sum() == 0
        assert grad[7].abs().sum() == 0
        # ID 9 is in active_ids but NOT in batch
        assert grad[9].abs().sum() == 0

    def test_grad_shape_matches_weight(self) -> None:
        weight = torch.randn(10, 4, requires_grad=True)
        ids = torch.tensor([1, 3])
        active_ids = torch.tensor([1, 3, 5])
        result = SparseEmbeddingLookup.apply(weight, ids, active_ids, None)
        result.sum().backward()
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape

    def test_grad_with_mask_applied(self) -> None:
        """Mask should affect gradients — masked positions get zero grad."""
        weight = torch.randn(5, 3, requires_grad=True)
        ids = torch.tensor([0, 1])
        active_ids = torch.tensor([0, 1])
        dense_mask = torch.ones(5, 3)
        dense_mask[0, 1] = 0.0  # mask out (0, 1)
        result = SparseEmbeddingLookup.apply(
            weight, ids, active_ids, dense_mask,
        )
        result.sum().backward()
        grad = weight.grad
        assert grad is not None
        # Position (0, 1) was masked, so gradient should be zero
        assert grad[0, 1].item() == 0.0
        # Unmasked positions should have gradient
        assert grad[0, 0].item() != 0.0
        assert grad[1, 0].item() != 0.0

    def test_no_grad_for_ids_and_active_ids(self) -> None:
        """ids and active_ids should not receive gradients."""
        weight = torch.randn(5, 3, requires_grad=True)
        ids = torch.tensor([0, 1])
        active_ids = torch.tensor([0, 1, 2])
        result = SparseEmbeddingLookup.apply(weight, ids, active_ids, None)
        grads = torch.autograd.grad(
            result.sum(), weight, create_graph=False,
        )
        # Should succeed — only weight gets gradient
        assert grads[0].shape == weight.shape

    def test_gradcheck_small_float64(self) -> None:
        """Numerical gradient check with float64 for precision."""
        weight = torch.randn(
            4, 3, dtype=torch.float64, requires_grad=True,
        )
        ids = torch.tensor([0, 2])
        active_ids = torch.tensor([0, 1, 2, 3])

        def fn(w: torch.Tensor) -> torch.Tensor:
            return SparseEmbeddingLookup.apply(w, ids, active_ids, None)

        assert torch.autograd.gradcheck(fn, (weight,), raise_exception=True)

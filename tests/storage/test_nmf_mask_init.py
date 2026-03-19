"""Tests for sparse mask generation (kairo.storage.nmf.mask_init)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from kairo.config import NMFConfig
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.types import NMFResult, SparseMask


@pytest.fixture
def nmf_result(small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig) -> NMFResult:
    """Pre-computed NMF result for mask tests."""
    return nmf_decompose(small_interaction_matrix, default_nmf_config)


class TestGenerateSparseMask:
    """Test suite for generate_sparse_mask function."""

    def test_returns_sparse_mask(self, nmf_result: NMFResult) -> None:
        mask = generate_sparse_mask(nmf_result, num_embeddings=100, embedding_dim=50, sparsity_ratio=0.5)
        assert isinstance(mask, SparseMask)

    def test_dense_shape_matches(self, nmf_result: NMFResult) -> None:
        num_emb, emb_dim = 100, 50
        mask = generate_sparse_mask(nmf_result, num_emb, emb_dim, sparsity_ratio=0.5)
        assert mask.dense_shape == (num_emb, emb_dim)

    @pytest.mark.parametrize("sparsity_ratio", [0.3, 0.5, 0.8, 0.95])
    def test_sparsity_ratio_honored(self, nmf_result: NMFResult, sparsity_ratio: float) -> None:
        """Actual sparsity should be within 5% of the requested ratio."""
        num_emb, emb_dim = 100, 50
        mask = generate_sparse_mask(nmf_result, num_emb, emb_dim, sparsity_ratio)

        total_params = num_emb * emb_dim
        active_params = mask.nnz
        actual_sparsity = 1.0 - active_params / total_params

        assert abs(actual_sparsity - sparsity_ratio) < 0.05, (
            f"Expected sparsity ~{sparsity_ratio}, got {actual_sparsity:.4f}"
        )

    def test_indices_within_bounds(self, nmf_result: NMFResult) -> None:
        num_emb, emb_dim = 100, 50
        mask = generate_sparse_mask(nmf_result, num_emb, emb_dim, sparsity_ratio=0.5)

        assert mask.indices.shape[0] == 2
        assert (mask.indices[0] >= 0).all() and (mask.indices[0] < num_emb).all()
        assert (mask.indices[1] >= 0).all() and (mask.indices[1] < emb_dim).all()

    def test_values_are_ones(self, nmf_result: NMFResult) -> None:
        """Active mask entries should be 1.0."""
        mask = generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)
        assert torch.allclose(mask.values, torch.ones_like(mask.values))

    def test_deterministic_output(self, nmf_result: NMFResult) -> None:
        mask1 = generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)
        mask2 = generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)

        assert torch.equal(mask1.indices, mask2.indices)
        assert torch.equal(mask1.values, mask2.values)

    def test_immutability_of_nmf_result(self, nmf_result: NMFResult) -> None:
        """NMFResult must not be modified."""
        W_clone = nmf_result.W.clone()
        H_clone = nmf_result.H.clone()

        generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)

        assert torch.equal(nmf_result.W, W_clone)
        assert torch.equal(nmf_result.H, H_clone)

    def test_result_is_frozen(self, nmf_result: NMFResult) -> None:
        mask = generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)
        assert dataclasses.is_dataclass(mask)

        with pytest.raises(dataclasses.FrozenInstanceError):
            mask.sparsity_ratio = 0.0  # type: ignore[misc]

    def test_zero_sparsity_keeps_all(self, nmf_result: NMFResult) -> None:
        """0% sparsity means all parameters are active."""
        num_emb, emb_dim = 100, 50
        mask = generate_sparse_mask(nmf_result, num_emb, emb_dim, sparsity_ratio=0.0)
        assert mask.nnz == num_emb * emb_dim

    def test_to_dense_roundtrip(self, nmf_result: NMFResult) -> None:
        """Converting mask to dense and checking active count."""
        mask = generate_sparse_mask(nmf_result, 100, 50, sparsity_ratio=0.5)
        dense = mask.to_dense()
        assert dense.shape == mask.dense_shape
        assert dense.sum().item() == mask.nnz

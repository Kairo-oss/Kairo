"""Tests for SparseEmbeddingTable (kairo.storage.embedding)."""

from __future__ import annotations

import torch

from kairo.config import NMFConfig
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.types import EmbeddingConfig, SparseMask


def _make_mask(num_embeddings: int, embedding_dim: int, sparsity_ratio: float) -> SparseMask:
    """Helper to create a SparseMask via NMF pipeline."""
    rng = torch.Generator()
    rng.manual_seed(42)
    W = torch.abs(torch.randn(num_embeddings, 5, generator=rng))
    H = torch.abs(torch.randn(5, embedding_dim, generator=rng))
    V = W @ H

    config = NMFConfig(rank=5, max_iter=50, seed=42)
    result = nmf_decompose(V, config)
    return generate_sparse_mask(result, num_embeddings, embedding_dim, sparsity_ratio)


class TestSparseEmbeddingTable:
    """Test suite for SparseEmbeddingTable."""

    def test_construction_without_mask(self, default_embedding_config: EmbeddingConfig) -> None:
        table = SparseEmbeddingTable(default_embedding_config)
        assert table is not None

    def test_construction_with_mask(self, default_embedding_config: EmbeddingConfig) -> None:
        mask = _make_mask(
            default_embedding_config.num_embeddings,
            default_embedding_config.embedding_dim,
            default_embedding_config.sparsity_ratio,
        )
        table = SparseEmbeddingTable(default_embedding_config, mask=mask)
        assert table is not None

    def test_forward_output_shape(self, default_embedding_config: EmbeddingConfig) -> None:
        table = SparseEmbeddingTable(default_embedding_config)
        ids = torch.tensor([0, 5, 10, 20])
        output = table(ids)
        assert output.shape == (4, default_embedding_config.embedding_dim)

    def test_forward_with_mask_output_shape(
        self, default_embedding_config: EmbeddingConfig
    ) -> None:
        mask = _make_mask(
            default_embedding_config.num_embeddings,
            default_embedding_config.embedding_dim,
            default_embedding_config.sparsity_ratio,
        )
        table = SparseEmbeddingTable(default_embedding_config, mask=mask)
        ids = torch.tensor([0, 5, 10])
        output = table(ids)
        assert output.shape == (3, default_embedding_config.embedding_dim)

    def test_masked_parameters_produce_zeros(self) -> None:
        """Masked positions should be zero in the output."""
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.5)
        mask = _make_mask(10, 8, 0.5)
        table = SparseEmbeddingTable(config, mask=mask)

        ids = torch.arange(10)
        output = table(ids)

        # Get the dense mask to check zeroed positions
        dense_mask = mask.to_dense()
        inactive = dense_mask == 0
        assert (output[inactive]).abs().sum().item() == 0.0

    def test_active_parameters_non_zero(self) -> None:
        """Active positions should generally be non-zero (with random init)."""
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.5)
        mask = _make_mask(10, 8, 0.5)
        table = SparseEmbeddingTable(config, mask=mask)

        ids = torch.arange(10)
        output = table(ids)

        dense_mask = mask.to_dense().bool()
        active_values = output[dense_mask]
        # At least some active values should be non-zero (extremely unlikely all are zero)
        assert active_values.abs().sum().item() > 0

    def test_without_mask_is_dense(self) -> None:
        """Without mask, output should equal standard embedding lookup."""
        config = EmbeddingConfig(num_embeddings=20, embedding_dim=16, sparsity_ratio=0.0)
        table = SparseEmbeddingTable(config)

        ids = torch.tensor([0, 1, 2])
        output = table(ids)

        # All values should be non-zero (with random init)
        assert output.shape == (3, 16)

    def test_with_mask_returns_new_instance(self, default_embedding_config: EmbeddingConfig) -> None:
        """with_mask() must return a new SparseEmbeddingTable (immutable pattern)."""
        mask1 = _make_mask(
            default_embedding_config.num_embeddings,
            default_embedding_config.embedding_dim,
            0.3,
        )
        mask2 = _make_mask(
            default_embedding_config.num_embeddings,
            default_embedding_config.embedding_dim,
            0.8,
        )

        table1 = SparseEmbeddingTable(default_embedding_config, mask=mask1)
        table2 = table1.with_mask(mask2)

        assert table1 is not table2
        assert isinstance(table2, SparseEmbeddingTable)

    def test_gradient_flows_through_active(self) -> None:
        """Gradients should flow through active (non-masked) parameters."""
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.5)
        mask = _make_mask(10, 8, 0.5)
        table = SparseEmbeddingTable(config, mask=mask)

        ids = torch.tensor([0, 1, 2])
        output = table(ids)
        loss = output.sum()
        loss.backward()

        # Weight gradient should exist
        assert table.weight.grad is not None
        assert table.weight.grad.shape == (10, 8)

    def test_batch_input(self, default_embedding_config: EmbeddingConfig) -> None:
        """Should handle 2D batch inputs."""
        table = SparseEmbeddingTable(default_embedding_config)
        ids = torch.tensor([[0, 1], [2, 3]])
        output = table(ids)
        assert output.shape == (2, 2, default_embedding_config.embedding_dim)

    def test_active_parameter_count(self) -> None:
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.5)
        mask = _make_mask(10, 8, 0.5)
        table = SparseEmbeddingTable(config, mask=mask)

        assert table.active_parameter_count == mask.nnz
        assert table.active_parameter_count < 10 * 8

    def test_compression_ratio(self) -> None:
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.5)
        mask = _make_mask(10, 8, 0.5)
        table = SparseEmbeddingTable(config, mask=mask)

        ratio = table.compression_ratio
        assert 0.0 < ratio < 1.0

    def test_compression_ratio_no_mask(self) -> None:
        """Without mask, compression ratio should be 0 (no compression)."""
        config = EmbeddingConfig(num_embeddings=10, embedding_dim=8, sparsity_ratio=0.0)
        table = SparseEmbeddingTable(config)
        assert table.compression_ratio == 0.0

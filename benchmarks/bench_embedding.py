"""Benchmarks: sparse vs dense embedding forward pass."""

from __future__ import annotations

import pytest
import torch

from kairo.config import NMFConfig
from kairo.ops.sparse_embedding import sparse_gather
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.types import EmbeddingConfig

from benchmarks.conftest import bench


def _make_sparse_table(
    num_embeddings: int, embedding_dim: int, sparsity: float, device: torch.device,
) -> SparseEmbeddingTable:
    torch.manual_seed(42)
    interaction = torch.rand(num_embeddings, embedding_dim, device=device) + 0.01
    nmf_cfg = NMFConfig(rank=8, max_iter=30, seed=42)
    nmf_result = nmf_decompose(interaction.cpu(), nmf_cfg)
    mask = generate_sparse_mask(nmf_result, num_embeddings, embedding_dim, sparsity)
    config = EmbeddingConfig(num_embeddings, embedding_dim, sparsity_ratio=sparsity)
    table = SparseEmbeddingTable(config, mask=mask).to(device)
    return table


@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.95])
@pytest.mark.parametrize("num_embeddings", [1000, 10000])
def test_bench_sparse_forward(
    sparsity: float, num_embeddings: int, device: torch.device,
) -> None:
    embedding_dim = 64
    batch_size = 256
    table = _make_sparse_table(num_embeddings, embedding_dim, sparsity, device)
    ids = torch.randint(0, num_embeddings, (batch_size,), device=device)

    result = bench(
        lambda: table(ids),
        name=f"sparse_forward(N={num_embeddings}, sp={sparsity})",
        warmup=5,
        iterations=20,
    )
    print(f"  {result}")


@pytest.mark.parametrize("num_embeddings", [1000, 10000])
def test_bench_dense_forward(num_embeddings: int, device: torch.device) -> None:
    embedding_dim = 64
    batch_size = 256
    config = EmbeddingConfig(num_embeddings, embedding_dim)
    table = SparseEmbeddingTable(config).to(device)
    ids = torch.randint(0, num_embeddings, (batch_size,), device=device)

    result = bench(
        lambda: table(ids),
        name=f"dense_forward(N={num_embeddings})",
        warmup=5,
        iterations=20,
    )
    print(f"  {result}")


@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.95])
def test_bench_sparse_gather_fallback(sparsity: float, device: torch.device) -> None:
    """Benchmark the ops.sparse_gather fallback path."""
    num_embeddings, embedding_dim = 5000, 64
    batch_size = 256
    table = _make_sparse_table(num_embeddings, embedding_dim, sparsity, device)
    ids = torch.randint(0, num_embeddings, (batch_size,), device=device)
    mask = table._mask

    result = bench(
        lambda: sparse_gather(table.weight, mask, ids),
        name=f"sparse_gather(sp={sparsity})",
        warmup=5,
        iterations=20,
    )
    print(f"  {result}")

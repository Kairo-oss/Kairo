"""Benchmarks: sparse vs dense backward pass.

These benchmarks run on CPU to fairly compare SparseRec sparse gradients
vs standard dense gradients through the PyTorch autograd path.
"""

from __future__ import annotations

import pytest
import torch

from kairo.config import NMFConfig
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.training.sparse_rec.sampler import compute_sampling_weights, sample_active_ids
from kairo.types import EmbeddingConfig

from benchmarks.conftest import bench


def _make_table_cpu(
    num_e: int, emb_dim: int, sparsity: float,
) -> SparseEmbeddingTable:
    torch.manual_seed(42)
    interaction = torch.rand(num_e, emb_dim) + 0.01
    nmf_cfg = NMFConfig(rank=8, max_iter=30, seed=42)
    result = nmf_decompose(interaction, nmf_cfg)
    mask = generate_sparse_mask(result, num_e, emb_dim, sparsity)
    config = EmbeddingConfig(num_e, emb_dim, sparsity_ratio=sparsity)
    return SparseEmbeddingTable(config, mask=mask)


@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.95])
def test_bench_sparse_backward(sparsity: float) -> None:
    num_e, emb_dim, batch_size = 5000, 64, 256
    table = _make_table_cpu(num_e, emb_dim, sparsity)
    ids = torch.randint(0, num_e, (batch_size,))

    access_counts = torch.ones(num_e)
    weights = compute_sampling_weights(access_counts)
    active_ids = sample_active_ids(weights, max(1, num_e // 10))

    def step():
        if table.weight.grad is not None:
            table.weight.grad.zero_()
        out = table(ids, active_ids=active_ids)
        out.sum().backward()

    result = bench(step, name=f"sparse_backward(sp={sparsity})", warmup=3, iterations=15)
    print(f"  {result}")


@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.95])
def test_bench_dense_backward(sparsity: float) -> None:
    num_e, emb_dim, batch_size = 5000, 64, 256
    table = _make_table_cpu(num_e, emb_dim, sparsity)
    ids = torch.randint(0, num_e, (batch_size,))

    def step():
        if table.weight.grad is not None:
            table.weight.grad.zero_()
        out = table(ids)  # no active_ids → standard autograd backward
        out.sum().backward()

    result = bench(step, name=f"dense_backward(sp={sparsity})", warmup=3, iterations=15)
    print(f"  {result}")

"""Benchmarks: mask topology update (prune + grow cycle) timing."""

from __future__ import annotations

import pytest
import torch

from kairo.config import SparseRecConfig
from kairo.training.sparse_rec.topology import GradientAccumulator, update_topology
from kairo.types import SparseMask

from benchmarks.conftest import bench


def _make_mask(num_e: int, emb_dim: int, sparsity: float) -> SparseMask:
    total = num_e * emb_dim
    nnz = max(1, int(total * (1.0 - sparsity)))

    torch.manual_seed(42)
    flat_indices = torch.randperm(total)[:nnz]
    rows = flat_indices // emb_dim
    cols = flat_indices % emb_dim
    indices = torch.stack([rows, cols])
    values = torch.ones(nnz)

    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=(num_e, emb_dim),
        sparsity_ratio=1.0 - nnz / total,
    )


@pytest.mark.parametrize("num_e", [1000, 10000, 100000])
def test_bench_topology_update(num_e: int) -> None:
    emb_dim = 64
    sparsity = 0.8
    mask = _make_mask(num_e, emb_dim, sparsity)
    weights = torch.randn(num_e, emb_dim)
    accumulator = GradientAccumulator(
        cumulative_magnitudes=torch.rand(num_e, emb_dim),
        num_steps=100,
    )
    config = SparseRecConfig(regrowth_fraction=0.1)

    result = bench(
        lambda: update_topology(mask, weights, accumulator, config),
        name=f"topology_update(N={num_e})",
        warmup=2,
        iterations=10,
    )
    print(f"  {result}")

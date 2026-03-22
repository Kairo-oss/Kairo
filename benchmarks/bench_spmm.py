"""Benchmarks: block-sparse matmul vs dense torch.mm."""

from __future__ import annotations

import pytest
import torch

from kairo.ops.spmm import block_sparse_mm
from kairo.types import SparseMask

from benchmarks.conftest import bench


def _make_block_mask(
    M: int, K: int, tile_size: int, sparsity: float, device: torch.device,
) -> SparseMask:
    """Create a block-level mask with given sparsity."""
    block_rows = M // tile_size
    block_cols = K // tile_size
    total_blocks = block_rows * block_cols
    num_active = max(1, int(total_blocks * (1.0 - sparsity)))

    torch.manual_seed(42)
    perm = torch.randperm(total_blocks)[:num_active]
    rows = perm // block_cols
    cols = perm % block_cols
    indices = torch.stack([rows, cols]).to(device)
    values = torch.ones(num_active, device=device)

    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=(block_rows, block_cols),
        sparsity_ratio=1.0 - num_active / total_blocks,
    )


@pytest.mark.parametrize("size", [256, 1024])
@pytest.mark.parametrize("sparsity", [0.8, 0.95])
def test_bench_block_sparse_mm(
    size: int, sparsity: float, device: torch.device,
) -> None:
    tile_size = 16
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    mask = _make_block_mask(size, size, tile_size, sparsity, device)

    result = bench(
        lambda: block_sparse_mm(A, B, mask, tile_size=tile_size),
        name=f"block_sparse_mm({size}x{size}, sp={sparsity})",
        warmup=3,
        iterations=15,
    )
    print(f"  {result}")


@pytest.mark.parametrize("size", [256, 1024])
def test_bench_dense_mm(size: int, device: torch.device) -> None:
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    result = bench(
        lambda: torch.mm(A, B),
        name=f"dense_mm({size}x{size})",
        warmup=3,
        iterations=15,
    )
    print(f"  {result}")

"""Tests for block-sparse matrix multiplication ops."""

from __future__ import annotations

import pytest
import torch

from kairo.ops.spmm import (
    _dense_to_block_sparse,
    block_sparse_mm,
)
from kairo.ops._autotuner import TileAutotuner, TileConfig
from kairo.types import SparseMask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_block_mask(
    num_tile_rows: int,
    num_tile_cols: int,
    sparsity: float,
    seed: int = 42,
) -> SparseMask:
    """Create a random block-level SparseMask at given sparsity."""
    torch.manual_seed(seed)
    total = num_tile_rows * num_tile_cols
    nnz = max(1, int(total * (1.0 - sparsity)))
    perm = torch.randperm(total)[:nnz]
    rows = perm // num_tile_cols
    cols = perm % num_tile_cols
    sort_idx = torch.argsort(rows, stable=True)
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    indices = torch.stack([rows, cols]).to(torch.int64)
    values = torch.ones(nnz)
    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=(num_tile_rows, num_tile_cols),
        sparsity_ratio=sparsity,
    )


@pytest.fixture
def small_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    """A (32, 32) and B (32, 16) for basic tests."""
    torch.manual_seed(42)
    A = torch.randn(32, 32)
    B = torch.randn(32, 16)
    return A, B


# ---------------------------------------------------------------------------
# _dense_to_block_sparse
# ---------------------------------------------------------------------------


class TestDenseToBlockSparse:
    def test_roundtrip(self) -> None:
        """Extracting tiles and reassembling should match original (at active tiles)."""
        torch.manual_seed(42)
        M, K, tile = 32, 32, 16
        A = torch.randn(M, K)
        mask = _make_block_mask(M // tile, K // tile, sparsity=0.5)

        tiles, row_ids, col_ids = _dense_to_block_sparse(A, mask, tile)

        assert tiles.shape[0] == mask.nnz
        assert tiles.shape[1] == tile
        assert tiles.shape[2] == tile
        assert row_ids.dtype == torch.int32
        assert col_ids.dtype == torch.int32

        # Verify each tile matches the original matrix
        for t in range(mask.nnz):
            r = int(row_ids[t].item()) * tile
            c = int(col_ids[t].item()) * tile
            expected = A[r : r + tile, c : c + tile]
            assert torch.allclose(tiles[t], expected)

    def test_single_tile(self) -> None:
        A = torch.eye(16)
        mask = SparseMask(
            indices=torch.tensor([[0], [0]], dtype=torch.int64),
            values=torch.ones(1),
            dense_shape=(1, 1),
            sparsity_ratio=0.0,
        )
        tiles, row_ids, col_ids = _dense_to_block_sparse(A, mask, 16)
        assert tiles.shape == (1, 16, 16)
        assert torch.allclose(tiles[0], A)


# ---------------------------------------------------------------------------
# block_sparse_mm (CPU fallback)
# ---------------------------------------------------------------------------


class TestBlockSparseMmCPU:
    def test_matches_torch_mm(
        self, small_matrices: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        A, B = small_matrices
        tile = 16
        # Fully dense block mask (no sparsity) should match torch.mm
        mask = _make_block_mask(
            A.shape[0] // tile, A.shape[1] // tile, sparsity=0.0
        )
        result = block_sparse_mm(A, B, mask, tile_size=tile)
        expected = A @ B

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4)

    def test_sparse_result_differs_from_dense(
        self, small_matrices: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        A, B = small_matrices
        tile = 16
        mask = _make_block_mask(
            A.shape[0] // tile, A.shape[1] // tile, sparsity=0.75
        )
        sparse_result = block_sparse_mm(A, B, mask, tile_size=tile)
        dense_result = A @ B

        # With 75% sparsity, results should differ
        assert not torch.allclose(sparse_result, dense_result, atol=1e-4)

    def test_high_sparsity_95(self) -> None:
        torch.manual_seed(42)
        M, K, N, tile = 64, 64, 32, 16
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        mask = _make_block_mask(M // tile, K // tile, sparsity=0.95)

        result = block_sparse_mm(A, B, mask, tile_size=tile)

        # Reconstruct expected: zero out inactive tiles, then matmul
        A_masked = torch.zeros_like(A)
        for t in range(mask.nnz):
            r = mask.indices[0, t].item() * tile
            c = mask.indices[1, t].item() * tile
            A_masked[r : r + tile, c : c + tile] = A[r : r + tile, c : c + tile]
        expected = A_masked @ B

        assert torch.allclose(result, expected, atol=1e-4)

    def test_different_tile_sizes(self) -> None:
        torch.manual_seed(42)
        M, K, N = 64, 64, 32

        for tile in [16, 32]:
            A = torch.randn(M, K)
            B = torch.randn(K, N)
            mask = _make_block_mask(M // tile, K // tile, sparsity=0.5)
            result = block_sparse_mm(A, B, mask, tile_size=tile)

            A_masked = torch.zeros_like(A)
            for t in range(mask.nnz):
                r = mask.indices[0, t].item() * tile
                c = mask.indices[1, t].item() * tile
                A_masked[r : r + tile, c : c + tile] = (
                    A[r : r + tile, c : c + tile]
                )
            expected = A_masked @ B
            assert torch.allclose(result, expected, atol=1e-4), (
                f"Failed for tile_size={tile}"
            )

    def test_empty_mask(self) -> None:
        torch.manual_seed(42)
        A = torch.randn(32, 32)
        B = torch.randn(32, 16)
        mask = SparseMask(
            indices=torch.zeros(2, 0, dtype=torch.int64),
            values=torch.zeros(0),
            dense_shape=(2, 2),
            sparsity_ratio=1.0,
        )
        result = block_sparse_mm(A, B, mask, tile_size=16)
        assert torch.all(result == 0)

    def test_dimension_mismatch_raises(self) -> None:
        A = torch.randn(32, 32)
        B = torch.randn(16, 16)  # K doesn't match
        mask = _make_block_mask(2, 2, sparsity=0.0)
        with pytest.raises(ValueError, match="Inner dimensions must match"):
            block_sparse_mm(A, B, mask, tile_size=16)


# ---------------------------------------------------------------------------
# TileAutotuner
# ---------------------------------------------------------------------------


class TestAutotuner:
    def test_returns_valid_config(self) -> None:
        autotuner = TileAutotuner()
        config = autotuner.best_config(M=256, K=256, N=128, sparsity=0.5)

        assert "tile_m" in config
        assert "tile_n" in config
        assert "tile_k" in config
        assert config["tile_m"] in (16, 32, 64)
        assert config["tile_k"] in (16, 32)

    def test_cache_returns_same_result(self) -> None:
        autotuner = TileAutotuner()
        c1 = autotuner.best_config(M=256, K=256, N=128, sparsity=0.5)
        c2 = autotuner.best_config(M=256, K=256, N=128, sparsity=0.5)
        assert c1 == c2

    def test_clear_cache(self) -> None:
        autotuner = TileAutotuner()
        autotuner.best_config(M=256, K=256, N=128, sparsity=0.5)
        assert len(autotuner._cache) > 0
        autotuner.clear_cache()
        assert len(autotuner._cache) == 0

    def test_default_config_for_cpu(self) -> None:
        """When CUDA not available in autotuner, should still return valid config."""
        autotuner = TileAutotuner()
        # This will use the CPU fallback path in best_config
        config = autotuner.best_config(M=64, K=64, N=32, sparsity=0.5)
        assert config["tile_m"] >= 16


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestBlockSparseMmCUDA:
    def test_cuda_matches_cpu(self) -> None:
        torch.manual_seed(42)
        M, K, N, tile = 32, 32, 16, 16
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        mask = _make_block_mask(M // tile, K // tile, sparsity=0.5)

        cpu_result = block_sparse_mm(A, B, mask, tile_size=tile)

        cuda_mask = SparseMask(
            indices=mask.indices.cuda(),
            values=mask.values.cuda(),
            dense_shape=mask.dense_shape,
            sparsity_ratio=mask.sparsity_ratio,
        )
        cuda_result = block_sparse_mm(
            A.cuda(), B.cuda(), cuda_mask, tile_size=tile
        )

        assert cuda_result.device.type == "cuda"
        assert torch.allclose(cpu_result, cuda_result.cpu(), atol=1e-3)

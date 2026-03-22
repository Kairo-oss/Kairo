"""Tests for sparse embedding gather/scatter ops."""

from __future__ import annotations

import pytest
import torch

from kairo.ops.sparse_embedding import (
    _coo_to_row_offsets,
    cuda_ext_available,
    sparse_gather,
    sparse_scatter_grad,
)
from kairo.types import SparseMask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def weight() -> torch.Tensor:
    """Small embedding weight matrix (10 rows, 8 dims)."""
    torch.manual_seed(42)
    return torch.randn(10, 8)


@pytest.fixture
def sparse_mask() -> SparseMask:
    """Sparse mask with ~50% sparsity for a (10, 8) table."""
    torch.manual_seed(42)
    N, D = 10, 8
    total = N * D
    nnz = total // 2
    perm = torch.randperm(total)[:nnz]
    rows = perm // D
    cols = perm % D
    # Sort by row for consistent behavior
    sort_idx = torch.argsort(rows, stable=True)
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    indices = torch.stack([rows, cols])
    values = torch.ones(nnz)
    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=(N, D),
        sparsity_ratio=1.0 - nnz / total,
    )


@pytest.fixture
def batch_ids() -> torch.Tensor:
    """Batch of embedding IDs to look up."""
    return torch.tensor([0, 3, 5, 7, 9], dtype=torch.int64)


# ---------------------------------------------------------------------------
# _coo_to_row_offsets
# ---------------------------------------------------------------------------


class TestCooToRowOffsets:
    def test_basic_correctness(self) -> None:
        rows = torch.tensor([0, 0, 1, 1, 1, 3], dtype=torch.int64)
        offsets = _coo_to_row_offsets(rows, num_rows=5)

        assert offsets.shape == (6,)
        expected = torch.tensor([0, 2, 5, 5, 6, 6], dtype=torch.int64)
        assert torch.equal(offsets, expected)

    def test_empty_mask(self) -> None:
        rows = torch.tensor([], dtype=torch.int64)
        offsets = _coo_to_row_offsets(rows, num_rows=4)

        assert offsets.shape == (5,)
        assert torch.all(offsets == 0)

    def test_single_row(self) -> None:
        rows = torch.tensor([2, 2, 2], dtype=torch.int64)
        offsets = _coo_to_row_offsets(rows, num_rows=4)

        expected = torch.tensor([0, 0, 0, 3, 3], dtype=torch.int64)
        assert torch.equal(offsets, expected)


# ---------------------------------------------------------------------------
# cuda_ext_available
# ---------------------------------------------------------------------------


class TestCudaExtAvailable:
    def test_returns_bool(self) -> None:
        result = cuda_ext_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# sparse_gather (CPU fallback)
# ---------------------------------------------------------------------------


class TestSparseGatherCPU:
    def test_matches_pytorch_reference(
        self, weight: torch.Tensor, sparse_mask: SparseMask, batch_ids: torch.Tensor
    ) -> None:
        result = sparse_gather(weight, sparse_mask, batch_ids)

        # Reference: manual dense mask application
        dense_mask = sparse_mask.to_dense()
        expected = weight[batch_ids] * dense_mask[batch_ids]

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-6)

    def test_with_repeated_ids(
        self, weight: torch.Tensor, sparse_mask: SparseMask
    ) -> None:
        ids = torch.tensor([2, 2, 5, 5, 2], dtype=torch.int64)
        result = sparse_gather(weight, sparse_mask, ids)

        dense_mask = sparse_mask.to_dense()
        expected = weight[ids] * dense_mask[ids]

        assert torch.allclose(result, expected, atol=1e-6)

    def test_empty_mask(self, weight: torch.Tensor) -> None:
        empty_mask = SparseMask(
            indices=torch.zeros(2, 0, dtype=torch.int64),
            values=torch.zeros(0),
            dense_shape=(10, 8),
            sparsity_ratio=1.0,
        )
        ids = torch.tensor([0, 1], dtype=torch.int64)
        result = sparse_gather(weight, empty_mask, ids)

        assert result.shape == (2, 8)
        assert torch.all(result == 0)

    def test_empty_ids(
        self, weight: torch.Tensor, sparse_mask: SparseMask
    ) -> None:
        ids = torch.tensor([], dtype=torch.int64)
        result = sparse_gather(weight, sparse_mask, ids)

        assert result.shape == (0, 8)

    def test_output_dtype_matches_weight(
        self, weight: torch.Tensor, sparse_mask: SparseMask, batch_ids: torch.Tensor
    ) -> None:
        result = sparse_gather(weight, sparse_mask, batch_ids)
        assert result.dtype == weight.dtype


# ---------------------------------------------------------------------------
# sparse_scatter_grad (CPU fallback)
# ---------------------------------------------------------------------------


class TestSparseScatterGradCPU:
    def test_matches_reference(self) -> None:
        torch.manual_seed(42)
        B, D, N = 4, 8, 10
        grad_output = torch.randn(B, D)
        batch_ids = torch.tensor([1, 3, 1, 5], dtype=torch.int64)
        active_ids = torch.tensor([1, 3, 5], dtype=torch.int64)

        result = sparse_scatter_grad(
            grad_output, active_ids, batch_ids, N, D
        )

        # Reference: accumulate manually
        expected = torch.zeros(N, D)
        expected[1] = grad_output[0] + grad_output[2]  # batch 0 and 2 map to row 1
        expected[3] = grad_output[1]
        expected[5] = grad_output[3]

        assert result.shape == (N, D)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_inactive_ids_get_zero_grad(self) -> None:
        B, D, N = 2, 4, 6
        grad_output = torch.ones(B, D)
        batch_ids = torch.tensor([0, 2], dtype=torch.int64)
        active_ids = torch.tensor([0], dtype=torch.int64)  # only row 0 active

        result = sparse_scatter_grad(
            grad_output, active_ids, batch_ids, N, D
        )

        # Row 2 is NOT in active_ids, so it should be zero
        assert torch.all(result[2] == 0)
        # Row 0 should have grad from batch 0 only
        assert torch.allclose(result[0], grad_output[0])

    def test_empty_active_ids(self) -> None:
        grad_output = torch.randn(3, 4)
        active_ids = torch.tensor([], dtype=torch.int64)
        batch_ids = torch.tensor([0, 1, 2], dtype=torch.int64)

        result = sparse_scatter_grad(grad_output, active_ids, batch_ids, 5, 4)
        assert torch.all(result == 0)

    def test_empty_batch(self) -> None:
        grad_output = torch.zeros(0, 4)
        active_ids = torch.tensor([0, 1], dtype=torch.int64)
        batch_ids = torch.tensor([], dtype=torch.int64)

        result = sparse_scatter_grad(grad_output, active_ids, batch_ids, 5, 4)
        assert torch.all(result == 0)


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestSparseGatherCUDA:
    @pytest.fixture
    def cuda_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.cuda()

    @pytest.fixture
    def cuda_mask(self, sparse_mask: SparseMask) -> SparseMask:
        return SparseMask(
            indices=sparse_mask.indices.cuda(),
            values=sparse_mask.values.cuda(),
            dense_shape=sparse_mask.dense_shape,
            sparsity_ratio=sparse_mask.sparsity_ratio,
        )

    @pytest.fixture
    def cuda_ids(self, batch_ids: torch.Tensor) -> torch.Tensor:
        return batch_ids.cuda()

    def test_cuda_matches_reference(
        self,
        cuda_weight: torch.Tensor,
        cuda_mask: SparseMask,
        cuda_ids: torch.Tensor,
    ) -> None:
        result = sparse_gather(cuda_weight, cuda_mask, cuda_ids)

        dense_mask = cuda_mask.to_dense()
        expected = cuda_weight[cuda_ids] * dense_mask[cuda_ids]

        assert result.device.type == "cuda"
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


@pytest.mark.cuda
class TestSparseScatterGradCUDA:
    def test_cuda_matches_cpu(self) -> None:
        torch.manual_seed(42)
        B, D, N = 4, 8, 10
        grad_output = torch.randn(B, D)
        batch_ids = torch.tensor([1, 3, 1, 5], dtype=torch.int64)
        active_ids = torch.tensor([1, 3, 5], dtype=torch.int64)

        cpu_result = sparse_scatter_grad(
            grad_output, active_ids, batch_ids, N, D
        )
        cuda_result = sparse_scatter_grad(
            grad_output.cuda(),
            active_ids.cuda(),
            batch_ids.cuda(),
            N, D,
        )

        assert cuda_result.device.type == "cuda"
        assert torch.allclose(cpu_result, cuda_result.cpu(), atol=1e-5)

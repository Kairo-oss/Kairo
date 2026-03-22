"""Sparse embedding gather/scatter ops with CUDA kernel + PyTorch fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from kairo.types import SparseMask


def cuda_ext_available() -> bool:
    """Check if kairo._C CUDA extension is loadable."""
    try:
        import kairo._C  # noqa: F401
        return True
    except ImportError:
        return False


def _coo_to_row_offsets(mask_rows: Tensor, num_rows: int) -> Tensor:
    """Convert COO row indices to CSR-style row_offsets for efficient lookup.

    Args:
        mask_rows: 1D int64 tensor of row indices (must be sorted).
        num_rows: Total number of rows in the sparse matrix.

    Returns:
        1D int64 tensor of shape (num_rows + 1,) where row_offsets[i] is the
        index into mask_rows where row i begins.
    """
    device = mask_rows.device
    row_offsets = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)

    if mask_rows.numel() == 0:
        return row_offsets

    # Count elements per row, then prefix-sum
    ones = torch.ones(mask_rows.shape[0], dtype=torch.int64, device=device)
    counts = torch.zeros(num_rows, dtype=torch.int64, device=device)
    counts.scatter_add_(0, mask_rows, ones)
    row_offsets[1:] = torch.cumsum(counts, dim=0)

    return row_offsets


def _sort_coo_by_row(
    mask: SparseMask,
) -> tuple[Tensor, Tensor]:
    """Return mask rows and cols sorted by row index."""
    rows = mask.indices[0]
    cols = mask.indices[1]
    sort_idx = torch.argsort(rows, stable=True)
    return rows[sort_idx], cols[sort_idx]


def sparse_gather(
    weight: Tensor,
    mask: SparseMask,
    ids: Tensor,
) -> Tensor:
    """Masked embedding lookup.

    For each id in ``ids``, returns the embedding row from ``weight`` with
    inactive columns (per ``mask``) zeroed out.

    Uses CUDA kernel when available, otherwise falls back to PyTorch indexing.

    Args:
        weight: Embedding weight matrix of shape (N, D).
        mask: SparseMask in COO format with dense_shape == (N, D).
        ids: 1D int64 tensor of batch IDs to look up.

    Returns:
        Tensor of shape (len(ids), D) with masked embeddings.
    """
    if ids.numel() == 0:
        return torch.zeros(
            0, weight.shape[1], dtype=weight.dtype, device=weight.device
        )

    sorted_rows, sorted_cols = _sort_coo_by_row(mask)
    num_rows = mask.dense_shape[0]

    if cuda_ext_available() and weight.is_cuda:
        import kairo._C

        row_offsets = _coo_to_row_offsets(sorted_rows, num_rows)
        return kairo._C.sparse_gather_forward(
            weight.contiguous(),
            sorted_rows.contiguous(),
            sorted_cols.contiguous(),
            ids.contiguous(),
            row_offsets.contiguous(),
        )

    # PyTorch fallback
    dense_mask = mask.to_dense()
    return weight[ids] * dense_mask[ids]


def sparse_scatter_grad(
    grad_output: Tensor,
    active_ids: Tensor,
    batch_ids: Tensor,
    num_embeddings: int,
    embedding_dim: int,
) -> Tensor:
    """Sparse gradient scatter.

    Accumulates gradients from ``grad_output`` into a (num_embeddings, embedding_dim)
    tensor, but only at rows specified by ``active_ids``.

    Uses CUDA kernel when available, otherwise falls back to PyTorch.

    Args:
        grad_output: Gradient tensor of shape (B, D).
        active_ids: 1D int64 tensor of active row IDs (sorted, unique).
        batch_ids: 1D int64 tensor of shape (B,) mapping each batch element
                   to its embedding row.
        num_embeddings: Total number of embedding rows (N).
        embedding_dim: Embedding dimension (D).

    Returns:
        Gradient weight tensor of shape (N, D).
    """
    if active_ids.numel() == 0 or batch_ids.numel() == 0:
        return torch.zeros(
            num_embeddings,
            embedding_dim,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

    if cuda_ext_available() and grad_output.is_cuda:
        import kairo._C

        return kairo._C.sparse_scatter_backward(
            grad_output.contiguous(),
            active_ids.contiguous(),
            batch_ids.contiguous(),
            num_embeddings,
            embedding_dim,
        )

    # PyTorch fallback (vectorized)
    grad_weight = torch.zeros(
        num_embeddings,
        embedding_dim,
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    active_mask = torch.isin(batch_ids, active_ids)
    active_positions = active_mask.nonzero(as_tuple=True)[0]
    if active_positions.numel() > 0:
        grad_weight.index_add_(
            0, batch_ids[active_positions], grad_output[active_positions],
        )
    return grad_weight

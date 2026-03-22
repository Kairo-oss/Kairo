"""Block-sparse matrix multiplication with CUDA acceleration + PyTorch fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from kairo.types import SparseMask

from kairo.ops.sparse_embedding import cuda_ext_available


def _dense_to_block_sparse(
    matrix: Tensor,
    block_mask: SparseMask,
    tile_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert dense matrix + block mask to tile format for kernel.

    Args:
        matrix: Dense matrix of shape (M, K).
        block_mask: SparseMask where indices refer to tile-level positions.
            dense_shape is (M // tile_size, K // tile_size).
        tile_size: Side length of each square tile.

    Returns:
        Tuple of (A_tiles, tile_row_ids, tile_col_ids) where:
        - A_tiles: float tensor of shape (num_tiles, tile_size, tile_size)
        - tile_row_ids: int32 tensor of shape (num_tiles,)
        - tile_col_ids: int32 tensor of shape (num_tiles,)
    """
    tile_rows = block_mask.indices[0]  # (nnz,)
    tile_cols = block_mask.indices[1]  # (nnz,)
    num_tiles = block_mask.nnz

    tile_row_ids = tile_rows.to(torch.int32)
    tile_col_ids = tile_cols.to(torch.int32)

    # Extract tile data from the dense matrix
    tiles = torch.zeros(
        num_tiles, tile_size, tile_size,
        dtype=matrix.dtype, device=matrix.device,
    )

    for t in range(num_tiles):
        r = tile_rows[t].item() * tile_size
        c = tile_cols[t].item() * tile_size
        r_end = min(r + tile_size, matrix.shape[0])
        c_end = min(c + tile_size, matrix.shape[1])
        tiles[t, : r_end - r, : c_end - c] = matrix[r:r_end, c:c_end]

    return tiles, tile_row_ids, tile_col_ids


def block_sparse_mm(
    A: Tensor,
    B: Tensor,
    block_mask: SparseMask,
    tile_size: int = 16,
) -> Tensor:
    """Block-sparse matrix multiply: C = blocksparse(A) @ B.

    Only tile positions indicated by ``block_mask`` are used from A.
    Auto-selects CUDA kernel vs PyTorch fallback.

    Args:
        A: Dense matrix of shape (M, K). Only tiles in block_mask are read.
        B: Dense matrix of shape (K, N).
        block_mask: SparseMask with indices at tile granularity.
        tile_size: Tile side length (default 16).

    Returns:
        Output tensor C of shape (M, N).
    """
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Inner dimensions must match: A is ({M}, {K}), B is ({K2}, {N})")

    A_tiles, tile_row_ids, tile_col_ids = _dense_to_block_sparse(
        A, block_mask, tile_size
    )
    num_tiles = A_tiles.shape[0]

    if num_tiles == 0:
        return torch.zeros(M, N, dtype=A.dtype, device=A.device)

    if cuda_ext_available() and A.is_cuda and B.is_cuda:
        import kairo._C

        return kairo._C.acc_spmm_forward(
            A_tiles.contiguous(),
            tile_row_ids.contiguous(),
            tile_col_ids.contiguous(),
            B.contiguous(),
            M, K, N, num_tiles, tile_size, tile_size,
        )

    # PyTorch fallback
    return _block_sparse_mm_fallback(
        A_tiles, tile_row_ids, tile_col_ids, B, M, K, N, tile_size
    )


def _block_sparse_mm_fallback(
    A_tiles: Tensor,
    tile_row_ids: Tensor,
    tile_col_ids: Tensor,
    B: Tensor,
    M: int,
    K: int,
    N: int,
    tile_size: int,
) -> Tensor:
    """Pure PyTorch fallback for block_sparse_mm."""
    C = torch.zeros(M, N, dtype=B.dtype, device=B.device)
    num_tiles = A_tiles.shape[0]

    for t in range(num_tiles):
        r = int(tile_row_ids[t].item()) * tile_size
        c = int(tile_col_ids[t].item()) * tile_size
        r_end = min(r + tile_size, M)
        c_end = min(c + tile_size, K)
        tm = r_end - r
        tk = c_end - c

        tile = A_tiles[t, :tm, :tk]  # (tm, tk)
        B_block = B[c:c_end, :]       # (tk, N)
        C[r:r_end, :] += tile @ B_block

    return C

"""Core frozen dataclasses used across all Kairo engines."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseMask:
    """Immutable sparse mask in COO format for embedding tables.

    Attributes:
        indices: 2D tensor of shape (2, nnz) with row and column indices of active parameters.
        values: 1D tensor of shape (nnz,) with values at active positions (typically all ones).
        dense_shape: Shape of the dense embedding table (num_embeddings, embedding_dim).
        sparsity_ratio: Fraction of parameters that are zeroed out (0.0 = dense, 1.0 = empty).
    """

    indices: torch.Tensor
    values: torch.Tensor
    dense_shape: tuple[int, int]
    sparsity_ratio: float

    def to_dense(self) -> torch.Tensor:
        """Convert sparse mask to dense boolean tensor."""
        dense = torch.zeros(self.dense_shape, dtype=self.values.dtype, device=self.values.device)
        dense[self.indices[0], self.indices[1]] = self.values
        return dense

    @property
    def nnz(self) -> int:
        """Number of non-zero (active) elements."""
        return self.indices.shape[1]

    @property
    def device(self) -> torch.device:
        """Device of the mask tensors."""
        return self.indices.device


@dataclass(frozen=True)
class NMFResult:
    """Immutable result of Non-negative Matrix Factorization.

    For an input matrix V of shape (m, n) and rank r:
        V ~ W @ H

    Attributes:
        W: Factor matrix of shape (m, r), non-negative.
        H: Factor matrix of shape (r, n), non-negative.
        reconstruction_error: Frobenius norm of (V - W @ H) at convergence.
        n_iterations: Number of iterations performed.
    """

    W: torch.Tensor
    H: torch.Tensor
    reconstruction_error: float
    n_iterations: int


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for a sparse embedding table.

    Attributes:
        num_embeddings: Size of the embedding dictionary (vocabulary size).
        embedding_dim: Dimensionality of each embedding vector.
        sparsity_ratio: Target fraction of parameters to be masked (0.0 to 1.0).
        sparse_format: Sparse storage format, one of "coo" or "csr".
    """

    num_embeddings: int
    embedding_dim: int
    sparsity_ratio: float = 0.0
    sparse_format: str = "coo"

    def __post_init__(self) -> None:
        if self.num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {self.num_embeddings}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not 0.0 <= self.sparsity_ratio <= 1.0:
            raise ValueError(
                f"sparsity_ratio must be in [0.0, 1.0], got {self.sparsity_ratio}"
            )
        if self.sparse_format not in ("coo", "csr"):
            raise ValueError(
                f"sparse_format must be 'coo' or 'csr', got '{self.sparse_format}'"
            )

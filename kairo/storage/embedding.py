"""SparseEmbeddingTable: embedding table with sparse mask support.

Applies a binary mask to the embedding weight matrix so that only
active (non-masked) parameters participate in the forward pass.
Masked positions are zeroed out, reducing effective parameter count.
"""

from __future__ import annotations

import torch
from torch import nn

from kairo.ops.sparse_embedding import cuda_ext_available, sparse_gather
from kairo.training.sparse_rec.sparse_grad import SparseEmbeddingLookup
from kairo.types import EmbeddingConfig, SparseMask


class SparseEmbeddingTable(nn.Module):
    """Embedding table that applies a sparse mask to zero out inactive parameters.

    Args:
        config: Embedding configuration specifying table dimensions and sparsity.
        mask: Optional sparse mask in COO format. If None, behaves as dense embedding.
    """

    def __init__(self, config: EmbeddingConfig, mask: SparseMask | None = None) -> None:
        super().__init__()
        self._config = config
        self._mask = mask

        self.weight = nn.Parameter(
            torch.randn(config.num_embeddings, config.embedding_dim)
        )

        # Pre-compute dense mask buffer for efficient forward pass
        if mask is not None:
            self.register_buffer("_dense_mask", mask.to_dense())
        else:
            self.register_buffer("_dense_mask", None)

    def forward(
        self, ids: torch.Tensor, active_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up embeddings for the given IDs, applying mask if present.

        Args:
            ids: Integer tensor of embedding IDs. Can be any shape.
            active_ids: Optional sorted 1D tensor of sampled embedding IDs for
                SparseRec sparse-gradient training. When provided, the backward
                pass computes gradients only for IDs in the intersection of
                ``ids`` and ``active_ids``.

        Returns:
            Tensor of shape (*ids.shape, embedding_dim) with masked positions zeroed.
        """
        dense_mask: torch.Tensor | None = getattr(self, "_dense_mask", None)

        # SparseRec path: sparse gradients via custom autograd Function
        if active_ids is not None:
            return SparseEmbeddingLookup.apply(
                self.weight, ids, active_ids, dense_mask,
            )

        # CUDA kernel path: accelerated masked gather
        if (
            self._mask is not None
            and self.weight.is_cuda
            and cuda_ext_available()
        ):
            return sparse_gather(self.weight, self._mask, ids)

        # Default PyTorch path (backward-compatible)
        embeddings = self.weight[ids]
        if dense_mask is not None:
            mask_rows = dense_mask[ids]
            embeddings = embeddings * mask_rows

        return embeddings

    def with_mask(self, new_mask: SparseMask) -> SparseEmbeddingTable:
        """Create a new SparseEmbeddingTable with an updated mask (immutable pattern).

        The new table shares the same weight data but applies a different mask.

        Args:
            new_mask: New sparse mask to apply.

        Returns:
            A new SparseEmbeddingTable instance with the updated mask.
        """
        new_config = EmbeddingConfig(
            num_embeddings=self._config.num_embeddings,
            embedding_dim=self._config.embedding_dim,
            sparsity_ratio=new_mask.sparsity_ratio,
            sparse_format=self._config.sparse_format,
        )
        new_table = SparseEmbeddingTable(new_config, mask=new_mask)
        # Copy weight data to the new instance
        new_table.weight = nn.Parameter(self.weight.data.clone())
        return new_table

    @property
    def num_embeddings(self) -> int:
        """Size of the embedding dictionary."""
        return self._config.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each embedding vector."""
        return self._config.embedding_dim

    @property
    def mask(self) -> SparseMask | None:
        """Current sparse mask, or None if dense."""
        return self._mask

    @property
    def active_parameter_count(self) -> int:
        """Number of non-masked (active) parameters."""
        if self._mask is None:
            return self._config.num_embeddings * self._config.embedding_dim
        return self._mask.nnz

    @property
    def compression_ratio(self) -> float:
        """Fraction of parameters that are masked (0.0 = no compression)."""
        if self._mask is None:
            return 0.0
        total = self._config.num_embeddings * self._config.embedding_dim
        return 1.0 - self.active_parameter_count / total

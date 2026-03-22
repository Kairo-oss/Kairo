"""Custom autograd Function for sparse backward pass in SparseRec.

The forward pass performs standard embedding lookup (optionally masked).
The backward pass restricts gradient computation to the intersection of
batch IDs and the active ID set, ensuring constant-sparsity gradients.
"""

from __future__ import annotations

import torch


class SparseEmbeddingLookup(torch.autograd.Function):
    """Embedding lookup with sparse backward restricted to active IDs.

    Forward: weight[ids] with optional dense_mask applied element-wise.
    Backward: gradients only for IDs in (batch_ids intersect active_ids).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        weight: torch.Tensor,
        ids: torch.Tensor,
        active_ids: torch.Tensor,
        dense_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Look up embeddings for requested IDs.

        Args:
            ctx: Autograd context for saving tensors.
            weight: Full embedding weight matrix (num_embeddings, embedding_dim).
            ids: 1D tensor of embedding IDs to look up.
            active_ids: 1D tensor of IDs allowed to receive gradients.
            dense_mask: Optional (num_embeddings, embedding_dim) mask. None = no mask.

        Returns:
            Looked-up embeddings of shape (len(ids), embedding_dim).
        """
        ctx.num_embeddings = weight.shape[0]  # type: ignore[attr-defined]
        ctx.embedding_dim = weight.shape[1]  # type: ignore[attr-defined]
        ctx.has_mask = dense_mask is not None  # type: ignore[attr-defined]

        # Single save_for_backward call with all needed tensors
        if dense_mask is not None:
            ctx.save_for_backward(ids, active_ids, dense_mask)
        else:
            ctx.save_for_backward(ids, active_ids)

        embeddings = weight[ids]
        if dense_mask is not None:
            embeddings = embeddings * dense_mask[ids]
        return embeddings

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None, None]:
        """Compute sparse gradients restricted to active IDs.

        Only IDs present in both the batch and the active set receive
        gradient updates. All other rows in grad_weight remain zero.

        Returns:
            Tuple of (grad_weight, None, None, None).
        """
        saved = ctx.saved_tensors
        if ctx.has_mask:  # type: ignore[attr-defined]
            ids, active_ids, dense_mask = saved
        else:
            ids, active_ids = saved
            dense_mask = None

        num_embeddings = ctx.num_embeddings  # type: ignore[attr-defined]
        embedding_dim = ctx.embedding_dim  # type: ignore[attr-defined]

        grad_weight = torch.zeros(
            num_embeddings, embedding_dim,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # Apply mask to grad_output if mask was used in forward
        effective_grad = grad_output
        if dense_mask is not None:
            effective_grad = grad_output * dense_mask[ids]

        # Find batch positions whose IDs are in the active set (vectorized)
        active_mask = torch.isin(ids, active_ids)
        active_positions = active_mask.nonzero(as_tuple=True)[0]

        # Scatter gradients for active positions in one vectorized operation
        if active_positions.numel() > 0:
            grad_weight.index_add_(
                0, ids[active_positions], effective_grad[active_positions],
            )

        return grad_weight, None, None, None

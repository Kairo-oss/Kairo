"""Sparse mask generation from NMF factors.

Computes importance scores from W @ H and thresholds them to produce
a binary sparse mask in COO format. Higher importance scores indicate
parameters that should remain active (non-masked).
"""

from __future__ import annotations

import torch

from kairo.types import NMFResult, SparseMask


def _compute_importance_scores(
    nmf_result: NMFResult,
    num_embeddings: int,
    embedding_dim: int,
) -> torch.Tensor:
    """Compute per-parameter importance from NMF factors.

    Reconstructs W @ H and reshapes/adapts to the target embedding shape.
    Returns a (num_embeddings, embedding_dim) importance matrix.
    """
    reconstruction = nmf_result.W @ nmf_result.H  # (m, n)

    # Adapt reconstruction to target embedding dimensions via interpolation
    # Normalize reconstruction to [0, 1] range for thresholding
    scores = reconstruction.unsqueeze(0).unsqueeze(0)  # (1, 1, m, n)
    scores = torch.nn.functional.interpolate(
        scores,
        size=(num_embeddings, embedding_dim),
        mode="bilinear",
        align_corners=False,
    )
    scores = scores.squeeze(0).squeeze(0)  # (num_embeddings, embedding_dim)

    # Normalize to [0, 1]
    s_min = scores.min()
    s_max = scores.max()
    scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else torch.ones_like(scores)

    return scores


def generate_sparse_mask(
    nmf_result: NMFResult,
    num_embeddings: int,
    embedding_dim: int,
    sparsity_ratio: float,
) -> SparseMask:
    """Generate a sparse mask from NMF factors by thresholding importance scores.

    Parameters with importance above the sparsity threshold are kept active.
    The mask is returned in COO format.

    Args:
        nmf_result: Result of NMF decomposition containing W and H factors.
        num_embeddings: Number of embedding rows in the target table.
        embedding_dim: Dimensionality of each embedding vector.
        sparsity_ratio: Fraction of parameters to mask (0.0 = keep all, 1.0 = mask all).

    Returns:
        Frozen SparseMask with active parameter indices in COO format.
    """
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError(f"sparsity_ratio must be in [0.0, 1.0], got {sparsity_ratio}")

    total_params = num_embeddings * embedding_dim
    num_active = round(total_params * (1.0 - sparsity_ratio))

    # Compute importance and select top-k active positions
    importance = _compute_importance_scores(nmf_result, num_embeddings, embedding_dim)

    if num_active >= total_params:
        # Keep all parameters
        rows = torch.arange(num_embeddings).repeat_interleave(embedding_dim)
        cols = torch.arange(embedding_dim).repeat(num_embeddings)
    else:
        # Flatten, find top-k, convert back to 2D indices
        flat_importance = importance.flatten()
        _, top_indices = torch.topk(flat_importance, k=num_active)
        rows = top_indices // embedding_dim
        cols = top_indices % embedding_dim

    indices = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.shape[1], dtype=importance.dtype, device=importance.device)

    actual_sparsity = 1.0 - indices.shape[1] / total_params

    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=(num_embeddings, embedding_dim),
        sparsity_ratio=actual_sparsity,
    )

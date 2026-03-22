"""Mask topology management for SparseRec: prune, grow, and update.

Implements cumulative gradient-based regrowth following the SparseRec
algorithm. Low-magnitude active weights are pruned, and inactive
positions with the highest cumulative gradient magnitude are activated.
The net active count stays roughly constant across updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kairo.config import SparseRecConfig
from kairo.types import MaskUpdateResult, SparseMask


@dataclass(frozen=True)
class GradientAccumulator:
    """Tracks cumulative gradient magnitudes across training steps.

    Immutable: with_update() returns a new instance with updated state.

    Attributes:
        cumulative_magnitudes: Tensor of shape (num_embeddings, embedding_dim)
            holding the running sum of absolute gradient values.
        num_steps: Number of gradient updates accumulated so far.
    """

    cumulative_magnitudes: torch.Tensor
    num_steps: int

    def with_update(self, new_gradients: torch.Tensor) -> GradientAccumulator:
        """Return a new accumulator with magnitudes updated by new gradients.

        Args:
            new_gradients: Gradient tensor of same shape as cumulative_magnitudes.

        Returns:
            New GradientAccumulator with updated magnitudes and step count.
        """
        updated_magnitudes = self.cumulative_magnitudes + new_gradients.abs()
        return GradientAccumulator(
            cumulative_magnitudes=updated_magnitudes,
            num_steps=self.num_steps + 1,
        )


def prune_mask(
    mask: SparseMask,
    weights: torch.Tensor,
    prune_fraction: float,
    criterion: str = "magnitude",
) -> SparseMask:
    """Remove lowest-score active weights from the mask.

    Args:
        mask: Current sparse mask in COO format.
        weights: Full weight matrix (num_embeddings, embedding_dim).
        prune_fraction: Fraction of active positions to prune (0.0 to 1.0).
        criterion: Scoring criterion, currently only "magnitude" supported.

    Returns:
        New SparseMask with pruned positions removed.
    """
    if criterion != "magnitude":
        raise NotImplementedError(
            f"Prune criterion '{criterion}' not yet implemented. Use 'magnitude'."
        )

    if prune_fraction <= 0.0 or mask.nnz == 0:
        return SparseMask(
            indices=mask.indices.clone(),
            values=mask.values.clone(),
            dense_shape=mask.dense_shape,
            sparsity_ratio=mask.sparsity_ratio,
        )

    num_to_prune = int(mask.nnz * prune_fraction)
    if num_to_prune == 0:
        return SparseMask(
            indices=mask.indices.clone(),
            values=mask.values.clone(),
            dense_shape=mask.dense_shape,
            sparsity_ratio=mask.sparsity_ratio,
        )

    # Score active positions by weight magnitude
    rows, cols = mask.indices[0], mask.indices[1]
    scores = weights[rows, cols].abs()

    # Keep positions with highest scores (prune lowest)
    num_to_keep = mask.nnz - num_to_prune
    _, keep_indices = scores.topk(num_to_keep, largest=True)
    keep_indices = keep_indices.sort().values

    new_rows = rows[keep_indices]
    new_cols = cols[keep_indices]
    new_indices = torch.stack([new_rows, new_cols], dim=0)
    new_values = torch.ones(num_to_keep, dtype=mask.values.dtype, device=mask.device)

    total = mask.dense_shape[0] * mask.dense_shape[1]
    new_sparsity = 1.0 - num_to_keep / total if total > 0 else 0.0

    return SparseMask(
        indices=new_indices,
        values=new_values,
        dense_shape=mask.dense_shape,
        sparsity_ratio=new_sparsity,
    )


def grow_mask(
    mask: SparseMask,
    accumulator: GradientAccumulator,
    grow_count: int,
) -> SparseMask:
    """Activate inactive positions with highest cumulative gradient magnitude.

    Args:
        mask: Current sparse mask in COO format.
        accumulator: Gradient accumulator tracking cumulative magnitudes.
        grow_count: Number of inactive positions to activate.

    Returns:
        New SparseMask with grown positions added.
    """
    if grow_count <= 0:
        return SparseMask(
            indices=mask.indices.clone(),
            values=mask.values.clone(),
            dense_shape=mask.dense_shape,
            sparsity_ratio=mask.sparsity_ratio,
        )

    # Build a dense mask of currently active positions
    dense_active = torch.zeros(
        mask.dense_shape,
        dtype=torch.bool,
        device=mask.device,
    )
    if mask.nnz > 0:
        dense_active[mask.indices[0], mask.indices[1]] = True

    # Score inactive positions by cumulative gradient magnitude
    inactive_mask = ~dense_active
    flat_scores = accumulator.cumulative_magnitudes.clone().view(-1)
    active_flat = dense_active.view(-1)
    flat_scores[active_flat] = -1.0
    flat_inactive = inactive_mask.view(-1)

    num_inactive = flat_inactive.sum().item()
    actual_grow = min(grow_count, int(num_inactive))

    if actual_grow == 0:
        return SparseMask(
            indices=mask.indices.clone(),
            values=mask.values.clone(),
            dense_shape=mask.dense_shape,
            sparsity_ratio=mask.sparsity_ratio,
        )

    _, top_flat_indices = flat_scores.topk(actual_grow, largest=True)
    num_cols = mask.dense_shape[1]
    new_rows = top_flat_indices // num_cols
    new_cols = top_flat_indices % num_cols

    # Combine existing and new indices
    combined_rows = torch.cat([mask.indices[0], new_rows])
    combined_cols = torch.cat([mask.indices[1], new_cols])
    combined_indices = torch.stack([combined_rows, combined_cols], dim=0)
    combined_values = torch.ones(
        combined_indices.shape[1],
        dtype=mask.values.dtype,
        device=mask.device,
    )

    total = mask.dense_shape[0] * mask.dense_shape[1]
    new_nnz = combined_indices.shape[1]
    new_sparsity = 1.0 - new_nnz / total if total > 0 else 0.0

    return SparseMask(
        indices=combined_indices,
        values=combined_values,
        dense_shape=mask.dense_shape,
        sparsity_ratio=new_sparsity,
    )


def update_topology(
    mask: SparseMask,
    weights: torch.Tensor,
    accumulator: GradientAccumulator,
    config: SparseRecConfig,
) -> MaskUpdateResult:
    """Combined prune+grow cycle keeping net active count constant.

    Prunes the lowest-scoring fraction of active weights, then grows
    the same number from inactive positions with highest cumulative
    gradient magnitude.

    Args:
        mask: Current sparse mask.
        weights: Full weight matrix.
        accumulator: Gradient accumulator with cumulative magnitudes.
        config: SparseRec configuration (regrowth_fraction, prune_criterion).

    Returns:
        Frozen MaskUpdateResult with new mask and update statistics.
    """
    # Step 1: Prune
    pruned_mask = prune_mask(
        mask, weights, config.regrowth_fraction, config.prune_criterion,
    )
    actual_pruned = mask.nnz - pruned_mask.nnz

    # Step 2: Grow the same number back
    grown_mask = grow_mask(pruned_mask, accumulator, actual_pruned)
    actual_grown = grown_mask.nnz - pruned_mask.nnz

    # Compute gradient norm for diagnostics
    grad_norm = float(
        accumulator.cumulative_magnitudes.norm(p=2).item()
    )

    return MaskUpdateResult(
        new_mask=grown_mask,
        num_pruned=actual_pruned,
        num_grown=actual_grown,
        cumulative_gradient_norm=grad_norm,
    )

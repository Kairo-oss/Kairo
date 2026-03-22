"""Multinomial sampling for SparseRec selective gradient computation.

Converts embedding access frequencies into sampling probabilities and
selects a subset of active embedding IDs for each training step.
Only the sampled IDs participate in gradient computation, yielding
constant sparsity in both forward and backward passes.
"""

from __future__ import annotations

import torch


def compute_sampling_weights(
    access_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert access frequencies to normalized sampling probabilities.

    Higher access count leads to higher sampling probability. Temperature
    controls exploration vs exploitation: higher temperature produces a
    more uniform distribution, lower temperature concentrates probability
    on frequently-accessed IDs.

    Args:
        access_counts: 1D tensor of non-negative access frequencies.
        temperature: Softmax temperature (> 0). Higher = more uniform.

    Returns:
        Normalized probability vector summing to 1.0, same shape as input.
    """
    if access_counts.numel() == 0:
        return access_counts.clone()

    # All-zero fallback: uniform distribution
    total = access_counts.sum()
    if total.item() == 0.0:
        n = access_counts.shape[0]
        return torch.full(
            (n,), 1.0 / n, dtype=access_counts.dtype, device=access_counts.device,
        )

    # Softmax with temperature for smooth probability assignment
    log_counts = torch.log(access_counts.clamp(min=1e-10))
    scaled = log_counts / temperature
    weights = torch.softmax(scaled, dim=0)
    return weights


def sample_active_ids(
    sampling_weights: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample embedding IDs via multinomial sampling without replacement.

    Args:
        sampling_weights: Normalized probability vector (must sum to ~1.0).
        num_samples: Number of IDs to sample. Must be <= len(sampling_weights).
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Sorted 1D tensor of shape (num_samples,) with sampled ID indices.

    Raises:
        ValueError: If num_samples exceeds the number of available IDs.
    """
    n = sampling_weights.shape[0]
    if num_samples > n:
        raise ValueError(
            f"num_samples ({num_samples}) exceeds available IDs ({n})"
        )

    if num_samples == n:
        return torch.arange(n, device=sampling_weights.device)

    sampled = torch.multinomial(
        sampling_weights,
        num_samples=num_samples,
        replacement=False,
        generator=generator,
    )
    return sampled.sort().values

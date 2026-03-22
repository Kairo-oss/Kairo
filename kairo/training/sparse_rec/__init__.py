"""SparseRec: selective gradient computation via multinomial sampling."""

from __future__ import annotations

from kairo.training.sparse_rec.sampler import compute_sampling_weights, sample_active_ids
from kairo.training.sparse_rec.sparse_grad import SparseEmbeddingLookup
from kairo.training.sparse_rec.topology import (
    GradientAccumulator,
    grow_mask,
    prune_mask,
    update_topology,
)

__all__ = [
    "compute_sampling_weights",
    "sample_active_ids",
    "SparseEmbeddingLookup",
    "GradientAccumulator",
    "grow_mask",
    "prune_mask",
    "update_topology",
]

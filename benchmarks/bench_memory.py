"""Benchmarks: peak GPU memory for sparse vs dense forward+backward."""

from __future__ import annotations

import pytest
import torch

from kairo.config import NMFConfig
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.training.sparse_rec.sampler import compute_sampling_weights, sample_active_ids
from kairo.types import EmbeddingConfig


def _measure_memory_mb(fn) -> float:
    """Measure peak GPU memory delta in MB for a function call."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - before) / (1024 * 1024)


@pytest.mark.cuda
@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.95])
def test_bench_memory_sparse_vs_dense(sparsity: float) -> None:
    device = torch.device("cuda")
    num_e, emb_dim, batch_size = 10000, 128, 512
    torch.manual_seed(42)

    interaction = torch.rand(num_e, emb_dim) + 0.01
    nmf_cfg = NMFConfig(rank=8, max_iter=30, seed=42)
    nmf_result = nmf_decompose(interaction, nmf_cfg)
    mask = generate_sparse_mask(nmf_result, num_e, emb_dim, sparsity)
    config = EmbeddingConfig(num_e, emb_dim, sparsity_ratio=sparsity)
    table = SparseEmbeddingTable(config, mask=mask).to(device)
    ids = torch.randint(0, num_e, (batch_size,), device=device)

    access_counts = torch.ones(num_e, device=device)
    weights = compute_sampling_weights(access_counts)
    active_ids = sample_active_ids(weights, max(1, num_e // 10))

    def sparse_fwd_bwd():
        table.zero_grad()
        out = table(ids, active_ids=active_ids)
        out.sum().backward()

    def dense_fwd_bwd():
        table.zero_grad()
        out = table(ids)
        out.sum().backward()

    sparse_mb = _measure_memory_mb(sparse_fwd_bwd)
    dense_mb = _measure_memory_mb(dense_fwd_bwd)

    print(
        f"  sparsity={sparsity}: sparse={sparse_mb:.1f}MB, "
        f"dense={dense_mb:.1f}MB, savings={dense_mb - sparse_mb:.1f}MB"
    )

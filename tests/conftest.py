"""Shared test fixtures for Kairo test suite."""

from __future__ import annotations

import pytest
import torch

from kairo.config import NMFConfig
from kairo.types import EmbeddingConfig


@pytest.fixture(params=["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrized device fixture. CUDA added when GPU tests are enabled."""
    return torch.device(request.param)


@pytest.fixture
def seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(seed: int) -> torch.Generator:
    """Seeded random number generator."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@pytest.fixture
def small_interaction_matrix(rng: torch.Generator) -> torch.Tensor:
    """Synthetic user-item interaction matrix (100 users x 50 items).

    Constructed as a low-rank matrix to ensure NMF can recover it well.
    """
    rank = 5
    W_true = torch.abs(torch.randn(100, rank, generator=rng))
    H_true = torch.abs(torch.randn(rank, 50, generator=rng))
    return W_true @ H_true


@pytest.fixture
def default_nmf_config() -> NMFConfig:
    """Default NMF configuration for tests."""
    return NMFConfig(rank=5, max_iter=100, tol=1e-4, seed=42)


@pytest.fixture
def default_embedding_config() -> EmbeddingConfig:
    """Default embedding configuration for tests."""
    return EmbeddingConfig(
        num_embeddings=100,
        embedding_dim=32,
        sparsity_ratio=0.5,
        sparse_format="coo",
    )

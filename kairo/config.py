"""Frozen configuration dataclasses for Kairo algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NMFConfig:
    """Configuration for Non-negative Matrix Factorization.

    Attributes:
        rank: Number of components (latent dimensions) for the factorization.
        max_iter: Maximum number of multiplicative update iterations.
        tol: Convergence tolerance on relative reconstruction error change.
        seed: Random seed for reproducibility. None for non-deterministic.
    """

    rank: int = 64
    max_iter: int = 200
    tol: float = 1e-4
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")

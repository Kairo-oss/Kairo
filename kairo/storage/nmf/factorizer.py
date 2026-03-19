"""Non-negative Matrix Factorization using multiplicative update rules.

Decomposes a non-negative matrix V ~ W @ H where W >= 0 and H >= 0.
Used to initialize sparse masks for embedding tables by identifying
importance structure in the interaction data.
"""

from __future__ import annotations

import torch

from kairo.config import NMFConfig
from kairo.types import NMFResult

_EPSILON = 1e-10


def _compute_reconstruction_error(V: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> float:
    """Frobenius norm of (V - W @ H)."""
    return torch.norm(V - W @ H, p="fro").item()


def _multiplicative_update_w(
    V: torch.Tensor, W: torch.Tensor, H: torch.Tensor
) -> torch.Tensor:
    """W_new = W * (V @ H^T) / (W @ H @ H^T + eps)."""
    numerator = V @ H.T
    denominator = W @ (H @ H.T) + _EPSILON
    return W * (numerator / denominator)


def _multiplicative_update_h(
    V: torch.Tensor, W: torch.Tensor, H: torch.Tensor
) -> torch.Tensor:
    """H_new = H * (W^T @ V) / (W^T @ W @ H + eps)."""
    numerator = W.T @ V
    denominator = (W.T @ W) @ H + _EPSILON
    return H * (numerator / denominator)


def nmf_decompose(interaction_matrix: torch.Tensor, config: NMFConfig) -> NMFResult:
    """Decompose a non-negative matrix V into W @ H using multiplicative updates.

    Args:
        interaction_matrix: Non-negative matrix of shape (m, n) to decompose.
        config: NMF configuration with rank, max_iter, tol, and seed.

    Returns:
        Frozen NMFResult with factor matrices W (m, rank) and H (rank, n),
        final reconstruction error, and number of iterations performed.

    Raises:
        ValueError: If interaction_matrix contains negative values.
    """
    V = interaction_matrix
    if (V < 0).any():
        raise ValueError("interaction_matrix must be non-negative")

    m, n = V.shape
    device = V.device
    dtype = V.dtype

    # Seeded initialization
    if config.seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(config.seed)
    else:
        gen = None

    # Random non-negative initialization
    W = torch.abs(torch.randn(m, config.rank, generator=gen, device=device, dtype=dtype)) + _EPSILON
    H = torch.abs(torch.randn(config.rank, n, generator=gen, device=device, dtype=dtype)) + _EPSILON

    prev_error = _compute_reconstruction_error(V, W, H)
    current_error = prev_error
    n_iterations = 0

    for i in range(1, config.max_iter + 1):
        W = _multiplicative_update_w(V, W, H)
        H = _multiplicative_update_h(V, W, H)

        current_error = _compute_reconstruction_error(V, W, H)
        n_iterations = i

        # Check convergence via relative change
        relative_change = abs(prev_error - current_error) / (prev_error + _EPSILON)
        if relative_change < config.tol:
            break

        prev_error = current_error

    return NMFResult(
        W=W,
        H=H,
        reconstruction_error=current_error,
        n_iterations=n_iterations,
    )

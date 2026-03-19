"""Tests for NMF decomposition (kairo.storage.nmf.factorizer)."""

from __future__ import annotations

import torch

from kairo.config import NMFConfig
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.types import NMFResult


class TestNMFDecompose:
    """Test suite for nmf_decompose function."""

    def test_returns_nmf_result(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        result = nmf_decompose(small_interaction_matrix, default_nmf_config)
        assert isinstance(result, NMFResult)

    def test_output_shapes(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        """W should be (m, rank) and H should be (rank, n)."""
        m, n = small_interaction_matrix.shape
        result = nmf_decompose(small_interaction_matrix, default_nmf_config)

        assert result.W.shape == (m, default_nmf_config.rank)
        assert result.H.shape == (default_nmf_config.rank, n)

    def test_non_negativity(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        """Both W and H must be non-negative."""
        result = nmf_decompose(small_interaction_matrix, default_nmf_config)

        assert (result.W >= 0).all(), "W contains negative values"
        assert (result.H >= 0).all(), "H contains negative values"

    def test_reconstruction_error_is_finite(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        result = nmf_decompose(small_interaction_matrix, default_nmf_config)
        assert result.reconstruction_error >= 0
        assert torch.isfinite(torch.tensor(result.reconstruction_error))

    def test_convergence_on_low_rank_matrix(self, rng: torch.Generator) -> None:
        """NMF should achieve low error on a matrix that is exactly low-rank."""
        rank = 3
        W_true = torch.abs(torch.randn(50, rank, generator=rng))
        H_true = torch.abs(torch.randn(rank, 30, generator=rng))
        V = W_true @ H_true

        config = NMFConfig(rank=rank, max_iter=300, tol=1e-6, seed=42)
        result = nmf_decompose(V, config)

        # Relative reconstruction error should be small
        V_approx = result.W @ result.H
        relative_error = torch.norm(V - V_approx) / torch.norm(V)
        assert relative_error < 0.05, f"Relative error too high: {relative_error:.4f}"

    def test_reconstruction_error_decreases(self, rng: torch.Generator) -> None:
        """Running more iterations should give equal or lower error."""
        V = torch.abs(torch.randn(40, 20, generator=rng)) + 0.1

        config_few = NMFConfig(rank=5, max_iter=10, tol=1e-10, seed=42)
        config_many = NMFConfig(rank=5, max_iter=200, tol=1e-10, seed=42)

        result_few = nmf_decompose(V, config_few)
        result_many = nmf_decompose(V, config_many)

        assert result_many.reconstruction_error <= result_few.reconstruction_error

    def test_n_iterations_within_bounds(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        result = nmf_decompose(small_interaction_matrix, default_nmf_config)
        assert 1 <= result.n_iterations <= default_nmf_config.max_iter

    def test_immutability_of_input(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        """Input matrix must not be modified."""
        original = small_interaction_matrix.clone()
        nmf_decompose(small_interaction_matrix, default_nmf_config)
        assert torch.equal(small_interaction_matrix, original)

    def test_reproducibility_with_seed(self, small_interaction_matrix: torch.Tensor) -> None:
        """Same seed should produce identical results."""
        config = NMFConfig(rank=5, max_iter=50, seed=123)

        result1 = nmf_decompose(small_interaction_matrix, config)
        result2 = nmf_decompose(small_interaction_matrix, config)

        assert torch.allclose(result1.W, result2.W)
        assert torch.allclose(result1.H, result2.H)
        assert result1.reconstruction_error == result2.reconstruction_error

    def test_result_is_frozen(
        self, small_interaction_matrix: torch.Tensor, default_nmf_config: NMFConfig
    ) -> None:
        """NMFResult should be immutable (frozen dataclass)."""
        import dataclasses

        result = nmf_decompose(small_interaction_matrix, default_nmf_config)
        assert dataclasses.is_dataclass(result)

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            result.reconstruction_error = 0.0  # type: ignore[misc]

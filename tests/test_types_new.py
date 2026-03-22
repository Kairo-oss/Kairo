"""Tests for new frozen types: SparseGradientResult and MaskUpdateResult."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from kairo.types import MaskUpdateResult, SparseMask, SparseGradientResult


class TestSparseGradientResult:
    """Tests for SparseGradientResult frozen dataclass."""

    def test_construction(self) -> None:
        indices = torch.tensor([0, 5, 10])
        gradients = torch.randn(3, 32)
        result = SparseGradientResult(indices=indices, gradients=gradients, batch_size=16)
        assert result.batch_size == 16
        assert result.indices.shape == (3,)
        assert result.gradients.shape == (3, 32)

    def test_frozen(self) -> None:
        result = SparseGradientResult(
            indices=torch.tensor([0]), gradients=torch.randn(1, 8), batch_size=4,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.batch_size = 8  # type: ignore[misc]


class TestMaskUpdateResult:
    """Tests for MaskUpdateResult frozen dataclass."""

    def test_construction(self) -> None:
        mask = SparseMask(
            indices=torch.tensor([[0, 1], [0, 1]]),
            values=torch.ones(2),
            dense_shape=(4, 4),
            sparsity_ratio=0.875,
        )
        result = MaskUpdateResult(
            new_mask=mask, num_pruned=3, num_grown=3, cumulative_gradient_norm=1.5,
        )
        assert result.num_pruned == 3
        assert result.num_grown == 3
        assert result.cumulative_gradient_norm == 1.5
        assert result.new_mask.nnz == 2

    def test_frozen(self) -> None:
        mask = SparseMask(
            indices=torch.tensor([[0], [0]]),
            values=torch.ones(1),
            dense_shape=(2, 2),
            sparsity_ratio=0.75,
        )
        result = MaskUpdateResult(
            new_mask=mask, num_pruned=1, num_grown=1, cumulative_gradient_norm=0.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.num_pruned = 0  # type: ignore[misc]

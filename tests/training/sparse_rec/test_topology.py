"""Tests for SparseRec mask topology updates (prune, grow, update)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from kairo.config import SparseRecConfig
from kairo.types import MaskUpdateResult, SparseMask
from kairo.training.sparse_rec.topology import (
    GradientAccumulator,
    grow_mask,
    prune_mask,
    update_topology,
)


def _make_mask(
    active_positions: list[tuple[int, int]],
    dense_shape: tuple[int, int],
) -> SparseMask:
    """Helper to create a SparseMask from a list of (row, col) positions."""
    if not active_positions:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0)
    else:
        rows = [p[0] for p in active_positions]
        cols = [p[1] for p in active_positions]
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(active_positions))
    total = dense_shape[0] * dense_shape[1]
    nnz = len(active_positions)
    sparsity = 1.0 - nnz / total if total > 0 else 0.0
    return SparseMask(
        indices=indices,
        values=values,
        dense_shape=dense_shape,
        sparsity_ratio=sparsity,
    )


class TestGradientAccumulator:
    """Tests for GradientAccumulator immutability and accumulation."""

    def test_with_update_returns_new_instance(self) -> None:
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.zeros(4, 3),
            num_steps=0,
        )
        new_acc = acc.with_update(torch.ones(4, 3))
        assert new_acc is not acc
        assert new_acc.num_steps == 1

    def test_original_not_mutated(self) -> None:
        original_mags = torch.zeros(4, 3)
        acc = GradientAccumulator(
            cumulative_magnitudes=original_mags.clone(),
            num_steps=0,
        )
        acc.with_update(torch.ones(4, 3))
        assert acc.num_steps == 0
        assert torch.equal(acc.cumulative_magnitudes, original_mags)

    def test_accumulates_magnitudes(self) -> None:
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.zeros(3, 2),
            num_steps=0,
        )
        grad = torch.tensor([[1.0, -2.0], [3.0, 0.0], [-1.0, 1.0]])
        new_acc = acc.with_update(grad)
        expected = grad.abs()
        assert torch.allclose(new_acc.cumulative_magnitudes, expected)

    def test_frozen(self) -> None:
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.zeros(2, 2),
            num_steps=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            acc.num_steps = 5  # type: ignore[misc]


class TestPruneMask:
    """Tests for prune_mask."""

    def test_prune_reduces_nnz(self) -> None:
        mask = _make_mask(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            dense_shape=(4, 4),
        )
        weights = torch.tensor([
            [10.0, 1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        pruned = prune_mask(mask, weights, prune_fraction=0.5)
        assert pruned.nnz < mask.nnz

    def test_prune_removes_lowest_magnitude(self) -> None:
        mask = _make_mask(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            dense_shape=(2, 2),
        )
        weights = torch.tensor([
            [10.0, 1.0],
            [5.0, 0.1],
        ])
        pruned = prune_mask(mask, weights, prune_fraction=0.5)
        # The 2 lowest magnitude: (0,1)=1.0 and (1,1)=0.1
        pruned_dense = pruned.to_dense()
        assert pruned_dense[0, 0] == 1.0  # kept (high magnitude)
        assert pruned_dense[1, 0] == 1.0  # kept (high magnitude)

    def test_prune_returns_new_mask(self) -> None:
        mask = _make_mask([(0, 0), (0, 1)], dense_shape=(2, 2))
        weights = torch.tensor([[5.0, 1.0], [0.0, 0.0]])
        pruned = prune_mask(mask, weights, prune_fraction=0.5)
        assert pruned is not mask

    def test_prune_fraction_zero_no_change(self) -> None:
        mask = _make_mask([(0, 0), (0, 1)], dense_shape=(2, 2))
        weights = torch.tensor([[5.0, 1.0], [0.0, 0.0]])
        pruned = prune_mask(mask, weights, prune_fraction=0.0)
        assert pruned.nnz == mask.nnz

    def test_prune_does_not_mutate_input_mask(self) -> None:
        mask = _make_mask([(0, 0), (0, 1)], dense_shape=(2, 2))
        original_nnz = mask.nnz
        weights = torch.tensor([[5.0, 1.0], [0.0, 0.0]])
        prune_mask(mask, weights, prune_fraction=0.5)
        assert mask.nnz == original_nnz


class TestGrowMask:
    """Tests for grow_mask."""

    def test_grow_increases_nnz(self) -> None:
        mask = _make_mask([(0, 0)], dense_shape=(2, 2))
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.tensor([
                [0.0, 5.0],
                [3.0, 1.0],
            ]),
            num_steps=1,
        )
        grown = grow_mask(mask, acc, grow_count=2)
        assert grown.nnz == mask.nnz + 2

    def test_grow_activates_highest_gradient(self) -> None:
        mask = _make_mask([(0, 0)], dense_shape=(2, 2))
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.tensor([
                [0.0, 10.0],  # (0,1) has highest grad among inactive
                [3.0, 1.0],
            ]),
            num_steps=1,
        )
        grown = grow_mask(mask, acc, grow_count=1)
        grown_dense = grown.to_dense()
        assert grown_dense[0, 1] == 1.0  # highest gradient activated

    def test_grow_returns_new_mask(self) -> None:
        mask = _make_mask([(0, 0)], dense_shape=(2, 2))
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.ones(2, 2),
            num_steps=1,
        )
        grown = grow_mask(mask, acc, grow_count=1)
        assert grown is not mask

    def test_grow_zero_count_no_change(self) -> None:
        mask = _make_mask([(0, 0)], dense_shape=(2, 2))
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.ones(2, 2),
            num_steps=1,
        )
        grown = grow_mask(mask, acc, grow_count=0)
        assert grown.nnz == mask.nnz

    def test_grow_does_not_mutate_input(self) -> None:
        mask = _make_mask([(0, 0)], dense_shape=(2, 2))
        original_nnz = mask.nnz
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.ones(2, 2),
            num_steps=1,
        )
        grow_mask(mask, acc, grow_count=1)
        assert mask.nnz == original_nnz


class TestUpdateTopology:
    """Tests for update_topology (combined prune+grow)."""

    def test_preserves_nnz(self) -> None:
        """Prune N + grow N should keep nnz roughly constant."""
        mask = _make_mask(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            dense_shape=(3, 3),
        )
        weights = torch.tensor([
            [10.0, 0.1, 0.0],
            [5.0, 0.2, 0.0],
            [0.0, 0.0, 0.0],
        ])
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.tensor([
                [0.0, 0.0, 8.0],
                [0.0, 0.0, 6.0],
                [9.0, 7.0, 5.0],
            ]),
            num_steps=10,
        )
        config = SparseRecConfig(
            sample_ratio=0.1,
            regrowth_interval=100,
            regrowth_fraction=0.5,
            prune_criterion="magnitude",
        )
        result = update_topology(mask, weights, acc, config)
        assert isinstance(result, MaskUpdateResult)
        # num_pruned == num_grown so nnz stays same
        assert result.num_pruned == result.num_grown
        assert result.new_mask.nnz == mask.nnz

    def test_returns_frozen_result(self) -> None:
        mask = _make_mask([(0, 0), (0, 1)], dense_shape=(2, 2))
        weights = torch.tensor([[5.0, 1.0], [0.0, 0.0]])
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.ones(2, 2),
            num_steps=1,
        )
        config = SparseRecConfig(
            regrowth_fraction=0.5,
        )
        result = update_topology(mask, weights, acc, config)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.num_pruned = 99  # type: ignore[misc]

    def test_cumulative_gradient_norm_positive(self) -> None:
        mask = _make_mask([(0, 0), (0, 1)], dense_shape=(2, 2))
        weights = torch.tensor([[5.0, 1.0], [0.0, 0.0]])
        acc = GradientAccumulator(
            cumulative_magnitudes=torch.ones(2, 2),
            num_steps=1,
        )
        config = SparseRecConfig(regrowth_fraction=0.5)
        result = update_topology(mask, weights, acc, config)
        assert result.cumulative_gradient_norm > 0.0

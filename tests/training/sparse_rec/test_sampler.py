"""Tests for SparseRec multinomial sampler."""

from __future__ import annotations

import pytest
import torch

from kairo.training.sparse_rec.sampler import (
    compute_sampling_weights,
    sample_active_ids,
)


class TestComputeSamplingWeights:
    """Tests for compute_sampling_weights."""

    def test_output_sums_to_one(self) -> None:
        counts = torch.tensor([10.0, 20.0, 30.0, 40.0])
        weights = compute_sampling_weights(counts)
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_higher_count_higher_probability(self) -> None:
        counts = torch.tensor([1.0, 10.0, 100.0])
        weights = compute_sampling_weights(counts, temperature=1.0)
        assert weights[0] < weights[1] < weights[2]

    def test_all_zero_returns_uniform(self) -> None:
        counts = torch.zeros(5)
        weights = compute_sampling_weights(counts)
        expected = torch.full((5,), 1.0 / 5.0)
        assert torch.allclose(weights, expected, atol=1e-6)

    def test_single_element(self) -> None:
        counts = torch.tensor([42.0])
        weights = compute_sampling_weights(counts)
        assert torch.isclose(weights[0], torch.tensor(1.0))

    def test_high_temperature_more_uniform(self) -> None:
        counts = torch.tensor([1.0, 100.0])
        low_temp = compute_sampling_weights(counts, temperature=0.1)
        high_temp = compute_sampling_weights(counts, temperature=10.0)
        # High temp should be closer to uniform (0.5, 0.5)
        low_temp_var = (low_temp - 0.5).abs().sum()
        high_temp_var = (high_temp - 0.5).abs().sum()
        assert high_temp_var < low_temp_var

    def test_output_non_negative(self) -> None:
        counts = torch.tensor([0.0, 5.0, 10.0])
        weights = compute_sampling_weights(counts)
        assert (weights >= 0).all()

    def test_does_not_mutate_input(self) -> None:
        counts = torch.tensor([1.0, 2.0, 3.0])
        original = counts.clone()
        compute_sampling_weights(counts)
        assert torch.equal(counts, original)


class TestSampleActiveIds:
    """Tests for sample_active_ids."""

    def test_returns_correct_count(self) -> None:
        weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
        ids = sample_active_ids(weights, num_samples=2)
        assert ids.shape == (2,)

    def test_returns_sorted(self) -> None:
        weights = torch.ones(100) / 100
        gen = torch.Generator().manual_seed(42)
        ids = sample_active_ids(weights, num_samples=20, generator=gen)
        assert torch.equal(ids, ids.sort().values)

    def test_no_duplicates(self) -> None:
        weights = torch.ones(10) / 10
        gen = torch.Generator().manual_seed(42)
        ids = sample_active_ids(weights, num_samples=5, generator=gen)
        assert ids.unique().shape == ids.shape

    def test_seed_reproducibility(self) -> None:
        weights = torch.ones(50) / 50
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        ids1 = sample_active_ids(weights, num_samples=10, generator=gen1)
        ids2 = sample_active_ids(weights, num_samples=10, generator=gen2)
        assert torch.equal(ids1, ids2)

    def test_sample_all(self) -> None:
        weights = torch.ones(5) / 5
        ids = sample_active_ids(weights, num_samples=5)
        assert torch.equal(ids, torch.arange(5))

    def test_raises_if_num_samples_exceeds_length(self) -> None:
        weights = torch.ones(3) / 3
        with pytest.raises(ValueError):
            sample_active_ids(weights, num_samples=4)

    def test_ids_within_range(self) -> None:
        n = 20
        weights = torch.ones(n) / n
        ids = sample_active_ids(weights, num_samples=10)
        assert (ids >= 0).all() and (ids < n).all()

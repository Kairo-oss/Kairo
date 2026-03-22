"""Tests for frozen configuration dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from kairo.config import AGENTConfig, NMFConfig, SparseRecConfig, TrainerConfig


class TestSparseRecConfig:
    """Tests for SparseRecConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = SparseRecConfig()
        assert config.sample_ratio == 0.1
        assert config.regrowth_interval == 100
        assert config.regrowth_fraction == 0.1
        assert config.prune_criterion == "magnitude"
        assert config.seed is None

    def test_custom_values(self) -> None:
        config = SparseRecConfig(
            sample_ratio=0.5, regrowth_interval=50, regrowth_fraction=0.2,
            prune_criterion="gradient", seed=42,
        )
        assert config.sample_ratio == 0.5
        assert config.prune_criterion == "gradient"
        assert config.seed == 42

    def test_frozen(self) -> None:
        config = SparseRecConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.sample_ratio = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize("sample_ratio", [0.0, -0.1, 1.1])
    def test_invalid_sample_ratio(self, sample_ratio: float) -> None:
        with pytest.raises(ValueError, match="sample_ratio"):
            SparseRecConfig(sample_ratio=sample_ratio)

    def test_sample_ratio_one_is_valid(self) -> None:
        config = SparseRecConfig(sample_ratio=1.0)
        assert config.sample_ratio == 1.0

    def test_invalid_regrowth_interval(self) -> None:
        with pytest.raises(ValueError, match="regrowth_interval"):
            SparseRecConfig(regrowth_interval=0)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_invalid_regrowth_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="regrowth_fraction"):
            SparseRecConfig(regrowth_fraction=fraction)

    def test_invalid_prune_criterion(self) -> None:
        with pytest.raises(ValueError, match="prune_criterion"):
            SparseRecConfig(prune_criterion="random")


class TestAGENTConfig:
    """Tests for AGENTConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = AGENTConfig()
        assert config.lr == 1e-3
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.eps == 1e-8
        assert config.weight_decay == 0.0
        assert config.correction_threshold == 0.1

    def test_frozen(self) -> None:
        config = AGENTConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.lr = 0.01  # type: ignore[misc]

    def test_invalid_lr(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            AGENTConfig(lr=0.0)

    @pytest.mark.parametrize("beta", [-0.1, 1.0])
    def test_invalid_beta1(self, beta: float) -> None:
        with pytest.raises(ValueError, match="beta1"):
            AGENTConfig(beta1=beta)

    @pytest.mark.parametrize("beta", [-0.1, 1.0])
    def test_invalid_beta2(self, beta: float) -> None:
        with pytest.raises(ValueError, match="beta2"):
            AGENTConfig(beta2=beta)

    def test_invalid_eps(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            AGENTConfig(eps=0.0)

    def test_invalid_weight_decay(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            AGENTConfig(weight_decay=-0.1)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_invalid_correction_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="correction_threshold"):
            AGENTConfig(correction_threshold=threshold)


class TestTrainerConfig:
    """Tests for TrainerConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = TrainerConfig()
        assert config.max_steps == 10000
        assert config.log_interval == 100
        assert config.eval_interval == 500
        assert config.checkpoint_interval == 1000
        assert config.device == "cuda"

    def test_frozen(self) -> None:
        config = TrainerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.max_steps = 100  # type: ignore[misc]

    def test_invalid_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            TrainerConfig(max_steps=0)

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            TrainerConfig(device="tpu")

    def test_cpu_device(self) -> None:
        config = TrainerConfig(device="cpu")
        assert config.device == "cpu"

"""Tests for SparseTrainer orchestrating the full sparse training pipeline."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from kairo.config import AGENTConfig, SparseRecConfig, TrainerConfig
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask
from kairo.training.trainer import SparseTrainer, TrainResult, TrainStepResult
from kairo.types import EmbeddingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(num_embeddings=50, embedding_dim=16, sparsity_ratio=0.5)


@pytest.fixture()
def sparse_table(embedding_config: EmbeddingConfig) -> SparseEmbeddingTable:
    """Create a table with a random sparse mask for testing."""
    torch.manual_seed(42)
    # Use NMF to generate a proper mask
    interaction = torch.rand(50, 16) + 0.01
    from kairo.config import NMFConfig
    nmf_cfg = NMFConfig(rank=4, max_iter=50, seed=42)
    nmf_result = nmf_decompose(interaction, nmf_cfg)
    mask = generate_sparse_mask(
        nmf_result,
        embedding_config.num_embeddings,
        embedding_config.embedding_dim,
        embedding_config.sparsity_ratio,
    )
    return SparseEmbeddingTable(embedding_config, mask=mask)


@pytest.fixture()
def sparse_rec_config() -> SparseRecConfig:
    return SparseRecConfig(
        sample_ratio=0.5,
        regrowth_interval=5,
        regrowth_fraction=0.1,
        seed=42,
    )


@pytest.fixture()
def agent_config() -> AGENTConfig:
    return AGENTConfig(lr=0.01, correction_threshold=0.1)


@pytest.fixture()
def trainer_config() -> TrainerConfig:
    return TrainerConfig(max_steps=20, log_interval=5, device="cpu")


def _simple_loss_fn(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple MSE loss for testing."""
    return ((output - targets) ** 2).mean()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainStepResult:
    """TrainStepResult and TrainResult are frozen."""

    def test_train_step_result_frozen(self) -> None:
        result = TrainStepResult(loss=1.0, step=0, active_ids_count=10, grad_norm=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.loss = 0.0  # type: ignore[misc]

    def test_train_result_frozen(self) -> None:
        result = TrainResult(
            final_loss=0.1, total_steps=100, mask_updates=5, final_sparsity=0.8,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_steps = 0  # type: ignore[misc]


class TestSparseTrainer:
    """Integration tests for the SparseTrainer pipeline."""

    def test_train_step_returns_result(
        self, sparse_table, sparse_rec_config, agent_config, trainer_config,
    ) -> None:
        trainer = SparseTrainer(
            table=sparse_table,
            sparse_rec_config=sparse_rec_config,
            agent_config=agent_config,
            trainer_config=trainer_config,
        )
        batch_ids = torch.randint(0, 50, (8,))
        targets = torch.randn(8, 16)
        result = trainer.train_step(batch_ids, targets, _simple_loss_fn)

        assert isinstance(result, TrainStepResult)
        assert result.step == 1
        assert result.loss > 0
        assert result.active_ids_count > 0
        assert result.grad_norm >= 0

    def test_loss_decreases_over_steps(
        self, sparse_table, sparse_rec_config, agent_config, trainer_config,
    ) -> None:
        torch.manual_seed(123)
        trainer = SparseTrainer(
            table=sparse_table,
            sparse_rec_config=sparse_rec_config,
            agent_config=agent_config,
            trainer_config=trainer_config,
        )
        batch_ids = torch.randint(0, 50, (16,))
        targets = torch.randn(16, 16)

        first_loss = trainer.train_step(batch_ids, targets, _simple_loss_fn).loss
        for _ in range(14):
            trainer.train_step(batch_ids, targets, _simple_loss_fn)
        last_loss = trainer.train_step(batch_ids, targets, _simple_loss_fn).loss

        assert last_loss < first_loss

    def test_topology_update_triggers(
        self, sparse_table, agent_config, trainer_config,
    ) -> None:
        """Topology update should occur at regrowth_interval boundaries."""
        config = SparseRecConfig(
            sample_ratio=0.5,
            regrowth_interval=3,
            regrowth_fraction=0.1,
            seed=42,
        )
        trainer = SparseTrainer(
            table=sparse_table,
            sparse_rec_config=config,
            agent_config=agent_config,
            trainer_config=trainer_config,
        )
        batch_ids = torch.randint(0, 50, (8,))
        targets = torch.randn(8, 16)

        initial_table_id = id(trainer.table)
        # Steps 1, 2 — no topology update
        trainer.train_step(batch_ids, targets, _simple_loss_fn)
        trainer.train_step(batch_ids, targets, _simple_loss_fn)
        assert trainer.mask_update_count == 0

        # Step 3 — topology update should trigger
        trainer.train_step(batch_ids, targets, _simple_loss_fn)
        assert trainer.mask_update_count == 1

    def test_train_loop(
        self, sparse_table, sparse_rec_config, agent_config, trainer_config,
    ) -> None:
        """Full train() loop produces valid TrainResult."""
        torch.manual_seed(99)
        trainer = SparseTrainer(
            table=sparse_table,
            sparse_rec_config=sparse_rec_config,
            agent_config=agent_config,
            trainer_config=trainer_config,
        )

        def data_iter():
            while True:
                yield torch.randint(0, 50, (8,)), torch.randn(8, 16)

        result = trainer.train(data_iter(), _simple_loss_fn, num_steps=10)

        assert isinstance(result, TrainResult)
        assert result.total_steps == 10
        assert result.final_loss > 0
        assert 0.0 <= result.final_sparsity <= 1.0

    def test_full_pipeline_nmf_to_training(self) -> None:
        """End-to-end: NMF init → sparse mask → SparseTrainer for 15 steps."""
        torch.manual_seed(7)
        from kairo.config import NMFConfig
        interaction = torch.rand(30, 20) + 0.01
        nmf_cfg = NMFConfig(rank=4, max_iter=50, seed=7)
        nmf_result = nmf_decompose(interaction, nmf_cfg)
        mask = generate_sparse_mask(nmf_result, 30, 20, 0.5)
        emb_config = EmbeddingConfig(30, 20, sparsity_ratio=0.5)
        table = SparseEmbeddingTable(emb_config, mask=mask)

        trainer = SparseTrainer(
            table=table,
            sparse_rec_config=SparseRecConfig(
                sample_ratio=0.5, regrowth_interval=5,
                regrowth_fraction=0.1, seed=7,
            ),
            agent_config=AGENTConfig(lr=0.01),
            trainer_config=TrainerConfig(max_steps=15, device="cpu"),
        )

        def data_iter():
            while True:
                yield torch.randint(0, 30, (8,)), torch.randn(8, 20)

        result = trainer.train(data_iter(), _simple_loss_fn, num_steps=15)
        assert result.total_steps == 15
        assert result.mask_updates >= 2  # regrowth_interval=5, 15 steps → ≥2 updates

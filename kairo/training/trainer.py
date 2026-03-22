"""SparseTrainer: orchestrates the full sparse training pipeline.

Each training step executes:
1. Sample active IDs via MultinomialSampler
2. Forward pass through SparseEmbeddingTable with sparse-gradient autograd
3. Compute loss
4. Backward pass (gradients only for sampled active IDs)
5. AGENT optimizer step (momentum-corrected update)
6. Accumulate gradient magnitudes
7. Periodically update mask topology (prune low-weight + grow high-gradient)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from kairo.config import AGENTConfig, SparseRecConfig, TrainerConfig
from kairo.storage.embedding import SparseEmbeddingTable
from kairo.training.agent_optim.optimizer import AGENTOptimizer
from kairo.training.sparse_rec.sampler import (
    compute_sampling_weights,
    sample_active_ids,
)
from kairo.training.sparse_rec.topology import (
    GradientAccumulator,
    update_topology,
)


@dataclass(frozen=True)
class TrainStepResult:
    """Immutable result of a single training step."""

    loss: float
    step: int
    active_ids_count: int
    grad_norm: float


@dataclass(frozen=True)
class TrainResult:
    """Immutable result of a full training run."""

    final_loss: float
    total_steps: int
    mask_updates: int
    final_sparsity: float


class SparseTrainer:
    """Orchestrates SparseRec sampling, AGENT optimization, and topology updates.

    Args:
        table: SparseEmbeddingTable to train (will be replaced on topology updates).
        sparse_rec_config: Configuration for sampling and topology management.
        agent_config: Configuration for the AGENT optimizer.
        trainer_config: Configuration for the training loop.
    """

    def __init__(
        self,
        table: SparseEmbeddingTable,
        sparse_rec_config: SparseRecConfig,
        agent_config: AGENTConfig,
        trainer_config: TrainerConfig,
    ) -> None:
        self._sparse_rec_config = sparse_rec_config
        self._trainer_config = trainer_config
        self._table = table

        self._optimizer = AGENTOptimizer(table.parameters(), agent_config)

        num_e = table.num_embeddings
        emb_dim = table.embedding_dim
        device = trainer_config.device

        # Access frequency tracker for multinomial sampling
        self._access_counts = torch.zeros(num_e, device=device)

        # Gradient accumulator for regrowth decisions
        self._grad_accumulator = GradientAccumulator(
            cumulative_magnitudes=torch.zeros(num_e, emb_dim, device=device),
            num_steps=0,
        )

        # Sampling RNG
        self._generator: torch.Generator | None = None
        if sparse_rec_config.seed is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(sparse_rec_config.seed)

        self._step_count = 0
        self._mask_update_count = 0

    @property
    def table(self) -> SparseEmbeddingTable:
        """Current embedding table (replaced on topology updates)."""
        return self._table

    @property
    def mask_update_count(self) -> int:
        """Number of topology updates performed so far."""
        return self._mask_update_count

    def train_step(
        self,
        batch_ids: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> TrainStepResult:
        """Execute one training step with sparse gradients.

        Args:
            batch_ids: Integer tensor of embedding IDs for this batch.
            targets: Target tensor for the loss function.
            loss_fn: Callable (output, targets) -> scalar loss tensor.

        Returns:
            Frozen TrainStepResult with loss, step, active count, and grad norm.
        """
        self._step_count += 1

        # Update access frequency histogram (vectorized)
        counts = torch.bincount(
            batch_ids.flatten(),
            minlength=self._access_counts.shape[0],
        )
        self._access_counts.add_(counts.float())

        # Sample active IDs
        num_e = self._table.num_embeddings
        num_samples = max(1, int(num_e * self._sparse_rec_config.sample_ratio))
        weights = compute_sampling_weights(self._access_counts)
        active_ids = sample_active_ids(weights, num_samples, self._generator)

        # Forward with sparse-gradient autograd
        self._optimizer.zero_grad()
        output = self._table(batch_ids, active_ids=active_ids)
        loss = loss_fn(output, targets)
        loss.backward()

        # Compute gradient norm for diagnostics
        grad_norm = 0.0
        if self._table.weight.grad is not None:
            grad_norm = float(self._table.weight.grad.norm(2).item())

            # Accumulate gradient magnitudes for regrowth
            self._grad_accumulator = self._grad_accumulator.with_update(
                self._table.weight.grad.detach(),
            )

        # AGENT optimizer step
        self._optimizer.step()

        # Maybe update mask topology
        self._maybe_update_topology()

        return TrainStepResult(
            loss=float(loss.item()),
            step=self._step_count,
            active_ids_count=num_samples,
            grad_norm=grad_norm,
        )

    def train(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int | None = None,
    ) -> TrainResult:
        """Run the full training loop.

        Args:
            dataloader: Iterable yielding (batch_ids, targets) tuples.
            loss_fn: Callable (output, targets) -> scalar loss tensor.
            num_steps: Number of steps to run. Defaults to trainer_config.max_steps.

        Returns:
            Frozen TrainResult with final metrics.
        """
        max_steps = num_steps if num_steps is not None else self._trainer_config.max_steps
        last_loss = float("nan")

        data_iter = iter(dataloader)
        for _ in range(max_steps):
            try:
                batch_ids, targets = next(data_iter)
            except StopIteration:
                break
            result = self.train_step(batch_ids, targets, loss_fn)
            last_loss = result.loss

        sparsity = self._table.compression_ratio

        return TrainResult(
            final_loss=last_loss,
            total_steps=self._step_count,
            mask_updates=self._mask_update_count,
            final_sparsity=sparsity,
        )

    def _maybe_update_topology(self) -> None:
        """Update mask topology if at a regrowth interval boundary."""
        interval = self._sparse_rec_config.regrowth_interval
        if self._step_count % interval != 0:
            return

        if self._table.mask is None:
            return

        old_weight = self._table.weight
        update_result = update_topology(
            self._table.mask,
            self._table.weight.data,
            self._grad_accumulator,
            self._sparse_rec_config,
        )

        # Replace table with new mask (immutable pattern)
        self._table = self._table.with_mask(update_result.new_mask)

        # Migrate optimizer state to new parameter
        new_weight = self._table.weight
        surviving_mask = update_result.new_mask.to_dense().bool().float()
        self._optimizer.migrate_state(old_weight, new_weight, surviving_mask)

        # Reset gradient accumulator after topology update
        shape = self._grad_accumulator.cumulative_magnitudes.shape
        device = self._grad_accumulator.cumulative_magnitudes.device
        self._grad_accumulator = GradientAccumulator(
            cumulative_magnitudes=torch.zeros(shape, device=device),
            num_steps=0,
        )

        self._mask_update_count += 1

"""Frozen configuration dataclasses for Kairo algorithms."""

from __future__ import annotations

from dataclasses import dataclass


_VALID_PRUNE_CRITERIA = frozenset({"magnitude", "gradient"})


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


@dataclass(frozen=True)
class SparseRecConfig:
    """Configuration for SparseRec selective gradient computation.

    Attributes:
        sample_ratio: Fraction of embedding IDs to sample per training step.
        regrowth_interval: Number of steps between mask topology updates.
        regrowth_fraction: Fraction of active weights to prune (and regrow) per update.
        prune_criterion: Weight selection criterion for pruning: "magnitude" or "gradient".
        seed: Random seed for reproducible sampling. None for non-deterministic.
    """

    sample_ratio: float = 0.1
    regrowth_interval: int = 100
    regrowth_fraction: float = 0.1
    prune_criterion: str = "magnitude"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError(
                f"sample_ratio must be in (0.0, 1.0], got {self.sample_ratio}"
            )
        if self.regrowth_interval <= 0:
            raise ValueError(
                f"regrowth_interval must be positive, got {self.regrowth_interval}"
            )
        if not 0.0 <= self.regrowth_fraction <= 1.0:
            raise ValueError(
                f"regrowth_fraction must be in [0.0, 1.0], got {self.regrowth_fraction}"
            )
        if self.prune_criterion not in _VALID_PRUNE_CRITERIA:
            raise ValueError(
                f"prune_criterion must be one of {sorted(_VALID_PRUNE_CRITERIA)}, "
                f"got '{self.prune_criterion}'"
            )


@dataclass(frozen=True)
class AGENTConfig:
    """Configuration for the AGENT optimizer with momentum correlation correction.

    At extreme sparsity (90-99%), standard optimizers lose directional accuracy.
    AGENT uses cosine similarity between the current gradient and historical momentum
    to adaptively amplify or dampen the update step.

    Attributes:
        lr: Learning rate.
        beta1: Exponential decay rate for the first moment (momentum).
        beta2: Exponential decay rate for the second moment (variance).
        eps: Term added to denominator for numerical stability.
        weight_decay: L2 regularization coefficient.
        correction_threshold: Minimum cosine similarity to apply momentum correction.
    """

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    correction_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {self.beta2}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0.0 <= self.correction_threshold <= 1.0:
            raise ValueError(
                f"correction_threshold must be in [0.0, 1.0], got {self.correction_threshold}"
            )


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the SparseTrainer training loop.

    Attributes:
        max_steps: Maximum number of training steps.
        log_interval: Steps between logging metrics.
        eval_interval: Steps between evaluation runs.
        checkpoint_interval: Steps between model checkpoints.
        device: Target device ("cuda" or "cpu").
    """

    max_steps: int = 10000
    log_interval: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'")

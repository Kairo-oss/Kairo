"""AGENT optimizer: Adaptive Gradient correction with historical momentum.

At high sparsity (90-99%), standard optimizers lose directional accuracy because
too few parameters update per step. AGENT computes cosine similarity between the
current gradient and historical momentum to adaptively correct the update.

Reference: https://arxiv.org/abs/2301.03573
"""

from __future__ import annotations

from typing import Any

import torch

from kairo.config import AGENTConfig


class AGENTOptimizer(torch.optim.Optimizer):
    """AGENT optimizer with momentum correlation correction for sparse training.

    Algorithm per parameter per step:
        1. m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           (momentum EMA)
        2. v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         (variance EMA)
        3. cos_sim = cosine_similarity(g_t, m_{t-1})            (direction alignment)
        4. correction = (1 + cos_sim) if cos_sim > threshold else 1.0
        5. m_hat = m_t / (1 - beta1^t)                          (bias correction)
        6. v_hat = v_t / (1 - beta2^t)                          (bias correction)
        7. param -= lr * correction * m_hat / (sqrt(v_hat) + eps)

    Plus optional decoupled weight decay (AdamW-style).
    """

    def __init__(self, params: Any, config: AGENTConfig) -> None:
        defaults = {
            "lr": config.lr,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "correction_threshold": config.correction_threshold,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Any | None = None,
    ) -> torch.Tensor | None:
        """Perform a single optimization step with momentum correlation correction.

        Args:
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            Loss value if closure was provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            threshold = group["correction_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                m = state["momentum"]
                v = state["variance"]

                # Decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Momentum correlation correction
                correction = 1.0
                if step > 1:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad.flatten().unsqueeze(0),
                        m.flatten().unsqueeze(0),
                    ).item()
                    if cos_sim > threshold:
                        correction = 1.0 + cos_sim

                # Update biased first moment estimate (momentum)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update biased second moment estimate (variance)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)

                # Parameter update with correction factor
                p.addcdiv_(
                    m_hat,
                    v_hat.sqrt().add_(eps),
                    value=-lr * correction,
                )

        return loss

    def migrate_state(
        self,
        old_param: torch.Tensor,
        new_param: torch.Tensor,
        surviving_mask: torch.Tensor,
    ) -> None:
        """Migrate optimizer state after mask topology update (prune/grow).

        Surviving positions keep their momentum and variance history.
        New (grown) positions get zero-initialized state.

        Args:
            old_param: Parameter tensor before the mask update.
            new_param: Parameter tensor after the mask update.
            surviving_mask: Dense boolean mask matching the parameter shape.
                True where positions survived the prune/grow cycle.
        """
        if old_param not in self.state:
            return

        old_state = self.state[old_param]
        new_state = {
            "step": old_state["step"],
            "momentum": old_state["momentum"] * surviving_mask,
            "variance": old_state["variance"] * surviving_mask,
        }

        # Move state from old param to new param
        del self.state[old_param]
        self.state[new_param] = new_state

        # Update param_groups to reference new param
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_param:
                    group["params"][i] = new_param

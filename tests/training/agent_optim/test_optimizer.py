"""Tests for the AGENT optimizer with momentum correlation correction."""

from __future__ import annotations

import copy

import pytest
import torch

from kairo.config import AGENTConfig
from kairo.training.agent_optim import AGENTOptimizer


@pytest.fixture()
def default_config() -> AGENTConfig:
    return AGENTConfig(lr=1e-2)


@pytest.fixture()
def simple_param() -> torch.nn.Parameter:
    """A small parameter for unit tests."""
    torch.manual_seed(42)
    return torch.nn.Parameter(torch.randn(4, 4))


class TestConvergence:
    """Verify AGENT can minimize a simple quadratic."""

    def test_convergence(self, default_config: AGENTConfig) -> None:
        """Minimize f(x) = sum(x^2) from random init, x should approach 0."""
        torch.manual_seed(0)
        x = torch.nn.Parameter(torch.randn(10))
        optimizer = AGENTOptimizer([x], config=default_config)

        for _ in range(200):
            optimizer.zero_grad()
            loss = (x**2).sum()
            loss.backward()
            optimizer.step()

        assert x.abs().max().item() < 0.1, (
            f"Failed to converge: max |x| = {x.abs().max().item()}"
        )


class TestStateInitialization:
    """First step must create momentum/variance with correct shapes."""

    def test_state_initialization(
        self, simple_param: torch.nn.Parameter, default_config: AGENTConfig
    ) -> None:
        optimizer = AGENTOptimizer([simple_param], config=default_config)

        # No state before first step
        assert len(optimizer.state) == 0

        simple_param.grad = torch.randn_like(simple_param)
        optimizer.step()

        state = optimizer.state[simple_param]
        assert state["step"] == 1
        assert state["momentum"].shape == simple_param.shape
        assert state["variance"].shape == simple_param.shape


class TestCorrectionBehavior:
    """Verify momentum correlation correction logic."""

    def test_correction_amplifies_aligned_gradients(self) -> None:
        """Repeated same-direction grads should produce correction > 1."""
        config = AGENTConfig(lr=1e-2, correction_threshold=0.0)
        x = torch.nn.Parameter(torch.ones(10))
        optimizer = AGENTOptimizer([x], config=config)

        # Use consistent gradient direction to build aligned momentum.
        # Run enough steps so correction kicks in (step > 1) and accumulates.
        grad = torch.ones(10)
        for _ in range(20):
            optimizer.zero_grad()
            x.grad = grad.clone()
            optimizer.step()

        x_agent = x.data.clone()

        # Compare with correction_threshold=1.0 (effectively disabling correction,
        # since cos_sim is always <= 1.0 and threshold is 1.0, so cos_sim > 1.0
        # is never true).
        x2 = torch.nn.Parameter(torch.ones(10))
        config_no_corr = AGENTConfig(lr=1e-2, correction_threshold=1.0)
        opt2 = AGENTOptimizer([x2], config=config_no_corr)
        for _ in range(20):
            opt2.zero_grad()
            x2.grad = grad.clone()
            opt2.step()

        # Agent with correction should have moved further from initial value
        dist_agent = (x_agent - 1.0).abs().mean().item()
        dist_no_corr = (x2.data - 1.0).abs().mean().item()
        assert dist_agent > dist_no_corr, (
            f"Correction did not amplify: agent={dist_agent}, no_corr={dist_no_corr}"
        )

    def test_correction_suppressed_for_orthogonal_gradients(self) -> None:
        """Alternating orthogonal grads should not trigger correction."""
        config = AGENTConfig(lr=1e-3, correction_threshold=0.1)
        x = torch.nn.Parameter(torch.zeros(4))
        optimizer = AGENTOptimizer([x], config=config)

        # Alternating orthogonal gradients
        grad_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        grad_b = torch.tensor([0.0, 1.0, 0.0, 0.0])

        # Build momentum from grad_a
        optimizer.zero_grad()
        x.grad = grad_a.clone()
        optimizer.step()

        # Now apply grad_b (orthogonal to momentum)
        param_before = x.data.clone()
        optimizer.zero_grad()
        x.grad = grad_b.clone()
        optimizer.step()

        # Compare with a run that uses correction_threshold=0.0
        # The key check: with orthogonal grads, cos_sim ~ 0, so
        # correction should be 1.0 regardless of threshold
        # We verify indirectly: both configs produce same result on step 2
        x2 = torch.nn.Parameter(torch.zeros(4))
        config2 = AGENTConfig(lr=1e-3, correction_threshold=0.0)
        opt2 = AGENTOptimizer([x2], config=config2)

        opt2.zero_grad()
        x2.grad = grad_a.clone()
        opt2.step()

        opt2.zero_grad()
        x2.grad = grad_b.clone()
        opt2.step()

        # With threshold=0.0, cos_sim for orthogonal should be ~0,
        # so correction = 1 + ~0 which is negligible difference.
        # The results should be very close.
        diff = (x.data - x2.data).abs().max().item()
        assert diff < 1e-5, f"Orthogonal grads produced different results: diff={diff}"


class TestWeightDecay:
    """Verify decoupled weight decay is applied."""

    def test_weight_decay_applied(self) -> None:
        config_wd = AGENTConfig(lr=1e-2, weight_decay=0.1)
        config_no_wd = AGENTConfig(lr=1e-2, weight_decay=0.0)

        torch.manual_seed(42)
        x_wd = torch.nn.Parameter(torch.ones(10) * 5.0)
        x_no_wd = torch.nn.Parameter(torch.ones(10) * 5.0)

        opt_wd = AGENTOptimizer([x_wd], config=config_wd)
        opt_no_wd = AGENTOptimizer([x_no_wd], config=config_no_wd)

        grad = torch.ones(10) * 0.01  # small gradient
        for _ in range(10):
            opt_wd.zero_grad()
            x_wd.grad = grad.clone()
            opt_wd.step()

            opt_no_wd.zero_grad()
            x_no_wd.grad = grad.clone()
            opt_no_wd.step()

        # Weight decay should shrink parameters more
        assert x_wd.data.abs().mean().item() < x_no_wd.data.abs().mean().item()


class TestSparseGradient:
    """Verify correct operation with mostly-zero gradients."""

    def test_sparse_gradient_handling(self) -> None:
        config = AGENTConfig(lr=1e-2)
        x = torch.nn.Parameter(torch.ones(10, 10))
        optimizer = AGENTOptimizer([x], config=config)

        # Create a mostly-zero gradient (only 10% non-zero)
        grad = torch.zeros(10, 10)
        grad[0, :] = 1.0  # only first row has gradient

        for _ in range(5):
            optimizer.zero_grad()
            x.grad = grad.clone()
            optimizer.step()

        # Only first row should have moved significantly
        first_row_change = (x.data[0] - 1.0).abs().mean().item()
        other_rows_change = (x.data[1:] - 1.0).abs().mean().item()
        assert first_row_change > other_rows_change * 10


class TestDeterminism:
    """Same gradient sequence must produce identical results."""

    def test_determinism(self) -> None:
        config = AGENTConfig(lr=1e-2)

        results = []
        for _ in range(2):
            torch.manual_seed(99)
            x = torch.nn.Parameter(torch.randn(8))
            optimizer = AGENTOptimizer([x], config=config)

            torch.manual_seed(123)
            for _ in range(10):
                optimizer.zero_grad()
                x.grad = torch.randn(8)
                optimizer.step()
            results.append(x.data.clone())

        assert torch.allclose(results[0], results[1]), (
            f"Non-deterministic: max diff = {(results[0] - results[1]).abs().max()}"
        )


class TestStateDictRoundtrip:
    """Save/load state_dict must preserve optimizer state."""

    def test_state_dict_roundtrip(
        self, simple_param: torch.nn.Parameter, default_config: AGENTConfig
    ) -> None:
        optimizer = AGENTOptimizer([simple_param], config=default_config)

        # Run a few steps to build state
        for _ in range(3):
            optimizer.zero_grad()
            simple_param.grad = torch.randn_like(simple_param)
            optimizer.step()

        state_dict = copy.deepcopy(optimizer.state_dict())

        # Create a new optimizer and load state
        optimizer2 = AGENTOptimizer([simple_param], config=default_config)
        optimizer2.load_state_dict(state_dict)

        # Verify state matches
        for key in ["step", "momentum", "variance"]:
            orig = optimizer.state[simple_param][key]
            loaded = optimizer2.state[simple_param][key]
            if isinstance(orig, torch.Tensor):
                assert torch.allclose(orig, loaded), f"Mismatch in {key}"
            else:
                assert orig == loaded, f"Mismatch in {key}: {orig} vs {loaded}"


class TestMigrateState:
    """Verify state migration for mask topology updates."""

    def test_migrate_state(self, default_config: AGENTConfig) -> None:
        old_param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = AGENTOptimizer([old_param], config=default_config)

        # Build some state
        for _ in range(3):
            optimizer.zero_grad()
            old_param.grad = torch.randn_like(old_param)
            optimizer.step()

        old_momentum = optimizer.state[old_param]["momentum"].clone()
        old_variance = optimizer.state[old_param]["variance"].clone()
        old_step = optimizer.state[old_param]["step"]

        # Create surviving mask (half positions survive)
        surviving_mask = torch.zeros(4, 4, dtype=torch.bool)
        surviving_mask[:2, :] = True

        new_param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer.migrate_state(old_param, new_param, surviving_mask)

        # Old param state should be removed
        assert old_param not in optimizer.state

        # New param state should exist with correct values
        new_state = optimizer.state[new_param]
        assert new_state["step"] == old_step

        # Surviving positions keep their state
        assert torch.allclose(
            new_state["momentum"][:2], old_momentum[:2]
        )
        assert torch.allclose(
            new_state["variance"][:2], old_variance[:2]
        )

        # Pruned/grown positions should be zeroed
        assert (new_state["momentum"][2:] == 0).all()
        assert (new_state["variance"][2:] == 0).all()

        # param_groups should reference new_param
        found = any(
            new_param is p
            for group in optimizer.param_groups
            for p in group["params"]
        )
        assert found, "new_param not found in param_groups"

    def test_migrate_state_no_existing_state(
        self, default_config: AGENTConfig
    ) -> None:
        """migrate_state should be a no-op when old_param has no state."""
        old_param = torch.nn.Parameter(torch.randn(4))
        new_param = torch.nn.Parameter(torch.randn(4))
        optimizer = AGENTOptimizer([old_param], config=default_config)

        # No steps taken, so no state exists
        surviving_mask = torch.ones(4, dtype=torch.bool)
        optimizer.migrate_state(old_param, new_param, surviving_mask)

        assert new_param not in optimizer.state


class TestFirstStepNoCorrection:
    """First step should use correction = 1.0 since there is no prior momentum."""

    def test_first_step_no_correction(self) -> None:
        config = AGENTConfig(lr=1e-2, correction_threshold=0.0)
        config_high = AGENTConfig(lr=1e-2, correction_threshold=1.0)

        x1 = torch.nn.Parameter(torch.ones(5))
        x2 = torch.nn.Parameter(torch.ones(5))

        opt1 = AGENTOptimizer([x1], config=config)
        opt2 = AGENTOptimizer([x2], config=config_high)

        grad = torch.ones(5)
        opt1.zero_grad()
        x1.grad = grad.clone()
        opt1.step()

        opt2.zero_grad()
        x2.grad = grad.clone()
        opt2.step()

        # First step: both should be identical since correction=1.0 on step 1
        assert torch.allclose(x1.data, x2.data), (
            f"First step differs: {(x1.data - x2.data).abs().max()}"
        )

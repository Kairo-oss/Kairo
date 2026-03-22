"""Shared fixtures and utilities for benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch


@dataclass
class BenchmarkResult:
    """Stores timing statistics from a benchmark run."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    iterations: int

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
            f"(min={self.min_ms:.3f} ms, n={self.iterations})"
        )


def bench(
    fn: Callable[[], object],
    name: str = "benchmark",
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Time a function with warmup, return statistics in milliseconds."""
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

    t = torch.tensor(times)
    return BenchmarkResult(
        name=name,
        mean_ms=float(t.mean()),
        std_ms=float(t.std()),
        min_ms=float(t.min()),
        iterations=iterations,
    )


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

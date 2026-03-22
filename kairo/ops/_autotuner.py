"""Runtime tile size selection for Acc-SpMM kernel."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileConfig:
    """Immutable tile configuration for block-sparse matmul."""

    tile_m: int
    tile_n: int
    tile_k: int


# Candidate configs to benchmark
CANDIDATE_CONFIGS: tuple[TileConfig, ...] = (
    TileConfig(tile_m=16, tile_n=16, tile_k=16),
    TileConfig(tile_m=32, tile_n=32, tile_k=16),
    TileConfig(tile_m=64, tile_n=64, tile_k=16),
    TileConfig(tile_m=16, tile_n=16, tile_k=32),
    TileConfig(tile_m=32, tile_n=32, tile_k=32),
)

DEFAULT_CONFIG = CANDIDATE_CONFIGS[0]

# Number of benchmark iterations
BENCH_WARMUP = 2
BENCH_ITERS = 5


def _cache_key(M: int, K: int, N: int, sparsity: float) -> str:
    """Create a hashable key for caching autotuner results."""
    # Bucket dimensions to avoid too many cache entries
    def bucket(x: int) -> int:
        if x <= 128:
            return 128
        if x <= 512:
            return 512
        if x <= 2048:
            return 2048
        return 4096

    sp_bucket = round(sparsity, 1)
    return f"{bucket(M)}_{bucket(K)}_{bucket(N)}_{sp_bucket}"


class TileAutotuner:
    """Runtime tile size selection for Acc-SpMM.

    Benchmarks candidate tile configs and caches the best one per
    (bucketed M, K, N, sparsity) combination.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def best_config(
        self,
        M: int,
        K: int,
        N: int,
        sparsity: float,
    ) -> dict[str, Any]:
        """Select the best tile configuration for the given problem size.

        If CUDA is not available or benchmarking fails, returns the default
        config (16x16x16).

        Args:
            M: Number of rows of A / C.
            K: Inner dimension.
            N: Number of columns of B / C.
            sparsity: Fraction of zero tiles (0.0 to 1.0).

        Returns:
            Dict with keys 'tile_m', 'tile_n', 'tile_k'.
        """
        key = _cache_key(M, K, N, sparsity)
        if key in self._cache:
            return self._cache[key]

        if not torch.cuda.is_available():
            result = _config_to_dict(DEFAULT_CONFIG)
            self._cache[key] = result
            return result

        best = self._benchmark(M, K, N, sparsity)
        self._cache[key] = best
        return best

    def _benchmark(
        self,
        M: int,
        K: int,
        N: int,
        sparsity: float,
    ) -> dict[str, Any]:
        """Benchmark candidates and return best config as dict."""
        best_time = float("inf")
        best_cfg = DEFAULT_CONFIG

        for cfg in CANDIDATE_CONFIGS:
            # Skip configs where tiles don't divide dimensions reasonably
            if cfg.tile_m > M or cfg.tile_k > K:
                continue

            elapsed = self._time_config(cfg, M, K, N, sparsity)
            if elapsed < best_time:
                best_time = elapsed
                best_cfg = cfg

        return _config_to_dict(best_cfg)

    def _time_config(
        self,
        cfg: TileConfig,
        M: int,
        K: int,
        N: int,
        sparsity: float,
    ) -> float:
        """Time a single config. Returns median of BENCH_ITERS runs."""
        try:
            from kairo.ops.spmm import block_sparse_mm, _dense_to_block_sparse
            from kairo.types import SparseMask

            device = torch.device("cuda")
            A = torch.randn(M, K, device=device)
            B = torch.randn(K, N, device=device)

            # Create a block mask at the given sparsity
            num_tile_rows = (M + cfg.tile_m - 1) // cfg.tile_m
            num_tile_cols = (K + cfg.tile_k - 1) // cfg.tile_k
            total_tiles = num_tile_rows * num_tile_cols
            nnz = max(1, int(total_tiles * (1.0 - sparsity)))

            perm = torch.randperm(total_tiles, device=device)[:nnz]
            rows = perm // num_tile_cols
            cols = perm % num_tile_cols
            indices = torch.stack([rows, cols]).to(torch.int64)
            values = torch.ones(nnz, device=device)

            mask = SparseMask(
                indices=indices,
                values=values,
                dense_shape=(num_tile_rows, num_tile_cols),
                sparsity_ratio=sparsity,
            )

            # Warmup
            for _ in range(BENCH_WARMUP):
                block_sparse_mm(A, B, mask, tile_size=cfg.tile_m)
                torch.cuda.synchronize()

            # Benchmark
            times: list[float] = []
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                block_sparse_mm(A, B, mask, tile_size=cfg.tile_m)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            times.sort()
            return times[len(times) // 2]  # median

        except Exception as exc:
            _logger.debug("Autotuner config %s failed: %s", cfg, exc)
            return float("inf")

    def clear_cache(self) -> None:
        """Clear the autotuner cache."""
        self._cache.clear()


def _config_to_dict(cfg: TileConfig) -> dict[str, Any]:
    """Convert TileConfig to dict."""
    return {
        "tile_m": cfg.tile_m,
        "tile_n": cfg.tile_n,
        "tile_k": cfg.tile_k,
    }

/**
 * Tile size constants for Acc-SpMM autotuning.
 */
#pragma once

// ---------------------------------------------------------------------------
// Default tile dimensions (used when no autotuner config is provided)
// ---------------------------------------------------------------------------

constexpr int DEFAULT_TILE_M = 16;
constexpr int DEFAULT_TILE_N = 16;
constexpr int DEFAULT_TILE_K = 16;

// Warp size (NVIDIA GPUs)
constexpr int WARP_SIZE = 32;

// Tensor Core fragment dimensions (wmma 16x16x16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ---------------------------------------------------------------------------
// Shared memory sizing
// ---------------------------------------------------------------------------

// Double buffering: 2 tiles in smem at once
constexpr int SMEM_STAGES = 2;

// Max shared memory per block (48 KB typical, 100+ KB on sm_89 with opt-in)
constexpr int MAX_SMEM_BYTES = 48 * 1024;

// ---------------------------------------------------------------------------
// Configurable tile parameters (template-friendly)
// ---------------------------------------------------------------------------

template <int TM = DEFAULT_TILE_M, int TN = DEFAULT_TILE_N,
          int TK = DEFAULT_TILE_K>
struct TileConfig {
    static constexpr int TILE_M = TM;
    static constexpr int TILE_N = TN;
    static constexpr int TILE_K = TK;
    static constexpr int TILE_AREA = TM * TN;
    static constexpr int SMEM_A = TM * (TK + 1);  // +1 for bank-conflict-free
    static constexpr int SMEM_B = TK * (TN + 1);
};

// Pre-instantiated configs for autotuner candidates
using TileConfig16 = TileConfig<16, 16, 16>;
using TileConfig32 = TileConfig<32, 32, 16>;
using TileConfig64 = TileConfig<64, 64, 16>;

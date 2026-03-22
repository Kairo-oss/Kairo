/**
 * Common macros, error checking, and type aliases for Kairo CUDA kernels.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Floating-type dispatch (float / half)
// ---------------------------------------------------------------------------

#define KAIRO_DISPATCH_FLOATING_TYPES(DTYPE, NAME, ...)                        \
    [&] {                                                                      \
        if ((DTYPE) == at::ScalarType::Float) {                                \
            using scalar_t = float;                                            \
            return __VA_ARGS__();                                              \
        } else if ((DTYPE) == at::ScalarType::Half) {                          \
            using scalar_t = __half;                                           \
            return __VA_ARGS__();                                              \
        } else {                                                               \
            AT_ERROR(NAME, " not implemented for dtype ", toString(DTYPE));     \
        }                                                                      \
    }()

// ---------------------------------------------------------------------------
// Thread / block helpers
// ---------------------------------------------------------------------------

constexpr int KAIRO_WARP_SIZE = 32;
constexpr int KAIRO_MAX_THREADS = 1024;
constexpr int KAIRO_DEFAULT_BLOCK = 256;

inline int kairo_grid_size(int total, int block) {
    return (total + block - 1) / block;
}

// Ceiling division
template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Shared-memory bank-conflict-free index (32-bank, +1 padding)
// ---------------------------------------------------------------------------

template <int BANKS = 32>
__device__ inline int bank_free_idx(int row, int col, int ncols) {
    return row * (ncols + 1) + col;
}

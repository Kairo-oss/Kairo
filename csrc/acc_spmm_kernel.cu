/**
 * Block-sparse tiled matrix multiplication (Acc-SpMM) CUDA kernels.
 *
 * Two paths:
 *   1. Tensor Core (wmma): FP16 input → FP32 accumulate → FP32 output
 *   2. FP32 fallback: standard tiled matmul for non-TC hardware
 *
 * Tile layout: A is stored as a flat array of non-zero tiles, each of size
 * (TILE_M x TILE_K). Metadata arrays (tile_row_ids, tile_col_ids) map each
 * tile to its position in the logical sparse matrix.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "include/common.cuh"
#include "include/tile_config.cuh"

// wmma is available on sm_70+ (Volta and later).
// We compile for sm_89, so wmma is always available at compile time.
// The __CUDA_ARCH__ guard is only valid in device code, so we use
// a simple compile-time flag based on our known target architecture.
#include <mma.h>
using namespace nvcuda;
#define KAIRO_HAS_WMMA 1

// ---------------------------------------------------------------------------
// Tensor Core kernel (FP16 → FP32)
// ---------------------------------------------------------------------------

#if KAIRO_HAS_WMMA

__global__ void acc_spmm_wmma_kernel(
    const __half* __restrict__ A_tiles,    // (num_tiles, TILE_M, TILE_K) flat
    const int* __restrict__ tile_row_ids,  // (num_tiles,)
    const int* __restrict__ tile_col_ids,  // (num_tiles,)
    const __half* __restrict__ B,          // (K, N) row-major
    float* __restrict__ C,                 // (M, N) row-major
    int M, int K, int N, int num_tiles)
{
    // Each block handles one output tile (WMMA_M x WMMA_N)
    // We iterate over all A-tiles that contribute to this output row-block
    // and accumulate via wmma

    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int a_row = tile_row_ids[tile_idx];  // logical row-block index
    int a_col = tile_col_ids[tile_idx];  // logical col-block index (= B row-block)

    // Pointer to this A tile
    const __half* A_tile_ptr = A_tiles + tile_idx * WMMA_M * WMMA_K;

    // Declare wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // We process WMMA_N columns of output at a time via warp index
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Load A fragment (same for all N-chunks)
    wmma::load_matrix_sync(a_frag, A_tile_ptr, WMMA_K);

    // Iterate over N-dimension in chunks of WMMA_N
    for (int n_start = warp_id * WMMA_N; n_start < N;
         n_start += num_warps * WMMA_N) {
        if (n_start + WMMA_N > N) break;  // skip partial tiles at boundary

        // Load B fragment: rows [a_col*TILE_K .. a_col*TILE_K+TILE_K), cols [n_start..n_start+WMMA_N)
        int b_row_offset = a_col * WMMA_K;
        const __half* B_ptr = B + b_row_offset * N + n_start;
        wmma::load_matrix_sync(b_frag, B_ptr, N);

        // Zero accumulator
        wmma::fill_fragment(c_frag, 0.0f);

        // MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Accumulate into global C with atomicAdd (multiple A-tiles may
        // contribute to the same output row-block)
        // Store to shared memory first, then atomicAdd to global
        __shared__ float c_shared[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(c_shared, c_frag, WMMA_N, wmma::mem_row_major);

        __syncthreads();

        // AtomicAdd to global output
        int c_row_offset = a_row * WMMA_M;
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_N;
             idx += blockDim.x) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            int global_row = c_row_offset + r;
            int global_col = n_start + c;
            if (global_row < M && global_col < N) {
                atomicAdd(&C[global_row * N + global_col], c_shared[idx]);
            }
        }

        __syncthreads();
    }
}

#endif  // KAIRO_HAS_WMMA

// ---------------------------------------------------------------------------
// FP32 fallback kernel
// ---------------------------------------------------------------------------

__global__ void acc_spmm_fp32_fallback_kernel(
    const float* __restrict__ A_tiles,     // (num_tiles, TILE_M, TILE_K) flat
    const int* __restrict__ tile_row_ids,  // (num_tiles,)
    const int* __restrict__ tile_col_ids,  // (num_tiles,)
    const float* __restrict__ B,           // (K, N) row-major
    float* __restrict__ C,                 // (M, N) row-major
    int M, int K, int N, int num_tiles,
    int tile_m, int tile_k)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int a_row = tile_row_ids[tile_idx];
    int a_col = tile_col_ids[tile_idx];

    const float* A_ptr = A_tiles + tile_idx * tile_m * tile_k;

    int c_row_base = a_row * tile_m;
    int b_row_base = a_col * tile_k;

    // Each thread computes one element of the partial C tile
    for (int idx = threadIdx.x; idx < tile_m * N; idx += blockDim.x) {
        int local_r = idx / N;
        int global_c = idx % N;
        int global_r = c_row_base + local_r;
        if (global_r >= M || global_c >= N) continue;

        float sum = 0.0f;
        for (int k = 0; k < tile_k; k++) {
            int b_row = b_row_base + k;
            if (b_row < K) {
                sum += A_ptr[local_r * tile_k + k] * B[b_row * N + global_c];
            }
        }

        atomicAdd(&C[global_r * N + global_c], sum);
    }
}

// ---------------------------------------------------------------------------
// C++ wrapper functions
// ---------------------------------------------------------------------------

torch::Tensor acc_spmm_forward_cuda(
    const torch::Tensor& A_tiles,
    const torch::Tensor& tile_row_ids,
    const torch::Tensor& tile_col_ids,
    const torch::Tensor& B,
    int M, int K, int N, int num_tiles,
    int tile_m, int tile_k)
{
    TORCH_CHECK(A_tiles.is_cuda(), "A_tiles must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");

    auto C = torch::zeros({M, N}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(B.device()));

    if (num_tiles == 0) return C;

    int block_size = KAIRO_DEFAULT_BLOCK;

#if KAIRO_HAS_WMMA
    if (A_tiles.scalar_type() == torch::kFloat16
        && tile_m == WMMA_M && tile_k == WMMA_K) {
        // Use Tensor Core path
        acc_spmm_wmma_kernel<<<num_tiles, block_size>>>(
            reinterpret_cast<const __half*>(A_tiles.data_ptr<at::Half>()),
            tile_row_ids.data_ptr<int>(),
            tile_col_ids.data_ptr<int>(),
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
            C.data_ptr<float>(),
            M, K, N, num_tiles);
    } else
#endif
    {
        // FP32 fallback
        TORCH_CHECK(A_tiles.scalar_type() == torch::kFloat32,
                     "FP32 fallback requires float32 A_tiles");
        TORCH_CHECK(B.scalar_type() == torch::kFloat32,
                     "FP32 fallback requires float32 B");

        acc_spmm_fp32_fallback_kernel<<<num_tiles, block_size>>>(
            A_tiles.data_ptr<float>(),
            tile_row_ids.data_ptr<int>(),
            tile_col_ids.data_ptr<int>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N, num_tiles, tile_m, tile_k);
    }

    CUDA_CHECK(cudaGetLastError());
    return C;
}

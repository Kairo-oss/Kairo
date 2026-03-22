/**
 * Sparse embedding gather (forward) and scatter (backward) CUDA kernels.
 *
 * Forward: For each batch ID, look up the embedding row, but only read
 *          columns where the sparse mask is active (COO → CSR row offsets).
 * Backward: Scatter grad_output rows into grad_weight only at active IDs.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "include/common.cuh"

// ---------------------------------------------------------------------------
// Forward kernel: sparse gather
// ---------------------------------------------------------------------------

__global__ void sparse_gather_forward_kernel(
    const float* __restrict__ weight,       // (N, D)
    const int64_t* __restrict__ mask_rows,  // (nnz,)
    const int64_t* __restrict__ mask_cols,  // (nnz,)
    const int64_t* __restrict__ batch_ids,  // (B,)
    const int64_t* __restrict__ row_offsets, // (N+1,)
    float* __restrict__ output,             // (B, D)
    int N, int D, int nnz, int B)
{
    int bid = blockIdx.x;  // one block per batch element
    if (bid >= B) return;

    int64_t row_id = batch_ids[bid];
    if (row_id < 0 || row_id >= N) return;

    // Find active columns for this row via CSR offsets
    int64_t start = row_offsets[row_id];
    int64_t end   = row_offsets[row_id + 1];
    int num_active = static_cast<int>(end - start);

    // Each thread strides across embedding dim
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = 0.0f;
        // Accumulate only active columns
        for (int k = 0; k < num_active; k++) {
            int64_t col = mask_cols[start + k];
            if (col == d) {
                val = weight[row_id * D + d];
                break;
            }
        }
        output[bid * D + d] = val;
    }
}

// Optimized path: when mask selects ALL columns for a row, do a dense copy
__global__ void sparse_gather_dense_row_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ batch_ids,
    const int64_t* __restrict__ row_offsets,
    const int64_t* __restrict__ mask_cols,
    float* __restrict__ output,
    int N, int D, int B)
{
    int bid = blockIdx.x;
    if (bid >= B) return;

    int64_t row_id = batch_ids[bid];
    if (row_id < 0 || row_id >= N) return;

    int64_t start = row_offsets[row_id];
    int64_t end   = row_offsets[row_id + 1];
    int num_active = static_cast<int>(end - start);

    // Build a dense mask in shared memory then multiply
    extern __shared__ char smem_raw[];
    bool* col_active = reinterpret_cast<bool*>(smem_raw);

    // Initialize shared memory
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        col_active[d] = false;
    }
    __syncthreads();

    // Mark active columns
    for (int k = threadIdx.x; k < num_active; k += blockDim.x) {
        int64_t col = mask_cols[start + k];
        if (col >= 0 && col < D) {
            col_active[col] = true;
        }
    }
    __syncthreads();

    // Gather with mask
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = col_active[d] ? weight[row_id * D + d] : 0.0f;
        output[bid * D + d] = val;
    }
}

// ---------------------------------------------------------------------------
// Backward kernel: sparse scatter
// ---------------------------------------------------------------------------

__global__ void sparse_scatter_backward_kernel(
    const float* __restrict__ grad_output,  // (B, D)
    const int64_t* __restrict__ active_ids, // (S,) sorted unique row IDs
    const int64_t* __restrict__ batch_ids,  // (B,)
    float* __restrict__ grad_weight,        // (N, D)
    int N, int D, int S, int B)
{
    // Grid: one block per (active_id, dim_chunk) pair
    int aid_idx = blockIdx.x;
    if (aid_idx >= S) return;

    int64_t active_row = active_ids[aid_idx];

    // Each thread handles a slice of embedding_dim
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float grad_sum = 0.0f;

        // Sum gradients from all batch elements that reference this row
        for (int b = 0; b < B; b++) {
            if (batch_ids[b] == active_row) {
                grad_sum += grad_output[b * D + d];
            }
        }

        if (grad_sum != 0.0f) {
            atomicAdd(&grad_weight[active_row * D + d], grad_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// C++ wrapper functions (called from bindings.cpp)
// ---------------------------------------------------------------------------

torch::Tensor sparse_gather_forward_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& mask_rows,
    const torch::Tensor& mask_cols,
    const torch::Tensor& batch_ids,
    const torch::Tensor& row_offsets)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(batch_ids.is_cuda(), "batch_ids must be on CUDA");

    int N = weight.size(0);
    int D = weight.size(1);
    int B = batch_ids.size(0);
    int nnz = mask_rows.size(0);

    auto output = torch::zeros({B, D}, weight.options());

    if (B == 0) return output;

    int block_size = std::min(D, KAIRO_MAX_THREADS);
    block_size = std::max(block_size, KAIRO_WARP_SIZE);
    // Round up to warp size
    block_size = ((block_size + KAIRO_WARP_SIZE - 1) / KAIRO_WARP_SIZE)
                 * KAIRO_WARP_SIZE;
    block_size = std::min(block_size, KAIRO_MAX_THREADS);

    // Use shared-memory path when D is manageable
    if (D <= 4096) {
        size_t smem_bytes = D * sizeof(bool);
        sparse_gather_dense_row_kernel<<<B, block_size, smem_bytes>>>(
            weight.data_ptr<float>(),
            batch_ids.data_ptr<int64_t>(),
            row_offsets.data_ptr<int64_t>(),
            mask_cols.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            N, D, B);
    } else {
        sparse_gather_forward_kernel<<<B, block_size>>>(
            weight.data_ptr<float>(),
            mask_rows.data_ptr<int64_t>(),
            mask_cols.data_ptr<int64_t>(),
            batch_ids.data_ptr<int64_t>(),
            row_offsets.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            N, D, nnz, B);
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor sparse_scatter_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& active_ids,
    const torch::Tensor& batch_ids,
    int N, int D)
{
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
    TORCH_CHECK(active_ids.is_cuda(), "active_ids must be on CUDA");
    TORCH_CHECK(batch_ids.is_cuda(), "batch_ids must be on CUDA");

    int S = active_ids.size(0);
    int B = batch_ids.size(0);

    auto grad_weight = torch::zeros({N, D}, grad_output.options());

    if (S == 0 || B == 0) return grad_weight;

    int block_size = std::min(D, KAIRO_MAX_THREADS);
    block_size = std::max(block_size, KAIRO_WARP_SIZE);
    block_size = ((block_size + KAIRO_WARP_SIZE - 1) / KAIRO_WARP_SIZE)
                 * KAIRO_WARP_SIZE;
    block_size = std::min(block_size, KAIRO_MAX_THREADS);

    sparse_scatter_backward_kernel<<<S, block_size>>>(
        grad_output.data_ptr<float>(),
        active_ids.data_ptr<int64_t>(),
        batch_ids.data_ptr<int64_t>(),
        grad_weight.data_ptr<float>(),
        N, D, S, B);

    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

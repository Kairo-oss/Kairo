/**
 * pybind11 / PyTorch extension bindings for Kairo CUDA kernels.
 *
 * Exposes:
 *   - sparse_gather_forward(weight, mask_rows, mask_cols, batch_ids, row_offsets)
 *   - sparse_scatter_backward(grad_output, active_ids, batch_ids, N, D)
 *   - acc_spmm_forward(A_tiles, tile_rows, tile_cols, B, M, K, N, num_tiles,
 *                       tile_m, tile_k)
 */

#include <torch/extension.h>

// Forward declarations (implemented in .cu files)
torch::Tensor sparse_gather_forward_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& mask_rows,
    const torch::Tensor& mask_cols,
    const torch::Tensor& batch_ids,
    const torch::Tensor& row_offsets);

torch::Tensor sparse_scatter_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& active_ids,
    const torch::Tensor& batch_ids,
    int N, int D);

torch::Tensor acc_spmm_forward_cuda(
    const torch::Tensor& A_tiles,
    const torch::Tensor& tile_row_ids,
    const torch::Tensor& tile_col_ids,
    const torch::Tensor& B,
    int M, int K, int N, int num_tiles,
    int tile_m, int tile_k);

// ---------------------------------------------------------------------------
// Input validation wrappers
// ---------------------------------------------------------------------------

torch::Tensor sparse_gather_forward(
    const torch::Tensor& weight,
    const torch::Tensor& mask_rows,
    const torch::Tensor& mask_cols,
    const torch::Tensor& batch_ids,
    const torch::Tensor& row_offsets)
{
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(mask_rows.is_contiguous(), "mask_rows must be contiguous");
    TORCH_CHECK(mask_cols.is_contiguous(), "mask_cols must be contiguous");
    TORCH_CHECK(batch_ids.is_contiguous(), "batch_ids must be contiguous");
    TORCH_CHECK(row_offsets.is_contiguous(), "row_offsets must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32,
                "weight must be float32");
    TORCH_CHECK(mask_rows.dtype() == torch::kInt64,
                "mask_rows must be int64");
    TORCH_CHECK(mask_cols.dtype() == torch::kInt64,
                "mask_cols must be int64");
    TORCH_CHECK(batch_ids.dtype() == torch::kInt64,
                "batch_ids must be int64");
    return sparse_gather_forward_cuda(weight, mask_rows, mask_cols,
                                      batch_ids, row_offsets);
}

torch::Tensor sparse_scatter_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& active_ids,
    const torch::Tensor& batch_ids,
    int N, int D)
{
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(active_ids.is_contiguous(), "active_ids must be contiguous");
    TORCH_CHECK(batch_ids.is_contiguous(), "batch_ids must be contiguous");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32,
                "grad_output must be float32");
    TORCH_CHECK(active_ids.dtype() == torch::kInt64,
                "active_ids must be int64");
    TORCH_CHECK(batch_ids.dtype() == torch::kInt64,
                "batch_ids must be int64");
    TORCH_CHECK(N > 0, "N must be positive");
    TORCH_CHECK(D > 0, "D must be positive");
    return sparse_scatter_backward_cuda(grad_output, active_ids, batch_ids,
                                        N, D);
}

torch::Tensor acc_spmm_forward(
    const torch::Tensor& A_tiles,
    const torch::Tensor& tile_row_ids,
    const torch::Tensor& tile_col_ids,
    const torch::Tensor& B,
    int M, int K, int N, int num_tiles,
    int tile_m, int tile_k)
{
    TORCH_CHECK(A_tiles.is_contiguous(), "A_tiles must be contiguous");
    TORCH_CHECK(tile_row_ids.is_contiguous(), "tile_row_ids must be contiguous");
    TORCH_CHECK(tile_col_ids.is_contiguous(), "tile_col_ids must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(tile_row_ids.dtype() == torch::kInt32,
                "tile_row_ids must be int32");
    TORCH_CHECK(tile_col_ids.dtype() == torch::kInt32,
                "tile_col_ids must be int32");
    TORCH_CHECK(M > 0 && K > 0 && N > 0, "dimensions must be positive");
    return acc_spmm_forward_cuda(A_tiles, tile_row_ids, tile_col_ids, B,
                                  M, K, N, num_tiles, tile_m, tile_k);
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kairo CUDA kernels for sparse embedding and block-sparse matmul";
    m.def("sparse_gather_forward", &sparse_gather_forward,
          "Sparse embedding gather forward (CUDA)",
          py::arg("weight"), py::arg("mask_rows"), py::arg("mask_cols"),
          py::arg("batch_ids"), py::arg("row_offsets"));
    m.def("sparse_scatter_backward", &sparse_scatter_backward,
          "Sparse embedding scatter backward (CUDA)",
          py::arg("grad_output"), py::arg("active_ids"),
          py::arg("batch_ids"), py::arg("N"), py::arg("D"));
    m.def("acc_spmm_forward", &acc_spmm_forward,
          "Block-sparse tiled matmul forward (CUDA)",
          py::arg("A_tiles"), py::arg("tile_row_ids"),
          py::arg("tile_col_ids"), py::arg("B"),
          py::arg("M"), py::arg("K"), py::arg("N"),
          py::arg("num_tiles"), py::arg("tile_m"), py::arg("tile_k"));
}

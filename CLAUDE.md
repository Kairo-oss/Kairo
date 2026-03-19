# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kairo is an end-to-end sparse computation framework for trillion-parameter recommendation systems (DLRM). It optimizes across storage, computation, and scheduling through four engines: Storage & Representation, Interaction, Training & Compute, and Scheduling & System.

Current status: **MVP complete** (NMF initialization + SparseEmbeddingTable). Next: full Phase 1 (SparseRec, AGENT optimizer, CUDA kernels).

## Environment Setup

```bash
conda create -n kairo python=3.10 -y
conda activate kairo
pip install -e ".[dev,test]"
```

Target server: 2x RTX 4090. When CUDA is available, add `"cuda"` to the `device` fixture params in `tests/conftest.py`.

## Commands

```bash
# Run all tests with coverage
conda run -n kairo python -m pytest tests/ -v --cov=kairo --cov-report=term-missing

# Run a single test file
conda run -n kairo python -m pytest tests/storage/test_embedding.py -v

# Run a single test
conda run -n kairo python -m pytest tests/storage/test_nmf_factorizer.py::TestNMFDecompose::test_non_negativity -v

# CUDA-only tests (when GPU available)
conda run -n kairo python -m pytest tests/ -m cuda -v

# Lint
conda run -n kairo ruff check kairo/

# Type check
conda run -n kairo pyright kairo/

# Run example
conda run -n kairo python examples/synthetic_sparse_embedding.py
```

## Architecture

### Data Flow Pipeline

```
Interaction Matrix → nmf_decompose() → NMFResult → generate_sparse_mask() → SparseMask → SparseEmbeddingTable
```

### Core Types (`kairo/types.py`)

All domain types are **frozen dataclasses** (immutable). This is a strict project invariant.

- `SparseMask`: COO-format binary mask with `indices` (2, nnz), `values` (nnz,), `dense_shape`, `sparsity_ratio`
- `NMFResult`: factorization output with `W`, `H`, `reconstruction_error`, `n_iterations`
- `EmbeddingConfig`: table configuration with validation in `__post_init__`

### Config Types (`kairo/config.py`)

Algorithm configs are also frozen dataclasses with validation:
- `NMFConfig`: rank, max_iter, tol, seed

### Immutability Contract

Functions that transform state **must return new objects**, never mutate inputs:
- `SparseEmbeddingTable.with_mask()` returns a new table instance
- `nmf_decompose()` does not modify the input matrix
- All `SparseMask` / `NMFResult` instances are frozen — tests verify this

### Module Organization (by engine)

```
kairo/storage/          # Engine 1: Storage & Representation
kairo/interaction/      # Engine 2: Interaction (Phase 2 — INFNet, HSTU)
kairo/training/         # Engine 3: Training & Compute (Phase 1 next — SparseRec, AGENT)
kairo/scheduling/       # Engine 4: Scheduling & System (Phase 3 — RecMG, RecOS, Prism)
kairo/ops/              # C++/CUDA kernel bindings (Phase 1 next)
```

Only `kairo/storage/` is populated in MVP. Create engine directories only when implementing them.

### `SparseEmbeddingTable` Design

- Stores full weight matrix as `nn.Parameter`, applies mask in `forward()`
- Dense mask is pre-computed via `register_buffer("_dense_mask", ...)` for efficiency
- Access `_dense_mask` via `getattr(self, "_dense_mask", None)` to satisfy pyright (registered buffers have type `Module` otherwise)

## Development Conventions

- **TDD mandatory**: write tests first (RED), implement (GREEN), then refactor
- **Test coverage**: 80%+ required. Currently at 93%
- **File size**: 200–400 lines typical, 800 max
- **Commit format**: `<type>: <description>` (feat, fix, refactor, docs, test, chore, perf, ci)
- **Sparse format**: COO is default (`indices` shape `(2, nnz)`). CSR support planned
- Tests mirror source structure: `tests/storage/` ↔ `kairo/storage/`
- Shared fixtures in `tests/conftest.py`: `device`, `seed`, `rng`, `small_interaction_matrix`, `default_nmf_config`, `default_embedding_config`

## Technical Approach (Four Engines)

### Engine 1: Storage & Representation — Semantic-Enhanced Memory Pool
- **LEAD (Learning to Collide)**: trains hash mapping by access frequency + semantic similarity → semantically similar IDs share representation space → 80%+ storage reduction
- **MSN (Memory Scaling Network)**: Product Key-based parameterized memory pool → Memory Gating activates tiny fraction at inference → decouples capacity from compute
- **NMF Data-Driven Init**: factorizes interaction matrix → initializes sparse mask at local optimum from day one

### Engine 2: Interaction — Task-Aware Dual-Stream
- **INFNet**: Task Proxy Tokens for MTL → cross-attention between feature tokens and K proxy tokens → O(N·K) instead of O(N²) → suppresses negative transfer
- **HSTU**: Pointwise Aggregated Attention replaces self-attention → 5.3–15.2x faster than FlashAttention2 on 8192-length sequences

### Engine 3: Training & Compute — Bi-Directional Sparse Optimization
- **SparseRec**: Multinomial Sampling selects active IDs → gradients computed only for subset → constant sparsity in both forward and backward → eliminates dense gradient memory
- **AGENT**: historical momentum correlation corrects gradient direction at high sparsity (99%) → +5.0% accuracy or 52.1% faster convergence

### Engine 4: Scheduling & System — Software-Hardware Codesign
- **RecMG**: dual ML models (cache eviction + prefetch prediction) for GPU HBM/DRAM/SSD tiered storage → 2.8x fewer on-demand fetches, 43% inference speedup
- **RecOS**: async tensor management with cross-CUDA-Stream inter-op parallelization + operator fusion → 68% latency reduction at peak
- **Prism**: CPU-intensive (Embedding Lookup) and GPU-intensive (Attention/MLP) subgraphs deployed on separate physical nodes → RDMA elastic scaling

## Roadmap & Progress

### Phase 1: Foundation — `In Progress`

| Component | Location | Status | Description |
|-----------|----------|--------|-------------|
| NMF Init | `kairo/storage/nmf/` | **Done** | Multiplicative update NMF → sparse mask generation |
| SparseEmbeddingTable | `kairo/storage/embedding.py` | **Done** | COO-masked embedding with immutable `with_mask()` |
| SparseRec | `kairo/training/sparse_rec/` | Planned | Multinomial sampler, sparse gradient autograd Function, mask grow/prune, cumulative gradient regrowth |
| AGENT Optimizer | `kairo/training/agent_optim/` | Planned | `torch.optim.Optimizer` subclass with momentum correction |
| Acc-SpMM Kernel | `csrc/acc_spmm_kernel.cu` | Planned | Block-sparse tiled matmul with Tensor Core support, build via `torch.utils.cpp_extension.CUDAExtension` |
| Sparse Embedding Kernel | `csrc/sparse_embedding_kernel.cu` | Planned | CUDA gather (forward) and scatter (backward) for masked embeddings |
| SparseTrainer | `kairo/training/trainer.py` | Planned | Training loop: forward → sparse backward → AGENT step → periodic prune/grow |
| Benchmarks | `benchmarks/` | Planned | Performance regression tracking with pytest-benchmark |

### Phase 2: Architecture — `Planned`
- `kairo/interaction/infnet/` — INFNet task proxy mechanism for unified multi-modal features
- `kairo/interaction/hstu/` — HSTU as core sequence modeling unit for ultra-long user behavior

### Phase 3: Ecosystem — `Planned`
- `kairo/scheduling/recmg/` — ML-guided tiered storage across HBM/NVMe
- `kairo/scheduling/recos/` — Dynamic graph scheduling for heterogeneous hardware (A800/RTX 4090)
- `kairo/scheduling/prism/` — CPU/GPU resource disaggregation with RDMA

### Phase 4: Evolution — `Planned`
- PEL-NAS — LLM-driven architecture search with complexity partitioning, generating optimal Kairo variants per device (edge NPU or cloud GPU) in minutes

## Comparison with SOTA

| Dimension | Traditional (Monolith/DLRMv2) | Kairo | Gain |
|-----------|-------------------------------|-------|------|
| Collision | Collision-free hash (linear storage growth) | LEAD (learned collision) | 80%+ storage reduction |
| Gradients | Sparse weights + dense gradients | SparseRec (bi-directional sparse) | 70% memory reduction |
| Interaction | Full-connected O(N²) | Proxy tokens O(N·K) | 2x performance |
| Storage | Static cache / LRU | RecMG (ML-guided prefetch) | 2.8x fewer fetches |
| Sequences | Transformer (quadratic) | HSTU (linear pointwise) | 10x+ speedup |
| Scheduling | Static graph (serial/blocking) | RecOS + Prism (dynamic + disaggregated) | 68% latency reduction |

## Research References

Technical specification and research survey documents:
`/Users/hayden/Projects/obsidian_prj/Kairo/`

Key papers:
- [LEAD](https://arxiv.org/abs/2203.15837) — Learning to Collide: semantic hash mapping for embedding compression
- [MSN](https://arxiv.org/abs/2602.07526) — Memory Scaling Network: product-key dynamic parameter activation
- [SparseRec](https://link.springer.com/article/10.1007/s41019-025-00327-5) — selective gradient computation via multinomial sampling
- [AGENT](https://arxiv.org/abs/2301.03573) — adaptive gradient correction with historical momentum
- [HSTU](https://arxiv.org/abs/2402.17152) — pointwise aggregated attention (5.3–15.2x faster than FlashAttention2)
- [INFNet](https://arxiv.org/abs/2508.11565) — task proxy tokens reducing O(N²) to O(N·K)
- [RecMG](https://arxiv.org/abs/2306.00103) — ML-guided tiered storage management
- [Acc-SpMM](https://arxiv.org/abs/2501.09251) — GPU Tensor Core accelerated sparse matrix multiplication

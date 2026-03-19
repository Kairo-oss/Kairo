"""End-to-end demo: NMF decomposition -> sparse mask -> SparseEmbeddingTable.

Demonstrates the core Kairo MVP pipeline on synthetic data.
"""

from __future__ import annotations

import torch

from kairo.config import NMFConfig
from kairo.storage import SparseEmbeddingTable
from kairo.storage.nmf import generate_sparse_mask, nmf_decompose
from kairo.types import EmbeddingConfig


def main() -> None:
    print("=" * 60)
    print("Kairo MVP Demo: Sparse Embedding via NMF Initialization")
    print("=" * 60)

    # 1. Generate synthetic interaction matrix (users x items)
    num_users, num_items = 500, 200
    true_rank = 10
    rng = torch.Generator()
    rng.manual_seed(42)

    W_true = torch.abs(torch.randn(num_users, true_rank, generator=rng))
    H_true = torch.abs(torch.randn(true_rank, num_items, generator=rng))
    interaction_matrix = W_true @ H_true

    print(f"\nInteraction matrix: {num_users} users x {num_items} items")
    print(f"True rank: {true_rank}")

    # 2. NMF decomposition
    nmf_config = NMFConfig(rank=true_rank, max_iter=200, tol=1e-5, seed=42)
    result = nmf_decompose(interaction_matrix, nmf_config)

    relative_error = torch.norm(interaction_matrix - result.W @ result.H) / torch.norm(
        interaction_matrix
    )
    print(f"\nNMF converged in {result.n_iterations} iterations")
    print(f"Reconstruction error: {result.reconstruction_error:.4f}")
    print(f"Relative error: {relative_error:.6f}")

    # 3. Generate sparse masks at various sparsity ratios
    num_embeddings, embedding_dim = 1000, 128
    sparsity_ratios = [0.5, 0.8, 0.95]

    print(f"\nEmbedding table: {num_embeddings} x {embedding_dim}")
    print(f"Total parameters: {num_embeddings * embedding_dim:,}")
    print()

    for ratio in sparsity_ratios:
        mask = generate_sparse_mask(result, num_embeddings, embedding_dim, ratio)

        emb_config = EmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_ratio=ratio,
        )
        table = SparseEmbeddingTable(emb_config, mask=mask)

        # Forward pass
        batch_ids = torch.randint(0, num_embeddings, (32,))
        output = table(batch_ids)

        print(f"Sparsity {ratio:.0%}:")
        print(f"  Active parameters: {table.active_parameter_count:,}")
        print(f"  Compression ratio: {table.compression_ratio:.1%}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Output norm (mean): {output.norm(dim=-1).mean():.4f}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()

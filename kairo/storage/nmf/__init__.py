"""NMF-based data-driven initialization for sparse embedding masks."""

from kairo.storage.nmf.factorizer import nmf_decompose
from kairo.storage.nmf.mask_init import generate_sparse_mask

__all__ = ["nmf_decompose", "generate_sparse_mask"]

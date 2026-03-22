"""Build script for Kairo CUDA extension kernels.

Usage:
    pip install -e . --no-build-isolation   # builds kairo._C alongside kairo
    python setup_cuda.py install            # standalone build

Requires: PyTorch with CUDA support, nvcc matching torch.version.cuda.
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="kairo_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "kairo._C",
            sources=[
                "csrc/sparse_embedding_kernel.cu",
                "csrc/acc_spmm_kernel.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_89",
                    "-lineinfo",
                    "--ptxas-options=-v",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

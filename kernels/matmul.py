"""
Triton Matrix Multiplication Kernel

This module implements a high-performance matrix multiplication kernel using OpenAI Triton.
The kernel uses tiling and L2 cache optimization to achieve performance comparable to cuBLAS.

Key concepts:
- Tiling: Divide matrices into blocks that fit in SRAM
- Block pointer arithmetic: Efficiently compute memory addresses for blocks
- L2 cache optimization: Group blocks to improve cache hit rate
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from utils.config import MATMUL_GROUP_SIZE_M
from utils.triton_helpers import TRITON_AVAILABLE, TritonKernelStub, require_triton, tl, triton

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:

    def get_autotune_configs():
        """Get autotuning configurations for different GPU architectures."""
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
        ]

    @triton.jit
    def _matmul_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Core matmul computation, shared by autotuned and manual kernels.

        Compute C = A @ B where:
        - A is (M, K)
        - B is (K, N)
        - C is (M, N)

        Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N block of C.
        """
        pid = tl.program_id(axis=0)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.autotune(
        configs=get_autotune_configs(),
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel_autotuned(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        _matmul_body(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
        )

    @triton.jit
    def _matmul_kernel_manual(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        _matmul_body(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
        )

else:

    def get_autotune_configs():
        return []

    _matmul_kernel_autotuned = TritonKernelStub()
    _matmul_kernel_manual = TritonKernelStub()


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Triton matrix multiplication: C = A @ B

    Args:
        a: Input matrix A of shape (M, K), dtype float16, float32, or bfloat16
        b: Input matrix B of shape (K, N), dtype float16, float32, or bfloat16
        block_m: Block size for M dimension (optional, uses autotune if None)
        block_n: Block size for N dimension (optional, uses autotune if None)
        block_k: Block size for K dimension (optional, uses autotune if None)
        use_autotune: Whether to use autotuning (ignored if block sizes provided)

    Returns:
        Output matrix C of shape (M, N), with dtype matching the kernel compute dtype.
        Float16/bfloat16 inputs preserve their dtype; float32 inputs are downcast to float16.

    Raises:
        ValueError: If tensors are not 2D/CUDA/on the same device, dimensions are incompatible, or block sizes are invalid
        TypeError: If input dtypes are unsupported or inconsistent
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got a.dim()={a.dim()}, b.dim()={b.dim()}")

    supported_dtypes = (torch.float16, torch.float32, torch.bfloat16)
    if a.dtype not in supported_dtypes or b.dtype not in supported_dtypes:
        raise TypeError(
            "Unsupported dtype for matmul. "
            f"Supported dtypes: {supported_dtypes}. Got a={a.dtype}, b={b.dtype}."
        )
    if a.dtype != b.dtype:
        raise TypeError(f"Input dtypes for matmul must match. Got a={a.dtype}, b={b.dtype}.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Triton matmul requires CUDA tensors for both inputs.")
    if a.device != b.device:
        raise ValueError(
            f"Input tensors must be on the same device. Got a={a.device}, b={b.device}."
        )

    require_triton()

    M, K = a.shape
    K2, N = b.shape

    if K != K2:
        raise ValueError(f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K2}, {N})")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    compute_dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else torch.float16
    if a.dtype != compute_dtype:
        a = a.to(compute_dtype)
    if b.dtype != compute_dtype:
        b = b.to(compute_dtype)

    c = torch.empty((M, N), device=a.device, dtype=compute_dtype)

    manual_blocks = block_m is not None and block_n is not None and block_k is not None

    if not manual_blocks and not use_autotune:
        raise ValueError("Manual block sizes are required when use_autotune=False")

    if manual_blocks:
        if block_m is None or block_n is None or block_k is None:
            raise ValueError(
                "All block sizes (block_m, block_n, block_k) must be specified for manual mode"
            )
        if block_m <= 0 or block_n <= 0 or block_k <= 0:
            raise ValueError(f"Block sizes must be positive, got ({block_m}, {block_n}, {block_k})")
        if block_m > M or block_n > N or block_k > K:
            raise ValueError(
                "Block sizes must not exceed matrix dimensions. "
                f"Got blocks=({block_m}, {block_n}, {block_k}), dims=({M}, {N}, {K})."
            )

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

        _matmul_kernel_manual[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=MATMUL_GROUP_SIZE_M,
        )

        logger.debug(
            f"Matmul kernel (manual): ({M}, {K}) @ ({K}, {N}) -> ({M}, {N}), "
            f"blocks=({block_m}, {block_n}, {block_k})"
        )
    else:

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

        _matmul_kernel_autotuned[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )

        logger.debug(f"Matmul kernel (autotuned): ({M}, {K}) @ ({K}, {N}) -> ({M}, {N})")

    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    triton_output = triton_matmul(a, b)
    torch_output = torch.matmul(a, b)

    max_diff = (triton_output - torch_output).abs().max().item()
    print(f"Max difference: {max_diff}")

    if max_diff < 1e-2:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

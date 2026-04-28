"""
Persistent Kernels: Thread-block Persistent Implementations

This module provides persistent kernel implementations that keep thread blocks
alive across multiple tiles, amortizing launch overhead. Educational focus:
teach students about occupancy/memory trade-offs in GPU programming.

Key concepts:
- Persistent threads: Thread blocks don't exit after one tile, they iterate
- Work distribution: Grid-stride loops or atomic work queues
- Trade-offs: Lower launch overhead vs. potential load imbalance
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:
    TRITON_AVAILABLE = False
    triton = SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y)
    tl = SimpleNamespace(constexpr=object())


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise ModuleNotFoundError("triton is required to run persistent kernels.")


if TRITON_AVAILABLE:

    @triton.jit
    def _persistent_matmul_kernel(
        A,
        B,
        C,
        M,
        K,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Persistent thread-block matmul kernel.

        Each thread block processes multiple output tiles in a grid-stride loop.
        This reduces kernel launch overhead for small matrices and demonstrates
        the persistent thread pattern.
        """
        # Grid-stride loop: each block handles multiple output tiles
        pid = tl.program_id(0)
        total_tiles = (M + BLOCK_M - 1) // BLOCK_M * ((N + BLOCK_N - 1) // BLOCK_N)

        for tile_idx in range(pid, total_tiles, tl.num_programs(0)):
            # Decode tile coordinates
            tiles_per_row = (N + BLOCK_N - 1) // BLOCK_N
            m_tile = tile_idx // tiles_per_row
            n_tile = tile_idx % tiles_per_row

            m_start = m_tile * BLOCK_M
            n_start = n_tile * BLOCK_N

            # Offsets for this tile
            offs_m = m_start + tl.arange(0, BLOCK_M)
            offs_n = n_start + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            # Pointers to A and B tiles
            a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

            # Accumulator
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            # Loop over K dimension
            for k_start in range(0, K, BLOCK_K):
                k_end = min(k_start + BLOCK_K, K)
                k_size = k_end - k_start

                # Load A block
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_size)
                a = tl.load(a_ptrs + k_start * stride_ak, mask=a_mask, other=0.0)

                # Load B block
                b_mask = (offs_k[:, None] < k_size) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs + k_start * stride_bk, mask=b_mask, other=0.0)

                # Accumulate
                acc = tl.dot(a, b, acc)

            # Store result
            c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)

else:

    class _TritonKernelStub:
        def __getitem__(self, _grid):
            def launcher(*args, **kwargs):
                _require_triton()

            return launcher

    _persistent_matmul_kernel = _TritonKernelStub()


def persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Persistent thread-block matrix multiplication.

    Computes: C = A @ B

    Uses persistent threads where each block processes multiple output tiles
    in a grid-stride loop. This demonstrates the persistent thread pattern
    which can reduce kernel launch overhead for small matrices.

    Args:
        a: Left matrix of shape (M, K)
        b: Right matrix of shape (K, N)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
        block_k: Block size for K dimension

    Returns:
        Result matrix of shape (M, N)
    """
    _require_triton()

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got A={a.dim()}D, B={b.dim()}D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: A={a.shape}, B={b.shape}")
    if a.device != b.device:
        raise ValueError(f"Tensors must be on same device: A={a.device}, B={b.device}")
    if a.device.type != "cuda":
        raise ValueError("Persistent kernels require CUDA tensors")

    m, k = a.shape
    k2, n = b.shape

    # Make contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Allocate output
    c = torch.empty(m, n, dtype=a.dtype, device=a.device)

    # Number of tiles
    num_m_tiles = (m + block_m - 1) // block_m
    num_n_tiles = (n + block_n - 1) // block_n
    total_tiles = num_m_tiles * num_n_tiles

    # Use enough blocks to saturate GPU but not too many
    # Aim for ~2-4 blocks per SM for good load balancing
    grid = (min(total_tiles, 256),)

    logger.debug(
        f"Persistent matmul: M={m}, K={k}, N={n}, "
        f"blocks=({block_m}, {block_n}, {block_k}), grid={grid}"
    )

    _persistent_matmul_kernel[grid](
        a,
        b,
        c,
        m,
        k,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return c


if __name__ == "__main__":
    torch.manual_seed(42)

    m, k, n = 512, 512, 512
    a = torch.randn(m, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, n, dtype=torch.float32, device="cuda")

    print("Testing persistent matmul...")
    c = persistent_matmul(a, b)
    ref = torch.mm(a, b)

    max_diff = (c - ref).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  {'✓ Passed' if max_diff < 1e-4 else '✗ Failed'}")

    print("\nTesting different sizes...")
    for size in [(64, 64, 64), (128, 256, 512), (1024, 1024, 1024)]:
        m, k, n = size
        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")
        c = persistent_matmul(a, b)
        ref = torch.mm(a, b)
        diff = (c - ref).abs().max().item()
        status = "✓" if diff < 1e-4 else "✗"
        print(f"  {size}: diff={diff:.2e} {status}")

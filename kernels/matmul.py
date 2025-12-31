"""
Triton Matrix Multiplication Kernel

This module implements a high-performance matrix multiplication kernel using OpenAI Triton.
The kernel uses tiling and L2 cache optimization to achieve performance comparable to cuBLAS.

Key concepts:
- Tiling: Divide matrices into blocks that fit in SRAM
- Block pointer arithmetic: Efficiently compute memory addresses for blocks
- L2 cache optimization: Group blocks to improve cache hit rate
"""

import torch
import triton
import triton.language as tl


def get_autotune_configs():
    """Get autotuning configurations for different GPU architectures."""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (number of elements to skip to get to next row/col)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (block sizes)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    
    Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N block of C.
    
    Tiling Strategy:
    1. Each program computes one block of C
    2. Iterate over K dimension in BLOCK_SIZE_K chunks
    3. Use L2 cache optimization via GROUP_SIZE_M (super-grouping)
    """
    # Program ID - which block of C we're computing
    pid = tl.program_id(axis=0)
    
    # Number of blocks in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 Cache Optimization: Group blocks to improve data reuse
    # Instead of simple row-major ordering, we use super-grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute starting row/col indices for this block
    # offs_am: row indices for block of A
    # offs_bn: col indices for block of B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to first block of A and B
    # a_ptrs: pointers to elements A[offs_am, offs_k]
    # b_ptrs: pointers to elements B[offs_k, offs_bn]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator for the block of C (in float32 for numerical stability)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute mask for valid elements (handle non-aligned K)
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Load blocks of A and B with masking for boundary conditions
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        # Compute block matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Convert accumulator to output dtype
    c = accumulator.to(tl.float16)
    
    # Compute output pointers and mask
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Write output block
    tl.store(c_ptrs, c, mask=c_mask)



@triton.jit
def matmul_kernel_no_autotune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Same as matmul_kernel but without autotuning.
    Used for manual block size experimentation.
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
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Triton matrix multiplication: C = A @ B
    
    Args:
        a: Input matrix A of shape (M, K), dtype float16
        b: Input matrix B of shape (K, N), dtype float16
        block_m: Block size for M dimension (optional, uses autotune if None)
        block_n: Block size for N dimension (optional, uses autotune if None)
        block_k: Block size for K dimension (optional, uses autotune if None)
        use_autotune: Whether to use autotuning (ignored if block sizes provided)
        
    Returns:
        Output matrix C of shape (M, N), dtype float16
        
    Raises:
        ValueError: If matrix dimensions are incompatible or invalid block sizes
        TypeError: If input dtypes are not supported
    """
    # Input validation
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got a.dim()={a.dim()}, b.dim()={b.dim()}")
    
    M, K = a.shape
    K2, N = b.shape
    
    if K != K2:
        raise ValueError(f"Incompatible matrix dimensions: A is ({M}, {K}), B is ({K2}, {N})")
    
    # Ensure contiguous tensors
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    
    # Convert to float16 if needed
    if a.dtype != torch.float16:
        a = a.to(torch.float16)
    if b.dtype != torch.float16:
        b = b.to(torch.float16)
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Determine if using manual block sizes or autotune
    manual_blocks = block_m is not None and block_n is not None and block_k is not None
    
    if manual_blocks:
        # Validate block sizes
        if block_m <= 0 or block_n <= 0 or block_k <= 0:
            raise ValueError(f"Block sizes must be positive, got ({block_m}, {block_n}, {block_k})")
        
        # Use non-autotuned kernel with specified block sizes
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        
        matmul_kernel_no_autotune[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=8,
        )
    else:
        # Use autotuned kernel
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    
    return c


def triton_matmul_fp32(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Triton matrix multiplication with float32 output.
    Useful for validation against PyTorch.
    """
    # Convert to float16 for computation
    a_fp16 = a.to(torch.float16) if a.dtype != torch.float16 else a
    b_fp16 = b.to(torch.float16) if b.dtype != torch.float16 else b
    
    result = triton_matmul(a_fp16, b_fp16)
    return result.to(torch.float32)


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(0)
    
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # Triton result
    triton_output = triton_matmul(a, b)
    
    # PyTorch reference
    torch_output = torch.matmul(a, b)
    
    # Compare
    max_diff = (triton_output - torch_output).abs().max().item()
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-2:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")

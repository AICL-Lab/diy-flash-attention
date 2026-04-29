"""
FlashAttention V2: Row-wise (Striped) Parallelism

This module implements FlashAttention V2 which uses row-wise parallelism
instead of the column-parallel approach in V1. Each thread block handles
one query row across all key blocks, providing better memory access patterns
on Ampere+ GPUs.

Key differences from V1:
- V1: Each thread block handles one query block (column parallel)
- V2: Each thread block handles one query row across all key blocks (row parallel)
- V2 typically 5-15% faster on Ampere+ due to better memory access patterns

⚠️ IMPORTANT: Tensor Layout Difference from V1 ⚠️
-------------------------------------------------
V2 uses a DIFFERENT tensor layout compared to V1:

- **V1 (flash_attention)**: `(batch, heads, seq_len, head_dim)`
- **V2 (flash_attention_v2)**: `(batch, seq_len, heads, head_dim)`

This means `heads` and `seq_len` dimensions are SWAPPED between versions!
When switching from V1 to V2, you must transpose your tensors accordingly.

Reference: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
           https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from utils.triton_helpers import TRITON_AVAILABLE, TritonKernelStub, require_triton, tl, triton

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:

    @triton.jit
    def _flash_attn_v2_fwd_kernel(
        Q,
        K,
        V,
        Out,
        L,
        SEQ_LENS,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        stride_lb,
        stride_lh,
        stride_lm,
        SM_SCALE,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_CTX: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        FlashAttention V2 forward kernel with row-wise parallelism.

        Each thread block processes:
        - One block of query rows (BLOCK_M rows)
        - Iterates over all key/value blocks

        This is different from V1 where each block iterates over query blocks.
        """
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        off_z = off_hz // NUM_HEADS
        off_h = off_hz % NUM_HEADS

        # Get sequence length for this batch element
        seq_len = tl.load(SEQ_LENS + off_z)
        seq_len = tl.minimum(seq_len, N_CTX)

        # Initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)

        # Pointers to Q, K, V blocks
        q_ptrs = (
            Q
            + off_z * stride_qb
            + off_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        k_ptrs = (
            K
            + off_z * stride_kb
            + off_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_k[None, :] * stride_kk
        )
        v_ptrs = (
            V
            + off_z * stride_vb
            + off_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )

        # Load Q block - shape (BLOCK_M, HEAD_DIM)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

        # Initialize accumulators
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

        # Loop over K/V blocks
        # For causal: only attend to keys up to current query position
        if IS_CAUSAL:
            loop_end = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
        else:
            loop_end = seq_len

        for start_n in range(0, loop_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)

            # Load K block - shape (BLOCK_N, HEAD_DIM)
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(start_n + offs_n[:, None]) < seq_len,
                other=0.0,
            )

            # Compute QK^T - shape (BLOCK_M, BLOCK_N)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k), qk)
            qk *= SM_SCALE

            # Apply causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Apply seq_len mask
            qk = tl.where((start_n + offs_n[None, :]) < seq_len, qk, float("-inf"))

            # Online softmax
            m_ij = tl.max(qk, axis=1)  # shape (BLOCK_M,)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(qk - m_new[:, None])
            l_new = alpha * l_i + tl.sum(beta, axis=1)

            # Load V block - shape (BLOCK_N, HEAD_DIM)
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(start_n + offs_n[:, None]) < seq_len,
                other=0.0,
            )

            # Update accumulator
            acc = acc * alpha[:, None]
            acc = tl.dot(beta.to(v.dtype), v, acc)

            # Update running max and sum
            m_i = m_new
            l_i = l_new

        # Normalize output
        acc = acc / tl.clamp(l_i[:, None], min=1e-8)

        # Store output
        out_ptrs = (
            Out
            + off_z * stride_ob
            + off_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)

        # Store log-sum-exp for backward pass
        l_ptrs = L + off_z * stride_lb + off_h * stride_lh + offs_m * stride_lm
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < seq_len)

else:
    _flash_attn_v2_fwd_kernel = TritonKernelStub()


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention V2: Memory-efficient attention with row-wise parallelism.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V

    V2 uses row-wise (striped) parallelism where each thread block handles
    one query row across all key blocks. This provides better memory access
    patterns on Ampere+ GPUs compared to V1's column-parallel approach.

    Args:
        q: Query tensor of shape (batch, seq_len, heads, head_dim)
        k: Key tensor of shape (batch, seq_len, heads, head_dim)
        v: Value tensor of shape (batch, seq_len, heads, head_dim)
        causal: Whether to apply causal masking (for autoregressive models)
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        seq_lens: Optional sequence lengths per batch element (shape: [batch])

    Returns:
        Attention output of same shape as input

    Raises:
        ValueError: If tensor shapes/devices are incompatible or inputs are not CUDA tensors
        TypeError: If input dtypes are unsupported or inconsistent
    """
    supported_dtypes = (torch.float16, torch.float32, torch.bfloat16)

    if q.dim() != 4:
        raise ValueError(f"Expected 4D tensor (batch, seq_len, heads, head_dim), got {q.dim()}D")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"K and V must be 4D. Got K={k.dim()}D, V={v.dim()}D")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"Q, K, V shapes must match. Got Q={q.shape}, K={k.shape}, V={v.shape}")
    if (
        q.dtype not in supported_dtypes
        or k.dtype not in supported_dtypes
        or v.dtype not in supported_dtypes
    ):
        raise TypeError(
            f"Unsupported dtype. Supported: {supported_dtypes}. Got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"Q, K, V dtypes must match. Got q={q.dtype}, k={q.dtype}, v={v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"Q, K, V must be on same device. Got q={q.device}, k={k.device}, v={v.device}"
        )
    if q.device.type != "cuda":
        raise ValueError("FlashAttention requires CUDA tensors")

    require_triton()

    batch, seq_len, heads, head_dim = q.shape

    supported_head_dims = (32, 64)
    if head_dim not in supported_head_dims:
        raise ValueError(f"Unsupported head_dim. Supported: {supported_head_dims}. Got {head_dim}")

    # Make contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Convert to float16 for computation
    compute_dtype = torch.float16
    if q.dtype != compute_dtype:
        q_compute = q.to(compute_dtype)
        k_compute = k.to(compute_dtype)
        v_compute = v.to(compute_dtype)
    else:
        q_compute = q
        k_compute = k
        v_compute = v

    # Handle seq_lens
    if seq_lens is None:
        seq_lens_tensor = torch.full((batch,), seq_len, device=q.device, dtype=torch.int32)
    else:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens_tensor = seq_lens.to(device=q.device, dtype=torch.int32)
        else:
            seq_lens_tensor = torch.tensor(seq_lens, device=q.device, dtype=torch.int32)

        if seq_lens_tensor.dim() != 1:
            raise ValueError("seq_lens must be 1D")
        if seq_lens_tensor.numel() != batch:
            raise ValueError(
                f"seq_lens length must match batch size. Got {seq_lens_tensor.numel()} vs {batch}"
            )
        if seq_lens_tensor.min().item() <= 0:
            raise ValueError("seq_lens values must be positive")
        if seq_lens_tensor.max().item() > seq_len:
            raise ValueError(f"seq_lens values must be <= seq_len ({seq_len})")

    seq_lens_tensor = seq_lens_tensor.contiguous()

    # Allocate output
    needs_zero_fill = seq_lens_tensor.min().item() < seq_len
    out = torch.zeros_like(q_compute) if needs_zero_fill else torch.empty_like(q_compute)
    L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

    # Scale factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Select block sizes
    BLOCK_M = 128 if head_dim == 64 else 64
    BLOCK_N = 64 if head_dim == 64 else 32

    # Clamp to seq_len
    BLOCK_M = min(BLOCK_M, seq_len)
    BLOCK_N = min(BLOCK_N, seq_len)

    num_m_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_m_blocks, batch * heads)

    logger.debug(
        f"FlashAttention V2: batch={batch}, heads={heads}, seq={seq_len}, "
        f"head_dim={head_dim}, block_m={BLOCK_M}, block_n={BLOCK_N}, causal={causal}"
    )

    _flash_attn_v2_fwd_kernel[grid](
        q_compute,
        k_compute,
        v_compute,
        out,
        L,
        seq_lens_tensor,
        q_compute.stride(0),
        q_compute.stride(2),
        q_compute.stride(1),
        q_compute.stride(3),
        k_compute.stride(0),
        k_compute.stride(2),
        k_compute.stride(1),
        k_compute.stride(3),
        v_compute.stride(0),
        v_compute.stride(2),
        v_compute.stride(1),
        v_compute.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        sm_scale,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_CTX=seq_len,
        NUM_HEADS=heads,
        IS_CAUSAL=causal,
    )

    # Convert back to original dtype
    if q.dtype != compute_dtype:
        out = out.to(q.dtype)

    return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    q = torch.randn((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.float16)

    print("Testing non-causal attention...")
    v2_out = flash_attention_v2(q, k, v, causal=False)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)

    max_diff = (v2_out - ref_out).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  {'✓ Passed' if max_diff < 1e-2 else '✗ Failed'}")

    print("\nTesting causal attention...")
    v2_out_causal = flash_attention_v2(q, k, v, causal=True)
    ref_out_causal = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    max_diff_causal = (v2_out_causal - ref_out_causal).abs().max().item()
    print(f"  Max difference: {max_diff_causal:.2e}")
    print(f"  {'✓ Passed' if max_diff_causal < 1e-2 else '✗ Failed'}")

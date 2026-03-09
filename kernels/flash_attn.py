"""
FlashAttention Implementation using OpenAI Triton

This module implements the FlashAttention algorithm for efficient attention computation.
FlashAttention reduces memory usage from O(N²) to O(N) by using tiled computation
and online softmax.

Key concepts:
- Online softmax: Compute softmax incrementally without materializing full attention matrix
- Tiled attention: Process Q, K, V in blocks that fit in SRAM
- IO-awareness: Minimize HBM reads/writes by keeping data in fast SRAM

Reference: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
           https://arxiv.org/abs/2205.14135
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_forward_kernel(
    # Input pointers
    Q,
    K,
    V,
    # Output pointer
    Out,
    # Softmax statistics (for potential backward pass)
    L,  # log-sum-exp
    # Sequence lengths (per batch)
    SEQ_LENS,
    # Dimensions
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lz,
    stride_lh,
    stride_lm,
    # Softmax scale
    SM_SCALE,
    # Head dimension
    HEAD_DIM: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Sequence lengths
    N_CTX: tl.constexpr,
    # Head count
    NUM_HEADS: tl.constexpr,
    # Causal mask flag
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention forward kernel.

    Computes: Out = softmax(Q @ K^T / sqrt(d)) @ V

    Uses online softmax to avoid materializing the full N×N attention matrix.

    Algorithm:
    1. Load Q block into SRAM
    2. For each K, V block:
       a. Compute S = Q @ K^T * scale
       b. Apply causal mask if needed
       c. Update running max (m) and sum (l) for online softmax
       d. Compute attention output incrementally
    3. Write final output to HBM
    """
    # Program IDs
    start_m = tl.program_id(0)  # Which Q block
    off_hz = tl.program_id(1)  # Which batch/head

    # Decode batch index for per-batch sequence length
    off_z = off_hz // NUM_HEADS
    seq_len = tl.load(SEQ_LENS + off_z)
    seq_len = tl.minimum(seq_len, N_CTX)

    # Scaling factor
    qk_scale = SM_SCALE

    # Offsets for this block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers to Q, K, V for this batch/head
    q_ptrs = Q + off_hz * stride_qz + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + off_hz * stride_kz + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + off_hz * stride_vz + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    # Load Q block - stays in SRAM for entire computation
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize accumulators for online softmax
    # m_i: running max (for numerical stability)
    # l_i: running sum of exp(x - m)
    # acc: running weighted sum of V
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Determine the range of K, V blocks to process
    # For causal attention, skip K/V blocks that are entirely masked out
    # (positions beyond the current Q block's causal boundary)
    if IS_CAUSAL:
        loop_end = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
    else:
        loop_end = seq_len

    # Iterate over K, V blocks (only up to loop_end for causal efficiency)
    for start_n in range(0, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block
        k = tl.load(
            k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0
        )

        # Compute Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= qk_scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Mask out-of-bounds positions
        qk = tl.where((start_n + offs_n[None, :]) < seq_len, qk, float("-inf"))

        # Online softmax update
        # Step 1: Compute new max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Step 2: Compute correction factors
        alpha = tl.exp(m_i - m_new)

        # Step 3: Update running sum
        l_new = alpha * l_i + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)

        # Step 4: Compute attention weights for this block
        p = tl.exp(qk - m_new[:, None])

        # Step 5: Load V block
        v = tl.load(
            v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0
        )

        # Step 6: Update accumulator
        # acc_new = (l_i * alpha * acc + p @ V) / l_new
        # But we defer the division to the end
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store log-sum-exp for potential backward pass
    l_ptrs = L + off_hz * stride_lz + offs_m * stride_lm
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < seq_len)

    # Store output
    out_ptrs = (
        Out + off_hz * stride_oz + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention: Memory-efficient attention computation.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
        causal: Whether to apply causal masking (for autoregressive models)
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        seq_lens: Optional sequence lengths per batch element (shape: [batch])

    Returns:
        Attention output of same shape as input

    Raises:
        ValueError: If tensor shapes are incompatible
    """
    # Handle both 3D and 4D inputs
    original_shape = q.shape
    if q.dim() == 4:
        batch, heads, seq_len, head_dim = q.shape
        # Reshape to (batch*heads, seq_len, head_dim)
        q = q.reshape(batch * heads, seq_len, head_dim)
        k = k.reshape(batch * heads, seq_len, head_dim)
        v = v.reshape(batch * heads, seq_len, head_dim)
        reshape_output = True
    elif q.dim() == 3:
        batch_heads, seq_len, head_dim = q.shape
        batch = 1
        heads = batch_heads
        reshape_output = False
    else:
        raise ValueError(f"Expected 3D or 4D tensors, got {q.dim()}D")

    # Validate shapes
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"Q, K, V shapes must match. Got Q={q.shape}, K={k.shape}, V={v.shape}")

    supported_head_dims = (32, 64)
    if head_dim not in supported_head_dims:
        raise ValueError(
            "Unsupported head_dim for FlashAttention. "
            f"Supported head_dim values: {supported_head_dims}. Got {head_dim}."
        )

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Ensure float16
    if q.dtype != torch.float16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    # Prepare per-batch sequence lengths
    if seq_lens is None:
        seq_lens_tensor = torch.full((batch,), seq_len, device=q.device, dtype=torch.int32)
    else:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens_tensor = seq_lens.to(device=q.device, dtype=torch.int32)
        else:
            seq_lens_tensor = torch.tensor(seq_lens, device=q.device, dtype=torch.int32)

        if seq_lens_tensor.dim() != 1:
            raise ValueError("seq_lens must be a 1D tensor or list")
        if reshape_output:
            if seq_lens_tensor.numel() != batch:
                raise ValueError(
                    f"seq_lens length must match batch size. Got {seq_lens_tensor.numel()} vs {batch}."
                )
        else:
            if seq_lens_tensor.numel() != 1:
                raise ValueError("seq_lens for 3D inputs must have length 1")

        max_len = int(seq_len)
        if seq_lens_tensor.min().item() <= 0:
            raise ValueError("seq_lens values must be positive")
        if seq_lens_tensor.max().item() > max_len:
            raise ValueError(
                f"seq_lens values must be <= seq_len ({max_len}). Got max={seq_lens_tensor.max().item()}."
            )

    seq_lens_tensor = seq_lens_tensor.contiguous()

    # Allocate output and log-sum-exp
    batch_heads = q.shape[0]
    needs_zero_fill = seq_lens_tensor.min().item() < seq_len
    out = torch.zeros_like(q) if needs_zero_fill else torch.empty_like(q)
    L = torch.empty((batch_heads, seq_len), device=q.device, dtype=torch.float32)

    # Softmax scaling factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Determine block sizes based on head_dim
    BLOCK_M = 128
    BLOCK_N = 64

    # Adjust for smaller head dimensions
    if head_dim <= 32:
        BLOCK_M = 64
        BLOCK_N = 32

    # Grid: one program per Q block per batch/head
    num_m_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_m_blocks, batch_heads)

    # Launch kernel
    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        out,
        L,
        seq_lens_tensor,
        # Q strides
        q.stride(0),
        0,
        q.stride(1),
        q.stride(2),
        # K strides
        k.stride(0),
        0,
        k.stride(1),
        k.stride(2),
        # V strides
        v.stride(0),
        0,
        v.stride(1),
        v.stride(2),
        # Out strides
        out.stride(0),
        0,
        out.stride(1),
        out.stride(2),
        # L strides
        L.stride(0),
        0,
        L.stride(1),
        sm_scale,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_CTX=seq_len,
        NUM_HEADS=heads,
        IS_CAUSAL=causal,
    )

    # Reshape output if needed
    if reshape_output:
        out = out.reshape(original_shape)

    return out


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Reference attention implementation for validation.

    This is a naive O(N²) implementation that materializes the full attention matrix.
    Used for correctness checking against FlashAttention.
    """
    # Compute attention scores
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax and weighted sum
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    return out


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    batch, heads, seq_len, head_dim = 2, 4, 128, 64

    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

    # Test non-causal
    print("Testing non-causal attention...")
    flash_out = flash_attention(q, k, v, causal=False)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    max_diff = (flash_out - ref_out).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  {'✓ Passed' if max_diff < 1e-2 else '✗ Failed'}")

    # Test causal
    print("\nTesting causal attention...")
    flash_out_causal = flash_attention(q, k, v, causal=True)
    ref_out_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff_causal = (flash_out_causal - ref_out_causal).abs().max().item()
    print(f"  Max difference: {max_diff_causal:.2e}")
    print(f"  {'✓ Passed' if max_diff_causal < 1e-2 else '✗ Failed'}")

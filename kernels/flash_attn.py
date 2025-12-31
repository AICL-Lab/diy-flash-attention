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
    Q, K, V,
    # Output pointer
    Out,
    # Softmax statistics (for potential backward pass)
    L,  # log-sum-exp
    # Dimensions
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lz, stride_lh, stride_lm,
    # Sequence lengths
    N_CTX,
    # Head dimension
    HEAD_DIM: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    off_hz = tl.program_id(1)   # Which batch/head
    
    # Compute batch and head indices
    off_z = off_hz // tl.num_programs(1)  # This doesn't work, need to pass num_heads
    off_h = off_hz % tl.num_programs(1)
    
    # Actually, let's compute it differently
    # off_hz encodes both batch and head: off_hz = batch_idx * num_heads + head_idx
    # We'll handle this in the wrapper
    
    # Scaling factor
    qk_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Offsets for this block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Pointers to Q, K, V for this batch/head
    q_ptrs = Q + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + off_hz * stride_vh + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    # Load Q block - stays in SRAM for entire computation
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize accumulators for online softmax
    # m_i: running max (for numerical stability)
    # l_i: running sum of exp(x - m)
    # acc: running weighted sum of V
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Determine the range of K, V blocks to process
    if IS_CAUSAL:
        # For causal attention, only process K, V up to current position
        end_n = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        end_n = N_CTX
    
    # Iterate over K, V blocks
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)
        
        # Compute Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk *= qk_scale
        
        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Mask out-of-bounds positions
        qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float("-inf"))
        
        # Online softmax update
        # Step 1: Compute new max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Step 2: Compute correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # Step 3: Update running sum
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Step 4: Compute attention weights for this block
        p = tl.exp(qk - m_new[:, None])
        
        # Step 5: Load V block
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)
        
        # Step 6: Update accumulator
        # acc_new = (l_i * alpha * acc + p @ V) / l_new
        # But we defer the division to the end
        acc = acc * (alpha * l_i)[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        
        # Update running statistics
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store log-sum-exp for potential backward pass
    l_ptrs = L + off_hz * stride_lh + offs_m * stride_lm
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)
    
    # Store output
    out_ptrs = Out + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)



def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
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
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Ensure float16
    if q.dtype != torch.float16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    
    # Allocate output and log-sum-exp
    batch_heads = q.shape[0]
    out = torch.empty_like(q)
    L = torch.empty((batch_heads, seq_len), device=q.device, dtype=torch.float32)
    
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
        q, k, v,
        out,
        L,
        # Q strides
        q.stride(0), 0, q.stride(1), q.stride(2),
        # K strides
        k.stride(0), 0, k.stride(1), k.stride(2),
        # V strides
        v.stride(0), 0, v.stride(1), v.stride(2),
        # Out strides
        out.stride(0), 0, out.stride(1), out.stride(2),
        # L strides
        L.stride(0), 0, L.stride(1),
        # Dimensions
        seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
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

"""
FlashAttention Implementation using OpenAI Triton

This module implements the FlashAttention algorithm for efficient attention computation.
FlashAttention reduces memory usage from O(N²) to O(N) by using tiled computation
and online softmax.

Key concepts:
- Online softmax: Compute softmax incrementally without materializing full attention matrix
- Tiled attention: Process Q, K, V in blocks that fit in SRAM
- IO-awareness: Minimize HBM reads/writes by keeping data in fast SRAM

Tensor Layout (V1):
-------------------
This implementation uses: `(batch, heads, seq_len, head_dim)`

Also accepts 3D input: `(batch*heads, seq_len, head_dim)`

⚠️ Note: FlashAttention V2 uses a different layout: `(batch, seq_len, heads, head_dim)`
The `heads` and `seq_len` dimensions are swapped in V2!

Reference: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
           https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from utils.config import FLASH_ATTN_DEFAULT_BLOCK_M, FLASH_ATTN_DEFAULT_BLOCK_N
from utils.triton_helpers import TRITON_AVAILABLE, TritonKernelStub, require_triton, tl, triton

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:

    @triton.jit
    def _flash_attention_forward_kernel(
        Q,
        K,
        V,
        Out,
        L,
        SEQ_LENS,
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
        SM_SCALE,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_CTX: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        off_z = off_hz // NUM_HEADS
        seq_len = tl.load(SEQ_LENS + off_z)
        seq_len = tl.minimum(seq_len, N_CTX)

        qk_scale = SM_SCALE

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)

        q_ptrs = (
            Q + off_hz * stride_qz + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        )
        k_ptrs = (
            K + off_hz * stride_kz + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        )
        v_ptrs = (
            V + off_hz * stride_vz + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        )

        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        if IS_CAUSAL:
            loop_end = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
        else:
            loop_end = seq_len

        for start_n in range(0, loop_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=(start_n + offs_n[:, None]) < seq_len,
                other=0.0,
            )

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k), qk)
            qk *= qk_scale

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(causal_mask, qk, float("-inf"))

            qk = tl.where((start_n + offs_n[None, :]) < seq_len, qk, float("-inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            l_new = alpha * l_i + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            p = tl.exp(qk - m_new[:, None])

            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(start_n + offs_n[:, None]) < seq_len,
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]

        l_ptrs = L + off_hz * stride_lz + offs_m * stride_lm
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < seq_len)

        out_ptrs = (
            Out + off_hz * stride_oz + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        )
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)

else:
    _flash_attention_forward_kernel = TritonKernelStub()


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
        ValueError: If tensor shapes/devices are incompatible or inputs are not CUDA tensors
        TypeError: If input dtypes are unsupported or inconsistent
    """
    supported_dtypes = (torch.float16, torch.float32, torch.bfloat16)

    if q.dim() not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensors, got {q.dim()}D")
    if k.dim() != q.dim() or v.dim() != q.dim():
        raise ValueError(
            f"Q, K, V must have the same rank. Got Q={q.dim()}D, K={k.dim()}D, V={v.dim()}D."
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"Q, K, V shapes must match. Got Q={q.shape}, K={k.shape}, V={v.shape}")
    if (
        q.dtype not in supported_dtypes
        or k.dtype not in supported_dtypes
        or v.dtype not in supported_dtypes
    ):
        raise TypeError(
            "Unsupported dtype for FlashAttention. "
            f"Supported dtypes: {supported_dtypes}. Got q={q.dtype}, k={k.dtype}, v={v.dtype}."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"Q, K, V dtypes must match. Got q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"Q, K, and V must be on the same device. Got q={q.device}, k={k.device}, v={v.device}."
        )
    if q.device.type != "cuda":
        raise ValueError("FlashAttention requires CUDA tensors for Q, K, and V.")

    require_triton()

    original_shape = q.shape
    if q.dim() == 4:
        batch, heads, seq_len, head_dim = q.shape
        q = q.reshape(batch * heads, seq_len, head_dim)
        k = k.reshape(batch * heads, seq_len, head_dim)
        v = v.reshape(batch * heads, seq_len, head_dim)
        reshape_output = True
    else:
        batch_heads, seq_len, head_dim = q.shape
        batch = 1
        heads = batch_heads
        reshape_output = False

    supported_head_dims = (32, 64)
    if head_dim not in supported_head_dims:
        raise ValueError(
            "Unsupported head_dim for FlashAttention. "
            f"Supported head_dim values: {supported_head_dims}. Got {head_dim}."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if q.dtype != torch.float16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

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

    batch_heads = q.shape[0]
    needs_zero_fill = seq_lens_tensor.min().item() < seq_len
    out = torch.zeros_like(q) if needs_zero_fill else torch.empty_like(q)
    L = torch.empty((batch_heads, seq_len), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    try:
        from kernels.modern_features import get_attention_config

        config = get_attention_config()
        block_m = int(config.get("BLOCK_M", FLASH_ATTN_DEFAULT_BLOCK_M))
        block_n = int(config.get("BLOCK_N", FLASH_ATTN_DEFAULT_BLOCK_N))
    except (ImportError, RuntimeError):
        block_m = FLASH_ATTN_DEFAULT_BLOCK_M
        block_n = FLASH_ATTN_DEFAULT_BLOCK_N

    if block_m <= 0 or block_n <= 0:
        raise ValueError(f"Invalid attention block config: BLOCK_M={block_m}, BLOCK_N={block_n}")

    if head_dim <= 32 and block_m > 64:
        block_m = 64
    if head_dim <= 32 and block_n > 32:
        block_n = 32

    block_m = max(min(block_m, seq_len), 1)
    block_n = max(min(block_n, seq_len), 1)

    num_m_blocks = triton.cdiv(seq_len, block_m)
    grid = (num_m_blocks, batch_heads)

    logger.debug(
        f"FlashAttention kernel: batch={batch}, heads={heads}, seq={seq_len}, "
        f"head_dim={head_dim}, block_m={block_m}, block_n={block_n}, causal={causal}"
    )

    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        out,
        L,
        seq_lens_tensor,
        q.stride(0),
        0,
        q.stride(1),
        q.stride(2),
        k.stride(0),
        0,
        k.stride(1),
        k.stride(2),
        v.stride(0),
        0,
        v.stride(1),
        v.stride(2),
        out.stride(0),
        0,
        out.stride(1),
        out.stride(2),
        L.stride(0),
        0,
        L.stride(1),
        sm_scale,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        N_CTX=seq_len,
        NUM_HEADS=heads,
        IS_CAUSAL=causal,
    )

    if reshape_output:
        out = out.reshape(original_shape)

    return out


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Reference attention implementation for validation."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch, heads, seq_len, head_dim = 2, 4, 128, 64
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

    print("Testing non-causal attention...")
    flash_out = flash_attention(q, k, v, causal=False)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    max_diff = (flash_out - ref_out).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  {'✓ Passed' if max_diff < 1e-2 else '✗ Failed'}")

    print("\nTesting causal attention...")
    flash_out_causal = flash_attention(q, k, v, causal=True)
    ref_out_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff_causal = (flash_out_causal - ref_out_causal).abs().max().item()
    print(f"  Max difference: {max_diff_causal:.2e}")
    print(f"  {'✓ Passed' if max_diff_causal < 1e-2 else '✗ Failed'}")

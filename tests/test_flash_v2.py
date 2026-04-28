"""Tests for FlashAttention V2 kernel with striped parallelism."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels import flash_attention_v2


class TestFlashAttentionV2Basic:
    """Basic correctness tests for FlashAttention V2."""

    @pytest.mark.cuda
    def test_flash_attention_v2_basic(self):
        """Test v2 forward pass matches PyTorch baseline."""
        torch.manual_seed(42)
        batch, seq_len, heads, head_dim = 1, 128, 8, 64

        q = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Our implementation
        output_v2 = flash_attention_v2(q, k, v)

        # PyTorch baseline (SDPA)
        q_t = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        output_baseline = F.scaled_dot_product_attention(q_t, k_t, v_t)
        output_baseline = output_baseline.transpose(
            1, 2
        )  # Back to (batch, seq_len, heads, head_dim)

        assert output_v2.shape == output_baseline.shape
        assert torch.allclose(output_v2, output_baseline, rtol=1e-2, atol=1e-3)

    @pytest.mark.cuda
    def test_flash_attention_v2_causal(self):
        """Test v2 causal masking matches PyTorch baseline."""
        torch.manual_seed(42)
        batch, seq_len, heads, head_dim = 2, 256, 8, 64

        q = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # V2 causal
        output_v2 = flash_attention_v2(q, k, v, causal=True)

        # PyTorch baseline with causal mask
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        output_baseline = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=True
        )
        output_baseline = output_baseline.transpose(1, 2)

        assert torch.allclose(output_v2, output_baseline, rtol=1e-2, atol=1e-3)


class TestFlashAttentionV2Dtypes:
    """Dtype support tests for FlashAttention V2."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_flash_attention_v2_dtypes(self, dtype):
        """Test v2 supports float16, bfloat16, float32."""
        torch.manual_seed(42)
        batch, seq_len, heads, head_dim = 1, 128, 8, 64

        q = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device="cuda")

        output = flash_attention_v2(q, k, v)

        assert output.dtype == dtype
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


class TestFlashAttentionV2SeqLens:
    """Variable sequence length tests for FlashAttention V2."""

    @pytest.mark.cuda
    def test_flash_attention_v2_seq_lens(self):
        """Test v2 variable sequence lengths (seq_lens masking)."""
        torch.manual_seed(42)
        batch, max_seq_len, heads, head_dim = 2, 256, 8, 64

        q = torch.randn(
            batch, max_seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, max_seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, max_seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Seq lens: first seq is 128, second is 200
        seq_lens = torch.tensor([128, 200], dtype=torch.int32, device="cuda")

        output_v2 = flash_attention_v2(q, k, v, seq_lens=seq_lens)

        assert output_v2.shape == q.shape
        # Verify no NaN values
        assert not torch.isnan(output_v2).any()


class TestFlashAttentionV2HeadDim:
    """Head dimension tests for FlashAttention V2."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_flash_attention_v2_head_dims(self, head_dim):
        """Test v2 supports head_dim 32 and 64."""
        torch.manual_seed(42)
        batch, seq_len, heads = 1, 128, 8

        q = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, seq_len, heads, head_dim, dtype=torch.float16, device="cuda"
        )

        output = flash_attention_v2(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

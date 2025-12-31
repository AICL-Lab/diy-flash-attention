"""
Tests for FlashAttention Kernel

This module contains unit tests and property-based tests for the FlashAttention kernel.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.flash_attn import flash_attention


class TestFlashAttentionBasic:
    """Basic unit tests for FlashAttention."""
    
    def test_non_causal(self):
        """Test non-causal attention."""
        batch, heads, seq_len, head_dim = 2, 4, 128, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    
    def test_causal(self):
        """Test causal attention."""
        batch, heads, seq_len, head_dim = 2, 4, 128, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=True)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch, heads, seq_len, head_dim = 2, 8, 256, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        out = flash_attention(q, k, v)
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype


class TestFlashAttentionConfigs:
    """Tests for different configurations."""
    
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_different_seq_lengths(self, seq_len):
        """Test different sequence lengths."""
        batch, heads, head_dim = 2, 4, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_different_head_dims(self, head_dim):
        """Test different head dimensions."""
        batch, heads, seq_len = 2, 4, 128
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("num_heads", [1, 4, 8])
    def test_different_num_heads(self, num_heads):
        """Test different number of heads."""
        batch, seq_len, head_dim = 2, 128, 64
        
        q = torch.randn((batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)


class TestFlashAttentionEdgeCases:
    """Edge case tests for FlashAttention."""
    
    def test_single_token(self):
        """Test with single token sequence."""
        batch, heads, seq_len, head_dim = 2, 4, 1, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        batch, heads, seq_len, head_dim = 1, 4, 128, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
        
        flash_out = flash_attention(q, k, v, causal=True)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)


class TestFlashAttentionErrorHandling:
    """Tests for error handling."""
    
    def test_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 256, 64), device="cuda", dtype=torch.float16)  # Different seq_len
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="match"):
            flash_attention(q, k, v)
    
    def test_invalid_ndim(self):
        """Test that invalid dimensions raise error."""
        q = torch.randn((128, 64), device="cuda", dtype=torch.float16)  # 2D instead of 3D/4D
        k = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="3D or 4D"):
            flash_attention(q, k, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

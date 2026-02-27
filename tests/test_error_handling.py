"""
Unit Tests for Error Handling

This module contains unit tests for error handling in matmul and attention kernels.
Tests verify that appropriate errors are raised for invalid inputs.

**Validates: Requirements 3.4**
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

from kernels.matmul import triton_matmul
from kernels.flash_attn import flash_attention


class TestMatmulErrorHandling:
    """Tests for matmul error handling."""
    
    def test_incompatible_dimensions(self):
        """Test error for incompatible matrix dimensions."""
        a = torch.randn((64, 32), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)  # K mismatch
        
        with pytest.raises(ValueError, match="Incompatible"):
            triton_matmul(a, b)
    
    def test_non_2d_tensor_a(self):
        """Test error for non-2D tensor A."""
        a = torch.randn((64, 64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="2D"):
            triton_matmul(a, b)
    
    def test_non_2d_tensor_b(self):
        """Test error for non-2D tensor B."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="2D"):
            triton_matmul(a, b)
    
    def test_1d_tensor(self):
        """Test error for 1D tensor."""
        a = torch.randn(64, device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="2D"):
            triton_matmul(a, b)
    
    def test_invalid_block_size_zero(self):
        """Test error for zero block size."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="positive"):
            triton_matmul(a, b, block_m=0, block_n=64, block_k=32)
    
    def test_invalid_block_size_negative(self):
        """Test error for negative block size."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="positive"):
            triton_matmul(a, b, block_m=-1, block_n=64, block_k=32)
    
    def test_partial_block_size_specification(self):
        """Test that partial block size specification uses autotune."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        # Only specifying some block sizes should fall back to autotune
        result = triton_matmul(a, b, block_m=64)  # Only block_m specified
        assert result.shape == (64, 64)
    
    def test_unsupported_dtype(self):
        """Test error for unsupported dtype (e.g. int32)."""
        a = torch.randint(0, 10, (64, 64), device="cuda", dtype=torch.int32)
        b = torch.randint(0, 10, (64, 64), device="cuda", dtype=torch.int32)
        
        with pytest.raises(TypeError, match="Unsupported dtype"):
            triton_matmul(a, b)
    
    def test_block_size_exceeds_dimension(self):
        """Test error when block size exceeds matrix dimension."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="must not exceed"):
            triton_matmul(a, b, block_m=128, block_n=64, block_k=32)


class TestFlashAttentionErrorHandling:
    """Tests for FlashAttention error handling."""
    
    def test_shape_mismatch_seq_len(self):
        """Test error for mismatched sequence lengths."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 256, 64), device="cuda", dtype=torch.float16)  # Different seq_len
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="match"):
            flash_attention(q, k, v)

    def test_unsupported_head_dim(self):
        """Test error for unsupported head_dim values."""
        q = torch.randn((1, 2, 64, 16), device="cuda", dtype=torch.float16)
        k = torch.randn((1, 2, 64, 16), device="cuda", dtype=torch.float16)
        v = torch.randn((1, 2, 64, 16), device="cuda", dtype=torch.float16)

        with pytest.raises(ValueError, match="head_dim"):
            flash_attention(q, k, v)

    def test_seq_lens_length_mismatch(self):
        """Test error for seq_lens length mismatch."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        seq_lens = torch.tensor([64], device="cuda", dtype=torch.int32)

        with pytest.raises(ValueError, match="batch size"):
            flash_attention(q, k, v, seq_lens=seq_lens)

    def test_seq_lens_invalid_values(self):
        """Test error for invalid seq_lens values."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)

        with pytest.raises(ValueError, match="positive"):
            flash_attention(q, k, v, seq_lens=torch.tensor([0, 64], device="cuda", dtype=torch.int32))

        with pytest.raises(ValueError, match="seq_len"):
            flash_attention(q, k, v, seq_lens=torch.tensor([129, 128], device="cuda", dtype=torch.int32))

    def test_seq_lens_3d_length_mismatch(self):
        """Test error for seq_lens length mismatch on 3D input."""
        q = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)

        with pytest.raises(ValueError, match="length 1"):
            flash_attention(q, k, v, seq_lens=torch.tensor([64, 64], device="cuda", dtype=torch.int32))
    
    def test_shape_mismatch_head_dim(self):
        """Test error for mismatched head dimensions."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 128, 32), device="cuda", dtype=torch.float16)  # Different head_dim
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="match"):
            flash_attention(q, k, v)
    
    def test_shape_mismatch_batch(self):
        """Test error for mismatched batch sizes."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((4, 4, 128, 64), device="cuda", dtype=torch.float16)  # Different batch
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="match"):
            flash_attention(q, k, v)
    
    def test_shape_mismatch_heads(self):
        """Test error for mismatched number of heads."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 8, 128, 64), device="cuda", dtype=torch.float16)  # Different heads
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="match"):
            flash_attention(q, k, v)
    
    def test_invalid_ndim_2d(self):
        """Test error for 2D tensors."""
        q = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="3D or 4D"):
            flash_attention(q, k, v)
    
    def test_invalid_ndim_5d(self):
        """Test error for 5D tensors."""
        q = torch.randn((1, 2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((1, 2, 4, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((1, 2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="3D or 4D"):
            flash_attention(q, k, v)
    
    def test_valid_3d_input(self):
        """Test that 3D input is accepted."""
        # 3D input: (batch*heads, seq_len, head_dim)
        q = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
        
        # Should not raise
        result = flash_attention(q, k, v)
        assert result.shape == q.shape
    
    def test_valid_4d_input(self):
        """Test that 4D input is accepted."""
        # 4D input: (batch, heads, seq_len, head_dim)
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        # Should not raise
        result = flash_attention(q, k, v)
        assert result.shape == q.shape


class TestErrorMessages:
    """Tests for error message quality."""
    
    def test_matmul_dimension_error_message(self):
        """Test that matmul dimension error message is informative."""
        a = torch.randn((64, 32), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        try:
            triton_matmul(a, b)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Error message should mention the incompatible dimensions
            assert "64" in error_msg or "32" in error_msg or "Incompatible" in error_msg
    
    def test_attention_shape_error_message(self):
        """Test that attention shape error message is informative."""
        q = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((2, 4, 256, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float16)
        
        try:
            flash_attention(q, k, v)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Error message should mention shape mismatch
            assert "match" in error_msg.lower() or "shape" in error_msg.lower()


class TestDtypeHandling:
    """Tests for dtype handling."""
    
    def test_matmul_float32_conversion(self):
        """Test that float32 input is converted to float16."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
        
        result = triton_matmul(a, b)
        assert result.dtype == torch.float16
    
    def test_matmul_bfloat16_input(self):
        """Test that bfloat16 input is accepted and produces correct result (Req 1.7)."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        
        result = triton_matmul(a, b)
        assert result.dtype == torch.float16
        
        ref = torch.matmul(a.float(), b.float()).half()
        assert torch.allclose(result, ref, rtol=1e-1, atol=1e-1)
    
    def test_attention_float32_conversion(self):
        """Test that float32 attention input is converted to float16."""
        q = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float32)
        k = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float32)
        v = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float32)
        
        result = flash_attention(q, k, v)
        assert result.dtype == torch.float16


class TestContiguityHandling:
    """Tests for non-contiguous tensor handling."""
    
    def test_matmul_non_contiguous(self):
        """Test matmul with non-contiguous tensors."""
        a = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((128, 64), device="cuda", dtype=torch.float16)
        
        # Create non-contiguous tensors via transpose
        a_t = a.t()  # Now (64, 128), non-contiguous
        b_t = b.t()  # Now (64, 128), non-contiguous
        
        # Should handle non-contiguous by making contiguous internally
        result = triton_matmul(a_t.contiguous(), b_t.t().contiguous())
        assert result.shape == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

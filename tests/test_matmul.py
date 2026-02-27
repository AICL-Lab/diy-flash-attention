"""
Tests for Triton Matrix Multiplication Kernel

This module contains unit tests and property-based tests for the matmul kernel.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

from kernels.matmul import triton_matmul


class TestMatmulBasic:
    """Basic unit tests for matmul kernel."""
    
    def test_square_matrix(self):
        """Test square matrix multiplication."""
        M, N, K = 512, 512, 512
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        triton_out = triton_matmul(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        
        assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
    
    def test_rectangular_matrix(self):
        """Test rectangular matrix multiplication."""
        M, N, K = 256, 512, 128
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        triton_out = triton_matmul(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        
        assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
    
    def test_non_power_of_2(self):
        """Test non-power-of-2 dimensions."""
        M, N, K = 33, 47, 61
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        triton_out = triton_matmul(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        
        assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        M, N, K = 128, 256, 64
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        c = triton_matmul(a, b)
        
        assert c.shape == (M, N)
        assert c.dtype == torch.float16


class TestMatmulEdgeCases:
    """Edge case tests for matmul kernel."""
    
    def test_zero_matrix(self):
        """Test multiplication with zero matrix."""
        M, N, K = 64, 64, 64
        a = torch.zeros((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        c = triton_matmul(a, b)
        
        assert torch.allclose(c, torch.zeros_like(c), atol=1e-5)
    
    def test_small_matrix(self):
        """Test very small matrices."""
        for size in [1, 2, 4, 8, 16]:
            a = torch.randn((size, size), device="cuda", dtype=torch.float16)
            b = torch.randn((size, size), device="cuda", dtype=torch.float16)
            
            triton_out = triton_matmul(a, b)
            torch_out = torch.matmul(a.float(), b.float()).half()
            
            assert torch.allclose(triton_out, torch_out, rtol=1e-1, atol=1e-1), f"Failed for size {size}"


class TestMatmulBlockSizes:
    """Tests for different block size configurations."""
    
    def test_different_block_sizes(self):
        """Test that different block sizes produce same results."""
        M, N, K = 256, 256, 256
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        # Reference with autotune
        ref = triton_matmul(a, b)
        
        # Test different block sizes
        block_configs = [
            (32, 32, 32),
            (64, 64, 32),
            (64, 64, 64),
            (128, 128, 32),
        ]
        
        for block_m, block_n, block_k in block_configs:
            result = triton_matmul(a, b, block_m=block_m, block_n=block_n, block_k=block_k)
            assert torch.allclose(result, ref, rtol=1e-2, atol=1e-2), \
                f"Block size ({block_m}, {block_n}, {block_k}) produced different result"


class TestMatmulErrorHandling:
    """Tests for error handling."""
    
    def test_incompatible_dimensions(self):
        """Test that incompatible dimensions raise error."""
        a = torch.randn((64, 32), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)  # K mismatch
        
        with pytest.raises(ValueError, match="Incompatible"):
            triton_matmul(a, b)
    
    def test_invalid_ndim(self):
        """Test that non-2D tensors raise error."""
        a = torch.randn((64, 64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="2D"):
            triton_matmul(a, b)
    
    def test_invalid_block_size(self):
        """Test that invalid block sizes raise error."""
        a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        
        with pytest.raises(ValueError, match="positive"):
            triton_matmul(a, b, block_m=0, block_n=64, block_k=32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

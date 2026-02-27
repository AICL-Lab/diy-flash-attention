"""
Property-Based Tests for DIY FlashAttention

This module contains property-based tests using Hypothesis to verify
correctness properties across many randomly generated inputs.

Each test is annotated with the property it validates from the design document.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

from kernels.matmul import triton_matmul
from kernels.flash_attn import flash_attention


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Matrix dimension strategies
small_dim = st.integers(min_value=1, max_value=64)
medium_dim = st.integers(min_value=16, max_value=256)
large_dim = st.integers(min_value=64, max_value=512)

# Block size strategies
block_sizes = st.sampled_from([16, 32, 64, 128])

# Attention dimension strategies
batch_size = st.integers(min_value=1, max_value=4)
num_heads = st.integers(min_value=1, max_value=8)
seq_length = st.integers(min_value=16, max_value=256)
head_dim = st.sampled_from([32, 64])


# =============================================================================
# Property 1: Matrix Multiplication Correctness
# Feature: diy-flash-attention, Property 1: Matrix Multiplication Correctness
# Validates: Requirements 1.1, 1.2, 6.1
# =============================================================================

class TestMatmulCorrectnessProperty:
    """
    Property 1: Matrix Multiplication Correctness
    
    *For any* matrices A of shape (M, K) and B of shape (K, N) with valid 
    floating-point values, the Triton matmul kernel output C should equal 
    torch.matmul(A, B) within relative tolerance of 1e-2.
    
    **Validates: Requirements 1.1, 1.2, 6.1**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        M=medium_dim,
        N=medium_dim,
        K=medium_dim,
    )
    def test_matmul_correctness_property(self, M, N, K):
        """
        Feature: diy-flash-attention, Property 1: Matrix Multiplication Correctness
        Validates: Requirements 1.1, 1.2, 6.1
        
        For any matrices A (M×K) and B (K×N), Triton matmul should match torch.matmul.
        """
        # Generate random matrices
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        # Compute with Triton
        triton_out = triton_matmul(a, b)
        
        # Compute reference with PyTorch (use float32 for better precision)
        torch_out = torch.matmul(a.float(), b.float()).half()
        
        # Verify correctness
        assert triton_out.shape == (M, N), f"Shape mismatch: {triton_out.shape} vs ({M}, {N})"
        assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2), \
            f"Max diff: {(triton_out - torch_out).abs().max().item()}"


# =============================================================================
# Property 2: Block Size Invariance
# Feature: diy-flash-attention, Property 2: Block Size Invariance
# Validates: Requirements 1.4
# =============================================================================

class TestBlockSizeInvarianceProperty:
    """
    Property 2: Block Size Invariance
    
    *For any* valid block size configuration (BLOCK_M, BLOCK_N, BLOCK_K) and 
    any input matrices A, B, the Triton matmul kernel should produce the same 
    result (within numerical tolerance) regardless of block size choice.
    
    **Validates: Requirements 1.4**
    """
    
    @settings(max_examples=50, deadline=None)
    @given(
        M=st.integers(min_value=64, max_value=256),
        N=st.integers(min_value=64, max_value=256),
        K=st.integers(min_value=64, max_value=256),
        block_m=st.sampled_from([32, 64, 128]),
        block_n=st.sampled_from([32, 64, 128]),
        block_k=st.sampled_from([32, 64]),
    )
    def test_block_size_invariance_property(self, M, N, K, block_m, block_n, block_k):
        """
        Feature: diy-flash-attention, Property 2: Block Size Invariance
        Validates: Requirements 1.4
        
        For any valid block sizes, the result should be the same as autotune.
        """
        # Generate random matrices
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        # Compute with autotune (reference)
        ref_out = triton_matmul(a, b)
        
        # Compute with manual block sizes
        manual_out = triton_matmul(a, b, block_m=block_m, block_n=block_n, block_k=block_k)
        
        # Results should match within tolerance
        assert torch.allclose(manual_out, ref_out, rtol=1e-2, atol=1e-2), \
            f"Block size ({block_m}, {block_n}, {block_k}) produced different result. " \
            f"Max diff: {(manual_out - ref_out).abs().max().item()}"


# =============================================================================
# Property 3: FlashAttention Correctness
# Feature: diy-flash-attention, Property 3: FlashAttention Correctness
# Validates: Requirements 4.1, 4.4, 6.1
# =============================================================================

class TestFlashAttentionCorrectnessProperty:
    """
    Property 3: FlashAttention Correctness
    
    *For any* query Q, key K, and value V tensors of compatible shapes, 
    the FlashAttention kernel output should equal the reference attention 
    computation softmax(Q @ K^T / sqrt(d)) @ V within relative tolerance of 1e-2.
    
    **Validates: Requirements 4.1, 4.4, 6.1**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_size,
        heads=num_heads,
        seq_len=seq_length,
        d=head_dim,
    )
    def test_flash_attention_correctness_property(self, batch, heads, seq_len, d):
        """
        Feature: diy-flash-attention, Property 3: FlashAttention Correctness
        Validates: Requirements 4.1, 4.4, 6.1
        
        For any Q, K, V tensors, FlashAttention should match reference implementation.
        """
        # Generate random tensors
        q = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        
        # Compute with FlashAttention
        flash_out = flash_attention(q, k, v, causal=False)
        
        # Compute reference with PyTorch
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # Verify correctness
        assert flash_out.shape == q.shape, f"Shape mismatch: {flash_out.shape} vs {q.shape}"
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2), \
            f"Max diff: {(flash_out - ref_out).abs().max().item()}"


# =============================================================================
# Property 4: Causal Masking Correctness
# Feature: diy-flash-attention, Property 4: Causal Masking Correctness
# Validates: Requirements 4.3
# =============================================================================

class TestCausalMaskingProperty:
    """
    Property 4: Causal Masking Correctness
    
    *For any* Q, K, V tensors, when causal masking is enabled, the FlashAttention 
    output should match the reference implementation with causal mask applied.
    
    **Validates: Requirements 4.3**
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        batch=batch_size,
        heads=num_heads,
        seq_len=seq_length,
        d=head_dim,
    )
    def test_causal_masking_property(self, batch, heads, seq_len, d):
        """
        Feature: diy-flash-attention, Property 4: Causal Masking Correctness
        Validates: Requirements 4.3
        
        For any Q, K, V tensors with causal=True, output should match causal reference.
        """
        # Generate random tensors
        q = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        
        # Compute with FlashAttention (causal)
        flash_out = flash_attention(q, k, v, causal=True)
        
        # Compute reference with PyTorch (causal)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Verify correctness
        assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2), \
            f"Causal attention mismatch. Max diff: {(flash_out - ref_out).abs().max().item()}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        batch=batch_size,
        heads=num_heads,
        seq_len=st.integers(min_value=8, max_value=64),
        d=head_dim,
    )
    def test_causal_future_independence(self, batch, heads, seq_len, d):
        """
        Feature: diy-flash-attention, Property 4: Causal Masking Correctness
        Validates: Requirements 4.3
        
        Verify that modifying future K/V doesn't affect past outputs (causality).
        """
        # Generate random tensors
        torch.manual_seed(42)  # For reproducibility within this test
        q = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        v = torch.randn((batch, heads, seq_len, d), device="cuda", dtype=torch.float16)
        
        # Compute original causal attention
        out_original = flash_attention(q, k, v, causal=True)
        
        # Modify the second half of K and V
        k_modified = k.clone()
        v_modified = v.clone()
        mid = seq_len // 2
        k_modified[:, :, mid:, :] = torch.randn_like(k_modified[:, :, mid:, :])
        v_modified[:, :, mid:, :] = torch.randn_like(v_modified[:, :, mid:, :])
        
        # Compute attention with modified K, V
        out_modified = flash_attention(q, k_modified, v_modified, causal=True)
        
        # First half of output should be identical (causality)
        first_half_diff = (out_original[:, :, :mid, :] - out_modified[:, :, :mid, :]).abs().max()
        assert first_half_diff < 1e-3, \
            f"Causal violation: modifying future affected past. Diff: {first_half_diff.item()}"


# =============================================================================
# Property 5: Memory Scaling
# Feature: diy-flash-attention, Property 5: Memory Scaling
# Validates: Requirements 5.4
# =============================================================================

class TestMemoryScalingProperty:
    """
    Property 5: Memory Scaling
    
    *For any* sequence length N, FlashAttention memory usage should scale as O(N) 
    rather than O(N²), meaning doubling the sequence length should approximately 
    double (not quadruple) memory usage.
    
    **Validates: Requirements 5.4**
    """
    
    def test_memory_scaling_property(self):
        """
        Feature: diy-flash-attention, Property 5: Memory Scaling
        Validates: Requirements 5.4
        
        Verify that memory scales linearly with sequence length.
        """
        batch, heads, head_dim = 2, 4, 64
        
        # Test different sequence lengths
        seq_lengths = [128, 256, 512]
        memory_usages = []
        
        for seq_len in seq_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
            k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
            v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
            
            # Run FlashAttention
            _ = flash_attention(q, k, v, causal=False)
            
            # Record peak memory
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usages.append(peak_memory)
            
            # Clean up
            del q, k, v
            torch.cuda.empty_cache()
        
        # Check scaling: memory should roughly double when seq_len doubles
        # For O(N²), it would quadruple
        # Allow some tolerance for overhead
        for i in range(1, len(seq_lengths)):
            ratio = memory_usages[i] / memory_usages[i-1]
            seq_ratio = seq_lengths[i] / seq_lengths[i-1]
            
            # For O(N) scaling, memory ratio should be close to seq_ratio
            # For O(N²) scaling, memory ratio would be seq_ratio²
            # We check that ratio is closer to linear than quadratic
            linear_expected = seq_ratio
            quadratic_expected = seq_ratio ** 2
            
            # Memory should be closer to linear scaling
            # This is a soft check due to memory allocation overhead
            assert ratio < quadratic_expected * 0.8, \
                f"Memory scaling appears quadratic: ratio={ratio:.2f}, " \
                f"linear={linear_expected:.2f}, quadratic={quadratic_expected:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

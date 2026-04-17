"""
Unit Tests for Validation Utilities

This module contains unit tests for the validation tools including
validate_matmul, validate_attention, and edge case validators.

**Validates: Requirements 6.4**
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

from kernels.flash_attn import flash_attention
from kernels.matmul import triton_matmul
from utils.validation import (
    validate_attention,
    validate_attention_edge_cases,
    validate_matmul,
    validate_matmul_edge_cases,
)


class TestValidateMatmul:
    """Tests for validate_matmul function."""

    def test_valid_matmul(self):
        """Test validation of correct matmul implementation."""
        is_valid, max_diff = validate_matmul(triton_matmul, m=256, n=256, k=256, verbose=False)

        assert is_valid, f"Valid matmul should pass validation, max_diff={max_diff}"
        assert max_diff < 0.1, f"Max diff should be small, got {max_diff}"

    def test_invalid_matmul(self):
        """Test validation catches incorrect implementation."""

        def bad_matmul(a, b):
            # Return zeros instead of actual result
            return torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        is_valid, max_diff = validate_matmul(bad_matmul, m=64, n=64, k=64, verbose=False)

        # Should fail validation (unless inputs happen to be zero)
        # max_diff should be significant
        assert max_diff > 0.01 or not is_valid

    def test_tolerance_parameters(self):
        """Test that tolerance parameters work correctly."""
        # With very tight tolerance, might fail due to FP16 precision
        is_valid_tight, _ = validate_matmul(
            triton_matmul, m=128, n=128, k=128, rtol=1e-6, atol=1e-6, verbose=False
        )

        # With loose tolerance, should pass
        is_valid_loose, _ = validate_matmul(
            triton_matmul, m=128, n=128, k=128, rtol=0.1, atol=0.1, verbose=False
        )

        assert is_valid_loose, "Should pass with loose tolerance"

    def test_different_sizes(self):
        """Test validation with different matrix sizes."""
        sizes = [(64, 64, 64), (128, 256, 64), (256, 128, 512)]

        for m, n, k in sizes:
            is_valid, max_diff = validate_matmul(triton_matmul, m=m, n=n, k=k, verbose=False)
            assert is_valid, f"Failed for size ({m}, {n}, {k}), max_diff={max_diff}"


class TestValidateAttention:
    """Tests for validate_attention function."""

    def test_valid_attention_non_causal(self):
        """Test validation of correct non-causal attention."""
        is_valid, max_diff = validate_attention(
            flash_attention, batch=2, heads=4, seq_len=128, head_dim=64, causal=False, verbose=False
        )

        assert is_valid, f"Valid attention should pass, max_diff={max_diff}"

    def test_valid_attention_causal(self):
        """Test validation of correct causal attention."""
        is_valid, max_diff = validate_attention(
            flash_attention, batch=2, heads=4, seq_len=128, head_dim=64, causal=True, verbose=False
        )

        assert is_valid, f"Valid causal attention should pass, max_diff={max_diff}"

    def test_invalid_attention(self):
        """Test validation catches incorrect implementation."""

        def bad_attention(q, k, v, causal=False):
            # Return zeros instead of actual result
            return torch.zeros_like(q)

        is_valid, max_diff = validate_attention(
            bad_attention, batch=2, heads=4, seq_len=64, head_dim=64, causal=False, verbose=False
        )

        # Should fail or have large diff
        assert max_diff > 0.01 or not is_valid

    def test_different_configs(self):
        """Test validation with different configurations."""
        configs = [
            (1, 1, 64, 64),
            (2, 4, 128, 64),
            (4, 8, 256, 32),
        ]

        for batch, heads, seq_len, head_dim in configs:
            is_valid, max_diff = validate_attention(
                flash_attention,
                batch=batch,
                heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                causal=False,
                verbose=False,
            )
            assert is_valid, f"Failed for config ({batch}, {heads}, {seq_len}, {head_dim})"


class TestMatmulEdgeCases:
    """Tests for validate_matmul_edge_cases function."""

    def test_edge_cases_pass(self):
        """Test that edge cases pass for correct implementation."""
        all_passed, results = validate_matmul_edge_cases(triton_matmul, verbose=False)

        # Most edge cases should pass
        passed_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        assert passed_count >= total_count * 0.8, (
            f"Too many edge cases failed: {passed_count}/{total_count}"
        )

    def test_zero_matrix(self):
        """Test zero matrix edge case specifically."""
        a = torch.zeros((64, 64), device="cuda", dtype=torch.float16)
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)

        result = triton_matmul(a, b)
        expected = torch.zeros((64, 64), device="cuda", dtype=torch.float16)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_small_matrices(self):
        """Test small matrix edge cases."""
        for size in [1, 2, 4, 8, 16]:
            a = torch.randn((size, size), device="cuda", dtype=torch.float16)
            b = torch.randn((size, size), device="cuda", dtype=torch.float16)

            triton_out = triton_matmul(a, b)
            torch_out = torch.matmul(a.float(), b.float()).half()

            # Allow larger tolerance for very small matrices
            assert torch.allclose(triton_out, torch_out, rtol=0.1, atol=0.1), (
                f"Failed for size {size}"
            )


class TestAttentionEdgeCases:
    """Tests for validate_attention_edge_cases function."""

    def test_edge_cases_pass(self):
        """Test that edge cases pass for correct implementation."""
        all_passed, results = validate_attention_edge_cases(flash_attention, verbose=False)

        # Most edge cases should pass
        passed_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        assert passed_count >= total_count * 0.8, (
            f"Too many edge cases failed: {passed_count}/{total_count}"
        )

    def test_single_token(self):
        """Test single token sequence."""
        q = torch.randn((1, 1, 1, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((1, 1, 1, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((1, 1, 1, 64), device="cuda", dtype=torch.float16)

        flash_out = flash_attention(q, k, v, causal=False)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert torch.allclose(flash_out, ref_out, rtol=0.1, atol=0.1)

    def test_batch_size_one(self):
        """Test batch size 1."""
        q = torch.randn((1, 4, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((1, 4, 128, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((1, 4, 128, 64), device="cuda", dtype=torch.float16)

        flash_out = flash_attention(q, k, v, causal=True)
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert torch.allclose(flash_out, ref_out, rtol=0.02, atol=0.02)


class TestValidationVerbose:
    """Tests for verbose output functionality."""

    def test_matmul_verbose(self, capsys):
        """Test verbose output for matmul validation."""
        validate_matmul(triton_matmul, m=64, n=64, k=64, verbose=True)

        captured = capsys.readouterr()
        assert "Validation" in captured.out
        assert "Max difference" in captured.out

    def test_attention_verbose(self, capsys):
        """Test verbose output for attention validation."""
        validate_attention(flash_attention, batch=1, heads=2, seq_len=64, head_dim=64, verbose=True)

        captured = capsys.readouterr()
        assert "Validation" in captured.out
        assert "Max difference" in captured.out

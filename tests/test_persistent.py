"""Tests for persistent kernel implementations."""

from __future__ import annotations

import pytest
import torch

from kernels import persistent_matmul


class TestPersistentMatmul:
    """Tests for persistent matmul kernel."""

    @pytest.mark.cuda
    def test_basic_correctness(self):
        """Test persistent matmul correctness against PyTorch."""
        torch.manual_seed(42)
        m, k, n = 512, 512, 512

        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")

        output = persistent_matmul(a, b)
        baseline = torch.mm(a, b)

        assert output.shape == baseline.shape
        assert torch.allclose(output, baseline, rtol=1e-4, atol=1e-5)

    @pytest.mark.cuda
    def test_rectangular_matrices(self):
        """Test persistent matmul with rectangular matrices."""
        torch.manual_seed(42)
        m, k, n = 256, 128, 512

        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")

        output = persistent_matmul(a, b)
        baseline = torch.mm(a, b)

        assert torch.allclose(output, baseline, rtol=1e-4, atol=1e-5)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_different_dtypes(self, dtype):
        """Test persistent matmul with different dtypes."""
        torch.manual_seed(42)
        m, k, n = 256, 256, 256

        a = torch.randn(m, k, dtype=dtype, device="cuda")
        b = torch.randn(k, n, dtype=dtype, device="cuda")

        output = persistent_matmul(a, b)
        baseline = torch.mm(a, b)

        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        assert torch.allclose(output, baseline, rtol=rtol, atol=1e-5)

    @pytest.mark.cuda
    def test_custom_block_sizes(self):
        """Test persistent matmul with custom block sizes."""
        torch.manual_seed(42)
        m, k, n = 512, 512, 512

        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")

        output = persistent_matmul(a, b, block_m=32, block_n=32)
        baseline = torch.mm(a, b)

        assert torch.allclose(output, baseline, rtol=1e-4, atol=1e-5)


class TestPersistentMatmulEdgeCases:
    """Edge case tests for persistent matmul."""

    @pytest.mark.cuda
    def test_small_matrices(self):
        """Test persistent matmul with small matrices."""
        torch.manual_seed(42)
        m, k, n = 32, 32, 32

        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")

        output = persistent_matmul(a, b)
        baseline = torch.mm(a, b)

        assert torch.allclose(output, baseline, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

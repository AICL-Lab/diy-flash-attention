"""Tests for Backend Selector - unified kernel dispatch registry."""

from __future__ import annotations

import pytest

from kernels import BackendSelector, KernelVariant, select_attention_kernel


class TestBackendSelectorBasic:
    """Basic functionality tests for BackendSelector."""

    def test_select_attention_v1(self):
        """Test selector returns v1 kernel when requested."""
        kernel = BackendSelector.select_attention(variant=KernelVariant.V1)
        assert callable(kernel)
        assert kernel.__name__ == "flash_attention"

    def test_select_attention_v2(self):
        """Test selector returns v2 kernel when requested."""
        kernel = BackendSelector.select_attention(variant=KernelVariant.V2)
        assert callable(kernel)
        assert kernel.__name__ == "flash_attention_v2"


class TestBackendSelectorHeuristics:
    """Test automatic kernel selection heuristics."""

    def test_auto_selection_small_problem(self):
        """Test auto selects V1 for small problems."""
        kernel = BackendSelector.select_attention(
            variant=KernelVariant.AUTO,
            q_shape=(1, 128, 8, 64),  # Small batch, short seq
            gpu_capability=80,  # Ampere
        )
        assert kernel.__name__ == "flash_attention"

    def test_auto_selection_large_problem(self):
        """Test auto selects V2 for large problems on Ampere+."""
        kernel = BackendSelector.select_attention(
            variant=KernelVariant.AUTO,
            q_shape=(8, 4096, 8, 64),  # Large batch, long seq
            gpu_capability=80,  # Ampere
        )
        assert kernel.__name__ == "flash_attention_v2"

    def test_auto_selection_pre_ampere_uses_v1(self):
        """Test auto selects V1 for pre-Ampere GPUs."""
        kernel = BackendSelector.select_attention(
            variant=KernelVariant.AUTO,
            q_shape=(8, 4096, 8, 64),  # Large problem
            gpu_capability=75,  # Turing
        )
        assert kernel.__name__ == "flash_attention"


class TestKernelVariant:
    """Test KernelVariant enum."""

    def test_kernel_variant_values(self):
        """Test all variant values are defined."""
        assert KernelVariant.V1.value == "v1"
        assert KernelVariant.V2.value == "v2"
        assert KernelVariant.PERSISTENT.value == "persistent"
        assert KernelVariant.AUTO.value == "auto"


class TestConvenienceFunction:
    """Test convenience wrapper function."""

    def test_select_attention_kernel_v1(self):
        """Test string-based variant selection."""
        kernel = select_attention_kernel(variant="v1")
        assert callable(kernel)
        assert kernel.__name__ == "flash_attention"

    def test_select_attention_kernel_v2(self):
        """Test string-based variant selection for v2."""
        kernel = select_attention_kernel(variant="v2")
        assert callable(kernel)
        assert kernel.__name__ == "flash_attention_v2"

    def test_select_attention_kernel_auto(self):
        """Test auto selection via convenience function."""
        kernel = select_attention_kernel(variant="auto", q_shape=(1, 256, 8, 64))
        assert callable(kernel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit Tests for Benchmark Utilities

This module contains unit tests for the benchmark tools including
TFLOPS calculation, BenchmarkResult, and BenchmarkRunner.

**Validates: Requirements 2.2**
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

from utils.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    calculate_matmul_flops,
    calculate_attention_flops,
    benchmark_fn,
)


class TestFlopsCalculation:
    """Tests for FLOPS calculation functions."""
    
    def test_matmul_flops_square(self):
        """Test FLOPS calculation for square matrices."""
        M, N, K = 1024, 1024, 1024
        flops = calculate_matmul_flops(M, N, K)
        
        # Expected: 2 * M * N * K
        expected = 2 * 1024 * 1024 * 1024
        assert flops == expected
    
    def test_matmul_flops_rectangular(self):
        """Test FLOPS calculation for rectangular matrices."""
        M, N, K = 512, 1024, 256
        flops = calculate_matmul_flops(M, N, K)
        
        expected = 2 * 512 * 1024 * 256
        assert flops == expected
    
    def test_matmul_flops_small(self):
        """Test FLOPS calculation for small matrices."""
        M, N, K = 1, 1, 1
        flops = calculate_matmul_flops(M, N, K)
        
        assert flops == 2
    
    def test_attention_flops(self):
        """Test FLOPS calculation for attention."""
        batch, heads, seq_len, head_dim = 2, 8, 512, 64
        flops = calculate_attention_flops(batch, heads, seq_len, head_dim)
        
        # Should be positive and reasonable
        assert flops > 0
        
        # QK^T: 2 * batch * heads * seq_len * seq_len * head_dim
        qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
        # Softmax: ~5 * batch * heads * seq_len * seq_len
        softmax_flops = 5 * batch * heads * seq_len * seq_len
        # AV: 2 * batch * heads * seq_len * head_dim * seq_len
        av_flops = 2 * batch * heads * seq_len * head_dim * seq_len
        
        expected = qk_flops + softmax_flops + av_flops
        assert flops == expected
    
    def test_attention_flops_scaling(self):
        """Test that attention FLOPS scales quadratically with seq_len."""
        batch, heads, head_dim = 1, 1, 64
        
        flops_128 = calculate_attention_flops(batch, heads, 128, head_dim)
        flops_256 = calculate_attention_flops(batch, heads, 256, head_dim)
        
        # Doubling seq_len should roughly quadruple FLOPS (due to N² attention)
        ratio = flops_256 / flops_128
        assert 3.5 < ratio < 4.5  # Allow some tolerance


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_creation(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            name="Test",
            size=(1024, 1024, 1024),
            time_ms=1.5,
            tflops=100.0,
        )
        
        assert result.name == "Test"
        assert result.size == (1024, 1024, 1024)
        assert result.time_ms == 1.5
        assert result.tflops == 100.0
        assert result.memory_mb == 0.0  # Default
    
    def test_with_memory(self):
        """Test BenchmarkResult with memory field."""
        result = BenchmarkResult(
            name="Test",
            size=(2, 8, 512, 64),
            time_ms=2.0,
            tflops=50.0,
            memory_mb=256.0,
        )
        
        assert result.memory_mb == 256.0
    
    def test_with_block_config(self):
        """Test BenchmarkResult with block configuration."""
        config = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}
        result = BenchmarkResult(
            name="Triton",
            size=(1024, 1024, 1024),
            time_ms=1.0,
            tflops=200.0,
            block_config=config,
        )
        
        assert result.block_config == config
    
    def test_str_representation(self):
        """Test string representation."""
        result = BenchmarkResult(
            name="Test",
            size=(512, 512, 512),
            time_ms=1.234,
            tflops=99.99,
        )
        
        str_repr = str(result)
        assert "Test" in str_repr
        assert "512×512×512" in str_repr
        assert "1.234" in str_repr
        assert "99.99" in str_repr


class TestBenchmarkFn:
    """Tests for benchmark_fn utility."""
    
    def test_benchmark_simple_fn(self):
        """Test benchmarking a simple function."""
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        
        ms, min_ms, max_ms = benchmark_fn(
            torch.matmul, a, b,
            warmup=5, rep=10
        )
        
        assert ms > 0
        assert min_ms > 0
        assert max_ms > 0
        assert min_ms <= ms <= max_ms
    
    def test_benchmark_with_kwargs(self):
        """Test benchmarking with keyword arguments."""
        x = torch.randn(64, 64, device="cuda")
        
        ms, _, _ = benchmark_fn(
            torch.sum, x,
            dim=0,
            warmup=5, rep=10
        )
        
        assert ms > 0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""
    
    def test_initialization(self):
        """Test BenchmarkRunner initialization."""
        runner = BenchmarkRunner(warmup=10, rep=50)
        
        assert runner.device == "cuda"
        assert runner.warmup == 10
        assert runner.rep == 50
        assert runner.results == []
    
    def test_clear_results(self):
        """Test clearing results."""
        runner = BenchmarkRunner()
        runner.results = [
            BenchmarkResult("Test", (64, 64, 64), 1.0, 10.0)
        ]
        
        runner.clear_results()
        assert runner.results == []
    
    def test_benchmark_matmul_basic(self):
        """Test basic matmul benchmarking."""
        from kernels.matmul import triton_matmul
        
        runner = BenchmarkRunner(warmup=5, rep=10)
        results = runner.benchmark_matmul(
            triton_matmul,
            sizes=[(64, 64, 64)],
        )
        
        # Should have PyTorch and Triton results
        assert len(results) >= 2
        
        # Check result structure
        for r in results:
            assert r.time_ms > 0
            assert r.tflops > 0
            assert r.size == (64, 64, 64)
    
    def test_benchmark_attention_basic(self):
        """Test basic attention benchmarking."""
        from kernels.flash_attn import flash_attention
        
        runner = BenchmarkRunner(warmup=5, rep=10)
        results = runner.benchmark_attention(
            flash_attention,
            seq_lengths=[64],
            batch_size=1,
            num_heads=2,
            head_dim=32,
        )
        
        # Should have PyTorch SDPA and FlashAttention results
        assert len(results) >= 2
        
        # Check result structure
        for r in results:
            assert r.time_ms > 0
            assert r.tflops > 0
    
    def test_results_accumulation(self):
        """Test that results accumulate across benchmarks."""
        from kernels.matmul import triton_matmul
        
        runner = BenchmarkRunner(warmup=5, rep=10)
        
        # First benchmark
        runner.benchmark_matmul(triton_matmul, sizes=[(64, 64, 64)])
        count1 = len(runner.results)
        
        # Second benchmark
        runner.benchmark_matmul(triton_matmul, sizes=[(128, 128, 128)])
        count2 = len(runner.results)
        
        assert count2 > count1
    
    def test_print_comparison_table(self, capsys):
        """Test printing comparison table."""
        runner = BenchmarkRunner()
        runner.results = [
            BenchmarkResult("PyTorch", (64, 64, 64), 1.0, 10.0),
            BenchmarkResult("Triton", (64, 64, 64), 0.5, 20.0),
        ]
        
        runner.print_comparison_table(title="Test Results")
        
        captured = capsys.readouterr()
        assert "Test Results" in captured.out
        assert "PyTorch" in captured.out
        assert "Triton" in captured.out
        assert "Speedup" in captured.out


class TestTflopsCalculation:
    """Tests for TFLOPS calculation correctness."""
    
    def test_tflops_from_time_and_flops(self):
        """Test TFLOPS calculation from time and FLOPS."""
        M, N, K = 1024, 1024, 1024
        flops = calculate_matmul_flops(M, N, K)
        
        # If operation takes 1ms, TFLOPS = flops / 1e-3 / 1e12 = flops / 1e9
        time_ms = 1.0
        tflops = flops / time_ms / 1e9
        
        # 2 * 1024^3 / 1e9 ≈ 2.147 TFLOPS
        expected_tflops = 2 * (1024 ** 3) / 1e9
        assert abs(tflops - expected_tflops) < 0.001
    
    def test_tflops_scaling_with_time(self):
        """Test that TFLOPS scales inversely with time."""
        flops = 1e12  # 1 TFLOP
        
        tflops_1ms = flops / 1.0 / 1e9  # 1000 TFLOPS
        tflops_2ms = flops / 2.0 / 1e9  # 500 TFLOPS
        
        assert tflops_1ms == 2 * tflops_2ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Benchmark utilities for comparing Triton kernels with PyTorch.

This module provides tools for measuring and comparing performance of
matrix multiplication and attention implementations.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import triton


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""
    name: str
    size: tuple  # (M, N, K) for matmul or (batch, heads, seq_len, head_dim) for attention
    time_ms: float
    tflops: float
    memory_mb: float = 0.0
    block_config: Optional[dict] = None
    
    def __str__(self) -> str:
        size_str = "×".join(map(str, self.size))
        return f"{self.name}: {size_str} | {self.time_ms:.3f} ms | {self.tflops:.2f} TFLOPS"


def calculate_matmul_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPs for matrix multiplication C = A @ B."""
    # Each element of C requires K multiplications and K-1 additions
    # Total: 2 * M * N * K FLOPs
    return 2 * M * N * K


def calculate_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """
    Calculate FLOPs for attention computation.
    
    Attention = softmax(Q @ K^T / sqrt(d)) @ V
    - Q @ K^T: 2 * batch * heads * seq_len * seq_len * head_dim
    - softmax: ~5 * batch * heads * seq_len * seq_len (exp, sum, div)
    - attn @ V: 2 * batch * heads * seq_len * head_dim * seq_len
    """
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * heads * seq_len * seq_len
    av_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    return qk_flops + softmax_flops + av_flops


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 25,
    rep: int = 100,
    quantiles: Optional[list] = None,
    **kwargs,
) -> tuple[float, float, float]:
    """
    Benchmark a function using Triton's benchmarking utilities.
    
    Args:
        fn: Function to benchmark
        *args: Positional arguments to pass to fn
        warmup: Number of warmup iterations
        rep: Number of repetitions for timing
        quantiles: Quantiles to compute (default: [0.5, 0.2, 0.8])
        **kwargs: Keyword arguments to pass to fn
        
    Returns:
        Tuple of (median_ms, min_ms, max_ms)
    """
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]
    
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(*args, **kwargs),
        warmup=warmup,
        rep=rep,
        quantiles=quantiles,
    )
    return ms, min_ms, max_ms


class BenchmarkRunner:
    """Runs and compares benchmarks between implementations."""
    
    def __init__(self, device: str = "cuda", warmup: int = 25, rep: int = 100):
        """
        Initialize benchmark runner.
        
        Args:
            device: Device to run benchmarks on
            warmup: Number of warmup iterations
            rep: Number of repetitions for timing
        """
        self.device = device
        self.warmup = warmup
        self.rep = rep
        self.results: list[BenchmarkResult] = []
    
    def benchmark_matmul(
        self,
        triton_fn: Callable,
        sizes: list[tuple[int, int, int]],
        block_configs: Optional[list[dict]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """
        Benchmark matrix multiplication implementations.
        
        Args:
            triton_fn: Triton matmul function to benchmark
            sizes: List of (M, N, K) tuples to test
            block_configs: Optional list of block size configurations
            dtype: Data type for matrices
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for M, N, K in sizes:
            # Generate random matrices
            a = torch.randn((M, K), device=self.device, dtype=dtype)
            b = torch.randn((K, N), device=self.device, dtype=dtype)
            
            flops = calculate_matmul_flops(M, N, K)
            
            # Benchmark PyTorch
            torch_ms, _, _ = benchmark_fn(
                torch.matmul, a, b,
                warmup=self.warmup, rep=self.rep
            )
            torch_tflops = flops / torch_ms / 1e9
            
            results.append(BenchmarkResult(
                name="PyTorch",
                size=(M, N, K),
                time_ms=torch_ms,
                tflops=torch_tflops,
            ))
            
            # Benchmark Triton (autotuned)
            triton_ms, _, _ = benchmark_fn(
                triton_fn, a, b,
                warmup=self.warmup, rep=self.rep
            )
            triton_tflops = flops / triton_ms / 1e9
            
            results.append(BenchmarkResult(
                name="Triton",
                size=(M, N, K),
                time_ms=triton_ms,
                tflops=triton_tflops,
            ))
            
            # Benchmark with specific block configs if provided
            if block_configs:
                for config in block_configs:
                    config_ms, _, _ = benchmark_fn(
                        triton_fn, a, b,
                        block_m=config.get("BLOCK_M", 128),
                        block_n=config.get("BLOCK_N", 256),
                        block_k=config.get("BLOCK_K", 64),
                        warmup=self.warmup, rep=self.rep
                    )
                    config_tflops = flops / config_ms / 1e9
                    
                    config_name = f"Triton({config.get('BLOCK_M')}×{config.get('BLOCK_N')}×{config.get('BLOCK_K')})"
                    results.append(BenchmarkResult(
                        name=config_name,
                        size=(M, N, K),
                        time_ms=config_ms,
                        tflops=config_tflops,
                        block_config=config,
                    ))
        
        self.results.extend(results)
        return results
    
    def benchmark_attention(
        self,
        flash_fn: Callable,
        seq_lengths: list[int],
        batch_size: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        causal: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """
        Benchmark attention implementations.
        
        Args:
            flash_fn: FlashAttention function to benchmark
            seq_lengths: List of sequence lengths to test
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            causal: Whether to use causal masking
            dtype: Data type for tensors
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for seq_len in seq_lengths:
            # Generate random Q, K, V
            q = torch.randn((batch_size, num_heads, seq_len, head_dim), 
                           device=self.device, dtype=dtype)
            k = torch.randn((batch_size, num_heads, seq_len, head_dim),
                           device=self.device, dtype=dtype)
            v = torch.randn((batch_size, num_heads, seq_len, head_dim),
                           device=self.device, dtype=dtype)
            
            flops = calculate_attention_flops(batch_size, num_heads, seq_len, head_dim)
            size = (batch_size, num_heads, seq_len, head_dim)
            
            # Benchmark PyTorch SDPA
            torch_ms, _, _ = benchmark_fn(
                torch.nn.functional.scaled_dot_product_attention,
                q, k, v, is_causal=causal,
                warmup=self.warmup, rep=self.rep
            )
            torch_tflops = flops / torch_ms / 1e9
            
            # Measure memory for PyTorch
            torch.cuda.reset_peak_memory_stats()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            torch_mem = torch.cuda.max_memory_allocated() / 1e6
            
            results.append(BenchmarkResult(
                name="PyTorch SDPA",
                size=size,
                time_ms=torch_ms,
                tflops=torch_tflops,
                memory_mb=torch_mem,
            ))
            
            # Benchmark FlashAttention
            flash_ms, _, _ = benchmark_fn(
                flash_fn, q, k, v, causal=causal,
                warmup=self.warmup, rep=self.rep
            )
            flash_tflops = flops / flash_ms / 1e9
            
            # Measure memory for FlashAttention
            torch.cuda.reset_peak_memory_stats()
            _ = flash_fn(q, k, v, causal=causal)
            flash_mem = torch.cuda.max_memory_allocated() / 1e6
            
            results.append(BenchmarkResult(
                name="FlashAttention",
                size=size,
                time_ms=flash_ms,
                tflops=flash_tflops,
                memory_mb=flash_mem,
            ))
        
        self.results.extend(results)
        return results
    
    def print_comparison_table(
        self,
        results: Optional[list[BenchmarkResult]] = None,
        title: str = "Benchmark Results",
    ) -> None:
        """Print formatted comparison table."""
        if results is None:
            results = self.results
        
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
        
        # Group results by size
        sizes = sorted(set(r.size for r in results))
        
        # Header
        print(f"{'Size':<20} | {'Implementation':<25} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'Memory (MB)':<12}")
        print("-" * 80)
        
        for size in sizes:
            size_results = [r for r in results if r.size == size]
            size_str = "×".join(map(str, size))
            
            for i, r in enumerate(size_results):
                if i == 0:
                    print(f"{size_str:<20} | {r.name:<25} | {r.time_ms:<12.3f} | {r.tflops:<10.2f} | {r.memory_mb:<12.1f}")
                else:
                    print(f"{'':<20} | {r.name:<25} | {r.time_ms:<12.3f} | {r.tflops:<10.2f} | {r.memory_mb:<12.1f}")
            
            # Print speedup if we have both PyTorch and Triton results
            pytorch_result = next((r for r in size_results if "PyTorch" in r.name), None)
            triton_result = next((r for r in size_results if r.name == "Triton" or r.name == "FlashAttention"), None)
            
            if pytorch_result and triton_result:
                speedup = pytorch_result.time_ms / triton_result.time_ms
                print(f"{'':<20} | {'Speedup:':<25} | {speedup:<12.2f}x |")
            
            print("-" * 80)
        
        print("=" * 80 + "\n")
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []


if __name__ == "__main__":
    # Quick test
    runner = BenchmarkRunner()
    
    print("Testing BenchmarkResult...")
    result = BenchmarkResult(
        name="Test",
        size=(1024, 1024, 1024),
        time_ms=1.5,
        tflops=100.0,
    )
    print(result)
    print("✓ BenchmarkResult works!")

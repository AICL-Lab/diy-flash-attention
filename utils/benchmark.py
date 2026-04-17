"""Benchmark utilities for comparing Triton kernels with PyTorch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import torch

from utils.config import BENCHMARK_QUANTILES, BENCHMARK_REPETITIONS, BENCHMARK_WARMUP, BYTES_PER_MB

logger = logging.getLogger(__name__)

try:
    import triton

    TRITON_AVAILABLE = True
except ModuleNotFoundError:
    TRITON_AVAILABLE = False
    triton = SimpleNamespace(testing=SimpleNamespace(do_bench=None))


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise ModuleNotFoundError("triton is required to run benchmark timing utilities.")


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""

    name: str
    size: tuple
    time_ms: float
    tflops: float
    memory_mb: float = 0.0
    block_config: Optional[dict] = None

    def __str__(self) -> str:
        size_str = "×".join(map(str, self.size))
        return f"{self.name}: {size_str} | {self.time_ms:.3f} ms | {self.tflops:.2f} TFLOPS"


def calculate_matmul_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPs for matrix multiplication C = A @ B."""
    return 2 * M * N * K


def calculate_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Calculate FLOPs for attention computation."""
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * heads * seq_len * seq_len
    av_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    return qk_flops + softmax_flops + av_flops


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = BENCHMARK_WARMUP,
    rep: int = BENCHMARK_REPETITIONS,
    quantiles: Optional[list] = None,
    **kwargs,
) -> tuple[float, float, float]:
    """Benchmark a function and return `(median_ms, p20_ms, p80_ms)`."""
    do_bench = getattr(getattr(triton, "testing", None), "do_bench", None)
    if do_bench is None or not callable(do_bench):
        _require_triton()
        raise RuntimeError("triton.testing.do_bench is not available")

    if quantiles is None:
        quantiles = BENCHMARK_QUANTILES

    logger.debug(f"Benchmarking {fn.__name__}: warmup={warmup}, rep={rep}")

    median_ms, p20_ms, p80_ms = do_bench(
        lambda: fn(*args, **kwargs),
        warmup=warmup,
        rep=rep,
        quantiles=quantiles,
    )
    return median_ms, p20_ms, p80_ms


class BenchmarkRunner:
    """Runs and compares benchmarks between implementations."""

    def __init__(
        self,
        device: str = "cuda",
        warmup: int = BENCHMARK_WARMUP,
        rep: int = BENCHMARK_REPETITIONS,
    ):
        self.device = device
        self.warmup = warmup
        self.rep = rep
        self.results: list[BenchmarkResult] = []
        logger.debug(f"BenchmarkRunner initialized: device={device}, warmup={warmup}, rep={rep}")

    def benchmark_matmul(
        self,
        triton_fn: Callable,
        sizes: list[tuple[int, int, int]],
        block_configs: Optional[list[dict]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        results = []

        for M, N, K in sizes:
            a = torch.randn((M, K), device=self.device, dtype=dtype)
            b = torch.randn((K, N), device=self.device, dtype=dtype)
            flops = calculate_matmul_flops(M, N, K)

            torch_ms, _, _ = benchmark_fn(torch.matmul, a, b, warmup=self.warmup, rep=self.rep)
            torch_tflops = flops / torch_ms / 1e9
            results.append(
                BenchmarkResult(
                    name="PyTorch", size=(M, N, K), time_ms=torch_ms, tflops=torch_tflops
                )
            )

            triton_ms, _, _ = benchmark_fn(triton_fn, a, b, warmup=self.warmup, rep=self.rep)
            triton_tflops = flops / triton_ms / 1e9
            results.append(
                BenchmarkResult(
                    name="Triton", size=(M, N, K), time_ms=triton_ms, tflops=triton_tflops
                )
            )

            if block_configs:
                for config in block_configs:
                    config_ms, _, _ = benchmark_fn(
                        triton_fn,
                        a,
                        b,
                        block_m=config.get("BLOCK_M", 128),
                        block_n=config.get("BLOCK_N", 256),
                        block_k=config.get("BLOCK_K", 64),
                        warmup=self.warmup,
                        rep=self.rep,
                    )
                    config_tflops = flops / config_ms / 1e9
                    config_name = f"Triton({config.get('BLOCK_M')}×{config.get('BLOCK_N')}×{config.get('BLOCK_K')})"
                    results.append(
                        BenchmarkResult(
                            name=config_name,
                            size=(M, N, K),
                            time_ms=config_ms,
                            tflops=config_tflops,
                            block_config=config,
                        )
                    )

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
        results = []

        for seq_len in seq_lengths:
            q = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=self.device, dtype=dtype
            )
            k = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=self.device, dtype=dtype
            )
            v = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=self.device, dtype=dtype
            )

            flops = calculate_attention_flops(batch_size, num_heads, seq_len, head_dim)
            size = (batch_size, num_heads, seq_len, head_dim)

            torch_ms, _, _ = benchmark_fn(
                torch.nn.functional.scaled_dot_product_attention,
                q,
                k,
                v,
                is_causal=causal,
                warmup=self.warmup,
                rep=self.rep,
            )
            torch_tflops = flops / torch_ms / 1e9

            torch.cuda.reset_peak_memory_stats()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            torch_mem = torch.cuda.max_memory_allocated() / BYTES_PER_MB
            results.append(
                BenchmarkResult(
                    name="PyTorch SDPA",
                    size=size,
                    time_ms=torch_ms,
                    tflops=torch_tflops,
                    memory_mb=torch_mem,
                )
            )

            flash_ms, _, _ = benchmark_fn(
                flash_fn, q, k, v, causal=causal, warmup=self.warmup, rep=self.rep
            )
            flash_tflops = flops / flash_ms / 1e9

            torch.cuda.reset_peak_memory_stats()
            _ = flash_fn(q, k, v, causal=causal)
            flash_mem = torch.cuda.max_memory_allocated() / BYTES_PER_MB
            results.append(
                BenchmarkResult(
                    name="FlashAttention",
                    size=size,
                    time_ms=flash_ms,
                    tflops=flash_tflops,
                    memory_mb=flash_mem,
                )
            )

        self.results.extend(results)
        return results

    def print_comparison_table(
        self,
        results: Optional[list[BenchmarkResult]] = None,
        title: str = "Benchmark Results",
    ) -> None:
        if results is None:
            results = self.results

        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
        print(
            f"{'Size':<20} | {'Implementation':<25} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'Memory (MB)':<12}"
        )
        print("-" * 80)

        sizes = sorted({r.size for r in results})
        for size in sizes:
            size_results = [r for r in results if r.size == size]
            size_str = "×".join(map(str, size))

            for i, result in enumerate(size_results):
                prefix = size_str if i == 0 else ""
                print(
                    f"{prefix:<20} | {result.name:<25} | {result.time_ms:<12.3f} | {result.tflops:<10.2f} | {result.memory_mb:<12.1f}"
                )

            pytorch_result = next((r for r in size_results if "PyTorch" in r.name), None)
            triton_result = next(
                (r for r in size_results if r.name == "Triton" or r.name == "FlashAttention"), None
            )
            if pytorch_result and triton_result:
                speedup = pytorch_result.time_ms / triton_result.time_ms
                print(f"{'':<20} | {'Speedup:':<25} | {speedup:<12.2f}x |")

            print("-" * 80)

        print("=" * 80 + "\n")

    def clear_results(self) -> None:
        self.results = []


if __name__ == "__main__":
    runner = BenchmarkRunner()
    result = BenchmarkResult(name="Test", size=(1024, 1024, 1024), time_ms=1.5, tflops=100.0)
    print(result)

#!/usr/bin/env python3
"""
FlashAttention Benchmark Script

Compares FlashAttention performance against PyTorch's scaled_dot_product_attention.
Tests various sequence lengths and measures memory usage.

Usage:
    python benchmarks/bench_flash.py
    python benchmarks/bench_flash.py --seq-lengths 128 256 512 1024 2048
    python benchmarks/bench_flash.py --causal
"""

import argparse
import gc
import sys

import torch

from kernels.flash_attn import flash_attention
from utils.benchmark import (
    BenchmarkRunner,
)
from utils.gpu_detect import detect_gpu, print_gpu_info
from utils.validation import validate_attention


def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention Benchmark")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Dimension of each head",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal masking",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Number of repetitions for timing",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate correctness before benchmarking",
    )
    parser.add_argument(
        "--memory-test",
        action="store_true",
        default=True,
        help="Run memory scaling test",
    )
    return parser.parse_args()


def measure_memory(fn, *args, **kwargs):
    """Measure peak memory usage of a function."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = fn(*args, **kwargs)
    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return peak_memory


def run_memory_scaling_test(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    device: str = "cuda",
):
    """
    Test memory scaling of FlashAttention vs naive attention.

    FlashAttention should scale as O(N) while naive scales as O(N²).
    """
    print("\n" + "=" * 60)
    print(" Memory Scaling Test")
    print("=" * 60)
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, causal={causal}")
    print("-" * 60)

    seq_lengths = [128, 256, 512, 1024, 2048]

    # Filter out sequence lengths that would cause OOM
    max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

    results = []

    print(
        f"{'Seq Len':<10} | {'PyTorch (MB)':<15} | {'Flash (MB)':<15} | {'Ratio':<10} | {'Scaling':<15}"
    )
    print("-" * 70)

    prev_pytorch_mem = None
    prev_flash_mem = None
    prev_seq_len = None

    for seq_len in seq_lengths:
        # Estimate memory requirement
        estimated_mem = batch_size * num_heads * seq_len * seq_len * 2 / (1024**3)  # GB
        if estimated_mem > max_memory * 0.8:
            print(f"{seq_len:<10} | {'OOM':<15} | {'OOM':<15} | {'N/A':<10} | {'N/A':<15}")
            continue

        try:
            q = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16
            )
            k = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16
            )
            v = torch.randn(
                (batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16
            )

            # Measure PyTorch SDPA memory
            pytorch_mem = measure_memory(
                torch.nn.functional.scaled_dot_product_attention, q, k, v, is_causal=causal
            )

            # Measure FlashAttention memory
            flash_mem = measure_memory(flash_attention, q, k, v, causal=causal)

            ratio = pytorch_mem / flash_mem if flash_mem > 0 else float("inf")

            # Calculate scaling factor
            if prev_seq_len is not None and prev_pytorch_mem is not None:
                seq_ratio = seq_len / prev_seq_len
                pytorch_scale = pytorch_mem / prev_pytorch_mem
                flash_scale = flash_mem / prev_flash_mem if prev_flash_mem > 0 else 1

                # O(N²) would give scale ≈ seq_ratio²
                # O(N) would give scale ≈ seq_ratio
                scaling_info = f"Py:{pytorch_scale:.1f}x, Fl:{flash_scale:.1f}x"
            else:
                scaling_info = "baseline"

            print(
                f"{seq_len:<10} | {pytorch_mem:<15.1f} | {flash_mem:<15.1f} | {ratio:<10.2f}x | {scaling_info:<15}"
            )

            results.append(
                {
                    "seq_len": seq_len,
                    "pytorch_mem": pytorch_mem,
                    "flash_mem": flash_mem,
                    "ratio": ratio,
                }
            )

            prev_pytorch_mem = pytorch_mem
            prev_flash_mem = flash_mem
            prev_seq_len = seq_len

            # Clean up
            del q, k, v
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:<10} | {'OOM':<15} | {'OOM':<15} | {'N/A':<10} | {'N/A':<15}")
            else:
                raise

    print("-" * 70)

    # Analyze scaling
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        seq_ratio = last["seq_len"] / first["seq_len"]

        pytorch_ratio = last["pytorch_mem"] / first["pytorch_mem"]
        flash_ratio = last["flash_mem"] / first["flash_mem"]

        print(
            f"\nScaling Analysis (seq_len: {first['seq_len']} → {last['seq_len']}, ratio: {seq_ratio}x):"
        )
        print(f"  PyTorch memory grew: {pytorch_ratio:.1f}x (O(N²) would be {seq_ratio**2:.1f}x)")
        print(f"  Flash memory grew:   {flash_ratio:.1f}x (O(N) would be {seq_ratio:.1f}x)")

        if flash_ratio < seq_ratio * 1.5:
            print("  ✓ FlashAttention shows approximately O(N) memory scaling!")
        else:
            print("  Note: Memory scaling may be affected by other factors")

    return results


def main():
    args = parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    # Print header
    print("\n" + "=" * 60)
    print(" DIY FlashAttention - Attention Benchmark")
    print("=" * 60)

    try:
        caps = detect_gpu()
        print_gpu_info(caps)
    except (RuntimeError, ImportError) as e:
        print(f"Warning: Could not detect GPU info: {e}")

    # Validate correctness first
    if args.validate:
        print("\n" + "-" * 60)
        print(" Validating Correctness")
        print("-" * 60)

        test_seq = min(args.seq_lengths)
        is_valid, max_diff = validate_attention(
            flash_attention,
            args.batch_size,
            args.num_heads,
            test_seq,
            args.head_dim,
            causal=args.causal,
            verbose=True,
        )

        if is_valid:
            print("✓ Correctness validated")
        else:
            print("✗ Correctness check failed")
            print("  Proceeding with benchmark anyway...")

    # Run performance benchmark
    print("\n" + "-" * 60)
    print(" Running Performance Benchmark")
    print("-" * 60)
    print(f"Config: batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Causal: {args.causal}")
    print(f"Warmup: {args.warmup}, Repetitions: {args.rep}")

    runner = BenchmarkRunner(warmup=args.warmup, rep=args.rep)
    results = runner.benchmark_attention(
        flash_fn=flash_attention,
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        causal=args.causal,
    )

    # Print results
    runner.print_comparison_table(results, title="FlashAttention Benchmark Results")

    # Print summary
    print("\n" + "-" * 60)
    print(" Performance Summary")
    print("-" * 60)

    for seq_len in args.seq_lengths:
        size = (args.batch_size, args.num_heads, seq_len, args.head_dim)
        size_results = [r for r in results if r.size == size]

        pytorch_result = next((r for r in size_results if "PyTorch" in r.name), None)
        flash_result = next((r for r in size_results if "Flash" in r.name), None)

        if pytorch_result and flash_result:
            speedup = pytorch_result.time_ms / flash_result.time_ms
            mem_reduction = (
                pytorch_result.memory_mb / flash_result.memory_mb
                if flash_result.memory_mb > 0
                else float("inf")
            )

            print(f"  Seq={seq_len}:")
            print(f"    Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
            print(f"    Memory:  {mem_reduction:.2f}x reduction")
            print(
                f"    PyTorch: {pytorch_result.tflops:.1f} TFLOPS, {pytorch_result.memory_mb:.1f} MB"
            )
            print(f"    Flash:   {flash_result.tflops:.1f} TFLOPS, {flash_result.memory_mb:.1f} MB")

    # Run memory scaling test
    if args.memory_test:
        run_memory_scaling_test(
            args.batch_size,
            args.num_heads,
            args.head_dim,
            args.causal,
        )

    print("\n" + "=" * 60)
    print(" Benchmark Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

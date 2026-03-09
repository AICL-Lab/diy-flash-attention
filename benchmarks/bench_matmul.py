#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark Script

Compares Triton matmul kernel performance against PyTorch's torch.matmul.
Tests various matrix sizes and block size configurations.

Usage:
    python benchmarks/bench_matmul.py
    python benchmarks/bench_matmul.py --sizes 1024 2048 4096
    python benchmarks/bench_matmul.py --test-block-sizes
"""

import argparse
import sys

import torch

from kernels.matmul import triton_matmul
from utils.benchmark import BenchmarkRunner, BenchmarkResult
from utils.gpu_detect import detect_gpu, print_gpu_info
from utils.validation import validate_matmul


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Multiplication Benchmark")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Matrix sizes to test (square matrices M=N=K)",
    )
    parser.add_argument(
        "--test-block-sizes",
        action="store_true",
        help="Test different block size configurations",
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)
    
    # Print GPU info
    print("\n" + "=" * 60)
    print(" DIY FlashAttention - Matrix Multiplication Benchmark")
    print("=" * 60)
    
    try:
        caps = detect_gpu()
        print_gpu_info(caps)
    except Exception as e:
        print(f"Warning: Could not detect GPU info: {e}")
    
    # Validate correctness first
    if args.validate:
        print("\n" + "-" * 60)
        print(" Validating Correctness")
        print("-" * 60)
        
        test_size = min(args.sizes)
        is_valid, max_diff = validate_matmul(triton_matmul, test_size, test_size, test_size)
        
        if is_valid:
            print(f"✓ Correctness validated (max diff: {max_diff:.2e})")
        else:
            print(f"✗ Correctness check failed (max diff: {max_diff:.2e})")
            print("  Proceeding with benchmark anyway...")
    
    # Create benchmark runner
    runner = BenchmarkRunner(warmup=args.warmup, rep=args.rep)
    
    # Prepare sizes (square matrices)
    sizes = [(s, s, s) for s in args.sizes]
    
    # Block size configurations to test
    block_configs = None
    if args.test_block_sizes:
        block_configs = [
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
        ]
    
    # Run benchmarks
    print("\n" + "-" * 60)
    print(" Running Benchmarks")
    print("-" * 60)
    print(f"Matrix sizes: {[s[0] for s in sizes]}")
    print(f"Warmup: {args.warmup}, Repetitions: {args.rep}")
    
    results = runner.benchmark_matmul(
        triton_fn=triton_matmul,
        sizes=sizes,
        block_configs=block_configs,
    )
    
    # Print results
    runner.print_comparison_table(results, title="Matrix Multiplication Benchmark Results")
    
    # Print summary
    print("\n" + "-" * 60)
    print(" Summary")
    print("-" * 60)
    
    for size in sizes:
        size_results = [r for r in results if r.size == size]
        pytorch_result = next((r for r in size_results if r.name == "PyTorch"), None)
        triton_result = next((r for r in size_results if r.name == "Triton"), None)
        
        if pytorch_result and triton_result:
            speedup = pytorch_result.time_ms / triton_result.time_ms
            size_str = f"{size[0]}×{size[1]}×{size[2]}"
            print(f"  {size_str}: Triton is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")
            print(f"    PyTorch: {pytorch_result.tflops:.1f} TFLOPS")
            print(f"    Triton:  {triton_result.tflops:.1f} TFLOPS")
    
    # Block size analysis
    if args.test_block_sizes:
        print("\n" + "-" * 60)
        print(" Block Size Analysis")
        print("-" * 60)
        
        for size in sizes:
            size_results = [r for r in results if r.size == size and r.block_config is not None]
            if size_results:
                best = max(size_results, key=lambda r: r.tflops)
                worst = min(size_results, key=lambda r: r.tflops)
                
                size_str = f"{size[0]}×{size[1]}×{size[2]}"
                print(f"\n  {size_str}:")
                print(f"    Best:  {best.name} ({best.tflops:.1f} TFLOPS)")
                print(f"    Worst: {worst.name} ({worst.tflops:.1f} TFLOPS)")
                print(f"    Difference: {(best.tflops / worst.tflops - 1) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print(" Benchmark Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

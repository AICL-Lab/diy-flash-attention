#!/usr/bin/env python3
"""
Block Size Experiment

This script helps you understand how different block sizes affect performance.
Modify the BLOCK_SIZE parameters and observe the TFLOPS changes.

This is the core "Vibe Coding" experience - feel the performance changes!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import triton

from kernels.matmul import triton_matmul
from utils.benchmark import benchmark_fn, calculate_matmul_flops


def experiment_block_sizes():
    """Experiment with different block size configurations."""
    
    print("=" * 70)
    print(" Block Size Experiment - Feel the Performance!")
    print("=" * 70)
    print("\nThis experiment shows how block sizes affect GPU performance.")
    print("Watch the TFLOPS numbers change as you modify block sizes!\n")
    
    # Test matrix size
    M, N, K = 4096, 4096, 4096
    print(f"Matrix size: {M} × {K} @ {K} × {N}")
    
    # Create test matrices
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    flops = calculate_matmul_flops(M, N, K)
    
    # Block size configurations to test
    # Format: (BLOCK_M, BLOCK_N, BLOCK_K)
    configs = [
        # Small blocks - more parallelism, but more memory accesses
        (32, 32, 32),
        (64, 64, 32),
        
        # Medium blocks - balanced
        (64, 64, 64),
        (128, 64, 32),
        (64, 128, 32),
        (128, 128, 32),
        
        # Large blocks - better data reuse, but less parallelism
        (128, 128, 64),
        (128, 256, 64),
        (256, 128, 64),
        (256, 256, 64),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Block Size (M×N×K)':<25} | {'Time (ms)':<12} | {'TFLOPS':<12} | {'Notes'}")
    print("-" * 70)
    
    results = []
    
    for block_m, block_n, block_k in configs:
        try:
            # Benchmark this configuration
            ms, _, _ = benchmark_fn(
                triton_matmul, a, b,
                block_m=block_m, block_n=block_n, block_k=block_k,
                warmup=10, rep=50
            )
            
            tflops = flops / ms / 1e9
            results.append((block_m, block_n, block_k, ms, tflops))
            
            # Add notes based on performance
            if tflops == max(r[4] for r in results):
                note = "⭐ Best so far!"
            elif tflops < min(r[4] for r in results[:-1]) if len(results) > 1 else float('inf'):
                note = "📉 Slowest"
            else:
                note = ""
            
            print(f"{block_m:>3} × {block_n:>3} × {block_k:>3}          | {ms:<12.3f} | {tflops:<12.2f} | {note}")
            
        except Exception as e:
            print(f"{block_m:>3} × {block_n:>3} × {block_k:>3}          | {'ERROR':<12} | {'N/A':<12} | {str(e)[:20]}")
    
    print("-" * 70)
    
    # Find best and worst
    if results:
        best = max(results, key=lambda x: x[4])
        worst = min(results, key=lambda x: x[4])
        
        print(f"\n📊 Results Summary:")
        print(f"   Best:  {best[0]}×{best[1]}×{best[2]} = {best[4]:.2f} TFLOPS")
        print(f"   Worst: {worst[0]}×{worst[1]}×{worst[2]} = {worst[4]:.2f} TFLOPS")
        print(f"   Difference: {(best[4]/worst[4] - 1) * 100:.1f}% performance gap!")
        
        print(f"\n💡 Key Insights:")
        print(f"   - Block size significantly affects performance")
        print(f"   - Larger blocks often better for data reuse")
        print(f"   - But too large can reduce parallelism")
        print(f"   - Optimal depends on matrix size and GPU architecture")
    
    # Compare with autotuned version
    print(f"\n🔧 Autotuned Performance:")
    ms_auto, _, _ = benchmark_fn(triton_matmul, a, b, warmup=10, rep=50)
    tflops_auto = flops / ms_auto / 1e9
    print(f"   Autotuned: {tflops_auto:.2f} TFLOPS")
    
    if results:
        best_manual = max(r[4] for r in results)
        if tflops_auto > best_manual:
            print(f"   Autotune is {(tflops_auto/best_manual - 1) * 100:.1f}% better than best manual config")
        else:
            print(f"   Best manual config is {(best_manual/tflops_auto - 1) * 100:.1f}% better than autotune!")


def interactive_experiment():
    """Interactive mode for experimenting with custom block sizes."""
    
    print("\n" + "=" * 70)
    print(" Interactive Block Size Experiment")
    print("=" * 70)
    
    M, N, K = 2048, 2048, 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    flops = calculate_matmul_flops(M, N, K)
    
    print(f"\nMatrix size: {M} × {K} @ {K} × {N}")
    print("Enter block sizes to test (or 'q' to quit)")
    print("Format: BLOCK_M BLOCK_N BLOCK_K (e.g., '128 256 64')\n")
    
    while True:
        try:
            user_input = input("Block sizes > ").strip()
            
            if user_input.lower() == 'q':
                break
            
            parts = user_input.split()
            if len(parts) != 3:
                print("Please enter 3 numbers: BLOCK_M BLOCK_N BLOCK_K")
                continue
            
            block_m, block_n, block_k = map(int, parts)
            
            if block_m <= 0 or block_n <= 0 or block_k <= 0:
                print("Block sizes must be positive")
                continue
            
            ms, _, _ = benchmark_fn(
                triton_matmul, a, b,
                block_m=block_m, block_n=block_n, block_k=block_k,
                warmup=5, rep=20
            )
            
            tflops = flops / ms / 1e9
            print(f"  → {block_m}×{block_n}×{block_k}: {ms:.3f} ms, {tflops:.2f} TFLOPS\n")
            
        except ValueError:
            print("Invalid input. Please enter 3 integers.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        sys.exit(1)
    
    experiment_block_sizes()
    
    # Uncomment to enable interactive mode
    # interactive_experiment()
    
    print("\n" + "=" * 70)
    print(" Experiment Complete!")
    print("=" * 70)
    print("\nTry modifying the block sizes in kernels/matmul.py to see the effect!")

#!/usr/bin/env python3
"""
Visualize Tiling Strategy

This script helps visualize how matrix multiplication is tiled
and how different block sizes affect the computation pattern.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_tiling(M: int, N: int, K: int, block_m: int, block_n: int, block_k: int):
    """
    Visualize the tiling strategy for matrix multiplication.
    
    C[M, N] = A[M, K] @ B[K, N]
    """
    print("=" * 70)
    print(" Matrix Multiplication Tiling Visualization")
    print("=" * 70)
    
    print(f"\nMatrix dimensions:")
    print(f"  A: {M} × {K}")
    print(f"  B: {K} × {N}")
    print(f"  C: {M} × {N}")
    
    print(f"\nBlock sizes:")
    print(f"  BLOCK_M: {block_m}")
    print(f"  BLOCK_N: {block_n}")
    print(f"  BLOCK_K: {block_k}")
    
    # Calculate number of blocks
    num_blocks_m = (M + block_m - 1) // block_m
    num_blocks_n = (N + block_n - 1) // block_n
    num_blocks_k = (K + block_k - 1) // block_k
    
    total_output_blocks = num_blocks_m * num_blocks_n
    
    print(f"\nTiling breakdown:")
    print(f"  Output blocks (M dimension): {num_blocks_m}")
    print(f"  Output blocks (N dimension): {num_blocks_n}")
    print(f"  K iterations per output block: {num_blocks_k}")
    print(f"  Total output blocks: {total_output_blocks}")
    print(f"  Total kernel invocations: {total_output_blocks}")
    
    # Memory analysis
    print(f"\n" + "-" * 70)
    print("Memory Analysis (per output block):")
    print("-" * 70)
    
    # Each output block needs to load:
    # - A block: BLOCK_M × BLOCK_K elements (loaded num_blocks_k times)
    # - B block: BLOCK_K × BLOCK_N elements (loaded num_blocks_k times)
    # - C block: BLOCK_M × BLOCK_N elements (written once)
    
    a_loads_per_block = block_m * block_k * num_blocks_k
    b_loads_per_block = block_k * block_n * num_blocks_k
    c_writes_per_block = block_m * block_n
    
    # In bytes (assuming float16)
    bytes_per_element = 2
    a_bytes = a_loads_per_block * bytes_per_element
    b_bytes = b_loads_per_block * bytes_per_element
    c_bytes = c_writes_per_block * bytes_per_element
    
    print(f"  A loads: {a_loads_per_block:,} elements ({a_bytes/1024:.1f} KB)")
    print(f"  B loads: {b_loads_per_block:,} elements ({b_bytes/1024:.1f} KB)")
    print(f"  C writes: {c_writes_per_block:,} elements ({c_bytes/1024:.1f} KB)")
    print(f"  Total memory traffic: {(a_bytes + b_bytes + c_bytes)/1024:.1f} KB")
    
    # SRAM usage
    print(f"\n" + "-" * 70)
    print("SRAM Usage (per block):")
    print("-" * 70)
    
    a_sram = block_m * block_k * bytes_per_element
    b_sram = block_k * block_n * bytes_per_element
    acc_sram = block_m * block_n * 4  # float32 accumulator
    
    total_sram = a_sram + b_sram + acc_sram
    
    print(f"  A block: {a_sram/1024:.1f} KB")
    print(f"  B block: {b_sram/1024:.1f} KB")
    print(f"  Accumulator (float32): {acc_sram/1024:.1f} KB")
    print(f"  Total SRAM needed: {total_sram/1024:.1f} KB")
    
    # Typical SRAM limits
    print(f"\n  Typical GPU SRAM per SM: ~192 KB (Ampere), ~228 KB (Hopper)")
    if total_sram > 192 * 1024:
        print(f"  ⚠️  Warning: Block size may exceed SRAM capacity!")
    else:
        print(f"  ✓ Block size fits in SRAM")
    
    # Compute intensity
    print(f"\n" + "-" * 70)
    print("Compute Intensity:")
    print("-" * 70)
    
    flops_per_block = 2 * block_m * block_n * K  # 2 ops per multiply-add
    bytes_per_block = a_bytes + b_bytes + c_bytes
    
    arithmetic_intensity = flops_per_block / bytes_per_block
    
    print(f"  FLOPs per output block: {flops_per_block:,}")
    print(f"  Bytes per output block: {bytes_per_block:,}")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")
    
    # GPU roofline analysis (rough estimates)
    print(f"\n  Typical GPU specs:")
    print(f"    A100: ~312 TFLOPS (FP16), ~2 TB/s HBM bandwidth")
    print(f"    H100: ~990 TFLOPS (FP16), ~3.35 TB/s HBM bandwidth")
    
    a100_ridge = 312e12 / 2e12  # FLOPs/byte at ridge point
    h100_ridge = 990e12 / 3.35e12
    
    print(f"\n  Ridge point (compute/memory balanced):")
    print(f"    A100: {a100_ridge:.1f} FLOPs/byte")
    print(f"    H100: {h100_ridge:.1f} FLOPs/byte")
    
    if arithmetic_intensity > a100_ridge:
        print(f"\n  ✓ Compute-bound on A100 (good!)")
    else:
        print(f"\n  ⚠️  Memory-bound on A100 (consider larger blocks)")
    
    # Visual representation
    print(f"\n" + "-" * 70)
    print("Visual Representation:")
    print("-" * 70)
    
    print(f"\n  Matrix C ({M}×{N}) divided into {num_blocks_m}×{num_blocks_n} blocks:")
    print()
    
    # Show a small grid representation
    max_show = min(8, num_blocks_m, num_blocks_n)
    
    print("    ", end="")
    for j in range(max_show):
        print(f"  B{j} ", end="")
    if num_blocks_n > max_show:
        print(" ...")
    print()
    
    for i in range(min(max_show, num_blocks_m)):
        print(f"  M{i}", end="")
        for j in range(max_show):
            block_id = i * num_blocks_n + j
            print(f" [{block_id:3d}]", end="")
        if num_blocks_n > max_show:
            print(" ...")
        print()
    
    if num_blocks_m > max_show:
        print("   ...")
    
    print(f"\n  Each block [{block_m}×{block_n}] computed by one GPU program")
    print(f"  Each program iterates {num_blocks_k} times over K dimension")


def main():
    print("\n🔍 Tiling Visualization for Matrix Multiplication\n")
    
    # Example configurations
    configs = [
        # (M, N, K, block_m, block_n, block_k)
        (1024, 1024, 1024, 128, 128, 32),
        (4096, 4096, 4096, 128, 256, 64),
    ]
    
    for M, N, K, bm, bn, bk in configs:
        visualize_tiling(M, N, K, bm, bn, bk)
        print("\n" + "=" * 70 + "\n")
    
    # Interactive mode
    print("Enter custom dimensions (or 'q' to quit):")
    print("Format: M N K BLOCK_M BLOCK_N BLOCK_K")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() == 'q':
                break
            
            parts = list(map(int, user_input.split()))
            if len(parts) != 6:
                print("Please enter 6 numbers: M N K BLOCK_M BLOCK_N BLOCK_K")
                continue
            
            visualize_tiling(*parts)
            
        except ValueError:
            print("Invalid input. Please enter 6 integers.")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

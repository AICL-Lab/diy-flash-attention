#!/usr/bin/env python3
"""
Quick Start Example for DIY FlashAttention

This script demonstrates the basic usage of the Triton matmul and FlashAttention kernels.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import triton_matmul, flash_attention
from utils import detect_gpu, print_gpu_info


def demo_matmul():
    """Demonstrate Triton matrix multiplication."""
    print("\n" + "=" * 60)
    print(" Matrix Multiplication Demo")
    print("=" * 60)
    
    # Create random matrices
    M, N, K = 1024, 1024, 1024
    print(f"\nMatrix sizes: A({M}×{K}) @ B({K}×{N}) = C({M}×{N})")
    
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # Triton matmul
    print("\nRunning Triton matmul...")
    c_triton = triton_matmul(a, b)
    
    # PyTorch reference
    print("Running PyTorch matmul...")
    c_torch = torch.matmul(a, b)
    
    # Compare
    max_diff = (c_triton - c_torch).abs().max().item()
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Result: {'✓ Correct!' if max_diff < 1e-2 else '✗ Mismatch'}")
    
    # Try different block sizes
    print("\n--- Block Size Experiment ---")
    block_configs = [
        (32, 32, 32),
        (64, 64, 32),
        (128, 128, 64),
    ]
    
    for bm, bn, bk in block_configs:
        c = triton_matmul(a, b, block_m=bm, block_n=bn, block_k=bk)
        diff = (c - c_torch).abs().max().item()
        print(f"Block({bm}×{bn}×{bk}): max_diff = {diff:.2e}")


def demo_flash_attention():
    """Demonstrate FlashAttention."""
    print("\n" + "=" * 60)
    print(" FlashAttention Demo")
    print("=" * 60)
    
    # Create random Q, K, V
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    print(f"\nShape: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    
    # Non-causal attention
    print("\n--- Non-Causal Attention ---")
    flash_out = flash_attention(q, k, v, causal=False)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    max_diff = (flash_out - ref_out).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    print(f"Result: {'✓ Correct!' if max_diff < 1e-2 else '✗ Mismatch'}")
    
    # Causal attention
    print("\n--- Causal Attention ---")
    flash_out_causal = flash_attention(q, k, v, causal=True)
    ref_out_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    max_diff_causal = (flash_out_causal - ref_out_causal).abs().max().item()
    print(f"Max difference: {max_diff_causal:.2e}")
    print(f"Result: {'✓ Correct!' if max_diff_causal < 1e-2 else '✗ Mismatch'}")
    
    # Memory comparison
    print("\n--- Memory Usage ---")
    torch.cuda.reset_peak_memory_stats()
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    pytorch_mem = torch.cuda.max_memory_allocated() / 1e6
    
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attention(q, k, v)
    flash_mem = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"PyTorch SDPA: {pytorch_mem:.1f} MB")
    print(f"FlashAttention: {flash_mem:.1f} MB")
    print(f"Memory reduction: {pytorch_mem / flash_mem:.2f}x")


def main():
    print("=" * 60)
    print(" DIY FlashAttention - Quick Start")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return
    
    # Print GPU info
    try:
        caps = detect_gpu()
        print_gpu_info(caps)
    except Exception as e:
        print(f"Warning: Could not detect GPU: {e}")
    
    # Run demos
    demo_matmul()
    demo_flash_attention()
    
    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run benchmarks: python benchmarks/bench_matmul.py")
    print("  - Run tests: pytest tests/")
    print("  - Experiment with block sizes in the kernel code")


if __name__ == "__main__":
    main()

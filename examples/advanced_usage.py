#!/usr/bin/env python3
"""
高级用法示例 - 展示更多 DIY FlashAttention 的功能

包含:
1. 自定义 Block Size 调优
2. 内存效率对比
3. Causal vs Non-Causal 对比
4. 批量处理性能
"""

import time

import torch

from kernels import flash_attention, triton_matmul
from utils import detect_gpu, print_gpu_info


def measure_time(fn, warmup: int = 10, repeat: int = 100) -> float:
    """测量函数执行时间"""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeat * 1000  # ms


def demo_block_size_tuning():
    """演示 Block Size 对矩阵乘法性能的影响"""
    print("\n" + "=" * 60)
    print("🔧 Block Size 调优演示")
    print("=" * 60)

    M, K, N = 4096, 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    configs = [
        (64, 64, 32),
        (64, 128, 32),
        (128, 128, 32),
        (128, 256, 64),
    ]

    print(f"\n矩阵大小: {M}×{K} @ {K}×{N}")
    print("-" * 50)
    print(f"{'Block Size (M×N×K)':<25} | {'时间 (ms)':<12} | {'TFLOPS':<10}")
    print("-" * 50)

    flops = 2 * M * N * K

    for bm, bn, bk in configs:
        try:
            ms = measure_time(
                lambda bm=bm, bn=bn, bk=bk: triton_matmul(a, b, block_m=bm, block_n=bn, block_k=bk),
                warmup=5,
                repeat=50,
            )
            tflops = flops / (ms / 1000) / 1e12
            print(f"{bm}×{bn}×{bk:<20} | {ms:>10.3f} | {tflops:>8.2f}")
        except RuntimeError as e:
            print(f"{bm}×{bn}×{bk:<20} | {'Error':<12} | {str(e)[:20]}")

    # Autotune
    ms = measure_time(lambda: triton_matmul(a, b), warmup=5, repeat=50)
    tflops = flops / (ms / 1000) / 1e12
    print(f"{'Autotune':<25} | {ms:>10.3f} | {tflops:>8.2f}")


def demo_memory_efficiency():
    """演示 FlashAttention 的内存效率"""
    print("\n" + "=" * 60)
    print("💾 内存效率对比演示")
    print("=" * 60)

    batch, heads, head_dim = 2, 8, 64

    print(f"\n配置: batch={batch}, heads={heads}, head_dim={head_dim}")
    print("-" * 60)
    print(f"{'序列长度':<12} | {'标准 Attention':<18} | {'FlashAttention':<18}")
    print("-" * 60)

    for seq_len in [512, 1024, 2048, 4096]:
        # 标准 attention 内存估算 (N×N attention matrix)
        standard_mem = batch * heads * seq_len * seq_len * 2 / (1024**2)  # MB, FP16

        # FlashAttention 内存估算 (只需要 O(N) 的中间状态)
        flash_mem = batch * heads * seq_len * head_dim * 2 / (1024**2)  # MB, FP16

        print(f"{seq_len:<12} | {standard_mem:>14.1f} MB | {flash_mem:>14.1f} MB")

    print("\n💡 FlashAttention 避免了 O(N²) 的 attention matrix 存储!")


def demo_causal_masking():
    """演示 Causal Masking 的效果"""
    print("\n" + "=" * 60)
    print("🎭 Causal Masking 演示")
    print("=" * 60)

    batch, heads, seq_len, head_dim = 1, 1, 8, 32

    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    # Non-causal attention
    out_full = flash_attention(q, k, v, causal=False)

    # Causal attention
    out_causal = flash_attention(q, k, v, causal=True)

    print(f"\n序列长度: {seq_len}")
    print("\n非因果注意力 (每个位置可以看到所有位置):")
    print(f"  输出形状: {out_full.shape}")

    print("\n因果注意力 (每个位置只能看到之前的位置):")
    print(f"  输出形状: {out_causal.shape}")

    # 验证因果性: 改变未来的 K/V 不应该影响当前位置的输出
    print("\n验证因果性...")
    k_modified = k.clone()
    k_modified[:, :, seq_len // 2 :, :] = torch.randn_like(k_modified[:, :, seq_len // 2 :, :])

    out_causal_modified = flash_attention(q, k_modified, v, causal=True)

    # 前半部分应该相同
    first_half_diff = (
        (out_causal[:, :, : seq_len // 2, :] - out_causal_modified[:, :, : seq_len // 2, :])
        .abs()
        .max()
    )
    print(f"  修改后半部分 K 后，前半部分输出差异: {first_half_diff.item():.2e}")
    print("  ✓ 因果性验证通过!" if first_half_diff < 1e-3 else "  ✗ 因果性验证失败!")


def demo_batch_performance():
    """演示批量处理性能"""
    print("\n" + "=" * 60)
    print("📦 批量处理性能演示")
    print("=" * 60)

    heads, seq_len, head_dim = 8, 1024, 64

    print(f"\n配置: heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    print("-" * 50)
    print(f"{'Batch Size':<12} | {'时间 (ms)':<12} | {'吞吐量 (samples/s)':<20}")
    print("-" * 50)

    for batch in [1, 2, 4, 8, 16]:
        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        ms = measure_time(
            lambda q=q, k=k, v=v: flash_attention(q, k, v, causal=True), warmup=5, repeat=50
        )
        throughput = batch / (ms / 1000)

        print(f"{batch:<12} | {ms:>10.3f} | {throughput:>18.1f}")


def main():
    """运行所有演示"""
    print("=" * 60)
    print("🚀 DIY FlashAttention 高级用法演示")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA GPU!")
        return

    caps = detect_gpu()
    print_gpu_info(caps)

    # 运行演示
    demo_block_size_tuning()
    demo_memory_efficiency()
    demo_causal_masking()
    demo_batch_performance()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

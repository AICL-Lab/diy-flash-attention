# Performance Guide

This guide covers optimization techniques for DIY FlashAttention, including configuration tuning, best practices, and common pitfalls.

> **Before you read:** You should already know the tensor layout and forward-only scope from [/tutorial](/tutorial).
>
> **Source files:** `kernels/flash_attn.py`, `kernels/flash_attn_v2.py`
>
> **Next step:** Continue to [/paper-guide](/paper-guide) or [/knowledge-map](/knowledge-map).

## Table of Contents

- [Performance Benchmarks](#performance-benchmarks)
- [Block Size Tuning](#block-size-tuning)
- [Data Type Selection](#data-type-selection)
- [Memory Optimization](#memory-optimization)
- [GPU Architecture Optimization](#gpu-architecture-optimization)
- [Profiling Tools](#profiling-tools)
- [Common Pitfalls](#common-pitfalls)
- [Performance Checklist](#performance-checklist)

---

## Performance Benchmarks

### Matrix Multiplication (MatMul)

Typical performance (RTX 4090, FP16):

| Matrix Size | PyTorch (TFLOPS) | Triton (TFLOPS) | Speedup |
|-------------|-----------------|-----------------|---------|
| 512×512 | 25 | 28 | 1.12x |
| 1024×1024 | 45 | 48 | 1.07x |
| 2048×2048 | 85 | 95 | 1.12x |
| 4096×4096 | 120 | 140 | 1.17x |
| 8192×8192 | 150 | 175 | 1.17x |

### FlashAttention

Typical performance (RTX 4090, FP16, batch=4, heads=8, head_dim=64):

| Seq Length | PyTorch SDPA (ms) | FlashAttention (ms) | Speedup | Memory Saved |
|------------|-------------------|---------------------|---------|--------------|
| 512 | 0.8 | 0.7 | 1.14x | 94% |
| 1024 | 2.5 | 2.0 | 1.25x | 97% |
| 2048 | 9.0 | 6.5 | 1.38x | 98% |
| 4096 | 35.0 | 22.0 | 1.59x | 99% |

### Memory Usage Comparison

| Seq Length | Standard Attention | FlashAttention | Savings |
|------------|-------------------|----------------|---------|
| 512 | 2 MB | 0.25 MB | 88% |
| 1024 | 8 MB | 0.5 MB | 94% |
| 2048 | 32 MB | 1 MB | 97% |
| 4096 | 128 MB | 2 MB | 98% |
| 8192 | 512 MB | 4 MB | 99% |

---

## Block Size Tuning

Block Size is the most critical parameter affecting Triton kernel performance.

### Core Principles

```
┌─────────────────────────────────────────────────────────────┐
│                    Block Size Trade-offs                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Small Block Size:            Large Block Size:             │
│  ┌───┬───┬───┐                ┌───────────┐                │
│  │   │   │   │                │           │                │
│  ├───┼───┼───┤                │           │                │
│  │   │   │   │                │   Single  │                │
│  ├───┼───┼───┤                │   Block   │                │
│  │   │   │   │                │           │                │
│  └───┴───┴───┘                └───────────┘                │
│                                                             │
│  ✅ More parallel blocks     ✅ Better data reuse          │
│  ✅ Good for small matrices  ✅ Less HBM access            │
│  ❌ More HBM access          ❌ May exceed SRAM            │
│  ❌ Lower data reuse         ❌ Less parallelism           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Configurations

| Matrix Size Range | BLOCK_M | BLOCK_N | BLOCK_K | num_stages | num_warps |
|-------------------|---------|---------|---------|------------|-----------|
| < 512 | 32 | 32 | 32 | 4 | 4 |
| 512 - 1024 | 64 | 64 | 32 | 4 | 4 |
| 1024 - 2048 | 64 | 128 | 32 | 4 | 4 |
| 2048 - 4096 | 128 | 128 | 32 | 4 | 4 |
| > 4096 | 128 | 256 | 64 | 3 | 8 |

### Using Autotune

**Recommended**: Use built-in autotune

```python
from kernels import triton_matmul

# Don't specify block size - automatic optimal selection
c = triton_matmul(a, b)
```

**Experimental**: Test different configurations manually

```bash
python examples/block_size_experiment.py
```

---

## Data Type Selection

### Type Comparison

| Data Type | Range | Mantissa | Performance | Recommended For |
|-----------|-------|----------|-------------|-----------------|
| FP32 | ±3.4e38 | 23 bits | 1x (baseline) | High precision/debug |
| FP16 | ±65504 | 10 bits | ~2x | Training/Inference |
| BF16 | ±3.4e38 | 7 bits | ~2x | Training (stable) |
| FP8 E4M3 | ±448 | 3 bits | ~4x | Inference (Hopper+) |
| FP8 E5M2 | ±57344 | 2 bits | ~4x | Gradient storage |

### Selection Guide

**Use FP16 for:**
- Most training and inference scenarios
- Standard LLM workloads

**Use BF16 for:**
- Training with large models (avoids overflow)
- When FP16 causes NaN issues

**Use FP32 for:**
- Debugging
- Numerical verification

---

## Memory Optimization

### Ensure Contiguous Memory

```python
# ❌ Bad: Non-contiguous tensor triggers extra copy
a = some_tensor.transpose(0, 1)
c = triton_matmul(a, b)  # Internally calls .contiguous()

# ✅ Good: Explicitly ensure contiguous
a = some_tensor.transpose(0, 1).contiguous()
c = triton_matmul(a, b)
```

### Monitor Memory Usage

```python
def memory_report():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")

# Clear cache before benchmarks
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

---

## GPU Architecture Optimization

### Ampere (SM80) - A100, RTX 30 Series

```python
ampere_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 3,
    "num_warps": 8,
}
# SRAM: ~164 KB per SM
```

### Ada (SM89) - RTX 40 Series

```python
ada_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 4,  # Larger SRAM
    "num_warps": 8,
}
# SRAM: ~192 KB per SM
```

### Hopper (SM90) - H100

```python
from kernels import check_hopper_features

features = check_hopper_features()

if features["tma_available"]:
    print("TMA available - async loading possible")

if features["fp8_available"]:
    print("FP8 available - low precision compute possible")

# SRAM: ~228 KB per SM
```

---

## Profiling Tools

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    result = triton_matmul(a, b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

### Triton Built-in Benchmarking

```python
from utils.benchmark import benchmark_fn

median_ms, p20_ms, p80_ms = benchmark_fn(
    triton_matmul, a, b,
    warmup=25,
    rep=100,
)
print(f"Median: {median_ms:.3f} ms, P20: {p20_ms:.3f} ms, P80: {p80_ms:.3f} ms")
```

---

## Common Pitfalls

### 1. Too Small Matrices

```python
# ❌ Bad: Kernel launch overhead dominates
a = torch.randn(32, 32, device="cuda")
for _ in range(1000):
    c = triton_matmul(a, a)

# ✅ Good: Use appropriately sized matrices
a = torch.randn(1024, 1024, device="cuda")
c = triton_matmul(a, a)
```

### 2. Frequent CPU-GPU Synchronization

```python
# ❌ Bad: Synchronizing every operation
for _ in range(100):
    result = triton_matmul(a, b)
    torch.cuda.synchronize()  # Blocks!

# ✅ Good: Batch operations, then sync
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()
```

### 3. Cold Start on First Run

```python
# ❌ Bad: First run includes compilation time
import time
start = time.time()
result = triton_matmul(a, b)  # Includes JIT compilation!
print(f"Time: {time.time() - start:.3f}s")

# ✅ Good: Warmup before timing
for _ in range(10):
    _ = triton_matmul(a, b)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) / 100 * 1000:.3f} ms")
```

---

## Performance Checklist

Before running benchmarks, ensure:

```
□ Data Types
  ├─ ☑ Use FP16 or BF16
  ├─ ☑ Avoid FP32 (unless high precision needed)
  └─ ☑ Consistent input/output dtypes

□ Memory
  ├─ ☑ Input tensors are contiguous
  ├─ ☑ Data already on GPU
  └─ ☑ Clear cache before benchmark

□ Configuration
  ├─ ☑ Use autotune (don't specify block size)
  ├─ ☑ Or choose appropriate block size for matrix size
  └─ ☑ Check SRAM capacity limits

□ Measurement
  ├─ ☑ Warm up (10+ iterations)
  ├─ ☑ Measure multiple times and average
  ├─ ☑ Use torch.cuda.synchronize()
  └─ ☑ Use GPU time, not CPU time

□ Code
  ├─ ☑ Avoid synchronization in loops
  ├─ ☑ Avoid CPU-GPU data movement in loops
  └─ ☑ Matrix size > 512
```

---

## Run Benchmarks

```bash
# Matrix multiplication benchmark
make bench-matmul

# FlashAttention benchmark
make bench-flash

# All benchmarks
make bench-all

# Generate report
make report
```

---

## References

- [Triton Performance Guide](https://triton-lang.org/main/programming-guide/chapter-3/performance.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

# DIY FlashAttention Cheatsheet

Quick reference for common APIs, commands, and configurations.

## Quick Start

```bash
# Installation
pip install -e ".[dev]"

# Run demo
make demo

# Run tests
make test
```

---

## Core APIs

### Matrix Multiplication

```python
from kernels import triton_matmul

# Basic usage (auto-selects optimal config)
c = triton_matmul(a, b)

# Specify block size
c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)

# Supported dtypes
a = torch.randn(..., dtype=torch.float16)   # ✅ Recommended
a = torch.randn(..., dtype=torch.bfloat16)  # ✅ Supported
a = torch.randn(..., dtype=torch.float32)   # ⚠️ Converts to float16 internally
```

### FlashAttention

```python
from kernels import flash_attention

# Basic usage
out = flash_attention(q, k, v)

# Causal attention (for autoregressive models)
out = flash_attention(q, k, v, causal=True)

# Variable-length sequences
seq_lens = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
out = flash_attention(q, k, v, seq_lens=seq_lens)

# 3D input: (batch*heads, seq_len, head_dim)
q_3d = torch.randn(16, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q_3d, k_3d, v_3d)
```

### GPU Detection

```python
from utils import detect_gpu, print_gpu_info

caps = detect_gpu()
print_gpu_info(caps)

# Check features
print(f"TMA: {caps.has_tma}")
print(f"FP8: {caps.has_fp8}")
```

### Benchmark

```python
from utils import BenchmarkRunner

runner = BenchmarkRunner(warmup=10, rep=50)

# Matrix multiplication
results = runner.benchmark_matmul(
    triton_matmul,
    sizes=[(1024, 1024, 1024), (2048, 2048, 2048)],
)
runner.print_comparison_table(results)

# FlashAttention
results = runner.benchmark_attention(
    flash_attention,
    seq_lengths=[512, 1024, 2048],
)
```

### Validation

```python
from utils import validate_matmul, validate_attention

# Validate matrix multiplication
is_valid, max_diff = validate_matmul(
    triton_matmul, m=1024, n=1024, k=1024
)

# Validate FlashAttention
is_valid, max_diff = validate_attention(
    flash_attention, batch=2, heads=8, seq_len=512, head_dim=64
)
```

---

## Useful Commands

| Command | Description |
|---------|-------------|
| `make demo` | Quick demo |
| `make test` | Run all tests |
| `make bench-all` | Run all benchmarks |
| `make gpu-info` | Show GPU info |
| `make experiment` | Block size experiment |
| `make lint` | Code linting |
| `make format` | Code formatting |
| `make clean` | Clean caches |

---

## Input Shapes

### Matrix Multiplication

```
A: (M, K) × B: (K, N) → C: (M, N)
```

### FlashAttention

```
4D Input: (batch, heads, seq_len, head_dim)
3D Input: (batch*heads, seq_len, head_dim)

seq_lens: (batch,) - Per-sample effective length
head_dim: Support for 32 or 64
```

---

## Block Size Recommendations

| Matrix Size | BLOCK_M | BLOCK_N | BLOCK_K |
|-------------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-1024 | 64 | 64 | 32 |
| 1024-2048 | 64 | 128 | 32 |
| 2048-4096 | 128 | 128 | 32 |
| > 4096 | 128 | 256 | 64 |

**Tip**: Use autotune (don't specify block size) for automatic optimal selection.

---

## Data Types

| Type | Precision | Performance | Recommended For |
|------|-----------|-------------|-----------------|
| `float16` | Medium | 2x | Training/Inference (Recommended) |
| `bfloat16` | Medium | 2x | Training (more stable) |
| `float32` | High | 1x | Debugging/High precision |
| `float8` | Low | 4x | Inference (Hopper+) |

---

## GPU Architectures

| Architecture | SM | GPUs | Features |
|--------------|-----|------|----------|
| Ampere | 80 | A100, RTX 30xx | Full support |
| Ada | 89 | RTX 40xx | Full support |
| Hopper | 90 | H100 | TMA, FP8 |
| Blackwell | 100 | B100 | Latest |

---

## Memory Complexity

| Method | Memory | Description |
|--------|--------|-------------|
| Standard Attention | O(N²) | Stores full attention matrix |
| FlashAttention | O(N) | Tiled computation, no full matrix |

**Memory Savings**: Up to **99%** for long sequences!

---

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Expected 2D tensors` | Non-2D matmul input | Use `.view()` or `.reshape()` |
| `Incompatible dimensions` | A.shape[1] != B.shape[0] | Check matrix dimensions |
| `CUDA tensors required` | Input on CPU | Use `.cuda()` or `.to("cuda")` |
| `Q, K, V shapes must match` | Shape mismatch | Ensure identical shapes |
| `Expected 3D or 4D tensors` | Wrong attention dims | Check input shapes |
| `Unsupported head_dim` | head_dim not 32/64 | Use 32 or 64 |
| `Unsupported dtype` | Wrong dtype | Use float16/bfloat16/float32 |
| `dtypes must match` | Dtype mismatch | Use consistent dtype |

---

## Performance Checklist

```
☑ Use FP16 or BF16
☑ Use autotune (don't specify block size)
☑ Warm up kernel (run a few times before timing)
☑ Ensure input is contiguous (.is_contiguous())
☑ Keep data on GPU
☑ Matrix size > 512
☑ Avoid synchronization in loops
☑ Avoid CPU-GPU data movement in loops
```

---

## File Structure

```
kernels/
├── matmul.py          # Matrix multiplication
├── flash_attn.py      # FlashAttention
└── modern_features.py # Modern GPU features

utils/
├── benchmark.py       # Benchmark tools
├── validation.py      # Validation tools
└── gpu_detect.py      # GPU detection

tests/
├── test_matmul.py     # Matrix multiplication tests
├── test_flash.py      # FlashAttention tests
├── test_properties.py # Property-based tests
└── test_error_handling.py # Error handling tests
```

---

## Links

- 📖 [Tutorial](./tutorial) - Step-by-step learning guide
- 📚 [API Reference](./api) - Complete API documentation
- 📊 [Performance Guide](./performance) - Optimization tips
- ❓ [FAQ](./faq) - Common questions
- 💻 [GitHub](https://github.com/LessUp/diy-flash-attention)

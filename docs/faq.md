# Frequently Asked Questions (FAQ)

Common questions and solutions for DIY FlashAttention.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Performance Issues](#performance-issues)
- [Correctness Issues](#correctness-issues)
- [GPU Compatibility](#gpu-compatibility)
- [Development](#development)

---

## Installation Issues

### Q: Triton installation fails? Help!

**A:** Triton requires specific CUDA and PyTorch versions. Try:

```bash
# Method 1: Via PyTorch (Recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton

# Method 2: Using conda
conda install -c conda-forge triton

# Method 3: Nightly (latest features)
pip install triton-nightly
```

**Requirements**:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10-3.11 |
| CUDA | 11.0 | 12.1 |
| PyTorch | 2.0.0 | 2.2+ |
| Triton | 2.1.0 | Latest |

---

### Q: "CUDA not available" error?

**A:** Check the following:

```bash
# 1. Check NVIDIA driver
nvidia-smi
# Should show GPU info and driver version

# 2. Check CUDA version
nvcc --version
# or
nvidia-smi | grep "CUDA Version"

# 3. Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

**Common causes**:
1. NVIDIA driver not installed or outdated
2. PyTorch CPU version installed
3. CUDA toolkit version mismatch with PyTorch

**Fix**:
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Q: Out of memory (OOM) error?

**A:** Try these solutions:

```python
# 1. Reduce batch size
batch = 1  # Instead of 4 or 8

# 2. Reduce sequence length
seq_len = 512  # Instead of 2048 or 4096

# 3. Use float16
dtype = torch.float16

# 4. Clear cache
torch.cuda.empty_cache()

# 5. Monitor memory usage
def print_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## Performance Issues

### Q: Why is Triton slower than PyTorch?

**A:** Possible causes and solutions:

| Cause | Symptom | Solution |
|-------|---------|----------|
| Small matrix | < 512 dimensions | Use PyTorch or increase matrix size |
| First run | Slow first time, fast after | Warm up kernel |
| Wrong block size | Performance varies | Use autotune |
| Non-contiguous | Unexpectedly slow | Ensure `.is_contiguous()` |

```python
# Warmup example
for _ in range(10):
    _ = triton_matmul(a, b)
torch.cuda.synchronize()

# Then benchmark
import time
start = time.time()
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) / 100 * 1000:.3f} ms")
```

---

### Q: How to choose optimal Block Size?

**A:**

**Recommended**: Use autotune (default)

```python
from kernels import triton_matmul

# Don't specify block size - automatic selection
c = triton_matmul(a, b)
```

**Manual selection guide**:

| Matrix Size | BLOCK_M | BLOCK_N | BLOCK_K |
|-------------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-1024 | 64 | 64 | 32 |
| 1024-2048 | 64 | 128 | 32 |
| > 2048 | 128 | 256 | 64 |

---

### Q: FlashAttention memory usage not reduced?

**A:** Check the following:

1. **Verify correct function usage**

```python
# ✅ Correct - using FlashAttention
from kernels import flash_attention
out = flash_attention(q, k, v)

# ❌ Wrong - using standard attention
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

2. **Sequence length matters**

```python
# Memory savings are significant for long sequences (> 256)
# N=256: ~75% savings
# N=1024: ~94% savings
# N=4096: ~98% savings
```

3. **Measure correctly**

```python
torch.cuda.reset_peak_memory_stats()
out = flash_attention(q, k, v)
peak_memory = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak_memory / 1e6:.1f} MB")
```

---

## Correctness Issues

### Q: Results don't match PyTorch exactly?

**A:** This is normal due to:

1. **Floating-point precision differences**
   - FP16 has precision loss
   - Different computation order causes rounding errors

2. **Acceptable tolerance**

```python
# Verification
import torch

# Recommended tolerance
rtol = 1e-2  # Relative tolerance
atol = 1e-2  # Absolute tolerance

assert torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)

# Check max diff
max_diff = (triton_out - torch_out).abs().max().item()
print(f"Max diff: {max_diff:.2e}")
# Usually < 1e-2 is acceptable
```

---

### Q: Causal attention results look wrong?

**A:** Check these steps:

```python
import torch
from kernels import flash_attention

# 1. Verify causal=True is set
out = flash_attention(q, k, v, causal=True)

# 2. Check input shapes
assert q.dim() == 4, f"Expected 4D, got {q.dim()}D"
assert q.shape == k.shape == v.shape, "Q, K, V shapes must match"

# 3. Compare with reference
ref = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True
)
max_diff = (out - ref).abs().max().item()
print(f"Max diff: {max_diff:.2e}")

# 4. Verify causality: modifying future shouldn't affect past
q_test = q.clone()
k_modified = k.clone()
k_modified[:, :, seq_len//2:, :] = torch.randn_like(
    k_modified[:, :, seq_len//2:, :]
)

out_orig = flash_attention(q_test, k, v, causal=True)
out_mod = flash_attention(q_test, k_modified, v, causal=True)

# First half should be identical
first_half_diff = (out_orig[:, :, :seq_len//2, :] -
                   out_mod[:, :, :seq_len//2, :]).abs().max()
print(f"Causality check: {first_half_diff.item():.2e} (should be ~0)")
```

---

## GPU Compatibility

### Q: What GPUs are supported?

**A:**

| GPU Architecture | Compute Capability | Models | Status |
|------------------|-------------------|--------|--------|
| Volta | SM70 | V100 | ✅ Basic support |
| Turing | SM75 | RTX 2080 | ✅ Basic support |
| Ampere | SM80 | A100, RTX 3090 | ✅ Full support |
| Ada | SM89 | RTX 4090 | ✅ Full support |
| Hopper | SM90 | H100 | ✅ Advanced features |
| Blackwell | SM100 | B100/B200 | ✅ Advanced features |

---

### Q: How to detect GPU features?

**A:**

```python
from utils import detect_gpu, print_gpu_info
from kernels import check_hopper_features

# Method 1: Detect GPU
caps = detect_gpu()
print(f"GPU: {caps.name}")
print(f"Architecture: {caps.arch.value}")
print(f"Compute Capability: {caps.compute_capability}")
print(f"TMA: {caps.has_tma}")
print(f"FP8: {caps.has_fp8}")

# Method 2: Print full info
print_gpu_info(caps)

# Method 3: Check Hopper+ features
features = check_hopper_features()
print(f"TMA: {features['tma_available']}")
print(f"FP8: {features['fp8_available']}")
```

---

### Q: Does it work on AMD GPUs?

**A:** Not currently. Triton primarily targets NVIDIA GPUs (CUDA). For AMD GPUs, ROCm support would be needed, which is not yet implemented.

---

## Development

### Q: How to debug Triton kernels?

**A:**

**Method 1: Small inputs for manual verification**

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                 device="cuda", dtype=torch.float16)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]],
                 device="cuda", dtype=torch.float16)

result = triton_matmul(a, b)
expected = torch.matmul(a.float(), b.float()).half()

print(f"Result:\n{result}")
print(f"Expected:\n{expected}")
print(f"Diff:\n{(result - expected).abs()}")
```

**Method 2: Check boundary conditions**

```python
a = torch.randn(33, 47, device="cuda", dtype=torch.float16)
b = torch.randn(47, 61, device="cuda", dtype=torch.float16)
result = triton_matmul(a, b)
assert result.shape == (33, 61)
```

**Method 3: Use validation tools**

```python
from utils import validate_matmul

is_valid, max_diff = validate_matmul(
    triton_matmul, m=128, n=128, k=128, verbose=True
)
```

---

## Quick Error Reference

| Error | Message | Cause | Solution |
|-------|---------|-------|----------|
| `ValueError` | "Expected 2D tensors" | Non-2D matmul input | Use `.view(M, K)` |
| `ValueError` | "Incompatible dimensions" | A.shape[1] != B.shape[0] | Check dimensions |
| `ValueError` | "CUDA tensors required" | Input on CPU | Use `.cuda()` |
| `ValueError` | "Expected 3D or 4D" | Wrong attention dims | Check shapes |
| `TypeError` | "Unsupported dtype" | Wrong dtype | Use fp16/bf16/fp32 |
| `RuntimeError` | "CUDA out of memory" | OOM | Reduce batch/seq_len |

---

## Getting Help

1. **Documentation**: [API Reference](./api) | [Tutorial](./tutorial) | [Performance Guide](./performance)

2. **GitHub Issues**: [Report bugs](https://github.com/LessUp/diy-flash-attention/issues)

3. **Bug Report Template**:
   ```markdown
   ## Environment
   - Python version:
   - PyTorch version:
   - Triton version:
   - CUDA version:
   - GPU model:

   ## Steps to Reproduce
   1. ...
   2. ...

   ## Error Message
   ```

   ```

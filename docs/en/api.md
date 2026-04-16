# API Reference

Complete API reference for DIY FlashAttention.

## Table of Contents

- [Kernels](#kernels)
  - [triton_matmul](#triton_matmul)
  - [flash_attention](#flash_attention)
  - [reference_attention](#reference_attention)
- [GPU Detection](#gpu-detection)
  - [detect_gpu](#detect_gpu)
  - [GPUCapabilities](#gpucapabilities)
  - [GPUArch](#gpuarch)
- [Benchmark Tools](#benchmark-tools)
- [Validation Tools](#validation-tools)

---

## Kernels

### `triton_matmul`

High-performance Triton matrix multiplication with autotune support.

```python
from kernels import triton_matmul

def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Matrix multiplication using Triton: C = A @ B

    Args:
        a: Input matrix A, shape (M, K). 
           Supported dtypes: float16, float32, bfloat16.
           Must be 2D CUDA tensor.
        
        b: Input matrix B, shape (K, N).
           Must be same dtype and device as a.
        
        block_m: M dimension block size (optional).
                 Must specify all three if used.
        
        block_n: N dimension block size.
        
        block_k: K dimension block size.
        
        use_autotune: Whether to use autotune (default: True).
                      Only active when block sizes not specified.

    Returns:
        torch.Tensor: Output matrix C, shape (M, N).
            - float16/bfloat16 input → same dtype output
            - float32 input → float16 output (converted internally)

    Raises:
        ValueError: Invalid tensor dimensions, non-CUDA tensor,
                    dimension mismatch, or invalid block sizes.
        TypeError: Unsupported dtype or dtype mismatch.

    Examples:
        Basic usage::

            import torch
            from kernels import triton_matmul

            a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
            b = torch.randn(512, 2048, device="cuda", dtype=torch.float16)
            c = triton_matmul(a, b)  # Uses autotune

        Manual block size::

            c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)

        BF16 support::

            a_bf16 = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
            c_bf16 = triton_matmul(a_bf16, b_bf16)
    """
```

---

### `flash_attention`

FlashAttention forward pass with O(N) memory complexity.

```python
from kernels import flash_attention

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention: Efficient attention computation.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        q: Query tensor.
           Shape: (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim).
           Supported dtypes: float16, float32, bfloat16.
           Must be CUDA tensor.
        
        k: Key tensor. Must match q's shape and dtype.
        
        v: Value tensor. Must match q's shape and dtype.
        
        causal: Whether to apply causal masking (for autoregressive models).
                Default: False. When True, position i can only attend to positions ≤ i.
        
        sm_scale: Softmax scale factor. Default: 1 / sqrt(head_dim).
        
        seq_lens: Effective sequence length per sample, shape (batch,).
                  dtype: int32. Used for variable-length sequences.
                  Positions beyond effective length are zeroed.

    Returns:
        torch.Tensor: Attention output, same shape as input q.
                      Computed internally in float16.

    Raises:
        ValueError: Invalid input dimensions, shape mismatch,
                    non-CUDA tensor, head_dim not 32 or 64,
                    or invalid seq_lens.
        TypeError: Unsupported or mismatched dtypes.

    Examples:
        Basic usage::

            q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
            out = flash_attention(q, k, v)

        Causal attention::

            out = flash_attention(q, k, v, causal=True)  # For GPT-style models

        Variable-length sequences::

            seq_lens = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
            out = flash_attention(q, k, v, seq_lens=seq_lens)
    """
```

---

## GPU Detection

### `detect_gpu`

Detect GPU capabilities and features.

```python
from utils import detect_gpu

def detect_gpu(device_id: int = 0) -> GPUCapabilities:
    """
    Detect capabilities of specified GPU.

    Args:
        device_id: CUDA device ID. Default: 0.

    Returns:
        GPUCapabilities: GPU capability information object.

    Example::

        from utils import detect_gpu

        caps = detect_gpu()
        print(f"GPU: {caps.name}")
        print(f"Architecture: {caps.arch.value}")
        print(f"Compute Capability: {caps.compute_capability}")
        print(f"TMA: {caps.has_tma}")
        print(f"FP8: {caps.has_fp8}")
    """
```

---

### `GPUCapabilities`

GPU capability information dataclass.

```python
from utils import GPUCapabilities
from dataclasses import dataclass

@dataclass
class GPUCapabilities:
    """GPU capability information."""

    name: str                    # GPU name, e.g., "NVIDIA GeForce RTX 4090"
    arch: GPUArch                # GPU architecture enum
    compute_capability: tuple    # e.g., (8, 9) for SM 89
    has_tma: bool                # Tensor Memory Accelerator (Hopper+)
    has_fp8: bool                # FP8 dtype support (Hopper+)
    has_warpgroup_mma: bool      # Warpgroup MMA (Hopper+)
    sram_per_sm: int             # Shared memory per SM (bytes)
    num_sms: int                 # Number of SMs
    total_memory_gb: float       # Total GPU memory (GB)
```

---

### `GPUArch`

GPU architecture enumeration.

```python
from utils import GPUArch

class GPUArch(Enum):
    """GPU Architecture Enumeration."""
    
    VOLTA = "sm_70"      # V100
    TURING = "sm_75"     # RTX 20 series
    AMPERE = "sm_80"     # A100, RTX 30 series
    ADA = "sm_89"        # RTX 40 series
    HOPPER = "sm_90"     # H100
    BLACKWELL = "sm_100" # B100/B200
    UNKNOWN = "unknown"
```

---

## Benchmark Tools

### `BenchmarkRunner`

Benchmark runner with formatted output.

```python
from utils import BenchmarkRunner

runner = BenchmarkRunner(warmup=10, rep=50)

# Matrix multiplication benchmark
results = runner.benchmark_matmul(
    triton_matmul,
    sizes=[(1024, 1024, 1024), (2048, 2048, 2048)],
)
runner.print_comparison_table(results)

# FlashAttention benchmark
results = runner.benchmark_attention(
    flash_attention,
    seq_lengths=[512, 1024, 2048],
)
```

---

## Validation Tools

### `validate_attention`

Validate FlashAttention correctness against PyTorch reference.

```python
from utils import validate_attention

is_valid, max_diff = validate_attention(
    flash_attention,
    batch=2,
    heads=8,
    seq_len=512,
    head_dim=64,
    causal=True,
)
print(f"Validation {'passed' if is_valid else 'failed'}, max diff: {max_diff:.2e}")
```

---

## Quick Reference

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Expected 2D tensors` | Non-2D matmul input | Use `.view()` or `.reshape()` |
| `ValueError: CUDA tensors required` | Input on CPU | Use `.to("cuda")` |
| `ValueError: Expected 3D or 4D tensors` | Wrong attention dimensions | Check input shape |
| `TypeError: Unsupported dtype` | Unsupported dtype | Use float16/bfloat16/float32 |

---

## Links

- [中文文档](../zh/api) - 完整中文 API 文档
- [Tutorial](./tutorial) - Step-by-step learning guide
- [Performance Guide](./performance) - Optimization tips
- [FAQ](./faq) - Common questions and solutions

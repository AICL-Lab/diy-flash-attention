# Spec: flash-attention-v2

## Overview

Implement FlashAttention v2 (striped/row-wise parallelism variant) alongside existing v1 block-column kernel. V2 targets modern GPU architectures (Ampere+) with better thread-block affinity and optional warp specialization for teaching.

## Public API

```python
def flash_attention_v2(
    q: torch.Tensor,       # (batch, seq_len, heads, head_dim)
    k: torch.Tensor,       # (batch, seq_len, heads, head_dim)
    v: torch.Tensor,       # (batch, seq_len, heads, head_dim)
    causal: bool = False,
    seq_lens: Optional[torch.Tensor] = None,  # (batch,)
    warp_specialize: bool = False,  # Teaching flag
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    FlashAttention v2: striped (row-wise) parallelism.
    
    Returns attention output with same shape as q.
    
    Args:
        q, k, v: Attention tensors
        causal: Enable causal masking
        seq_lens: Per-sequence lengths for variable batches
        warp_specialize: Use warp-level specialization (teaching tool)
        dtype: Compute dtype (default: infer from input)
    
    Raises:
        ValueError: Unsupported head_dim, causal+seq_lens conflict
        RuntimeError: CUDA/Triton unavailable
    """
```

## Behavior

### Forward Pass

- **Attention formula**: `output = softmax(Q @ K^T / sqrt(d_k) [+ mask]) @ V`
- **Causal masking**: Set future tokens to -inf before softmax if `causal=True`
- **Variable lengths**: Apply `seq_lens` mask when provided
- **Dtype handling**: Convert non-float16 inputs to float16 for kernel (output preserves input dtype)
- **Output shape**: `(batch, seq_len, heads, head_dim)` (same as input)

### Kernel Characteristics

- **Parallelization**: Row-wise (each thread block handles one row of attention scores)
- **Host descriptors** (Hopper+): Use TensorDescriptor for variable-stride inputs when available
- **Optional warp specialization**: Flag to demonstrate scheduling differences vs v1
- **BLOCK_M, BLOCK_N autotune**: Automatically selected based on GPU and head_dim

### Supported Configurations

| Parameter | Values | Notes |
|-----------|--------|-------|
| Batch | 1-256 | No hard limit; heuristics apply |
| Seq_len | 1-32768+ | Tested up to 4K, scaling tested |
| Heads | 8-40 | Per-head processing |
| head_dim | 32, 64 | Required (matching v1 for parity) |
| Dtype | float16, bfloat16, float32 | float32 converted to float16 internally |
| Causal | True, False | Tested with both |

### GPU Support

| Arch | Capability | Status |
|------|-----------|--------|
| Ampere | SM80 | Full support |
| Ada | SM89 | Full support |
| Hopper | SM90 | Full support + host descriptors |
| Blackwell | SM100 | Full support (assumed) |

Note: Volta/Turing supported via fallback to basic kernels.

## Correctness Criteria

1. **Numerical correctness**: Output matches PyTorch `F.scaled_dot_product_attention()` within floating-point tolerance (rtol=1e-5, atol=1e-6)
2. **Causal masking**: Causal v2 output identical to causal v1
3. **Variable lengths**: seq_lens mask applied correctly (output for padding positions should reflect masked attention)
4. **Dtype preservation**: Output dtype matches input dtype
5. **Gradient-ready**: Forward pass structure compatible with torch.autograd (backward not implemented in Phase A)

## Integration Points

- **API entry**: `kernels/__init__.py` exports `flash_attention_v2`
- **Backend selector**: `backend_selector.select_attention()` routes `variant="v2"`
- **Benchmarks**: Updated `benchmarks/bench_flash.py` to include v1 vs v2 comparison
- **Tests**: New tests in `tests/test_flash_v2.py` (standalone) and merged into `test_properties.py` (hypothesis)

## Testing Strategy

### Unit Tests (test_flash_v2.py)

```python
def test_flash_attention_v2_basic():
    """V2 kernel forward pass on standard shapes."""
def test_flash_attention_v2_causal():
    """Causal masking correctness."""
def test_flash_attention_v2_seq_lens():
    """Variable sequence lengths handling."""
def test_flash_attention_v2_dtypes():
    """float16, bfloat16, float32 I/O compatibility."""
def test_flash_attention_v2_vs_v1():
    """Numerical parity with v1 on identical inputs."""
```

### Property-Based Tests (test_properties.py additions)

```python
@hypothesis.given(
    batch=..., seq_len=..., heads=..., head_dim=..., 
    dtype=..., causal=...
)
def test_flash_attention_v2_properties(...):
    """Randomized correctness against PyTorch baseline."""
```

### Benchmark Comparisons

- `bench_flash.py`: v1 vs v2 FLOPS, memory bandwidth, latency
- Scaling: seq_len from 256 to 16384, batch from 1 to 64
- Output: TFLOPS, GB/s, timing comparison

## Deferred (Phase B+)

- Warp group MMA specialization (Hopper-only)
- FP8 output optimization
- Multi-query/grouped attention
- Backward pass (gradient computation)
- Sliding window, prefix_lm mask types

## Success Criteria

- [ ] Kernel compiles on Triton 2.1+
- [ ] All unit tests pass
- [ ] Property-based tests pass on 100+ generated inputs
- [ ] Numerical correctness validated vs PyTorch (rtol=1e-5)
- [ ] V1 vs V2 comparison benchmark shows 5-15% speedup on Ampere+ (expected range)
- [ ] GPU memory usage within ~5% of v1
- [ ] Code reviewed for numerical stability

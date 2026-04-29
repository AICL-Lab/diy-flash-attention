# Tensor Layout Guide

Understanding tensor layout differences between FlashAttention V1 and V2.

## The Critical Difference

⚠️ **V1 and V2 use different tensor layouts** - the `heads` and `seq_len` dimensions are swapped!

| Version | Layout |
|---------|--------|
| V1 (`flash_attention`) | `(batch, heads, seq_len, head_dim)` |
| V2 (`flash_attention_v2`) | `(batch, seq_len, heads, head_dim)` |

## Why the Difference?

FlashAttention V2 uses row-wise (striped) parallelism, which requires the `seq_len` dimension to be contiguous in memory for optimal memory access patterns on Ampere+ GPUs.

V1 uses column-parallel processing, where `heads` being contiguous is more natural.

## Example: Correct Usage

### V1 (flash_attention)

```python
import torch
from kernels import flash_attention

batch, heads, seq_len, head_dim = 2, 8, 512, 64

# V1 expects: (batch, heads, seq_len, head_dim)
q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)
print(out.shape)  # (2, 8, 512, 64)
```

### V2 (flash_attention_v2)

```python
import torch
from kernels import flash_attention_v2

batch, heads, seq_len, head_dim = 2, 8, 512, 64

# V2 expects: (batch, seq_len, heads, head_dim)
q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)

out = flash_attention_v2(q, k, v, causal=True)
print(out.shape)  # (2, 512, 8, 64)
```

## Converting Between Versions

To switch between V1 and V2, you need to transpose dimensions 1 and 2:

```python
# From V1 to V2 format
q_v2 = q_v1.transpose(1, 2)  # (b, h, s, d) -> (b, s, h, d)
k_v2 = k_v1.transpose(1, 2)
v_v2 = v_v1.transpose(1, 2)

# Run V2
out_v2 = flash_attention_v2(q_v2, k_v2, v_v2, causal=True)

# Convert back to V1 format
out_v1 = out_v2.transpose(1, 2)  # (b, s, h, d) -> (b, h, s, d)
```

## Which Should You Use?

| Scenario | Recommendation |
|----------|----------------|
| Ampere+ GPU (A100, RTX 30xx, RTX 40xx) | V2 for 5-15% better performance |
| Volta/Turing GPU (V100, RTX 20xx) | V1 (V2 not optimized for older architectures) |
| Large batch + long sequences | V2 |
| Code compatibility priority | V1 (standard PyTorch attention layout) |

## BackendSelector

If you want automatic selection, use `BackendSelector`:

```python
from kernels import BackendSelector, flash_attention, flash_attention_v2

# The selector handles layout differences internally
selector = BackendSelector()
kernel = selector.select_attention(batch=2, heads=8, seq_len=1024, head_dim=64)

# Or use flash_attention with variant parameter
from kernels import flash_attention
out = flash_attention(q, k, v, causal=True, variant="auto")
```

## Summary

- **V1**: `(batch, heads, seq_len, head_dim)` - standard layout, universal support
- **V2**: `(batch, seq_len, heads, head_dim)` - optimized for Ampere+, 5-15% faster
- **Always check tensor shapes** when switching versions!

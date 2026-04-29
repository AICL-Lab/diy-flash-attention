# Spec: kernel-selector

## Overview

Implement a minimal backend selector registry that routes kernel calls based on GPU capability, user preference, and problem size heuristics. Unifies v1/v2/persistent kernel dispatch without adding framework complexity.

## Public API

```python
from enum import Enum
from typing import Optional, Literal

class KernelVariant(str, Enum):
    """Kernel implementation variant."""
    V1 = "v1"                      # Baseline block-column attention
    V2 = "v2"                      # Striped row-wise attention
    PERSISTENT = "persistent"      # Persistent thread-block variant
    AUTO = "auto"                  # Automatic selection based on heuristics

class BackendSelector:
    """Unified kernel dispatch registry."""
    
    @staticmethod
    def select_attention(
        variant: KernelVariant = KernelVariant.AUTO,
        q_shape: Tuple[int, int, int, int] = None,  # (batch, seq_len, heads, head_dim)
        gpu_capability: Optional[int] = None,       # Compute capability (e.g., 80)
    ) -> Callable:
        """
        Select attention kernel implementation.
        
        Args:
            variant: Preferred variant (AUTO uses heuristics)
            q_shape: Query tensor shape for heuristic-based selection
            gpu_capability: GPU compute capability (auto-detected if None)
        
        Returns:
            Callable kernel function with signature:
            fn(q, k, v, causal=False, seq_lens=None, dtype=None) -> Tensor
        
        Raises:
            ValueError: Unsupported variant or GPU
            RuntimeError: GPU detection failed
        """
    
    @staticmethod
    def select_matmul(
        variant: KernelVariant = KernelVariant.AUTO,
        shape: Tuple[int, int, int] = None,  # (m, k, n)
        gpu_capability: Optional[int] = None,
    ) -> Callable:
        """
        Select matmul kernel implementation.
        
        Returns:
            Callable with signature: fn(a, b) -> Tensor
        """
    
    @staticmethod
    def get_available_variants(
        kernel_type: Literal["attention", "matmul"],
        gpu_capability: Optional[int] = None,
    ) -> list[KernelVariant]:
        """
        List available kernel variants for given GPU.
        
        Returns:
            Subset of [V1, V2, PERSISTENT] available on GPU
        """
    
    @staticmethod
    def set_default_variant(
        kernel_type: Literal["attention", "matmul"],
        variant: KernelVariant,
    ) -> None:
        """
        Override default variant selection.
        
        Example:
            BackendSelector.set_default_variant("attention", KernelVariant.V2)
        """

def select_attention_kernel(variant: str = "auto", **kwargs) -> Callable:
    """Convenience function wrapping BackendSelector.select_attention()."""

def select_matmul_kernel(variant: str = "auto", **kwargs) -> Callable:
    """Convenience function wrapping BackendSelector.select_matmul()."""

def benchmark_kernel_selection(
    kernel_type: str,
    shape: Tuple,
    variants: Optional[list[str]] = None,
    iterations: int = 5,
) -> dict:
    """
    Benchmark available kernel variants on given shape.
    
    Returns:
        {
            'v1': {'time_ms': 1.5, 'gbps': 400, ...},
            'v2': {'time_ms': 1.2, 'gbps': 450, ...},
            ...
        }
    """
```

## Selection Heuristics

### Attention Kernel Selection

| Condition | Selected Variant | Rationale |
|-----------|-----------------|-----------|
| `variant="v1"` | V1 | User override |
| `variant="v2"` | V2 | User override |
| GPU < Ampere (SM80) | V1 | V2 requires modern hardware |
| batch < 2 | V1 | V1 better for small batch (lower overhead) |
| seq_len < 512 | V1 | V1 simpler, competitive for short sequences |
| batch >= 4, seq_len >= 4096 | V2 | V2 advantages: better parallelism, TMA-ready |
| else | V1 | Conservative default |

### Matmul Kernel Selection

| Condition | Selected Variant | Rationale |
|-----------|-----------------|-----------|
| GPU < Ampere | V1 | Baseline only |
| persistent preferred + m×k >> n (memory-bound) | Persistent | Persistent better for memory-bound ops |
| else | V1 | V1 (or V2) default |

## Supported Kernels by GPU Architecture

| Arch | V1 | V2 | Persistent |
|------|----|----|------------|
| Volta (SM70) | ✅ | ⚠️ (basic) | ⚠️ (basic) |
| Turing (SM75) | ✅ | ⚠️ (basic) | ⚠️ (basic) |
| Ampere (SM80) | ✅ | ✅ | ✅ |
| Ada (SM89) | ✅ | ✅ | ✅ |
| Hopper (SM90) | ✅ | ✅ (TMA) | ✅ (TMA) |
| Blackwell (SM100) | ✅ | ✅ | ✅ |

**Legend:**
- ✅ = Full support
- ⚠️ = Supported but not optimized (may fall back to V1)

## Integration Points

### In Kernel Modules

```python
# In kernels/__init__.py
from .backend_selector import select_attention_kernel, select_matmul_kernel

def flash_attention(
    q, k, v, 
    causal=False, 
    seq_lens=None,
    variant="auto",
    **kwargs
) -> Tensor:
    """Public API with automatic kernel selection."""
    kernel = select_attention_kernel(variant, q_shape=q.shape)
    return kernel(q, k, v, causal=causal, seq_lens=seq_lens, **kwargs)
```

### In Benchmarks

```python
# benchmarks/bench_selector.py
from kernels.backend_selector import benchmark_kernel_selection

results = benchmark_kernel_selection(
    "attention",
    shape=(2, 4096, 8, 64),
    variants=["v1", "v2"]
)
```

### In Tests

```python
# tests/test_backend_selector.py
def test_selector_routes_correctly():
    """Verify selector picks expected kernel."""
    kernel = select_attention_kernel(variant="v2")
    assert callable(kernel)
```

## Behavior

### Kernel Dispatch

1. **User specifies variant** (or "auto" for heuristics)
2. **GPU capability detected** via `detect_gpu()`
3. **Shape heuristics applied** (if variant="auto")
4. **Availability check**: Raise error if kernel unavailable on this GPU
5. **Return kernel function** with configured parameters

### Backward Compatibility

- Existing calls to `flash_attention()` / `triton_matmul()` remain unchanged
- New `variant=` parameter is optional; defaults to "auto"
- Default behavior (v1 for conservative/small problems) preserves existing performance

### Configuration State

```python
# Global default can be overridden
BackendSelector._default_variants = {
    "attention": KernelVariant.AUTO,
    "matmul": KernelVariant.V1,
}

# Per-context override (future: context manager)
with kernel_selector_context(default_variant="v2"):
    out = flash_attention(q, k, v)  # Uses v2
```

## Testing Strategy

### Unit Tests (tests/test_backend_selector.py)

```python
def test_select_attention_v1():
    """Explicit v1 selection returns v1 kernel."""

def test_select_attention_v2():
    """Explicit v2 selection returns v2 kernel."""

def test_select_attention_auto_heuristics():
    """Auto variant applies heuristics correctly."""

def test_selector_respects_gpu_capability():
    """V2 unavailable on Volta; selection falls back or errors."""

def test_benchmark_kernel_selection():
    """Benchmark variant produces timing results."""

def test_default_variant_override():
    """Global default can be overridden."""
```

### Integration Tests

- Selector + flash_attention integration: output matches kernel call
- Selector + benchmark: benchmark runs all available variants
- Selector state isolation: different threads get consistent results

## Configuration & Defaults

```python
# kernels/config.py additions
KERNEL_VARIANT_DEFAULTS = {
    "attention": "auto",      # Auto-select based on GPU + shape
    "matmul": "v1",          # Conservative: stick with v1 matmul
}

KERNEL_HEURISTICS = {
    "attention_small_batch_threshold": 2,
    "attention_short_seq_threshold": 512,
    "attention_large_batch_threshold": 4,
    "attention_long_seq_threshold": 4096,
    "matmul_memory_bound_ratio": 1.5,  # m*k / n ratio for persistent
}
```

## Performance Expectations

| Scenario | Expected Behavior | Notes |
|----------|-------------------|-------|
| Auto on Ampere, batch=1, seq=256 | Select V1 | Conservative (small problem) |
| Auto on Ampere, batch=8, seq=8192 | Select V2 | V2 advantages kick in |
| Auto on Hopper | Select V2 (TMA-optimized) | Host descriptor path |
| Auto on Volta | Select V1 | No V2 available |
| Explicit V2 on Volta | Error | Kernel unavailable |

## Deferred (Phase B+)

- Automatic cost model (dynamic kernel selection based on profiling)
- JIT module generation (FlashInfer-style kernel caching)
- Per-context kernel preferences (context manager)
- Kernel caching and reuse (avoid re-selection overhead)
- Fallback chains (if v2 fails, try v1)

## Success Criteria

- [ ] Selector correctly routes based on variant parameter
- [ ] GPU capability detection working
- [ ] Heuristics produce sensible kernel choices
- [ ] All kernel variants available on supported GPUs
- [ ] Fallback/error handling for unsupported GPU+variant combos
- [ ] Unit tests all pass
- [ ] Integration with flash_attention() and triton_matmul() transparent
- [ ] Benchmark selector produces timing comparisons
- [ ] Documentation includes selection heuristic table

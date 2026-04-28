# Spec: persistent-kernels

## Overview

Implement persistent thread-block matmul and minimal persistent attention reference with built-in profiling hooks. These kernels teach occupancy/memory-hierarchy trade-offs without replacing existing v1 baseline kernels.

## Public API

```python
def persistent_matmul(
    a: torch.Tensor,          # (m, k)
    b: torch.Tensor,          # (k, n)
    block_m: int = 64,
    block_n: int = 64,
    profile: bool = False,    # Enable profiling metrics
) -> torch.Tensor:
    """
    Persistent thread-block matmul (reference implementation).
    
    Args:
        a: Left operand (m, k)
        b: Right operand (k, n)
        block_m, block_n: Thread block tile sizes
        profile: Enable GPU memory profiling
    
    Returns:
        Output (m, n)
    
    Raises:
        ValueError: Unsupported shapes/dtypes
        RuntimeError: CUDA/Triton unavailable
    """

def persistent_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    profile: bool = False,
) -> torch.Tensor:
    """
    Persistent attention kernel (row-wise, reference).
    Similar to flash_attention_v2 but emphasizing occupancy analysis.
    """
```

## Behavior

### Persistent Matmul Kernel

- **Strategy**: Single kernel invocation processes entire (m, n) matrix via persistent thread blocks
- **Register pressure**: Configurable; higher BLOCK_M/BLOCK_N → more registers per block → lower occupancy
- **Shared memory**: Tuned per GPU arch (Ampere: 96KB baseline)
- **Output**: Identical to standard matmul (within float precision)

### Persistent Attention Kernel

- **Strategy**: Row-wise parallelism (one thread block per attention head/query row)
- **Persistence**: All attention heads computed by persistent blocks in single pass
- **Memory hierarchy**: Emphasizes shared memory buffering vs global memory stalls
- **Profiling hooks**: Metrics on SRAM utilization, register spilling

### Profiling Integration

When `profile=True`, kernel emits:
- GPU occupancy % (threads active / max hardware threads)
- Shared memory usage (bytes, % of available)
- Register count per thread
- Estimated stall cycles (L2 miss penalty)

## Supported Configurations

### Persistent Matmul

| Parameter | Values | Notes |
|-----------|--------|-------|
| M, N, K | 64-16384 | Tested up to 8K×8K |
| Dtype | float16, float32, bfloat16 | Mixed precision deferred |
| block_m | 16-256 | Trade occupancy vs register pressure |
| block_n | 16-256 | Tuned per GPU arch |

### Persistent Attention

| Parameter | Values | Notes |
|-----------|--------|-------|
| Batch | 1-256 | Per-head row processing |
| Seq_len | 64-16384 | Minimum 64 for occupancy benefits |
| Heads | 8-40 | Standard range |
| head_dim | 32, 64 | Matching v1/v2 |
| Causal | True, False | Tested both |

## Correctness Criteria

1. **Numerical correctness**: Output matches naive matmul within float precision (rtol=1e-5)
2. **Occupancy trade-off**: Demonstrable relationship between BLOCK_M/BLOCK_N and GPU occupancy
3. **Profiling accuracy**: GPU metrics within 5% of torch.profiler baseline
4. **Stability**: No register spilling on GPU with sufficient registers (no OOM exceptions)

## Integration Points

- **API entry**: `kernels/__init__.py` exports `persistent_matmul`, `persistent_attention`
- **Backend selector**: `backend_selector.select_kernel(variant="persistent")`
- **Profiling module**: Hooks into `utils/profiling.py` for metric extraction
- **Benchmarks**: `benchmarks/bench_persistent.py` dedicated script with occupancy analysis
- **Tests**: `tests/test_persistent.py` with correctness + profiling validation

## Testing Strategy

### Unit Tests (test_persistent.py)

```python
def test_persistent_matmul_correctness():
    """Persistent matmul output matches torch.mm()."""

def test_persistent_matmul_occupancy_scaling():
    """Verify occupancy decreases with larger block sizes."""

def test_persistent_attention_correctness():
    """Output matches reference attention."""

def test_persistent_attention_causal():
    """Causal masking correctness."""

def test_profiling_metrics_extraction():
    """GPU metrics extracted and returned correctly."""
```

### Benchmark Script (bench_persistent.py)

- Occupancy vs block size sweep
- Persistent vs naive vs tiled matmul comparison
- Attention: persistent vs v1 vs v2 timing + occupancy
- Output: CSV with metrics, occupancy plot

## GPU Support

| Arch | Capability | Minimum | Recommendation |
|------|-----------|---------|-----------------|
| Ampere | SM80 | Supported | Recommended baseline |
| Ada | SM89 | Supported | Full feature set |
| Hopper | SM90 | Supported | Optimal TMA integration (future) |
| Blackwell | SM100 | Assumed | Optimal warp group MMA (future) |

**Note**: Volta/Turing less suitable (lower occupancy, smaller SMEM); fallback available.

## Profiling Utilities Interface

```python
@dataclass
class GPUMemoryProfile:
    occupancy_pct: float       # 0-100
    sram_used_bytes: int
    sram_available_bytes: int
    registers_per_thread: int
    l2_misses: int
    estimated_stall_cycles: int

def profile_kernel(kernel_fn, *args, **kwargs) -> GPUMemoryProfile:
    """Context manager wrapping torch.profiler."""
```

## Deferred (Phase B+)

- TMA/async copy optimization (reference only now)
- Automatic occupancy tuning (fixed heuristics in Phase A)
- Register pressure prediction/spilling mitigation
- Per-warp scheduling analysis
- Mixed-precision persistent kernels

## Success Criteria

- [ ] Persistent matmul compiles and produces correct output
- [ ] Persistent attention compiles and produces correct output
- [ ] Profiling metrics extracted for both kernels
- [ ] Unit tests all pass
- [ ] Occupancy scaling visible in benchmark (e.g., 4×BLOCK_M → 50% occupancy)
- [ ] Benchmark comparison shows trade-offs (persistent good for memory-bound, not for compute-bound)
- [ ] GPU memory usage tracked and reported
- [ ] Code enables clear teaching of occupancy/register pressure concepts

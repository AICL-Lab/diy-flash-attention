# Spec: profiling-toolkit

## Overview

Create `utils/profiling.py` with GPU memory analysis, performance counter extraction, and educational metrics. Integrates with torch.profiler and optional Nsight Compute parsing for teaching GPU performance fundamentals.

## Public API

```python
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class GPUMemoryProfile:
    """GPU memory hierarchy metrics."""
    occupancy_pct: float             # Active threads / max hardware threads (0-100)
    smem_used_bytes: int             # Shared memory per thread block
    smem_available_bytes: int        # Total shared memory on GPU
    reg_per_thread: int              # Registers per thread
    l2_hit_rate: float               # L2 cache hit rate (0-1)
    estimated_stall_cycles: int      # Estimated stall cycles from memory
    
    def smem_pressure_pct(self) -> float:
        """Shared memory utilization %."""
        return 100 * self.smem_used_bytes / max(1, self.smem_available_bytes)
    
    def can_increase_occupancy(self) -> bool:
        """Heuristic: sufficient SMEM headroom to increase block size."""
        return self.smem_pressure_pct() < 70

@dataclass
class KernelBenchmark:
    """Kernel performance metrics."""
    kernel_name: str
    elapsed_ms: float
    tflops: float               # Tensor operations per second
    gbps: float                 # Global memory bandwidth
    memory_profile: GPUMemoryProfile

class GPUProfiler:
    """Context manager for profiling GPU kernels."""
    
    def __init__(self, gpu_id: int = 0, trace_memory: bool = True):
        """Initialize profiler for given GPU."""
    
    def __enter__(self):
        """Start profiling."""
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and extract metrics."""
    
    def get_profile(self) -> GPUMemoryProfile:
        """Return accumulated profiling metrics."""
    
    def reset(self):
        """Clear profiling state."""

def profile_kernel(
    kernel_fn: Callable,
    *args,
    gpu_id: int = 0,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs
) -> KernelBenchmark:
    """
    Profile a kernel function with warmup and averaging.
    
    Args:
        kernel_fn: Function to profile
        *args, **kwargs: Arguments to pass to kernel_fn
        gpu_id: GPU device ID
        warmup: Warm-up iterations (not measured)
        iterations: Measurement iterations (averaged)
    
    Returns:
        KernelBenchmark with aggregated metrics
    """

def estimate_occupancy(
    block_size: int,
    registers_per_thread: int,
    shared_memory_bytes: int,
    gpu_capability: int,
) -> float:
    """
    Estimate GPU occupancy (%) given kernel parameters.
    
    Args:
        block_size: Threads per thread block
        registers_per_thread: Register allocation per thread
        shared_memory_bytes: Shared memory per thread block
        gpu_capability: GPU compute capability (e.g., 80 for Ampere)
    
    Returns:
        Occupancy percentage (0-100)
    """

def get_gpu_memory_hierarchy(gpu_capability: int) -> dict:
    """
    Return GPU memory hierarchy specs for given capability.
    
    Returns:
        {
            'smem_per_block': int,        # Shared memory per block (bytes)
            'registers_per_warp': int,    # Register file capacity
            'l1_cache_kb': int,
            'l2_cache_mb': int,
            'hbm_gb': int,
            'peak_bandwidth_gbps': float,
        }
    """

def get_roofline_model(
    gpu_capability: int,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Roofline model parameters for given GPU and dtype.
    
    Returns:
        {
            'compute_roof_tflops': float,     # Peak compute TFLOPS
            'memory_roof_gbps': float,        # Peak bandwidth GB/s
            'arithmetic_intensity_threshold': float,  # AI where compute/memory limited
        }
    """
```

## Behavior

### GPU Memory Profile Extraction

1. **Occupancy calculation**: Based on block size, registers, shared memory per GPU arch
2. **Memory hierarchy awareness**: L1/L2 cache hit rates, SMEM pressure
3. **Stall estimation**: Rough model of memory latency impact
4. **Roofline model**: Compute vs memory bandwidth ceiling for benchmark analysis

### Profiling Integration

- Uses `torch.profiler` as primary data source (available in PyTorch 1.10+)
- Optional integration with Nsight Compute output (JSON/CSV parsing)
- Context manager pattern for easy measurement encapsulation
- Warmup iterations + averaging for stable measurements

### Educational Output

- Simple TFLOPS/GB/s calculations
- Occupancy % visualization (text bars, optional plots)
- Memory hierarchy stress (SMEM %, register pressure)
- Roofline analysis: "kernel is memory-bound" vs "compute-bound"

## Supported Hardware

| Arch | Capability | Specs Available |
|------|-----------|-----------------|
| Volta | SM70 | Basic (SMEM 96KB, no L1) |
| Turing | SM75 | Basic |
| Ampere | SM80 | Full (96KB SMEM, 128KB registers/thread) |
| Ada | SM89 | Full (96KB SMEM, 260KB registers/thread) |
| Hopper | SM90 | Full + TMA support metadata |
| Blackwell | SM100 | Full + WGMMA support (assumed) |

## Configuration & Integration

### Integration with Existing Modules

```python
# In kernels/modern_features.py
from utils.profiling import estimate_occupancy, get_gpu_memory_hierarchy

# Use occupancy estimates to select kernel variants
occupancy = estimate_occupancy(
    block_size=256,
    registers_per_thread=64,
    shared_memory_bytes=32768,
    gpu_capability=detect_gpu().compute_capability
)
```

### Benchmark Integration

```python
# In benchmarks/bench_flash.py
from utils.profiling import profile_kernel

result = profile_kernel(
    flash_attention_v2,
    q, k, v, causal=True,
    iterations=10
)
print(f"V2: {result.tflops:.1f} TFLOPS, {result.gbps:.1f} GB/s, {result.memory_profile.occupancy_pct:.1f}%")
```

## Testing Strategy

### Unit Tests (tests/test_profiling.py)

```python
def test_occupancy_estimation():
    """Occupancy calculation matches known GPU specs."""

def test_memory_hierarchy_specs():
    """GPU hierarchy parameters correct per architecture."""

def test_profiling_context_manager():
    """Profile metrics extracted correctly."""

def test_roofline_model():
    """Roofline calculation produces sensible bounds."""

def test_profiling_decorator():
    """Profile wrapper works with various kernel signatures."""
```

### Integration Tests

- Profile v1 attention kernel; verify occupancy < v2
- Profile persistent matmul; verify increasing BLOCK_M decreases occupancy
- Profile memory-bound vs compute-bound kernels; verify roofline classification

## Deferred (Phase B+)

- Automated register pressure reduction strategies
- Per-warp stall analysis (requires Nsight Systems)
- Custom roofline plots (matplotlib integration optional)
- Dynamic occupancy-based kernel selection
- Hardware profiling counters (advanced NSight Compute integration)

## Success Criteria

- [ ] GPU memory hierarchy specs correct for Ampere/Ada/Hopper
- [ ] Occupancy estimation within 5% of actual measurements
- [ ] Profile metrics extracted for all supported GPU architectures
- [ ] torch.profiler integration working without errors
- [ ] Unit tests all pass
- [ ] Benchmark output includes profiling metrics (TFLOPS, GB/s, occupancy)
- [ ] Educational metrics ("memory-bound" / "compute-bound" classification) working
- [ ] Documentation includes roofline interpretation guide

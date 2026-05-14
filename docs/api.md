# API Reference

Complete API reference for DIY FlashAttention.

## Table of Contents

- [Kernels](#kernels)
  - [triton_matmul](#triton_matmul)
  - [flash_attention](#flash_attention)
  - [flash_attention_v2](#flash_attention_v2)
  - [persistent_matmul](#persistent_matmul)
  - [reference_attention](#reference_attention)
- [Backend Selector](#backend-selector)
  - [BackendSelector](#backendselector)
  - [KernelVariant](#kernelvariant)
  - [select_attention_kernel](#select_attention_kernel)
- [Mask DSL](#mask-dsl)
  - [BlockMask](#blockmask)
  - [create_block_mask](#create_block_mask)
  - [compose_block_masks](#compose_block_masks)
- [GPU Detection](#gpu-detection)
  - [detect_gpu](#detect_gpu)
  - [GPUCapabilities](#gpucapabilities)
  - [GPUArch](#gpuarch)
- [Profiling Tools](#profiling-tools)
  - [GPUMemoryProfile](#gpumemoryprofile)
  - [KernelBenchmark](#kernelbenchmark)
  - [estimate_occupancy](#estimate_occupancy)
  - [get_gpu_memory_hierarchy](#get_gpu_memory_hierarchy)
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

    ⚠️ Tensor Layout: (batch, heads, seq_len, head_dim)
    This is DIFFERENT from flash_attention_v2 which uses (batch, seq_len, heads, head_dim).

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

### `flash_attention_v2`

FlashAttention V2 with row-wise parallelism for better performance on Ampere+.

```python
from kernels import flash_attention_v2

def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention V2: Memory-efficient attention with row-wise parallelism.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V

    V2 uses row-wise (striped) parallelism where each thread block handles
    one query row across all key blocks. This provides better memory access
    patterns on Ampere+ GPUs (5-15% faster than V1).

    ⚠️ Tensor Layout: (batch, seq_len, heads, head_dim)
    This is DIFFERENT from flash_attention (V1) which uses (batch, heads, seq_len, head_dim).
    The `heads` and `seq_len` dimensions are SWAPPED!

    Args:
        q: Query tensor of shape (batch, seq_len, heads, head_dim)
        k: Key tensor of shape (batch, seq_len, heads, head_dim)
        v: Value tensor of shape (batch, seq_len, heads, head_dim)
        causal: Whether to apply causal masking (for autoregressive models)
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        seq_lens: Optional sequence lengths per batch element (shape: [batch])

    Returns:
        Attention output of same shape as input

    Raises:
        ValueError: If tensor shapes/devices are incompatible or inputs are not CUDA tensors
        TypeError: If input dtypes are unsupported or inconsistent

    Example:
        ::

            import torch
            from kernels import flash_attention_v2

            batch, seq_len, heads, head_dim = 2, 512, 8, 64

            # Note: V2 layout is (batch, seq_len, heads, head_dim)!
            q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)

            out = flash_attention_v2(q, k, v, causal=True)
    """
```

---

### `persistent_matmul`

Persistent thread-block matrix multiplication.

```python
from kernels import persistent_matmul

def persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Persistent thread-block matrix multiplication.

    Uses persistent threads where each block processes multiple output tiles
    in a grid-stride loop. This demonstrates the persistent thread pattern
    which can reduce kernel launch overhead for small matrices.

    Args:
        a: Left matrix of shape (M, K)
        b: Right matrix of shape (K, N)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
        block_k: Block size for K dimension

    Returns:
        Result matrix of shape (M, N)

    Example:
        ::

            import torch
            from kernels import persistent_matmul

            a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
            b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
            c = persistent_matmul(a, b)
    """
```

---

### `reference_attention`

PyTorch reference implementation for validation.

```python
from kernels.flash_attn import reference_attention

def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard attention implementation for validation."""
```

---

## Backend Selector

### `BackendSelector`

Unified kernel dispatch registry.

```python
from kernels import BackendSelector

class BackendSelector:
    """
    Unified kernel dispatch registry.

    Selects optimal kernel implementation based on:
    - GPU compute capability
    - Problem size (batch, seq_len, heads, head_dim)
    - User preference

    Heuristics:
    - V2 requires Ampere+ (SM80) for optimal performance
    - Small problems (batch < 2 or seq < 512) → V1
    - Large problems (batch >= 4 and seq >= 4096) → V2
    - Default → V1

    Example:
        ::

            from kernels import BackendSelector

            selector = BackendSelector()

            # Get optimal kernel for your problem
            kernel = selector.select_attention(
                q_shape=(4, 2048, 8, 64),
                gpu_capability=80  # Ampere
            )
            out = kernel(q, k, v, causal=True)
    """

    @staticmethod
    def select_attention(
        variant: KernelVariant = KernelVariant.AUTO,
        q_shape: Optional[Tuple[int, int, int, int]] = None,
        gpu_capability: Optional[int] = None,
    ) -> Callable:
        """
        Select attention kernel implementation.

        Args:
            variant: Preferred variant (AUTO uses heuristics)
            q_shape: Query tensor shape (batch, seq_len, heads, head_dim)
            gpu_capability: GPU compute capability major version

        Returns:
            Callable kernel function
        """
```

---

### `KernelVariant`

Kernel implementation variant enumeration.

```python
from kernels import KernelVariant

class KernelVariant(str, Enum):
    """Kernel implementation variant."""

    V1 = "v1"            # Standard FlashAttention
    V2 = "v2"            # Row-wise parallel (Ampere+)
    PERSISTENT = "persistent"
    AUTO = "auto"        # Use heuristics
```

---

### `select_attention_kernel`

Convenience wrapper for BackendSelector.

```python
from kernels import select_attention_kernel

def select_attention_kernel(
    variant: str = "auto",
    q_shape: Optional[Tuple[int, int, int, int]] = None,
    gpu_capability: Optional[int] = None,
) -> Callable:
    """
    Convenience wrapper for BackendSelector.select_attention.

    Args:
        variant: Variant name string ("v1", "v2", "persistent", "auto")
        q_shape: Query tensor shape for heuristic selection
        gpu_capability: GPU compute capability major version

    Returns:
        Callable kernel function

    Example:
        ::

            from kernels import select_attention_kernel

            # Let it choose automatically
            kernel = select_attention_kernel("auto", q_shape=(4, 2048, 8, 64))
    """
```

---

## Mask DSL

### `BlockMask`

Block-level attention mask abstraction.

```python
from kernels import BlockMask

@dataclass
class BlockMask:
    """
    Block-level attention mask abstraction.

    Instead of a full (seq_len, seq_len) token-level mask, we use a
    (n_query_blocks, n_key_blocks) block-level mask. This is more memory
    efficient and aligns with FlashAttention's tiled computation.

    Attributes:
        mask_matrix: Boolean tensor of shape (n_query_blocks, n_key_blocks)
        query_block_size: Number of query tokens per block
        key_block_size: Number of key tokens per block
        mask_type: Pattern name for debugging/logging

    Example:
        ::

            from kernels import BlockMask, create_block_mask

            # Create a causal mask
            mask = create_block_mask("causal", query_len=256, key_len=256)

            # Apply to attention scores
            masked_scores = mask.apply_to_scores(scores)
    """

    def apply_to_scores(
        self,
        scores: torch.Tensor,
        score_mod: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Apply block mask to attention scores."""
```

---

### `create_block_mask`

Create a block-level attention mask.

```python
from kernels import create_block_mask

def create_block_mask(
    pattern: str,
    query_len: int,
    key_len: int,
    block_size: int = 128,
    causal: bool = False,
    sliding_window: Optional[int] = None,
    prefix_len: Optional[int] = None,
) -> BlockMask:
    """
    Create a block-level attention mask.

    Args:
        pattern: Mask pattern name
            - "causal": Lower triangular (autoregressive)
            - "full": All positions attend to all positions
            - "sliding_window": Local attention within window
            - "prefix_lm": Prefix fully attended, rest causal
        query_len: Total query sequence length
        key_len: Total key sequence length
        block_size: Number of tokens per block (default: 128)
        causal: Alias for pattern="causal"
        sliding_window: Window size for sliding_window pattern
        prefix_len: Prefix length for prefix_lm pattern

    Returns:
        BlockMask with the specified pattern

    Raises:
        ValueError: If pattern is unknown or required parameters are missing

    Example:
        ::

            from kernels import create_block_mask

            # Causal mask for autoregressive models
            causal = create_block_mask("causal", 256, 256, block_size=32)

            # Sliding window attention
            sliding = create_block_mask("sliding_window", 256, 256, sliding_window=64)

            # Prefix LM (encoder-decoder)
            prefix = create_block_mask("prefix_lm", 256, 256, prefix_len=64)
    """
```

---

### `compose_block_masks`

Compose two block masks.

```python
from kernels import compose_block_masks

def compose_block_masks(
    mask1: BlockMask,
    mask2: BlockMask,
    operation: str = "intersect",
) -> BlockMask:
    """
    Compose two block masks.

    Args:
        mask1: First mask
        mask2: Second mask (must have same shape as mask1)
        operation: Composition operation
            - "intersect": AND (only positions allowed by both)
            - "union": OR (positions allowed by either)

    Returns:
        New BlockMask with composed pattern

    Raises:
        ValueError: If masks have different shapes or operation is unknown
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

## Profiling Tools

### `GPUMemoryProfile`

GPU memory hierarchy metrics for a kernel.

```python
from utils import GPUMemoryProfile

@dataclass
class GPUMemoryProfile:
    """
    GPU memory hierarchy metrics for a kernel.

    Captures key metrics that affect kernel performance:
    - Occupancy: Higher is generally better (more latency hiding)
    - SMEM pressure: Lower leaves room for more blocks per SM
    - L2 hit rate: Higher means less HBM traffic
    """

    occupancy_pct: float  # 0-100
    smem_used_bytes: int
    smem_available_bytes: int
    reg_per_thread: int
    l2_hit_rate: float  # 0-1
    estimated_stall_cycles: int

    def smem_pressure_pct(self) -> float:
        """Calculate shared memory utilization percentage."""

    def can_increase_occupancy(self) -> bool:
        """Heuristic check if occupancy can be increased."""
```

---

### `KernelBenchmark`

Kernel performance metrics from a benchmark run.

```python
from utils import KernelBenchmark

@dataclass
class KernelBenchmark:
    """
    Kernel performance metrics from a benchmark run.

    Combines timing data with memory profile for comprehensive analysis.
    """

    kernel_name: str
    elapsed_ms: float
    tflops: float
    gbps: float
    memory_profile: GPUMemoryProfile
```

---

### `estimate_occupancy`

Estimate GPU occupancy percentage.

```python
from utils import estimate_occupancy

def estimate_occupancy(
    block_size: int,
    registers_per_thread: int,
    shared_memory_bytes: int,
    gpu_capability: int,
) -> float:
    """
    Estimate GPU occupancy percentage.

    Occupancy is the ratio of active warps to the maximum number of warps
    that can run concurrently on an SM. Higher occupancy helps hide latency.

    Args:
        block_size: Number of threads per block
        registers_per_thread: Register usage per thread
        shared_memory_bytes: Shared memory usage per block
        gpu_capability: GPU compute capability major version (80, 89, 90)

    Returns:
        Estimated occupancy percentage (0-100)

    Example:
        ::

            from utils import estimate_occupancy

            # Estimate for FlashAttention block
            occ = estimate_occupancy(
                block_size=128,
                registers_per_thread=64,
                shared_memory_bytes=16384,
                gpu_capability=80
            )
            print(f"Occupancy: {occ:.1f}%")
    """
```

---

### `get_gpu_memory_hierarchy`

Return GPU memory hierarchy specifications.

```python
from utils import get_gpu_memory_hierarchy

def get_gpu_memory_hierarchy(gpu_capability: int) -> Dict[str, int]:
    """
    Return GPU memory hierarchy specifications.

    Args:
        gpu_capability: GPU compute capability major version (80, 89, 90, etc.)

    Returns:
        Dictionary with memory hierarchy specs:
        - smem_per_block: Shared memory per block (bytes)
        - registers_per_warp: Registers available per warp
        - l1_cache_kb, l2_cache_mb: Cache sizes
        - hbm_gb: High Bandwidth Memory size
        - peak_bandwidth_gbps, peak_compute_tflops: Peak performance
        - max_warps_per_sm, max_threads_per_warp: Thread limits

    Example:
        ::

            from utils import get_gpu_memory_hierarchy

            specs = get_gpu_memory_hierarchy(80)  # Ampere
            print(f"SMEM per block: {specs['smem_per_block'] // 1024}KB")
            print(f"Peak bandwidth: {specs['peak_bandwidth_gbps']} GB/s")
    """
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

### Tensor Layout Summary

| Function | Layout |
|----------|--------|
| `flash_attention` (V1) | `(batch, heads, seq_len, head_dim)` |
| `flash_attention_v2` (V2) | `(batch, seq_len, heads, head_dim)` |

---

## Links

- [Tensor Layout Guide](./tensor-layout) - Understanding V1 vs V2 layout differences
- [中文文档](/zh/api) - Complete Chinese API documentation
- [Tutorial](./tutorial) - Step-by-step learning guide
- [Performance Guide](./performance) - Optimization tips
- [FAQ](./faq) - Common questions and solutions

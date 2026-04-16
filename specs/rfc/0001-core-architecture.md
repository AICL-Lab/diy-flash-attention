# RFC 0001: Core Architecture - DIY FlashAttention

## Status
**Accepted**

## Context

This project implements the FlashAttention algorithm using Python and OpenAI Triton. The project is divided into two main phases:

1. **Foundation Phase**: Implement Triton matrix multiplication kernel to understand Triton's programming model
2. **Advanced Phase**: Implement FlashAttention core algorithm including online softmax and tiled attention

Triton is a Python DSL for writing efficient GPU kernels that automatically handles memory tiling and coalesced access, allowing developers to focus on algorithm logic.

The project also includes complete engineering infrastructure: benchmark tools, numerical validation, GPU architecture adaptation, documentation, and example code.

## Architecture

### Project Structure

```
diy-flash-attention/
├── kernels/               # GPU Kernels
│   ├── matmul.py          # Matrix multiplication with autotune
│   ├── flash_attn.py      # FlashAttention implementation
│   └── modern_features.py # GPU capability detection
├── benchmarks/            # Performance benchmarks
├── tests/                 # Comprehensive test suite
├── utils/                 # Validation & GPU detection
├── examples/              # Usage examples
├── docs/                  # Documentation
├── specs/                 # Specification documents
└── scripts/               # Automation scripts
```

### Modern CUDA Features (Hopper/Blackwell)

CUDA 13.x and Hopper (SM90) / Blackwell (SM100) architectures introduce several important optimizations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Modern GPU Features                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TMA (Tensor Memory Accelerator)                            │
│  ├── Hardware-accelerated async data transfer               │
│  ├── Automatic 2D/3D tensor layout handling                 │
│  └── Reduced address computation overhead                   │
│                                                              │
│  Warpgroup MMA                                               │
│  ├── 4 warps collaborating on matrix multiplication         │
│  ├── Larger tile size (64x256x16)                           │
│  └── Higher compute throughput                              │
│                                                              │
│  FP8 Support                                                 │
│  ├── E4M3 and E5M2 formats                                  │
│  ├── 2x compute throughput vs FP16                          │
│  └── Suitable for inference scenarios                       │
│                                                              │
│  Async Barriers                                              │
│  ├── Software pipeline optimization                         │
│  └── Hide memory latency                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### GPU Memory Hierarchy

FlashAttention's core optimization is based on GPU memory hierarchy:

```
┌─────────────────────────────────────────┐
│              HBM (High Bandwidth Memory) │
│              ~1.5 TB/s, ~40GB            │
│                    ↕                     │
│              L2 Cache                    │
│              ~4 TB/s, ~40MB              │
│                    ↕                     │
│              SRAM (Shared Memory)        │
│              ~19 TB/s, ~192KB per SM     │
│                    ↕                     │
│              Registers                   │
│              Fastest, ~256KB per SM      │
└─────────────────────────────────────────┘
```

FlashAttention avoids writing the full N×N attention matrix to HBM by blocking computation into SRAM.

## Components and Interfaces

### 1. Matrix Multiplication Kernel (`kernels/matmul.py`)

Provides two kernel variants: `matmul_kernel` with autotune and `matmul_kernel_no_autotune` for manual tuning.

```python
import torch
import triton
import triton.language as tl

def get_autotune_configs():
    """Get autotuning configurations for different GPU architectures."""
    # Returns 7 (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M) configurations
    ...

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B with autotune.

    Tiling Strategy:
    1. Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N block of C
    2. Iterate over K dimension in BLOCK_SIZE_K chunks
    3. Use L2 cache optimization via GROUP_SIZE_M (super-grouping)
    4. Accumulate in float32, output in float16
    """
    ...

def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Triton matrix multiplication: C = A @ B

    Args:
        a: Input matrix A of shape (M, K), dtype float16/float32/bfloat16
        b: Input matrix B of shape (K, N), dtype float16/float32/bfloat16
        block_m, block_n, block_k: Block sizes (None = use autotune)
        use_autotune: Whether to use autotuning (ignored if block sizes provided)

    Returns:
        Output matrix C of shape (M, N), dtype float16

    Raises:
        ValueError: If matrix dimensions are incompatible or invalid block sizes
        TypeError: If input dtypes are not supported (supported: float16, float32, bfloat16)
    """
    ...
```

### 2. FlashAttention Kernel (`kernels/flash_attn.py`)

Supports 4D (batch, heads, seq_len, head_dim) and 3D (batch*heads, seq_len, head_dim) input, with per-batch variable length sequences.

```python
@triton.jit
def _flash_attention_forward_kernel(
    # Input/output pointers
    Q, K, V, Out,
    # Softmax statistics (log-sum-exp for potential backward pass)
    L,
    # Per-batch sequence lengths
    SEQ_LENS,
    # Strides
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lz, stride_lh, stride_lm,
    # Softmax scale
    SM_SCALE,
    # Compile-time constants
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_CTX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention forward pass using online softmax.

    Grid: (num_m_blocks, batch * heads)

    Algorithm:
    1. Load Q block into SRAM
    2. For each K, V block:
       a. Compute S = Q @ K^T * scale
       b. Apply causal mask if needed
       c. Update running max (m_i) and sum (l_i) for online softmax
       d. Accumulate attention output incrementally (defer normalization)
    3. Final normalization: acc / l_i
    4. Store log-sum-exp (m_i + log(l_i)) for potential backward pass
    5. Write final output to HBM
    """
    ...

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention: Memory-efficient attention computation.

    Args:
        q: Query tensor, shape (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
        k: Key tensor, same shape as q
        v: Value tensor, same shape as q
        causal: Whether to apply causal masking (for autoregressive models)
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        seq_lens: Per-batch sequence lengths, shape [batch] (optional)

    Returns:
        Attention output of same shape as input

    Raises:
        ValueError: If tensor shapes are incompatible or head_dim not in {32, 64}
    """
    ...
```

### 3. GPU Detection (`utils/gpu_detect.py`)

```python
from dataclasses import dataclass
from enum import Enum

class GPUArch(Enum):
    """Supported GPU architectures."""
    VOLTA = "sm_70"       # V100
    TURING = "sm_75"      # RTX 20xx
    AMPERE = "sm_80"      # A100, RTX 30xx
    ADA = "sm_89"         # RTX 40xx
    HOPPER = "sm_90"      # H100
    BLACKWELL = "sm_100"  # B100/B200
    UNKNOWN = "unknown"

@dataclass
class GPUCapabilities:
    """GPU capabilities detection result."""
    name: str
    arch: GPUArch
    compute_capability: tuple[int, int]
    has_tma: bool           # Tensor Memory Accelerator (Hopper+)
    has_fp8: bool           # FP8 support (Hopper+)
    has_warpgroup_mma: bool # Warpgroup MMA (Hopper+)
    sram_per_sm: int        # Shared memory per SM in bytes
    num_sms: int            # Number of streaming multiprocessors
    total_memory_gb: float  # Total GPU memory in GB

def detect_gpu(device_id: int = 0) -> GPUCapabilities:
    """Detect current GPU capabilities."""
    ...

def get_optimal_config(caps: GPUCapabilities, operation: str) -> dict:
    """Get optimal kernel configuration for detected GPU."""
    ...
```

### 4. Modern Features (`kernels/modern_features.py`)

Architecture-adaptive kernel selection and modern GPU feature support (TMA, FP8, Warpgroup MMA).

```python
class AdaptiveKernelSelector:
    """
    Selects optimal kernel implementation based on GPU architecture.
    Lazy initialization; provides automatic fallback on older GPUs.
    """
    def get_matmul_config(self) -> Dict[str, Any]: ...
    def get_attention_config(self) -> Dict[str, Any]: ...
    def select_matmul_kernel(self) -> Callable: ...
    def select_attention_kernel(self) -> Callable: ...
```

## Data Models

### Tensor Specifications

```python
# Matrix Multiplication
A: torch.Tensor  # Shape: (M, K), dtype: float16/float32/bfloat16
B: torch.Tensor  # Shape: (K, N), dtype: float16/float32/bfloat16
C: torch.Tensor  # Shape: (M, N), dtype: float16 (accumulation in float32)

# FlashAttention (4D input, internally reshaped to 3D)
Q: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
K: torch.Tensor  # Same shape as Q, dtype: float16
V: torch.Tensor  # Same shape as Q, dtype: float16
Out: torch.Tensor  # Same shape as input Q, dtype: float16

# Softmax statistics (kernel internal, stored for potential backward pass)
L: torch.Tensor  # Shape: (batch*heads, seq_len), dtype: float32 - log-sum-exp

# Per-batch sequence lengths (optional)
seq_lens: torch.Tensor  # Shape: (batch,), dtype: int32
```

### Autotune Configurations (matmul)

```python
# 7 autotune configs, keyed by [M, N, K]
AUTOTUNE_CONFIGS = [
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_SIZE_M": 32,  "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 4, "num_warps": 4},
]
```

### FlashAttention Block Size Selection

```python
# Block sizes are dynamically chosen based on head_dim
if head_dim <= 32:
    BLOCK_M, BLOCK_N = 64, 32
else:  # head_dim == 64
    BLOCK_M, BLOCK_N = 128, 64
```

## Algorithms

### Online Softmax Algorithm

FlashAttention's core is online softmax, enabling block-wise softmax computation without storing the full attention matrix:

```
Algorithm: Online Softmax for Tiled Attention

Input: Q block (BLOCK_M × d), K blocks, V blocks
Output: O (attention output)

Initialize:
  m_i = -inf  (running max)
  l_i = 0     (running sum)
  O_i = 0     (running output)

For each K, V block j:
  1. S_ij = Q_i @ K_j^T * scale
  2. m_ij = max(S_ij)
  3. m_new = max(m_i, m_ij)
  4. l_new = exp(m_i - m_new) * l_i + sum(exp(S_ij - m_new))
  5. O_new = (l_i * exp(m_i - m_new) * O_i + exp(S_ij - m_new) @ V_j) / l_new
  6. Update: m_i = m_new, l_i = l_new, O_i = O_new

Return: O_i
```

## Correctness Properties

### Property 1: Matrix Multiplication Correctness

For any matrices A of shape (M, K) and B of shape (K, N) with valid floating-point values, the Triton matmul kernel output C should equal torch.matmul(A, B) within relative tolerance of 1e-3.

**Validates: Requirements 1.1, 1.2, 6.1**

### Property 2: Block Size Invariance

For any valid block size configuration (BLOCK_M, BLOCK_N, BLOCK_K) and any input matrices A, B, the Triton matmul kernel should produce the same result (within numerical tolerance) regardless of block size choice.

**Validates: Requirements 1.4**

### Property 3: FlashAttention Correctness

For any query Q, key K, and value V tensors of compatible shapes, the FlashAttention kernel output should equal the reference attention computation `softmax(Q @ K^T / sqrt(d)) @ V` within relative tolerance of 1e-3.

**Validates: Requirements 4.1, 4.4, 6.1**

### Property 4: Causal Masking Correctness

For any Q, K, V tensors, when causal masking is enabled, the FlashAttention output should match the reference implementation with causal mask applied (upper triangular positions set to -inf before softmax).

**Validates: Requirements 4.3**

### Property 5: Memory Scaling

For any sequence length N, FlashAttention memory usage should scale as O(N) rather than O(N²), meaning doubling the sequence length should approximately double (not quadruple) memory usage.

**Validates: Requirements 5.4**

## Error Handling

### Matrix Multiplication Errors

| Error Condition | Handling |
|----------------|----------|
| Non-2D tensors | Raise `ValueError` with actual dim info |
| Incompatible matrix dimensions (A.shape[1] != B.shape[0]) | Raise `ValueError` with descriptive message |
| Invalid block size (≤ 0 or > matrix dimension) | Raise `ValueError` with valid range |
| Manual block sizes required but not all provided | Raise `ValueError` |
| Non-contiguous tensors | Silently convert to contiguous |
| Unsupported dtype (not float16/float32/bfloat16) | Raise `TypeError` listing supported dtypes |
| Supported but non-float16 dtype | Silently convert to float16 |

### FlashAttention Errors

| Error Condition | Handling |
|----------------|----------|
| Non-3D/4D tensors | Raise `ValueError` with actual dim |
| Q, K, V shape mismatch | Raise `ValueError` with expected shapes |
| head_dim not in {32, 64} | Raise `ValueError` with supported values |
| seq_lens dim != 1 | Raise `ValueError` |
| seq_lens length != batch size | Raise `ValueError` |
| seq_lens values ≤ 0 or > seq_len | Raise `ValueError` |
| Non-float16 dtype | Silently convert to float16 |

### GPU Detection Errors

| Error Condition | Handling |
|----------------|----------|
| CUDA not available | Raise `RuntimeError` |
| Unknown GPU architecture | Return `GPUArch.UNKNOWN` with conservative config |
| Modern features unavailable | Fallback to standard implementations |

## Testing Strategy

### Test File Structure

| Test File | Responsibility |
|-----------|----------------|
| `tests/test_matmul.py` | Matrix multiplication unit tests |
| `tests/test_flash.py` | FlashAttention unit tests |
| `tests/test_properties.py` | Hypothesis property-based tests (5 Properties) |
| `tests/test_benchmark.py` | Benchmark tool unit tests |
| `tests/test_validation.py` | Validation tool unit tests |
| `tests/test_error_handling.py` | Error handling unit tests |
| `tests/conftest.py` | Shared fixtures and skip conditions |

### Dual Testing Approach

This project uses a combination of unit tests and property-based tests:

1. **Unit Tests**
   - Test specific examples and edge cases
   - Verify error handling logic
   - Test API interface correctness

2. **Property-Based Tests**
   - Use Hypothesis library to generate random inputs
   - Verify universal properties hold across all inputs
   - Each property test runs at least 100 iterations

### Test Annotation Format

Each property test must include the following annotation:

```python
# Feature: diy-flash-attention, Property 1: Matrix Multiplication Correctness
# Validates: Requirements 1.1, 1.2, 6.1
```

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

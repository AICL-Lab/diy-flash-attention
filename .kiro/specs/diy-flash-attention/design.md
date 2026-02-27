# Design Document: DIY FlashAttention

## Overview

本项目使用 Python 和 OpenAI Triton 实现 FlashAttention 算法。项目分为两个主要阶段：

1. **基础阶段**：实现 Triton 矩阵乘法 Kernel，理解 Triton 编程模型
2. **进阶阶段**：实现 FlashAttention 核心算法，包括 online softmax 和 tiled attention

Triton 是一个用于编写高效 GPU kernel 的 Python DSL，它自动处理内存分块（Tiling）和合并访问（Coalescing），让开发者专注于算法逻辑。

项目还包含完整的工程化基础设施：benchmark 工具、数值验证、GPU 架构自适应、文档和示例代码。

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Project Structure                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌───────────────────────┐    ┌───────────────────────┐           │
│  │   kernels/            │    │   benchmarks/         │           │
│  │                       │    │                       │           │
│  │  matmul.py            │    │  bench_matmul.py      │           │
│  │  flash_attn.py        │    │  bench_flash.py       │           │
│  │  modern_features.py   │    │                       │           │
│  └───────────────────────┘    └───────────────────────┘           │
│                                                                    │
│  ┌───────────────────────┐    ┌───────────────────────┐           │
│  │   tests/              │    │   utils/              │           │
│  │                       │    │                       │           │
│  │  test_matmul.py       │    │  benchmark.py         │           │
│  │  test_flash.py        │    │  validation.py        │           │
│  │  test_properties.py   │    │  gpu_detect.py        │           │
│  │  test_benchmark.py    │    │                       │           │
│  │  test_validation.py   │    └───────────────────────┘           │
│  │  test_error_handling  │                                        │
│  │  conftest.py          │    ┌───────────────────────┐           │
│  └───────────────────────┘    │   docs/               │           │
│                                │                       │           │
│  ┌───────────────────────┐    │  api.md               │           │
│  │   examples/           │    │  tutorial.md          │           │
│  │                       │    │  cheatsheet.md        │           │
│  │  quick_start.py       │    │  performance.md       │           │
│  │  advanced_usage.py    │    │  faq.md               │           │
│  │  block_size_exp.py    │    └───────────────────────┘           │
│  │  visualize_tiling.py  │                                        │
│  └───────────────────────┘    ┌───────────────────────┐           │
│                                │   scripts/            │           │
│                                │  run_all_benchmarks   │           │
│                                └───────────────────────┘           │
│                                                                    │
│  .gitignore   README.md      pyproject.toml   Makefile            │
│  LICENSE      CONTRIBUTING.md CHANGELOG.md     .github/           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Modern CUDA Features (Hopper/Blackwell)

CUDA 13.x 和 Hopper (SM90) / Blackwell (SM100) 架构引入了多项重要优化：

```
┌─────────────────────────────────────────────────────────────┐
│                    Modern GPU Features                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TMA (Tensor Memory Accelerator)                            │
│  ├── 硬件加速的异步数据传输                                   │
│  ├── 自动处理 2D/3D tensor 布局                              │
│  └── 减少 address 计算开销                                   │
│                                                              │
│  Warpgroup MMA                                               │
│  ├── 4 个 warp 协同执行矩阵乘法                              │
│  ├── 更大的 tile size (64x256x16)                           │
│  └── 更高的计算吞吐量                                        │
│                                                              │
│  FP8 Support                                                 │
│  ├── E4M3 和 E5M2 格式                                       │
│  ├── 2x 计算吞吐量 vs FP16                                   │
│  └── 适用于推理场景                                          │
│                                                              │
│  Async Barriers                                              │
│  ├── 软件流水线优化                                          │
│  └── 隐藏内存延迟                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### GPU Memory Hierarchy

FlashAttention 的核心优化基于 GPU 内存层次结构：

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

FlashAttention 通过将计算分块到 SRAM 中，避免将完整的 N×N attention matrix 写入 HBM。

## Components and Interfaces

### 1. Matrix Multiplication Kernel (`kernels/matmul.py`)

提供两个 kernel 变体：带 autotune 的 `matmul_kernel` 和用于手动调参的 `matmul_kernel_no_autotune`。

```python
import torch
import triton
import triton.language as tl

def get_autotune_configs():
    """Get autotuning configurations for different GPU architectures."""
    # 返回 7 组 (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M) 配置
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

@triton.jit
def matmul_kernel_no_autotune(...):
    """Same as matmul_kernel but without autotuning. Used for manual block size experimentation."""
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

def triton_matmul_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton matmul with float32 output. Useful for validation against PyTorch."""
    ...
```

### 2. FlashAttention Kernel (`kernels/flash_attn.py`)

支持 4D (batch, heads, seq_len, head_dim) 和 3D (batch*heads, seq_len, head_dim) 输入，支持 per-batch 变长序列。

```python
@triton.jit
def _flash_attention_forward_kernel(
    # Input/output pointers
    Q, K, V, Out,
    # Softmax statistics (log-sum-exp for potential backward pass)
    L,
    # Per-batch sequence lengths
    SEQ_LENS,
    # Strides (batch*head 维度合并后的 4 组 stride)
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

def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Reference O(N²) attention for correctness validation."""
    ...
```

### 3. Benchmark Utilities (`utils/benchmark.py`)

```python
@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""
    name: str
    size: tuple  # (M, N, K) for matmul or (batch, heads, seq_len, head_dim) for attention
    time_ms: float
    tflops: float
    memory_mb: float = 0.0
    block_config: Optional[dict] = None

def calculate_matmul_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPs for matrix multiplication: 2 * M * N * K."""
    ...

def calculate_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Calculate FLOPs for attention: Q@K^T + softmax + attn@V."""
    ...

def benchmark_fn(fn, *args, warmup=25, rep=100, **kwargs) -> tuple[float, float, float]:
    """Benchmark a function using triton.testing.do_bench. Returns (median_ms, min_ms, max_ms)."""
    ...

class BenchmarkRunner:
    """Runs and compares benchmarks between implementations."""
    
    def __init__(self, device="cuda", warmup=25, rep=100): ...
    
    def benchmark_matmul(
        self,
        triton_fn: Callable,
        sizes: list[tuple[int, int, int]],
        block_configs: Optional[list[dict]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """Benchmark Triton matmul vs torch.matmul, optionally with specific block configs."""
        ...
    
    def benchmark_attention(
        self,
        flash_fn: Callable,
        seq_lengths: list[int],
        batch_size: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        causal: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """Benchmark FlashAttention vs PyTorch SDPA, including memory measurement."""
        ...
    
    def print_comparison_table(self, results=None, title="Benchmark Results") -> None:
        """Print formatted comparison table with speedup ratios."""
        ...
    
    def clear_results(self) -> None:
        """Clear stored results."""
        ...
```

### 4. Validation Utilities (`utils/validation.py`)

```python
def validate_matmul(
    triton_fn: Callable,
    m: int, n: int, k: int,
    rtol: float = 1e-3, atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda", verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Validate Triton matmul against torch.matmul (float32 reference).
    Returns: (is_valid, max_diff)
    """
    ...

def validate_attention(
    flash_fn: Callable,
    batch: int, heads: int, seq_len: int, head_dim: int,
    causal: bool = False,
    rtol: float = 1e-3, atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda", verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Validate FlashAttention against PyTorch's scaled_dot_product_attention.
    Returns: (is_valid, max_diff)
    """
    ...

def validate_matmul_edge_cases(
    triton_fn: Callable, rtol=1e-3, atol=1e-3, device="cuda", verbose=True,
) -> Tuple[bool, dict]:
    """
    Validate matmul edge cases: zero matrices, non-power-of-2 dims,
    very small matrices, rectangular matrices.
    Returns: (all_passed, results_dict)
    """
    ...

def validate_attention_edge_cases(
    flash_fn: Callable, rtol=1e-3, atol=1e-3, device="cuda", verbose=True,
) -> Tuple[bool, dict]:
    """
    Validate attention edge cases: different seq lengths, causal/non-causal,
    different head dimensions (32, 64).
    Returns: (all_passed, results_dict)
    """
    ...
```

### 5. GPU Detection Utilities (`utils/gpu_detect.py`)

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
    """
    Detect current GPU capabilities.
    
    Returns:
        GPUCapabilities with detected features
    Raises:
        RuntimeError: If CUDA is not available
    """
    ...

def get_optimal_config(caps: GPUCapabilities, operation: str) -> dict:
    """
    Get optimal kernel configuration for detected GPU.
    
    Args:
        caps: GPU capabilities
        operation: "matmul" or "flash_attention"
        
    Returns:
        Optimal block sizes and other parameters
    Raises:
        ValueError: If operation is unknown
    """
    ...

def print_gpu_info(caps: Optional[GPUCapabilities] = None) -> None:
    """Print GPU information in a formatted way."""
    ...
```

### 6. Modern Features (`kernels/modern_features.py`)

架构自适应 kernel 选择和现代 GPU 特性支持（TMA、FP8、Warpgroup MMA）。

```python
def check_hopper_features() -> Dict[str, bool]:
    """Check availability of Hopper+ features. Returns feature flags dict."""
    ...

def supports_fp8() -> bool:
    """Check if current GPU supports FP8."""
    ...

def to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Convert to FP8 E4M3 format. Fallback to float16 if unsupported."""
    ...

def to_fp8_e5m2(tensor: torch.Tensor) -> torch.Tensor:
    """Convert to FP8 E5M2 format. Fallback to float16 if unsupported."""
    ...

class AdaptiveKernelSelector:
    """
    Selects optimal kernel implementation based on GPU architecture.
    Lazy initialization; provides automatic fallback on older GPUs.
    """
    def get_matmul_config(self) -> Dict[str, Any]: ...
    def get_attention_config(self) -> Dict[str, Any]: ...
    def select_matmul_kernel(self) -> Callable: ...
    def select_attention_kernel(self) -> Callable: ...

# Convenience functions (use global AdaptiveKernelSelector instance)
def get_optimal_matmul() -> Callable: ...
def get_optimal_attention() -> Callable: ...
def get_matmul_config() -> Dict[str, Any]: ...
def get_attention_config() -> Dict[str, Any]: ...

def create_tma_descriptor(tensor, block_shape) -> Optional[Any]:
    """Create TMA descriptor (placeholder, requires Hopper+ and Triton 3.0+)."""
    ...

def print_feature_status() -> None:
    """Print status of modern CUDA features."""
    ...
```

## Data Models

### Tensor Specifications

```python
# Matrix Multiplication
A: torch.Tensor  # Shape: (M, K), dtype: float16/float32/bfloat16 (internally converted to float16)
B: torch.Tensor  # Shape: (K, N), dtype: float16/float32/bfloat16
C: torch.Tensor  # Shape: (M, N), dtype: float16 (accumulation in float32)

# FlashAttention (4D input, internally reshaped to 3D)
Q: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim) or (batch*heads, seq_len, head_dim)
K: torch.Tensor  # Same shape as Q, dtype: float16
V: torch.Tensor  # Same shape as Q, dtype: float16
Out: torch.Tensor  # Same shape as input Q, dtype: float16

# Softmax statistics (kernel internal, stored for potential backward pass)
L: torch.Tensor  # Shape: (batch*heads, seq_len), dtype: float32 - log-sum-exp (m_i + log(l_i))

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

### Architecture-Adaptive Configurations

```python
# AdaptiveKernelSelector returns different configs per GPU arch:
# Hopper/Blackwell: larger blocks, more stages/warps, TMA enabled (if available)
# Ampere/Ada:       same block sizes, fewer stages, no TMA
# Older GPUs:       smaller blocks, conservative settings
```

### Online Softmax Algorithm

FlashAttention 的核心是 online softmax，允许分块计算 softmax 而无需存储完整的 attention matrix：

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

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Matrix Multiplication Correctness

*For any* matrices A of shape (M, K) and B of shape (K, N) with valid floating-point values, the Triton matmul kernel output C should equal torch.matmul(A, B) within relative tolerance of 1e-3.

**Validates: Requirements 1.1, 1.2, 6.1**

### Property 2: Block Size Invariance

*For any* valid block size configuration (BLOCK_M, BLOCK_N, BLOCK_K) and any input matrices A, B, the Triton matmul kernel should produce the same result (within numerical tolerance) regardless of block size choice.

**Validates: Requirements 1.4**

### Property 3: FlashAttention Correctness

*For any* query Q, key K, and value V tensors of compatible shapes, the FlashAttention kernel output should equal the reference attention computation `softmax(Q @ K^T / sqrt(d)) @ V` within relative tolerance of 1e-3.

**Validates: Requirements 4.1, 4.4, 6.1**

### Property 4: Causal Masking Correctness

*For any* Q, K, V tensors, when causal masking is enabled, the FlashAttention output should match the reference implementation with causal mask applied (upper triangular positions set to -inf before softmax).

**Validates: Requirements 4.3**

### Property 5: Memory Scaling

*For any* sequence length N, FlashAttention memory usage should scale as O(N) rather than O(N²), meaning doubling the sequence length should approximately double (not quadruple) memory usage.

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

### Validation Errors

| Error Condition | Handling |
|----------------|----------|
| Numerical mismatch exceeds tolerance | Return `(False, max_diff)` with optional verbose output |

## Testing Strategy

### Test File Structure

| 测试文件 | 职责 |
|---------|------|
| `tests/test_matmul.py` | 矩阵乘法单元测试 |
| `tests/test_flash.py` | FlashAttention 单元测试 |
| `tests/test_properties.py` | Hypothesis 属性测试（5 个 Property） |
| `tests/test_benchmark.py` | Benchmark 工具单元测试 |
| `tests/test_validation.py` | 验证工具单元测试 |
| `tests/test_error_handling.py` | 错误处理单元测试 |
| `tests/conftest.py` | 共享 fixtures 和 skip 条件 |

### Dual Testing Approach

本项目采用单元测试和属性测试相结合的策略：

1. **单元测试 (Unit Tests)**
   - 测试特定示例和边界情况
   - 验证错误处理逻辑
   - 测试 API 接口正确性

2. **属性测试 (Property-Based Tests)**
   - 使用 Hypothesis 库生成随机输入
   - 验证普遍性质在所有输入上成立
   - 每个属性测试运行至少 100 次迭代

### Property-Based Testing Configuration

```python
# 使用 Hypothesis 进行属性测试
from hypothesis import given, strategies as st, settings

# 矩阵维度策略
matrix_dims = st.integers(min_value=16, max_value=1024)
block_sizes = st.sampled_from([32, 64, 128, 256])

# 测试配置
@settings(max_examples=100, deadline=None)
```

### Test Annotation Format

每个属性测试必须包含以下注释：

```python
# Feature: diy-flash-attention, Property 1: Matrix Multiplication Correctness
# Validates: Requirements 1.1, 1.2, 6.1
```

### Test Coverage Matrix

| Property | Unit Tests | Property Tests | 对应文件 |
|----------|------------|----------------|---------|
| Matmul Correctness | Specific sizes, edge cases | Random M, N, K | test_matmul, test_properties |
| Block Size Invariance | Fixed configs | Random block sizes | test_matmul, test_properties |
| FlashAttention Correctness | Standard shapes | Random batch, seq, heads | test_flash, test_properties |
| Causal Masking | Known patterns | Random sequences | test_flash, test_properties |
| Memory Scaling | Fixed sequence lengths | N/A (benchmark) | test_properties |
| Error Handling | Invalid inputs, dtypes | N/A | test_error_handling |
| Benchmark Utils | TFLOPS calc, output format | N/A | test_benchmark |
| Validation Utils | Edge cases, tolerances | N/A | test_validation |

### Edge Cases to Test

- Zero matrices
- Very small matrices (1×1, 2×2, ..., 16×16, smaller than block size)
- Non-power-of-2 dimensions (33×47×61, 100×200×150)
- Rectangular matrices
- Different head dimensions (32, 64)
- Variable sequence lengths (per-batch)
- Causal vs non-causal attention

# Design Document: DIY FlashAttention

## Overview

本项目使用 Python 和 OpenAI Triton 实现 FlashAttention 算法。项目分为两个主要阶段：

1. **基础阶段**：实现 Triton 矩阵乘法 Kernel，理解 Triton 编程模型
2. **进阶阶段**：实现 FlashAttention 核心算法，包括 online softmax 和 tiled attention

Triton 是一个用于编写高效 GPU kernel 的 Python DSL，它自动处理内存分块（Tiling）和合并访问（Coalescing），让开发者专注于算法逻辑。

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Project Structure                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │   kernels/       │    │   benchmarks/    │               │
│  │                  │    │                  │               │
│  │  matmul.py       │    │  bench_matmul.py │               │
│  │  flash_attn.py   │    │  bench_flash.py  │               │
│  │                  │    │                  │               │
│  └──────────────────┘    └──────────────────┘               │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │   tests/         │    │   utils/         │               │
│  │                  │    │                  │               │
│  │  test_matmul.py  │    │  benchmark.py    │               │
│  │  test_flash.py   │    │  validation.py   │               │
│  │                  │    │  gpu_detect.py   │               │
│  └──────────────────┘    └──────────────────┘               │
│                                                              │
│  .gitignore              README.md                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
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

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (meta-parameters)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B
    
    Tiling Strategy:
    - Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N block of C
    - Iterate over K dimension in BLOCK_SIZE_K chunks
    - Use L2 cache optimization via GROUP_SIZE_M
    """
    pass

def triton_matmul(a: torch.Tensor, b: torch.Tensor, 
                  block_m: int = 128, block_n: int = 256, block_k: int = 64) -> torch.Tensor:
    """
    Wrapper function for Triton matrix multiplication.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        block_m, block_n, block_k: Block size parameters
        
    Returns:
        Output matrix C of shape (M, N)
    """
    pass
```

### 2. FlashAttention Kernel (`kernels/flash_attn.py`)

```python
@triton.jit
def flash_attention_forward_kernel(
    # Input pointers
    Q, K, V, 
    # Output pointer
    Out,
    # Softmax statistics for backward pass
    L, M_ptr,
    # Dimensions
    seq_len, head_dim,
    # Strides
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    # Scaling factor
    sm_scale,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Causal mask flag
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention forward pass using online softmax.
    
    Algorithm:
    1. Load Q block into SRAM
    2. For each K, V block:
       a. Compute S = Q @ K^T * scale
       b. Apply causal mask if needed
       c. Update running max and sum for online softmax
       d. Compute attention output incrementally
    3. Write final output to HBM
    """
    pass

def flash_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None
) -> torch.Tensor:
    """
    FlashAttention wrapper function.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal masking
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        
    Returns:
        Attention output of shape (batch, heads, seq_len, head_dim)
    """
    pass
```

### 3. Benchmark Utilities (`utils/benchmark.py`)

```python
@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""
    name: str
    matrix_size: tuple
    time_ms: float
    tflops: float
    memory_mb: float = 0.0

class BenchmarkRunner:
    """Runs and compares benchmarks between implementations."""
    
    def benchmark_matmul(
        self, 
        sizes: list[tuple[int, int, int]],
        block_configs: list[dict] = None
    ) -> list[BenchmarkResult]:
        """Benchmark matrix multiplication implementations."""
        pass
    
    def benchmark_attention(
        self,
        seq_lengths: list[int],
        head_dim: int = 64,
        num_heads: int = 8
    ) -> list[BenchmarkResult]:
        """Benchmark attention implementations."""
        pass
    
    def print_comparison_table(self, results: list[BenchmarkResult]) -> None:
        """Print formatted comparison table."""
        pass
```

### 4. Validation Utilities (`utils/validation.py`)

```python
def validate_matmul(
    triton_fn: Callable,
    m: int, n: int, k: int,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> tuple[bool, float]:
    """
    Validate Triton matmul against torch.matmul.
    
    Returns:
        (is_valid, max_diff)
    """
    pass

def validate_attention(
    flash_fn: Callable,
    batch: int, heads: int, seq_len: int, head_dim: int,
    causal: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> tuple[bool, float]:
    """
    Validate FlashAttention against PyTorch's scaled_dot_product_attention.
    
    Returns:
        (is_valid, max_diff)
    """
    pass
```

### 5. GPU Detection Utilities (`utils/gpu_detect.py`)

```python
from dataclasses import dataclass
from enum import Enum

class GPUArch(Enum):
    """Supported GPU architectures."""
    AMPERE = "sm_80"      # A100
    HOPPER = "sm_90"      # H100
    BLACKWELL = "sm_100"  # B100/B200
    UNKNOWN = "unknown"

@dataclass
class GPUCapabilities:
    """GPU capabilities detection result."""
    arch: GPUArch
    compute_capability: tuple[int, int]
    has_tma: bool           # Tensor Memory Accelerator
    has_fp8: bool           # FP8 support
    has_warpgroup_mma: bool # Warpgroup MMA
    sram_per_sm: int        # Shared memory per SM in bytes

def detect_gpu() -> GPUCapabilities:
    """
    Detect current GPU capabilities.
    
    Returns:
        GPUCapabilities with detected features
    """
    pass

def get_optimal_config(caps: GPUCapabilities, operation: str) -> dict:
    """
    Get optimal kernel configuration for detected GPU.
    
    Args:
        caps: GPU capabilities
        operation: "matmul" or "flash_attention"
        
    Returns:
        Optimal block sizes and other parameters
    """
    pass
```

## Data Models

### Tensor Specifications

```python
# Matrix Multiplication
A: torch.Tensor  # Shape: (M, K), dtype: float16/float32
B: torch.Tensor  # Shape: (K, N), dtype: float16/float32
C: torch.Tensor  # Shape: (M, N), dtype: float16/float32

# FlashAttention
Q: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim), dtype: float16
K: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim), dtype: float16
V: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim), dtype: float16
Out: torch.Tensor  # Shape: (batch, heads, seq_len, head_dim), dtype: float16

# Softmax statistics (for numerical stability)
L: torch.Tensor  # Shape: (batch, heads, seq_len), dtype: float32 - log-sum-exp
M: torch.Tensor  # Shape: (batch, heads, seq_len), dtype: float32 - running max
```

### Block Size Configurations

```python
# Recommended block sizes for different GPU architectures
MATMUL_CONFIGS = {
    "default": {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
    "small": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
    "large": {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
}

FLASH_ATTN_CONFIGS = {
    "default": {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 64},
    "small_seq": {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 64},
    "large_seq": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 64},
}
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
| Incompatible matrix dimensions (A.shape[1] != B.shape[0]) | Raise ValueError with descriptive message |
| Invalid block size (≤ 0 or > matrix dimension) | Raise ValueError with valid range |
| Non-contiguous tensors | Convert to contiguous or raise warning |
| Unsupported dtype | Raise TypeError listing supported dtypes |

### FlashAttention Errors

| Error Condition | Handling |
|----------------|----------|
| Q, K, V shape mismatch | Raise ValueError with expected shapes |
| head_dim not supported by block size | Raise ValueError with supported dimensions |
| CUDA out of memory | Raise RuntimeError with memory estimate |
| Invalid causal mask configuration | Raise ValueError |

### Validation Errors

| Error Condition | Handling |
|----------------|----------|
| Numerical mismatch exceeds tolerance | Return (False, max_diff) with detailed info |
| Reference implementation unavailable | Skip validation with warning |

## Testing Strategy

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

| Property | Unit Tests | Property Tests |
|----------|------------|----------------|
| Matmul Correctness | Specific sizes, edge cases | Random M, N, K |
| Block Size Invariance | Fixed configs | Random block sizes |
| FlashAttention Correctness | Standard shapes | Random batch, seq, heads |
| Causal Masking | Known patterns | Random sequences |
| Memory Scaling | Fixed sequence lengths | N/A (benchmark) |

### Edge Cases to Test

- Zero matrices
- Identity matrices
- Very small matrices (smaller than block size)
- Non-power-of-2 dimensions
- Single element matrices
- Maximum supported dimensions

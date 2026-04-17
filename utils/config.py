"""
Configuration constants for DIY FlashAttention.

Centralizes all hardcoded values for easy tuning and consistency.
"""

from __future__ import annotations

# =============================================================================
# Matmul Kernel Configuration
# =============================================================================

#: Group size for L2 cache optimization in matmul kernel
MATMUL_GROUP_SIZE_M: int = 8

#: Default block size for M dimension in matmul
MATMUL_DEFAULT_BLOCK_M: int = 128

#: Default block size for N dimension in matmul
MATMUL_DEFAULT_BLOCK_N: int = 256

#: Default block size for K dimension in matmul
MATMUL_DEFAULT_BLOCK_K: int = 64

# =============================================================================
# FlashAttention Kernel Configuration
# =============================================================================

#: Default block size for M (query sequence) dimension in FlashAttention
FLASH_ATTN_DEFAULT_BLOCK_M: int = 128

#: Default block size for N (key/value sequence) dimension in FlashAttention
FLASH_ATTN_DEFAULT_BLOCK_N: int = 64

# =============================================================================
# Benchmark Configuration
# =============================================================================

#: Number of warmup iterations before timing
BENCHMARK_WARMUP: int = 25

#: Number of repetitions for timing
BENCHMARK_REPETITIONS: int = 100

#: Default quantiles for benchmark statistics [median, lower, upper]
BENCHMARK_QUANTILES: list[float] = [0.5, 0.2, 0.8]

#: Bytes per megabyte (consistent unit across codebase)
BYTES_PER_MB: int = 1024**2

#: Bytes per gigabyte
BYTES_PER_GB: int = 1024**3

# =============================================================================
# GPU Detection Configuration
# =============================================================================

#: Minimum compute capability for FP8 support (Hopper)
FP8_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (9, 0)

#: Minimum compute capability for TMA support (Hopper)
TMA_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (9, 0)

# =============================================================================
# Validation Configuration
# =============================================================================

#: Default relative tolerance for validation comparisons
DEFAULT_RTOL: float = 1e-3

#: Default absolute tolerance for validation comparisons
DEFAULT_ATOL: float = 1e-3

#: Pass/fail tolerance threshold for kernel smoke tests
SMOKE_TEST_RTOL: float = 1e-2

#: Default random seed for reproducible test data
DEFAULT_SEED: int = 42

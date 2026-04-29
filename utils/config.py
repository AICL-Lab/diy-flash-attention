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

# =============================================================================
# Validation Configuration
# =============================================================================

#: Default relative tolerance for validation comparisons
DEFAULT_RTOL: float = 1e-3

#: Default absolute tolerance for validation comparisons
DEFAULT_ATOL: float = 1e-3

#: Default random seed for reproducible test data
DEFAULT_SEED: int = 42

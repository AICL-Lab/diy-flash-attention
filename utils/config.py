"""
Configuration constants for DIY FlashAttention.

This module centralizes all hardcoded configuration values for easy tuning
and consistency across the codebase.
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

# =============================================================================
# GPU Detection Configuration
# =============================================================================

#: Minimum compute capability for FP8 support (Hopper)
FP8_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (9, 0)

#: Minimum compute capability for TMA support (Hopper)
TMA_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (9, 0)

# =============================================================================
# Logging Configuration
# =============================================================================

#: Default log level
LOG_LEVEL: str = "INFO"

#: Log format string
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#: Log date format
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

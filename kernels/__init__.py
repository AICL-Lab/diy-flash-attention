"""Triton GPU kernels for matrix multiplication and FlashAttention."""

from .matmul import triton_matmul
from .flash_attn import flash_attention
from .modern_features import (
    get_optimal_matmul,
    get_optimal_attention,
    get_matmul_config,
    get_attention_config,
    check_hopper_features,
    supports_fp8,
    AdaptiveKernelSelector,
)

__all__ = [
    "triton_matmul",
    "flash_attention",
    "get_optimal_matmul",
    "get_optimal_attention",
    "get_matmul_config",
    "get_attention_config",
    "check_hopper_features",
    "supports_fp8",
    "AdaptiveKernelSelector",
]

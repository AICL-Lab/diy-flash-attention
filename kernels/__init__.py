"""Triton GPU kernels for matrix multiplication and FlashAttention."""

from .flash_attn import flash_attention
from .flash_attn_v2 import flash_attention_v2
from .matmul import triton_matmul
from .modern_features import (
    AdaptiveKernelSelector,
    check_hopper_features,
    get_attention_config,
    get_matmul_config,
    get_optimal_attention,
    get_optimal_matmul,
    supports_fp8,
)

__all__ = [
    "triton_matmul",
    "flash_attention",
    "flash_attention_v2",
    "get_optimal_matmul",
    "get_optimal_attention",
    "get_matmul_config",
    "get_attention_config",
    "check_hopper_features",
    "supports_fp8",
    "AdaptiveKernelSelector",
]

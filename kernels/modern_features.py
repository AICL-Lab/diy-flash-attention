"""
Modern CUDA Features for Hopper+ GPUs

This module provides:
- Feature detection for TMA, FP8, Warpgroup MMA
- FP8 dtype conversion utilities with graceful fallback
- Architecture-adaptive kernel selection (delegates to gpu_detect for configs)

Note: These features require Hopper (SM90) or newer GPUs.
On older GPUs, the module provides fallback to standard implementations.
"""

from typing import Any, Callable, Dict

import torch
import triton.language as tl

from utils.gpu_detect import detect_gpu, get_optimal_config

# =============================================================================
# Feature Detection
# =============================================================================


def check_hopper_features() -> Dict[str, Any]:
    """
    Check availability of Hopper+ features.

    Returns:
        Dictionary with feature availability flags
    """
    try:
        caps = detect_gpu()
        is_hopper_plus = caps.compute_capability >= (9, 0)

        return {
            "tma_available": is_hopper_plus and hasattr(tl, "make_tensor_descriptor"),
            "fp8_available": is_hopper_plus,
            "wgmma_available": is_hopper_plus,
            "arch": caps.arch.value,
            "compute_capability": caps.compute_capability,
        }
    except Exception:
        return {
            "tma_available": False,
            "fp8_available": False,
            "wgmma_available": False,
            "arch": "unknown",
            "compute_capability": (0, 0),
        }


# =============================================================================
# FP8 Support
# =============================================================================


def supports_fp8() -> bool:
    """Check if current GPU supports FP8."""
    features = check_hopper_features()
    return features["fp8_available"]


def to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to FP8 E4M3 format.

    FP8 E4M3 has 4 exponent bits and 3 mantissa bits.
    Range: ~±448, good for weights and activations.

    Args:
        tensor: Input tensor (float16 or float32)

    Returns:
        Tensor in FP8 E4M3 format (or float16 if FP8 not supported)
    """
    if not supports_fp8():
        return tensor.to(torch.float16)

    try:
        return tensor.to(torch.float8_e4m3fn)
    except (AttributeError, RuntimeError):
        return tensor.to(torch.float16)


def to_fp8_e5m2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to FP8 E5M2 format.

    FP8 E5M2 has 5 exponent bits and 2 mantissa bits.
    Larger range but less precision, good for gradients.

    Args:
        tensor: Input tensor (float16 or float32)

    Returns:
        Tensor in FP8 E5M2 format (or float16 if FP8 not supported)
    """
    if not supports_fp8():
        return tensor.to(torch.float16)

    try:
        return tensor.to(torch.float8_e5m2)
    except (AttributeError, RuntimeError):
        return tensor.to(torch.float16)


# =============================================================================
# Architecture-Adaptive Kernel Selection
# =============================================================================


class AdaptiveKernelSelector:
    """
    Selects optimal kernel implementation based on GPU architecture.

    Delegates configuration to gpu_detect.get_optimal_config and adds
    feature flags (TMA, FP8) for Hopper+ GPUs.
    """

    def __init__(self):
        self._caps = None
        self._features = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of GPU detection."""
        if not self._initialized:
            try:
                self._caps = detect_gpu()
                self._features = check_hopper_features()
            except Exception:
                self._caps = None
                self._features = {
                    "tma_available": False,
                    "fp8_available": False,
                    "wgmma_available": False,
                }
            self._initialized = True

    def get_matmul_config(self) -> Dict[str, Any]:
        """Get optimal matmul configuration with feature flags."""
        self._initialize()
        if self._caps is None:
            return {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "num_stages": 2,
                "num_warps": 4,
                "use_tma": False,
                "use_fp8": False,
            }

        config = get_optimal_config(self._caps, "matmul")
        config["use_tma"] = self._features.get("tma_available", False)
        config["use_fp8"] = False
        return config

    def get_attention_config(self) -> Dict[str, Any]:
        """Get optimal attention configuration with feature flags."""
        self._initialize()
        if self._caps is None:
            return {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 2, "num_warps": 4, "use_tma": False}

        config = get_optimal_config(self._caps, "flash_attention")
        config["use_tma"] = self._features.get("tma_available", False)
        return config

    def select_matmul_kernel(self) -> Callable:
        """
        Select the best matmul kernel for current GPU.

        Returns:
            Matmul function optimized for current architecture
        """
        from kernels.matmul import triton_matmul

        # TMA and FP8 variants would be added here when available
        return triton_matmul

    def select_attention_kernel(self) -> Callable:
        """
        Select the best attention kernel for current GPU.

        Returns:
            Attention function optimized for current architecture
        """
        from kernels.flash_attn import flash_attention

        return flash_attention


# Global selector instance
_kernel_selector = AdaptiveKernelSelector()


def get_optimal_matmul() -> Callable:
    """Get the optimal matmul implementation for current GPU."""
    return _kernel_selector.select_matmul_kernel()


def get_optimal_attention() -> Callable:
    """Get the optimal attention implementation for current GPU."""
    return _kernel_selector.select_attention_kernel()


def get_matmul_config() -> Dict[str, Any]:
    """Get optimal matmul configuration for current GPU."""
    return _kernel_selector.get_matmul_config()


def get_attention_config() -> Dict[str, Any]:
    """Get optimal attention configuration for current GPU."""
    return _kernel_selector.get_attention_config()


# =============================================================================
# Utility Functions
# =============================================================================


def print_feature_status():
    """Print status of modern CUDA features."""
    features = check_hopper_features()

    print("=" * 50)
    print("Modern CUDA Feature Status")
    print("=" * 50)
    print(f"Architecture:        {features['arch']}")
    print(
        f"Compute Capability:  {features['compute_capability'][0]}.{features['compute_capability'][1]}"
    )
    print("-" * 50)
    print("Feature Availability:")
    print(
        f"  TMA:               {'✓ Available' if features['tma_available'] else '✗ Not available'}"
    )
    print(
        f"  FP8:               {'✓ Available' if features['fp8_available'] else '✗ Not available'}"
    )
    print(
        f"  Warpgroup MMA:     {'✓ Available' if features['wgmma_available'] else '✗ Not available'}"
    )
    print("=" * 50)

    if not any([features["tma_available"], features["fp8_available"], features["wgmma_available"]]):
        print("\nNote: Modern features require Hopper (SM90) or newer GPU.")
        print("Standard implementations will be used as fallback.")


if __name__ == "__main__":
    print_feature_status()

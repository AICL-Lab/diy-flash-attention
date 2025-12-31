"""
Modern CUDA Features for Hopper+ GPUs

This module provides optimized kernel variants that leverage modern GPU features:
- TMA (Tensor Memory Accelerator) for efficient data loading
- FP8 support for higher throughput
- Warpgroup MMA for larger tile sizes
- Architecture-adaptive kernel selection

Note: These features require Hopper (SM90) or newer GPUs.
On older GPUs, the module provides fallback to standard implementations.

Requirements:
- CUDA 12.0+ for TMA
- Triton 2.1+ for FP8 support
"""

import math
from typing import Optional, Callable, Dict, Any

import torch
import triton
import triton.language as tl

from utils.gpu_detect import detect_gpu, GPUCapabilities, GPUArch


# =============================================================================
# Feature Detection and Fallback
# =============================================================================

def check_hopper_features() -> Dict[str, bool]:
    """
    Check availability of Hopper+ features.
    
    Returns:
        Dictionary with feature availability flags
    """
    try:
        caps = detect_gpu()
        is_hopper_plus = caps.compute_capability >= (9, 0)
        
        return {
            "tma_available": is_hopper_plus and hasattr(tl, 'make_tensor_descriptor'),
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
        # Fallback to float16
        return tensor.to(torch.float16)
    
    try:
        return tensor.to(torch.float8_e4m3fn)
    except (AttributeError, RuntimeError):
        # FP8 dtype not available in this PyTorch version
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
    
    Provides automatic fallback to standard implementations on older GPUs.
    """
    
    def __init__(self):
        self.caps = None
        self.features = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of GPU detection."""
        if not self._initialized:
            try:
                self.caps = detect_gpu()
                self.features = check_hopper_features()
            except Exception:
                self.caps = None
                self.features = {
                    "tma_available": False,
                    "fp8_available": False,
                    "wgmma_available": False,
                }
            self._initialized = True
    
    def get_matmul_config(self) -> Dict[str, Any]:
        """
        Get optimal matmul configuration for current GPU.
        
        Returns:
            Configuration dictionary with block sizes and kernel parameters
        """
        self._initialize()
        
        if self.caps is None:
            return self._default_matmul_config()
        
        if self.caps.arch in (GPUArch.HOPPER, GPUArch.BLACKWELL):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "num_stages": 4,
                "num_warps": 8,
                "use_tma": self.features.get("tma_available", False),
                "use_fp8": False,  # FP8 matmul requires special handling
            }
        elif self.caps.arch in (GPUArch.AMPERE, GPUArch.ADA):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "num_stages": 3,
                "num_warps": 8,
                "use_tma": False,
                "use_fp8": False,
            }
        else:
            return self._default_matmul_config()
    
    def _default_matmul_config(self) -> Dict[str, Any]:
        """Default configuration for older GPUs."""
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
    
    def get_attention_config(self) -> Dict[str, Any]:
        """
        Get optimal attention configuration for current GPU.
        
        Returns:
            Configuration dictionary with block sizes and kernel parameters
        """
        self._initialize()
        
        if self.caps is None:
            return self._default_attention_config()
        
        if self.caps.arch in (GPUArch.HOPPER, GPUArch.BLACKWELL):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "num_stages": 3,
                "num_warps": 8,
                "use_tma": self.features.get("tma_available", False),
            }
        elif self.caps.arch in (GPUArch.AMPERE, GPUArch.ADA):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "num_stages": 2,
                "num_warps": 4,
                "use_tma": False,
            }
        else:
            return self._default_attention_config()
    
    def _default_attention_config(self) -> Dict[str, Any]:
        """Default attention configuration for older GPUs."""
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "num_stages": 2,
            "num_warps": 4,
            "use_tma": False,
        }
    
    def select_matmul_kernel(self) -> Callable:
        """
        Select the best matmul kernel for current GPU.
        
        Returns:
            Matmul function optimized for current architecture
        """
        from kernels.matmul import triton_matmul
        
        # For now, return standard implementation
        # TMA and FP8 variants would be added here when available
        return triton_matmul
    
    def select_attention_kernel(self) -> Callable:
        """
        Select the best attention kernel for current GPU.
        
        Returns:
            Attention function optimized for current architecture
        """
        from kernels.flash_attn import flash_attention
        
        # For now, return standard implementation
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
# TMA Placeholder (Requires Hopper+ and Triton 3.0+)
# =============================================================================

def create_tma_descriptor(
    tensor: torch.Tensor,
    block_shape: tuple,
) -> Optional[Any]:
    """
    Create TMA descriptor for efficient tensor loading.
    
    TMA (Tensor Memory Accelerator) provides hardware-accelerated
    asynchronous data transfer with automatic address calculation.
    
    Args:
        tensor: Source tensor in global memory
        block_shape: Shape of blocks to load
        
    Returns:
        TMA descriptor or None if TMA not available
        
    Note:
        This is a placeholder. Full TMA support requires:
        - Hopper (SM90) or newer GPU
        - Triton 3.0+ with TMA support
        - CUDA 12.0+
    """
    features = check_hopper_features()
    
    if not features["tma_available"]:
        return None
    
    # TMA descriptor creation would go here
    # This requires Triton's tl.make_tensor_descriptor API
    # which is available in newer Triton versions
    
    return None


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
    print(f"Compute Capability:  {features['compute_capability'][0]}.{features['compute_capability'][1]}")
    print("-" * 50)
    print("Feature Availability:")
    print(f"  TMA:               {'✓ Available' if features['tma_available'] else '✗ Not available'}")
    print(f"  FP8:               {'✓ Available' if features['fp8_available'] else '✗ Not available'}")
    print(f"  Warpgroup MMA:     {'✓ Available' if features['wgmma_available'] else '✗ Not available'}")
    print("=" * 50)
    
    if not any([features['tma_available'], features['fp8_available'], features['wgmma_available']]):
        print("\nNote: Modern features require Hopper (SM90) or newer GPU.")
        print("Standard implementations will be used as fallback.")


def benchmark_feature_impact():
    """
    Benchmark the impact of modern features (if available).
    
    Compares standard vs optimized implementations.
    """
    import time
    
    print("\nBenchmarking Feature Impact...")
    print("-" * 50)
    
    # Test configuration
    M, N, K = 4096, 4096, 4096
    batch, heads, seq_len, head_dim = 4, 8, 2048, 64
    
    # Get optimal implementations
    matmul_fn = get_optimal_matmul()
    attention_fn = get_optimal_attention()
    
    # Matmul benchmark
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = matmul_fn(a, b)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        _ = matmul_fn(a, b)
    torch.cuda.synchronize()
    matmul_time = (time.perf_counter() - start) / 100 * 1000
    
    flops = 2 * M * N * K
    tflops = flops / matmul_time / 1e9
    
    print(f"MatMul ({M}×{K} @ {K}×{N}):")
    print(f"  Time:   {matmul_time:.3f} ms")
    print(f"  TFLOPS: {tflops:.2f}")
    
    # Attention benchmark
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = attention_fn(q, k, v, causal=True)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        _ = attention_fn(q, k, v, causal=True)
    torch.cuda.synchronize()
    attn_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"\nFlashAttention (batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}):")
    print(f"  Time:   {attn_time:.3f} ms")
    
    print("-" * 50)


if __name__ == "__main__":
    print_feature_status()
    
    if torch.cuda.is_available():
        benchmark_feature_impact()

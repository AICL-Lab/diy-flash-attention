"""GPU detection utilities for optimal kernel configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


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


def _get_arch_from_cc(major: int, minor: int) -> GPUArch:
    """Map compute capability to GPU architecture."""
    cc = (major, minor)
    if cc >= (10, 0):
        return GPUArch.BLACKWELL
    elif cc >= (9, 0):
        return GPUArch.HOPPER
    elif cc >= (8, 9):
        return GPUArch.ADA
    elif cc >= (8, 0):
        return GPUArch.AMPERE
    elif cc >= (7, 5):
        return GPUArch.TURING
    elif cc >= (7, 0):
        return GPUArch.VOLTA
    else:
        return GPUArch.UNKNOWN


def detect_gpu(device_id: int = 0) -> GPUCapabilities:
    """
    Detect current GPU capabilities.
    
    Args:
        device_id: CUDA device ID to query
        
    Returns:
        GPUCapabilities with detected features
        
    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install CUDA and PyTorch with CUDA support.")
    
    props = torch.cuda.get_device_properties(device_id)
    major, minor = props.major, props.minor
    arch = _get_arch_from_cc(major, minor)
    
    # Feature detection based on architecture
    is_hopper_plus = (major, minor) >= (9, 0)
    is_ampere_plus = (major, minor) >= (8, 0)
    
    return GPUCapabilities(
        name=props.name,
        arch=arch,
        compute_capability=(major, minor),
        has_tma=is_hopper_plus,
        has_fp8=is_hopper_plus,
        has_warpgroup_mma=is_hopper_plus,
        sram_per_sm=props.max_shared_memory_per_multiprocessor,
        num_sms=props.multi_processor_count,
        total_memory_gb=props.total_memory / (1024**3),
    )


def get_optimal_config(caps: GPUCapabilities, operation: str) -> dict:
    """
    Get optimal kernel configuration for detected GPU.
    
    Args:
        caps: GPU capabilities from detect_gpu()
        operation: "matmul" or "flash_attention"
        
    Returns:
        Optimal block sizes and other parameters
    """
    if operation == "matmul":
        if caps.arch in (GPUArch.HOPPER, GPUArch.BLACKWELL):
            # Larger blocks for Hopper+ with more SRAM
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            }
        elif caps.arch in (GPUArch.AMPERE, GPUArch.ADA):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            }
        else:
            # Conservative config for older GPUs
            return {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "num_stages": 2,
                "num_warps": 4,
            }
    
    elif operation == "flash_attention":
        if caps.arch in (GPUArch.HOPPER, GPUArch.BLACKWELL):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_D": 64,
                "num_stages": 3,
                "num_warps": 8,
            }
        elif caps.arch in (GPUArch.AMPERE, GPUArch.ADA):
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_D": 64,
                "num_stages": 2,
                "num_warps": 4,
            }
        else:
            return {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_D": 64,
                "num_stages": 2,
                "num_warps": 4,
            }
    
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'matmul' or 'flash_attention'.")


def print_gpu_info(caps: Optional[GPUCapabilities] = None) -> None:
    """Print GPU information in a formatted way."""
    if caps is None:
        caps = detect_gpu()
    
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    print(f"Name:                {caps.name}")
    print(f"Architecture:        {caps.arch.value}")
    print(f"Compute Capability:  {caps.compute_capability[0]}.{caps.compute_capability[1]}")
    print(f"Total Memory:        {caps.total_memory_gb:.1f} GB")
    print(f"SMs:                 {caps.num_sms}")
    print(f"SRAM per SM:         {caps.sram_per_sm // 1024} KB")
    print("-" * 50)
    print("Feature Support:")
    print(f"  TMA:               {'✓' if caps.has_tma else '✗'}")
    print(f"  FP8:               {'✓' if caps.has_fp8 else '✗'}")
    print(f"  Warpgroup MMA:     {'✓' if caps.has_warpgroup_mma else '✗'}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        caps = detect_gpu()
        print_gpu_info(caps)
        
        print("\nOptimal MatMul Config:")
        config = get_optimal_config(caps, "matmul")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        print("\nOptimal FlashAttention Config:")
        config = get_optimal_config(caps, "flash_attention")
        for k, v in config.items():
            print(f"  {k}: {v}")
    except RuntimeError as e:
        print(f"Error: {e}")

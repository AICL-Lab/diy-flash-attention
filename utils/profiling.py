"""
GPU Profiling Toolkit

This module provides utilities for understanding GPU memory hierarchy
and estimating kernel occupancy. Educational focus: teach students
why FlashAttention is fast by showing memory/occupancy trade-offs.

Key concepts:
- Occupancy: Ratio of active warps to maximum warps per SM
- Shared memory pressure: How much SMEM is used vs available
- Memory hierarchy: HBM → L2 → L1 → SMEM → Registers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GPUMemoryProfile:
    """
    GPU memory hierarchy metrics for a kernel.

    Captures key metrics that affect kernel performance:
    - Occupancy: Higher is generally better (more latency hiding)
    - SMEM pressure: Lower leaves room for more blocks per SM
    - L2 hit rate: Higher means less HBM traffic
    """

    occupancy_pct: float  # 0-100
    smem_used_bytes: int
    smem_available_bytes: int
    reg_per_thread: int
    l2_hit_rate: float  # 0-1
    estimated_stall_cycles: int

    def smem_pressure_pct(self) -> float:
        """Calculate shared memory utilization percentage."""
        if self.smem_available_bytes == 0:
            return 100.0
        return 100.0 * self.smem_used_bytes / self.smem_available_bytes

    def can_increase_occupancy(self) -> bool:
        """
        Heuristic check if occupancy can be increased.

        Returns True if there's sufficient SMEM headroom (< 70% used).
        """
        return self.smem_pressure_pct() < 70.0


@dataclass
class KernelBenchmark:
    """
    Kernel performance metrics from a benchmark run.

    Combines timing data with memory profile for comprehensive analysis.
    """

    kernel_name: str
    elapsed_ms: float
    tflops: float
    gbps: float
    memory_profile: GPUMemoryProfile


def get_gpu_memory_hierarchy(gpu_capability: int) -> Dict[str, int]:
    """
    Return GPU memory hierarchy specifications.

    Args:
        gpu_capability: GPU compute capability major version (80, 89, 90, etc.)

    Returns:
        Dictionary with memory hierarchy specs
    """
    specs = {
        # Ampere (SM80) - A100
        80: {
            "smem_per_block": 96 * 1024,  # 96KB
            "registers_per_warp": 128 * 32,  # 4096 per warp
            "l1_cache_kb": 128,
            "l2_cache_mb": 40,
            "hbm_gb": 16,  # Typical A100
            "peak_bandwidth_gbps": 1935,
            "peak_compute_tflops": 19500,  # FP32
            "max_warps_per_sm": 32,
            "max_threads_per_warp": 32,
        },
        # Ada (SM89) - RTX 4090
        89: {
            "smem_per_block": 96 * 1024,
            "registers_per_warp": 128 * 32,
            "l1_cache_kb": 128,
            "l2_cache_mb": 96,
            "hbm_gb": 24,  # Typical RTX 4090
            "peak_bandwidth_gbps": 960,
            "peak_compute_tflops": 16384,
            "max_warps_per_sm": 32,
            "max_threads_per_warp": 32,
        },
        # Hopper (SM90) - H100
        90: {
            "smem_per_block": 144 * 1024,  # Up to 144KB
            "registers_per_warp": 256 * 32,  # Increased
            "l1_cache_kb": 192,
            "l2_cache_mb": 50,
            "hbm_gb": 80,  # Typical H100
            "peak_bandwidth_gbps": 2133,
            "peak_compute_tflops": 67000,  # TF32
            "max_warps_per_sm": 32,
            "max_threads_per_warp": 32,
        },
    }
    # Default to Ampere for unknown architectures
    return specs.get(gpu_capability, specs[80])


def estimate_occupancy(
    block_size: int,
    registers_per_thread: int,
    shared_memory_bytes: int,
    gpu_capability: int,
) -> float:
    """
    Estimate GPU occupancy percentage.

    Occupancy is the ratio of active warps to the maximum number of warps
    that can run concurrently on an SM. Higher occupancy helps hide latency.

    Args:
        block_size: Number of threads per block
        registers_per_thread: Register usage per thread
        shared_memory_bytes: Shared memory usage per block
        gpu_capability: GPU compute capability major version

    Returns:
        Estimated occupancy percentage (0-100)

    Note:
        This is a simplified model. Real occupancy depends on:
        - Register file size limit
        - Shared memory limit
        - Max threads per SM
        - Max warps per SM
    """
    specs = get_gpu_memory_hierarchy(gpu_capability)

    # Calculate warps per block
    threads_per_warp = specs.get("max_threads_per_warp", 32)
    warps_per_block = (block_size + threads_per_warp - 1) // threads_per_warp

    # Limit by register file
    # Each warp needs registers_per_thread * 32 registers
    registers_per_warp = registers_per_thread * threads_per_warp
    max_warps_by_regs = specs["registers_per_warp"] // max(registers_per_warp, 1)

    # Limit by shared memory
    smem_available = specs["smem_per_block"]
    max_blocks_by_smem = smem_available // max(shared_memory_bytes, 1)

    # Max warps considering both limits
    max_warps = min(
        max_warps_by_regs,
        max_blocks_by_smem * warps_per_block,
        specs.get("max_warps_per_sm", 32),
    )

    # Occupancy = active warps / max warps
    # Each SM can run multiple blocks, limited by resources
    active_warps = warps_per_block
    occupancy = min(100.0, (active_warps / max(max_warps, 1)) * 100.0)

    return occupancy


if __name__ == "__main__":
    print("GPU Memory Hierarchy Specs:")
    print("=" * 60)

    for cap, name in [(80, "Ampere"), (89, "Ada"), (90, "Hopper")]:
        specs = get_gpu_memory_hierarchy(cap)
        print(f"\n{name} (SM{cap}):")
        print(f"  SMEM per block:  {specs['smem_per_block'] // 1024}KB")
        print(f"  L2 cache:        {specs['l2_cache_mb']}MB")
        print(f"  HBM:             {specs['hbm_gb']}GB")
        print(f"  Peak bandwidth:  {specs['peak_bandwidth_gbps']} GB/s")

    print("\n" + "=" * 60)
    print("Occupancy Estimation Examples:")
    print("=" * 60)

    # FlashAttention typical block sizes
    for block_m, block_n in [(64, 32), (128, 64)]:
        block_size = block_m * 1  # Simplified: 1 thread per row
        smem = block_m * 64 * 2 + block_n * 64 * 2  # Q + K blocks in float16
        occ = estimate_occupancy(block_size, 64, smem, 80)
        print(f"\nBlock ({block_m}, {block_n}):")
        print(f"  Block size: {block_size}")
        print(f"  SMEM:       {smem} bytes")
        print(f"  Occupancy:  {occ:.1f}%")

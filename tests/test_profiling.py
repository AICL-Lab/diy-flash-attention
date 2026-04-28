"""Tests for GPU profiling utilities."""

from __future__ import annotations

import pytest

from utils.profiling import (
    GPUMemoryProfile,
    KernelBenchmark,
    estimate_occupancy,
    get_gpu_memory_hierarchy,
)


class TestGPUMemoryProfile:
    """Test GPUMemoryProfile dataclass."""

    def test_creation(self):
        """Test GPUMemoryProfile creation."""
        profile = GPUMemoryProfile(
            occupancy_pct=75.0,
            smem_used_bytes=32768,
            smem_available_bytes=98304,
            reg_per_thread=64,
            l2_hit_rate=0.8,
            estimated_stall_cycles=1000,
        )
        assert profile.occupancy_pct == 75.0
        assert profile.smem_used_bytes == 32768
        assert profile.smem_pressure_pct() > 0

    def test_smem_pressure_calculation(self):
        """Test shared memory pressure calculation."""
        profile = GPUMemoryProfile(
            occupancy_pct=50.0,
            smem_used_bytes=49152,
            smem_available_bytes=98304,
            reg_per_thread=64,
            l2_hit_rate=0.5,
            estimated_stall_cycles=500,
        )
        assert profile.smem_pressure_pct() == 50.0

    def test_can_increase_occupancy(self):
        """Test occupancy increase check."""
        low_pressure = GPUMemoryProfile(
            occupancy_pct=50.0,
            smem_used_bytes=32768,
            smem_available_bytes=98304,
            reg_per_thread=64,
            l2_hit_rate=0.8,
            estimated_stall_cycles=100,
        )
        assert low_pressure.can_increase_occupancy() is True

        high_pressure = GPUMemoryProfile(
            occupancy_pct=90.0,
            smem_used_bytes=86016,
            smem_available_bytes=98304,
            reg_per_thread=64,
            l2_hit_rate=0.8,
            estimated_stall_cycles=100,
        )
        assert high_pressure.can_increase_occupancy() is False


class TestGPUMemoryHierarchy:
    """Test GPU memory hierarchy specs."""

    def test_ampere_specs(self):
        """Test Ampere memory specs."""
        specs = get_gpu_memory_hierarchy(80)
        assert specs["smem_per_block"] == 96 * 1024  # 96KB
        assert specs["peak_bandwidth_gbps"] > 0
        assert specs["hbm_gb"] > 0

    def test_ada_specs(self):
        """Test Ada memory specs."""
        specs = get_gpu_memory_hierarchy(89)
        assert specs["smem_per_block"] == 96 * 1024
        assert specs["l2_cache_mb"] == 96

    def test_hopper_specs(self):
        """Test Hopper memory specs."""
        specs = get_gpu_memory_hierarchy(90)
        assert specs["smem_per_block"] == 144 * 1024  # Up to 144KB
        assert specs["peak_bandwidth_gbps"] > 2000

    def test_unknown_arch_defaults_to_ampere(self):
        """Test unknown architecture defaults to Ampere specs."""
        specs = get_gpu_memory_hierarchy(0)
        assert specs == get_gpu_memory_hierarchy(80)


class TestEstimateOccupancy:
    """Test occupancy estimation."""

    def test_basic_estimation(self):
        """Test basic occupancy calculation."""
        occ = estimate_occupancy(
            block_size=256,
            registers_per_thread=64,
            shared_memory_bytes=32768,
            gpu_capability=80,
        )
        assert 0 <= occ <= 100

    def test_high_smem_reduces_occupancy(self):
        """Test that high SMEM usage reduces occupancy."""
        low_smem = estimate_occupancy(
            block_size=256,
            registers_per_thread=64,
            shared_memory_bytes=16384,
            gpu_capability=80,
        )
        high_smem = estimate_occupancy(
            block_size=256,
            registers_per_thread=64,
            shared_memory_bytes=65536,
            gpu_capability=80,
        )
        assert low_smem >= high_smem

    def test_different_architectures(self):
        """Test occupancy estimation for different architectures."""
        for cap in [80, 89, 90]:
            occ = estimate_occupancy(
                block_size=256,
                registers_per_thread=64,
                shared_memory_bytes=32768,
                gpu_capability=cap,
            )
            assert 0 <= occ <= 100, f"Occupancy out of range for SM{cap}"


class TestKernelBenchmark:
    """Test KernelBenchmark dataclass."""

    def test_creation(self):
        """Test KernelBenchmark creation."""
        profile = GPUMemoryProfile(
            occupancy_pct=75.0,
            smem_used_bytes=32768,
            smem_available_bytes=98304,
            reg_per_thread=64,
            l2_hit_rate=0.8,
            estimated_stall_cycles=1000,
        )
        benchmark = KernelBenchmark(
            kernel_name="flash_attention_v2",
            elapsed_ms=1.5,
            tflops=100.0,
            gbps=500.0,
            memory_profile=profile,
        )
        assert benchmark.kernel_name == "flash_attention_v2"
        assert benchmark.tflops == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""CPU-safe tests for GPU detection helpers."""

import pytest

from utils.gpu_detect import (
    GPUArch,
    GPUCapabilities,
    _get_arch_from_cc,
    detect_gpu,
    get_optimal_config,
)


class TestArchMapping:
    def test_arch_mapping_boundaries(self):
        assert _get_arch_from_cc(10, 0) == GPUArch.BLACKWELL
        assert _get_arch_from_cc(9, 0) == GPUArch.HOPPER
        assert _get_arch_from_cc(8, 9) == GPUArch.ADA
        assert _get_arch_from_cc(8, 0) == GPUArch.AMPERE
        assert _get_arch_from_cc(7, 5) == GPUArch.TURING
        assert _get_arch_from_cc(7, 0) == GPUArch.VOLTA
        assert _get_arch_from_cc(6, 1) == GPUArch.UNKNOWN


class TestDetectGpuFallback:
    def test_detect_gpu_raises_without_cuda(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            detect_gpu()


class TestOptimalConfig:
    @pytest.mark.parametrize(
        ("arch", "operation", "expected"),
        [
            (GPUArch.HOPPER, "matmul", {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}),
            (GPUArch.AMPERE, "matmul", {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}),
            (GPUArch.UNKNOWN, "matmul", {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}),
            (GPUArch.HOPPER, "flash_attention", {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 64}),
            (GPUArch.UNKNOWN, "flash_attention", {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 64}),
        ],
    )
    def test_optimal_configs(self, arch, operation, expected):
        caps = GPUCapabilities(
            name="Test GPU",
            arch=arch,
            compute_capability=(9, 0),
            has_tma=arch in (GPUArch.HOPPER, GPUArch.BLACKWELL),
            has_fp8=arch in (GPUArch.HOPPER, GPUArch.BLACKWELL),
            has_warpgroup_mma=arch in (GPUArch.HOPPER, GPUArch.BLACKWELL),
            sram_per_sm=228 * 1024,
            num_sms=80,
            total_memory_gb=80.0,
        )

        config = get_optimal_config(caps, operation)

        for key, value in expected.items():
            assert config[key] == value

    def test_unknown_operation_raises(self):
        caps = GPUCapabilities(
            name="Test GPU",
            arch=GPUArch.AMPERE,
            compute_capability=(8, 0),
            has_tma=False,
            has_fp8=False,
            has_warpgroup_mma=False,
            sram_per_sm=164 * 1024,
            num_sms=108,
            total_memory_gb=40.0,
        )

        with pytest.raises(ValueError, match="Unknown operation"):
            get_optimal_config(caps, "unknown")

"""CPU-safe tests for modern feature detection and adaptive selectors."""

from kernels.modern_features import (
    AdaptiveKernelSelector,
    check_hopper_features,
    get_attention_config,
    get_matmul_config,
    supports_fp8,
)
from utils.gpu_detect import GPUArch, GPUCapabilities


class TestFeatureDetection:
    def test_check_hopper_features_fallback(self, monkeypatch):
        monkeypatch.setattr("kernels.modern_features.detect_gpu", lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))

        features = check_hopper_features()

        assert features["tma_available"] is False
        assert features["fp8_available"] is False
        assert features["wgmma_available"] is False
        assert features["arch"] == "unknown"
        assert features["compute_capability"] == (0, 0)

    def test_supports_fp8_uses_feature_flag(self, monkeypatch):
        monkeypatch.setattr("kernels.modern_features.check_hopper_features", lambda: {"fp8_available": True})
        assert supports_fp8() is True


class TestAdaptiveSelector:
    def test_selector_fallback_config_when_detection_fails(self, monkeypatch):
        monkeypatch.setattr("kernels.modern_features.detect_gpu", lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))

        selector = AdaptiveKernelSelector()

        matmul_config = selector.get_matmul_config()
        attention_config = selector.get_attention_config()

        assert matmul_config["BLOCK_M"] == 64
        assert matmul_config["BLOCK_N"] == 64
        assert matmul_config["BLOCK_K"] == 32
        assert matmul_config["use_tma"] is False
        assert matmul_config["use_fp8"] is False
        assert attention_config["BLOCK_M"] == 64
        assert attention_config["BLOCK_N"] == 32
        assert attention_config["use_tma"] is False

    def test_selector_uses_detected_config(self, monkeypatch):
        caps = GPUCapabilities(
            name="H100",
            arch=GPUArch.HOPPER,
            compute_capability=(9, 0),
            has_tma=True,
            has_fp8=True,
            has_warpgroup_mma=True,
            sram_per_sm=228 * 1024,
            num_sms=120,
            total_memory_gb=80.0,
        )
        monkeypatch.setattr("kernels.modern_features.detect_gpu", lambda: caps)
        monkeypatch.setattr(
            "kernels.modern_features.check_hopper_features",
            lambda: {
                "tma_available": True,
                "fp8_available": True,
                "wgmma_available": True,
                "arch": caps.arch.value,
                "compute_capability": caps.compute_capability,
            },
        )

        selector = AdaptiveKernelSelector()
        matmul_config = selector.get_matmul_config()
        attention_config = selector.get_attention_config()

        assert matmul_config["BLOCK_M"] == 128
        assert matmul_config["BLOCK_N"] == 256
        assert matmul_config["BLOCK_K"] == 64
        assert matmul_config["use_tma"] is True
        assert matmul_config["use_fp8"] is False
        assert attention_config["BLOCK_M"] == 128
        assert attention_config["BLOCK_N"] == 64
        assert attention_config["use_tma"] is True

    def test_selected_kernels_are_callable(self):
        selector = AdaptiveKernelSelector()
        assert callable(selector.select_matmul_kernel())
        assert callable(selector.select_attention_kernel())


class TestModuleLevelHelpers:
    def test_module_level_config_helpers_return_dicts(self):
        assert isinstance(get_matmul_config(), dict)
        assert isinstance(get_attention_config(), dict)

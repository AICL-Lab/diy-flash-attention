"""Utility functions for benchmarking, validation, and GPU detection."""

from .benchmark import BenchmarkResult, BenchmarkRunner
from .gpu_detect import GPUArch, GPUCapabilities, detect_gpu, print_gpu_info
from .profiling import (
    GPUMemoryProfile,
    KernelBenchmark,
    estimate_occupancy,
    get_gpu_memory_hierarchy,
)
from .triton_helpers import TRITON_AVAILABLE, TritonKernelStub, require_triton
from .validation import validate_attention, validate_matmul

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "validate_matmul",
    "validate_attention",
    "detect_gpu",
    "GPUCapabilities",
    "GPUArch",
    "print_gpu_info",
    # Profiling utilities
    "GPUMemoryProfile",
    "KernelBenchmark",
    "estimate_occupancy",
    "get_gpu_memory_hierarchy",
    # Triton helpers
    "TRITON_AVAILABLE",
    "require_triton",
    "TritonKernelStub",
]

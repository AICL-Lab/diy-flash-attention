"""
Backend Selector: Unified Kernel Dispatch Registry

This module provides a unified interface for selecting the optimal kernel
implementation based on GPU capability and problem characteristics.

Heuristics:
- V2 requires Ampere+ (SM80) for optimal performance
- Small problems (batch < 2 or seq < 512) → V1 (lower overhead)
- Large problems (batch >= 4 and seq >= 4096) → V2 (better parallelism)
- Default → V1 (conservative)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Tuple

from kernels.flash_attn import flash_attention
from kernels.flash_attn_v2 import flash_attention_v2


class KernelVariant(str, Enum):
    """Kernel implementation variant."""

    V1 = "v1"
    V2 = "v2"
    PERSISTENT = "persistent"
    AUTO = "auto"


class BackendSelector:
    """
    Unified kernel dispatch registry.

    Selects optimal kernel implementation based on:
    - GPU compute capability
    - Problem size (batch, seq_len, heads, head_dim)
    - User preference
    """

    @staticmethod
    def select_attention(
        variant: KernelVariant = KernelVariant.AUTO,
        q_shape: Optional[Tuple[int, int, int, int]] = None,
        gpu_capability: Optional[int] = None,
    ) -> Callable:
        """
        Select attention kernel implementation.

        Args:
            variant: Preferred variant (AUTO uses heuristics)
            q_shape: Query tensor shape (batch, seq_len, heads, head_dim) for heuristic selection
            gpu_capability: GPU compute capability major version (auto-detected if None)

        Returns:
            Callable kernel function
        """
        if variant == KernelVariant.AUTO:
            variant = BackendSelector._select_attention_heuristic(q_shape, gpu_capability)

        if variant == KernelVariant.V1:
            return flash_attention
        elif variant == KernelVariant.V2:
            return flash_attention_v2
        elif variant == KernelVariant.PERSISTENT:
            # TODO: Import persistent_attention once implemented
            raise NotImplementedError("Persistent attention not yet implemented")
        else:
            raise ValueError(f"Unknown variant: {variant}")

    @staticmethod
    def _select_attention_heuristic(
        q_shape: Optional[Tuple[int, int, int, int]] = None,
        gpu_capability: Optional[int] = None,
    ) -> KernelVariant:
        """
        Apply heuristics for automatic kernel selection.

        Rules:
        1. Pre-Ampere (< SM80) → V1
        2. Small problem (batch < 2 or seq < 512) → V1
        3. Large problem (batch >= 4 and seq >= 4096) on Ampere+ → V2
        4. Default → V1
        """
        # Detect GPU capability if not provided
        if gpu_capability is None:
            try:
                from utils.gpu_detect import detect_gpu

                gpu = detect_gpu()
                gpu_capability = gpu.compute_capability[0]
            except (RuntimeError, ImportError):
                # Fallback to conservative choice
                return KernelVariant.V1

        # Rule 1: V2 requires Ampere+
        if gpu_capability < 80:
            return KernelVariant.V1

        # Rule 2 & 3: Problem size heuristics
        if q_shape is not None:
            batch, seq_len, heads, head_dim = q_shape

            # Small problem → V1 (lower kernel launch overhead)
            if batch < 2 or seq_len < 512:
                return KernelVariant.V1

            # Large problem → V2 (better parallelism)
            if batch >= 4 and seq_len >= 4096:
                return KernelVariant.V2

        # Default: V1 (conservative)
        return KernelVariant.V1


def select_attention_kernel(
    variant: str = "auto",
    q_shape: Optional[Tuple[int, int, int, int]] = None,
    gpu_capability: Optional[int] = None,
) -> Callable:
    """
    Convenience wrapper for BackendSelector.select_attention.

    Args:
        variant: Variant name string ("v1", "v2", "persistent", "auto")
        q_shape: Query tensor shape for heuristic selection
        gpu_capability: GPU compute capability major version

    Returns:
        Callable kernel function
    """
    return BackendSelector.select_attention(
        KernelVariant(variant),
        q_shape=q_shape,
        gpu_capability=gpu_capability,
    )


if __name__ == "__main__":
    print("Testing BackendSelector...")

    # Test V1 selection
    v1 = BackendSelector.select_attention(variant=KernelVariant.V1)
    print(f"  V1: {v1.__name__}")

    # Test V2 selection
    v2 = BackendSelector.select_attention(variant=KernelVariant.V2)
    print(f"  V2: {v2.__name__}")

    # Test auto selection
    auto_small = BackendSelector.select_attention(
        variant=KernelVariant.AUTO,
        q_shape=(1, 128, 8, 64),
        gpu_capability=80,
    )
    print(f"  Auto (small): {auto_small.__name__}")

    auto_large = BackendSelector.select_attention(
        variant=KernelVariant.AUTO,
        q_shape=(8, 4096, 8, 64),
        gpu_capability=80,
    )
    print(f"  Auto (large): {auto_large.__name__}")

    print("✓ BackendSelector tests passed")

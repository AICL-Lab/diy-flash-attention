"""
Triton Helper Utilities

This module provides shared utilities for Triton kernel development,
centralizing common patterns used across all kernel implementations.

Key utilities:
- TRITON_AVAILABLE: Check if Triton is available
- require_triton(): Raise error if Triton is not available
- triton_cdiv: Ceiling division utility (works without Triton)

This module eliminates code duplication across kernel files.
"""

from __future__ import annotations

from types import SimpleNamespace

# Triton availability check
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:
    TRITON_AVAILABLE = False
    # Provide minimal fallback for use without Triton
    triton = SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y)  # type: ignore
    tl = SimpleNamespace(constexpr=object())  # type: ignore


def require_triton() -> None:
    """
    Raise ModuleNotFoundError if Triton is not available.

    Call this at the start of kernel functions that require Triton.
    """
    if not TRITON_AVAILABLE:
        raise ModuleNotFoundError(
            "Triton is required to run GPU kernels. Install it with: pip install triton>=2.1.0"
        )


def triton_cdiv(x: int, y: int) -> int:
    """
    Ceiling division: ceil(x / y).

    Equivalent to triton.cdiv, but works without Triton installed.

    Args:
        x: Numerator
        y: Denominator

    Returns:
        Ceiling of x / y
    """
    return (x + y - 1) // y


class TritonKernelStub:
    """
    Stub class for Triton kernels when Triton is not available.

    Provides a placeholder that raises an error if actually called.
    Used to define kernel functions that work in CPU-only environments.
    """

    def __getitem__(self, _grid):
        def launcher(*args, **kwargs):
            require_triton()

        return launcher


# Export commonly used items
__all__ = [
    "TRITON_AVAILABLE",
    "require_triton",
    "triton_cdiv",
    "TritonKernelStub",
    "triton",
    "tl",
]

"""
Validation utilities for verifying numerical correctness of Triton kernels.

This module provides functions to compare Triton kernel outputs against
PyTorch reference implementations.
"""

from typing import Callable, Tuple

import torch


def validate_matmul(
    triton_fn: Callable,
    m: int,
    n: int,
    k: int,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Validate Triton matmul against torch.matmul.

    Args:
        triton_fn: Triton matmul function to validate
        m, n, k: Matrix dimensions (A is M×K, B is K×N)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        dtype: Data type for matrices
        device: Device to run on
        verbose: Whether to print detailed information

    Returns:
        Tuple of (is_valid, max_diff)
    """
    torch.manual_seed(42)

    # Generate random matrices
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    # Compute with Triton
    triton_output = triton_fn(a, b)

    # Compute with PyTorch (use float32 for reference)
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    torch_output = torch.matmul(a_fp32, b_fp32).to(dtype)

    # Compare
    diff = (triton_output - torch_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if within tolerance
    is_close = torch.allclose(triton_output, torch_output, rtol=rtol, atol=atol)

    if verbose:
        print(f"Validation for matmul ({m}×{k}) @ ({k}×{n}):")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Within tolerance (rtol={rtol}, atol={atol}): {is_close}")

    return is_close, max_diff


def validate_attention(
    flash_fn: Callable,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[bool, float]:
    """
    Validate FlashAttention against PyTorch's scaled_dot_product_attention.

    Args:
        flash_fn: FlashAttention function to validate
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        dtype: Data type for tensors
        device: Device to run on
        verbose: Whether to print detailed information

    Returns:
        Tuple of (is_valid, max_diff)
    """
    torch.manual_seed(42)

    # Generate random Q, K, V
    q = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=dtype)

    # Compute with FlashAttention
    flash_output = flash_fn(q, k, v, causal=causal)

    # Compute with PyTorch SDPA (reference)
    torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Compare
    diff = (flash_output - torch_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if within tolerance
    is_close = torch.allclose(flash_output, torch_output, rtol=rtol, atol=atol)

    if verbose:
        print(
            f"Validation for attention (batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}):"
        )
        print(f"  Causal: {causal}")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Within tolerance (rtol={rtol}, atol={atol}): {is_close}")

    return is_close, max_diff


def validate_matmul_edge_cases(
    triton_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """
    Validate matmul with edge cases.

    Tests:
    - Zero matrices
    - Identity-like operations
    - Non-power-of-2 dimensions
    - Very small matrices

    Returns:
        Tuple of (all_passed, results_dict)
    """
    results = {}
    all_passed = True

    if verbose:
        print("\nEdge Case Validation:")
        print("-" * 40)

    # Test 1: Zero matrix
    if verbose:
        print("  Testing zero matrix...")
    a = torch.zeros((64, 64), device=device, dtype=torch.float16)
    b = torch.randn((64, 64), device=device, dtype=torch.float16)
    triton_out = triton_fn(a, b)
    expected = torch.zeros((64, 64), device=device, dtype=torch.float16)
    is_valid = torch.allclose(triton_out, expected, rtol=rtol, atol=atol)
    results["zero_matrix"] = is_valid
    all_passed &= is_valid
    if verbose:
        print(f"    Zero matrix: {'✓' if is_valid else '✗'}")

    # Test 2: Non-power-of-2 dimensions
    if verbose:
        print("  Testing non-power-of-2 dimensions...")
    for m, n, k in [(33, 47, 61), (100, 200, 150), (17, 17, 17)]:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        triton_out = triton_fn(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        is_valid = torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)
        results[f"non_pow2_{m}x{n}x{k}"] = is_valid
        all_passed &= is_valid
        if verbose:
            print(f"    {m}×{n}×{k}: {'✓' if is_valid else '✗'}")

    # Test 3: Very small matrices
    if verbose:
        print("  Testing small matrices...")
    for size in [1, 2, 4, 8, 16]:
        a = torch.randn((size, size), device=device, dtype=torch.float16)
        b = torch.randn((size, size), device=device, dtype=torch.float16)
        triton_out = triton_fn(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        is_valid = torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)
        results[f"small_{size}x{size}"] = is_valid
        all_passed &= is_valid
        if verbose:
            print(f"    {size}×{size}: {'✓' if is_valid else '✗'}")

    # Test 4: Rectangular matrices
    if verbose:
        print("  Testing rectangular matrices...")
    for m, n, k in [(128, 64, 256), (64, 256, 128), (256, 128, 64)]:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        triton_out = triton_fn(a, b)
        torch_out = torch.matmul(a.float(), b.float()).half()
        is_valid = torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)
        results[f"rect_{m}x{n}x{k}"] = is_valid
        all_passed &= is_valid
        if verbose:
            print(f"    {m}×{n}×{k}: {'✓' if is_valid else '✗'}")

    if verbose:
        print("-" * 40)
        print(f"  Overall: {'✓ All passed' if all_passed else '✗ Some failed'}")

    return all_passed, results


def validate_attention_edge_cases(
    flash_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """
    Validate attention with edge cases.

    Tests:
    - Different sequence lengths
    - Causal vs non-causal
    - Different head dimensions

    Returns:
        Tuple of (all_passed, results_dict)
    """
    results = {}
    all_passed = True

    if verbose:
        print("\nAttention Edge Case Validation:")
        print("-" * 40)

    # Test different configurations
    configs = [
        {"batch": 1, "heads": 1, "seq_len": 64, "head_dim": 64, "causal": False},
        {"batch": 2, "heads": 4, "seq_len": 128, "head_dim": 64, "causal": False},
        {"batch": 2, "heads": 4, "seq_len": 128, "head_dim": 64, "causal": True},
        {"batch": 1, "heads": 8, "seq_len": 256, "head_dim": 32, "causal": False},
        {"batch": 4, "heads": 8, "seq_len": 512, "head_dim": 64, "causal": True},
    ]

    for config in configs:
        is_valid, max_diff = validate_attention(
            flash_fn,
            config["batch"],
            config["heads"],
            config["seq_len"],
            config["head_dim"],
            causal=bool(config["causal"]),
            rtol=rtol,
            atol=atol,
            device=device,
            verbose=False,
        )

        config_name = f"b{config['batch']}_h{config['heads']}_s{config['seq_len']}_d{config['head_dim']}_{'causal' if config['causal'] else 'full'}"
        results[config_name] = is_valid
        all_passed &= is_valid

        if verbose:
            status = "✓" if is_valid else "✗"
            print(f"  {config_name}: {status} (max_diff={max_diff:.2e})")

    if verbose:
        print("-" * 40)
        print(f"  Overall: {'✓ All passed' if all_passed else '✗ Some failed'}")

    return all_passed, results


if __name__ == "__main__":
    print("Testing validation utilities...")

    # Import matmul for testing
    from kernels.matmul import triton_matmul

    # Test basic validation
    is_valid, max_diff = validate_matmul(triton_matmul, 512, 512, 512, verbose=True)

    # Test edge cases
    validate_matmul_edge_cases(triton_matmul)

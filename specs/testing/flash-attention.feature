# BDD Testing Specification: DIY FlashAttention

## Overview

This document specifies the Behavior-Driven Development (BDD) testing approach for the DIY FlashAttention project. Tests are written using pytest and Hypothesis for property-based testing.

## Test Structure

### Feature: Matrix Multiplication

#### Scenario: Basic Matrix Multiplication

```gherkin
Feature: Matrix Multiplication Kernel
  As a developer
  I want to multiply matrices using Triton kernel
  So that I can understand Triton's programming model

  Scenario: Multiply two valid matrices
    Given matrix A with shape (M, K) and dtype float16
    And matrix B with shape (K, N) and dtype float16
    When I call triton_matmul(A, B)
    Then the result should have shape (M, N)
    And the result should match torch.matmul(A, B) within 1e-3 tolerance

  Scenario: Multiply with incompatible dimensions
    Given matrix A with shape (128, 64)
    And matrix B with shape (128, 64)
    When I call triton_matmul(A, B)
    Then a ValueError should be raised
    And the error message should mention dimension mismatch

  Scenario: Multiply with unsupported dtype
    Given matrix A with shape (128, 128) and dtype int32
    And matrix B with shape (128, 128) and dtype int32
    When I call triton_matmul(A, B)
    Then a TypeError should be raised
```

#### Scenario: Block Size Tuning

```gherkin
Feature: Block Size Parameter Tuning
  As a developer
  I want to adjust block size parameters
  So that I can optimize performance

  Scenario: Use autotune for optimal block size
    Given matrices A (512, 512) and B (512, 512)
    When I call triton_matmul(A, B) with use_autotune=True
    Then the kernel should automatically select optimal block sizes
    And the result should be numerically correct

  Scenario: Use manual block sizes
    Given matrices A (512, 512) and B (512, 512)
    When I call triton_matmul(A, B) with block_m=128, block_n=128, block_k=32
    Then the kernel should use the specified block sizes
    And the result should be numerically correct

  Scenario: Invalid block size
    Given matrix A with shape (64, 64)
    When I call triton_matmul with block_m=128
    Then a ValueError should be raised
```

### Feature: FlashAttention

#### Scenario: Basic Attention Computation

```gherkin
Feature: FlashAttention Forward Pass
  As a developer
  I want to compute attention efficiently
  So that I can process long sequences in LLMs

  Scenario: Compute attention for standard input
    Given Q, K, V tensors with shape (2, 8, 512, 64)
    When I call flash_attention(Q, K, V)
    Then the output should have shape (2, 8, 512, 64)
    And the output should match reference attention within 1e-3 tolerance

  Scenario: Compute attention with causal masking
    Given Q, K, V tensors with shape (2, 8, 512, 64)
    When I call flash_attention(Q, K, V, causal=True)
    Then the output should apply causal mask
    And future positions should be masked with -inf before softmax

  Scenario: Compute attention with variable sequence lengths
    Given Q, K, V tensors with shape (2, 8, 512, 64)
    And seq_lens tensor with shape (2,) containing [256, 512]
    When I call flash_attention(Q, K, V, seq_lens=seq_lens)
    Then each batch should use its respective sequence length
```

#### Scenario: Edge Cases

```gherkin
Feature: FlashAttention Edge Cases
  As a developer
  I want FlashAttention to handle edge cases correctly
  So that I can trust the implementation

  Scenario: Unsupported head dimension
    Given Q, K, V tensors with head_dim=128
    When I call flash_attention(Q, K, V)
    Then a ValueError should be raised
    And the error should mention supported head dimensions (32, 64)

  Scenario: 3D input tensor
    Given Q, K, V tensors with shape (16, 512, 64)
    When I call flash_attention(Q, K, V)
    Then the kernel should handle 3D input correctly
    And the output should be numerically correct

  Scenario: Mismatched Q, K, V shapes
    Given Q with shape (2, 8, 512, 64)
    And K with shape (2, 8, 256, 64)
    And V with shape (2, 8, 512, 64)
    When I call flash_attention(Q, K, V)
    Then a ValueError should be raised
```

### Feature: Performance Benchmarks

#### Scenario: Matrix Multiplication Benchmark

```gherkin
Feature: Performance Benchmarking
  As a developer
  I want to benchmark kernel performance
  So that I can quantify optimization effects

  Scenario: Benchmark matmul against PyTorch
    When I run the matmul benchmark
    Then TFLOPS should be reported for both implementations
    And a comparison table should be displayed
    And speedup ratio should be calculated

  Scenario: FlashAttention memory benchmark
    When I run the FlashAttention benchmark
    Then memory usage should be measured for different sequence lengths
    And O(N) vs O(N²) scaling should be demonstrated
```

### Feature: GPU Architecture Detection

#### Scenario: GPU Capability Detection

```gherkin
Feature: GPU Architecture Detection
  As a developer
  I want automatic GPU detection
  So that optimal kernels are selected

  Scenario: Detect Hopper GPU
    Given a Hopper (SM90) GPU
    When I call detect_gpu()
    Then GPUArch.HOPPER should be returned
    And has_tma should be True
    And has_fp8 should be True

  Scenario: Detect older GPU
    Given a Volta (SM70) GPU
    When I call detect_gpu()
    Then GPUArch.VOLTA should be returned
    And has_tma should be False
    And fallback kernels should be used
```

## Property-Based Tests

### Property 1: Matrix Multiplication Correctness

```gherkin
Feature: diy-flash-attention, Property 1: Matrix Multiplication Correctness
Validates: Requirements 1.1, 1.2, 6.1

For any matrices A (M, K) and B (K, N) with valid floating-point values:
  When triton_matmul(A, B) is computed
  Then the result should equal torch.matmul(A, B) within 1e-3 relative tolerance
```

### Property 2: Block Size Invariance

```gherkin
Feature: diy-flash-attention, Property 2: Block Size Invariance
Validates: Requirements 1.4

For any valid block size configuration and input matrices A, B:
  When triton_matmul is called with different block sizes
  Then all results should be numerically equivalent within tolerance
```

### Property 3: FlashAttention Correctness

```gherkin
Feature: diy-flash-attention, Property 3: FlashAttention Correctness
Validates: Requirements 4.1, 4.4, 6.1

For any Q, K, V tensors of compatible shapes:
  When flash_attention(Q, K, V) is computed
  Then the output should equal softmax(Q @ K^T / sqrt(d)) @ V within 1e-3 tolerance
```

### Property 4: Causal Masking Correctness

```gherkin
Feature: diy-flash-attention, Property 4: Causal Masking Correctness
Validates: Requirements 4.3

For any Q, K, V tensors with causal=True:
  When flash_attention(Q, K, V, causal=True) is computed
  Then the output should match reference with upper triangular mask set to -inf
```

### Property 5: Memory Scaling

```gherkin
Feature: diy-flash-attention, Property 5: Memory Scaling
Validates: Requirements 5.4

For any sequence length N:
  When FlashAttention memory usage is measured
  Then memory should scale as O(N), not O(N²)
  And doubling N should approximately double memory, not quadruple it
```

## Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| kernels/ | 85% |
| utils/ | 90% |
| benchmarks/ | 80% |

## Test Execution

```bash
# Run all tests
make test

# Run property tests only
pytest tests/test_properties.py -v

# Run with coverage
pytest --cov=kernels --cov=utils --cov=benchmarks --cov-report=html
```

## Test Data Generation

Tests should use Hypothesis strategies for generating random inputs:

```python
from hypothesis import given, strategies as st, settings

# Matrix dimension strategies
matrix_dims = st.integers(min_value=16, max_value=1024)
block_sizes = st.sampled_from([32, 64, 128, 256])

# Attention shape strategies
batch_sizes = st.integers(min_value=1, max_value=8)
head_counts = st.integers(min_value=1, max_value=16)
seq_lengths = st.integers(min_value=32, max_value=2048)
head_dims = st.sampled_from([32, 64])

# Test configuration
@settings(max_examples=100, deadline=None)
```

## Continuous Integration

All tests must pass in CI before merging:
- Unit tests run on CPU (skipped if CUDA unavailable)
- Property tests run with reduced examples in CI (max_examples=10)
- Full property tests run on GPU-enabled CI runners

## Purpose

Define the stable educational contract for the Triton matmul and FlashAttention kernels, their architecture-adaptation helpers, and the benchmark/correctness surface that explains and validates them.

## Requirements

### Requirement: Educational Triton matmul kernel
The project SHALL expose a Triton matrix multiplication entrypoint for learning and benchmarking tiled GPU kernels from Python.

#### Scenario: Valid CUDA matmul inputs
- **WHEN** a caller passes two compatible 2D CUDA tensors with supported floating-point dtypes to `triton_matmul`
- **THEN** the project SHALL return a tensor with shape `(M, N)` and numerically match the PyTorch reference within documented tolerance

#### Scenario: Manual block sizes override autotune
- **WHEN** a caller provides explicit block sizes to `triton_matmul`
- **THEN** the project SHALL run the manual kernel path instead of autotune and preserve the same output contract

### Requirement: Educational FlashAttention forward kernel
The project SHALL expose a forward-only FlashAttention implementation for 3D and 4D CUDA tensors so readers can study online softmax, tiling, and causal masking.

#### Scenario: Standard attention invocation
- **WHEN** a caller passes matching CUDA Q, K, and V tensors to `flash_attention`
- **THEN** the project SHALL return an output tensor matching the input layout and numerically track the reference scaled-dot-product attention result within documented tolerance

#### Scenario: Causal masking
- **WHEN** a caller enables `causal=True`
- **THEN** the project SHALL apply autoregressive masking semantics before softmax normalization

#### Scenario: Variable per-batch sequence lengths
- **WHEN** a caller supplies `seq_lens`
- **THEN** the project SHALL restrict computation to each batch element's declared sequence length instead of assuming the full context length is valid

#### Scenario: Unsupported head dimension
- **WHEN** a caller uses a head dimension outside the project's supported set
- **THEN** the project SHALL raise a descriptive error that documents the supported head dimensions

### Requirement: Architecture adaptation helpers
The project SHALL provide architecture detection and Hopper-or-newer feature helpers without breaking older CUDA environments.

#### Scenario: Modern feature detection
- **WHEN** a caller requests architecture-aware configuration or feature status
- **THEN** the project SHALL return explicit information about architecture, compute capability, and modern feature availability such as TMA or FP8 support

#### Scenario: Fallback on unsupported hardware
- **WHEN** Hopper-specific capabilities are unavailable
- **THEN** the project SHALL fall back to compatible defaults instead of failing during feature inspection

### Requirement: Benchmarks and correctness surface
The project SHALL include benchmark and test surfaces that explain performance and validate correctness for the educational kernels.

#### Scenario: Benchmark comparison
- **WHEN** a user runs the benchmark tooling
- **THEN** the project SHALL compare the custom Triton kernels against the relevant PyTorch baseline and report timing or throughput metrics

#### Scenario: Correctness validation
- **WHEN** kernel tests run in a compatible environment
- **THEN** the project SHALL verify numerical agreement against reference implementations and cover expected edge cases

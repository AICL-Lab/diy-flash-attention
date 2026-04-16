# Product Requirements Document: DIY FlashAttention

## Introduction

This project implements the FlashAttention algorithm using Python and OpenAI Triton. FlashAttention is the core attention mechanism acceleration algorithm most critical to LLMs. Starting from a basic matrix multiplication kernel, the project progressively builds a complete FlashAttention implementation and validates performance through benchmarks.

The project also includes complete engineering infrastructure:
- Numerical validation tools and benchmark framework
- GPU architecture auto-detection and adaptive configuration
- Documentation (API, tutorials, cheatsheets, FAQs) and example code
- pyproject.toml project packaging, Makefile automation, CI configuration
- Git version control with modern GPU architecture features:
  - TMA (Tensor Memory Accelerator) async data transfers
  - Warpgroup MMA instructions
  - FP8 data type support
  - Async barriers and pipeline optimization

## Glossary

| Term | Definition |
|------|------------|
| **Triton_Kernel** | GPU compute kernel written using OpenAI Triton |
| **Block_Size** | Block size parameters in GPU computing, affecting memory access efficiency and compute performance |
| **Tiling** | GPU memory blocking technique that splits large matrices into smaller chunks for computation |
| **Coalescing** | GPU memory coalesced access technique that optimizes memory bandwidth utilization |
| **TFLOPS** | Trillion floating-point operations per second, measuring compute performance |
| **FlashAttention** | A memory-efficient attention mechanism implementation that reduces HBM access through block computation |
| **Benchmark_Script** | Performance testing scripts for comparing different implementations |
| **TMA** | Tensor Memory Accelerator, hardware-accelerated async data transfer unit introduced in Hopper architecture |
| **Warpgroup_MMA** | Warpgroup-level matrix multiply-accumulate instructions in Hopper architecture |
| **FP8** | 8-bit floating-point data type for accelerating inference |
| **SDPA** | Scaled Dot-Product Attention, PyTorch's built-in attention implementation |
| **Autotune** | Triton's auto-tuning mechanism that selects optimal block size configurations |

## Requirements

### Requirement 1: Basic Matrix Multiplication Kernel

**User Story:** As a developer, I want to implement a basic Triton matrix multiplication kernel to understand Triton's programming model and block pointer arithmetic.

#### Acceptance Criteria

1. The Triton_Kernel shall compute matrix multiplication C = A @ B for two input matrices
2. When the Triton_Kernel receives matrices A (M×K) and B (K×N), the Triton_Kernel shall produce output matrix C (M×N)
3. The Triton_Kernel shall use configurable Block_Size parameters for M, N, and K dimensions
4. When Block_Size parameters are modified, the Triton_Kernel shall still produce correct results
5. The Triton_Kernel shall handle matrices where dimensions are not multiples of Block_Size
6. The Triton_Kernel shall support Autotune to automatically select optimal configurations from multiple Block_Size settings
7. The Triton_Kernel shall support float16, float32, bfloat16 input dtypes with internal float32 accumulation and float16 output

### Requirement 2: Performance Benchmark Script

**User Story:** As a developer, I want a benchmark script to compare Triton kernel performance against PyTorch native implementations to quantify optimization effects.

#### Acceptance Criteria

1. The Benchmark_Script shall compare Triton_Kernel performance against torch.matmul
2. When running benchmarks, the Benchmark_Script shall report TFLOPS for both implementations
3. The Benchmark_Script shall test multiple matrix sizes to show scaling behavior
4. The Benchmark_Script shall report execution time in milliseconds for each test case
5. When benchmark completes, the Benchmark_Script shall display a formatted comparison table

### Requirement 3: Block Size Parameter Tuning

**User Story:** As a developer, I want to adjust Block Size parameters and observe performance changes to understand key factors in GPU optimization.

#### Acceptance Criteria

1. The Triton_Kernel shall expose Block_Size as tunable parameters
2. When different Block_Size values are used, the Benchmark_Script shall report corresponding TFLOPS changes
3. The Benchmark_Script shall support testing multiple Block_Size configurations in a single run
4. If an invalid Block_Size is provided, then the Triton_Kernel shall raise a descriptive error

### Requirement 4: FlashAttention Core Implementation

**User Story:** As a developer, I want to implement FlashAttention's core algorithm to understand its memory-efficient attention computation mechanism.

#### Acceptance Criteria

1. The Triton_Kernel shall implement the forward pass of FlashAttention algorithm
2. When computing attention, the Triton_Kernel shall use online softmax to avoid materializing full attention matrix
3. The Triton_Kernel shall support causal masking for autoregressive models
4. When processing Q, K, V tensors, the Triton_Kernel shall produce numerically correct attention output
5. The Triton_Kernel shall handle variable sequence lengths within a batch via per-batch seq_lens parameter
6. The Triton_Kernel shall support both 4D (batch, heads, seq_len, head_dim) and 3D (batch*heads, seq_len, head_dim) input
7. The Triton_Kernel shall support head_dim of 32 and 64
8. The Triton_Kernel shall store log-sum-exp statistics for potential backward pass

### Requirement 5: FlashAttention Performance Validation

**User Story:** As a developer, I want to validate the correctness and performance of the FlashAttention implementation to ensure it can be used for practical LLM inference.

#### Acceptance Criteria

1. The Benchmark_Script shall compare FlashAttention against PyTorch's scaled_dot_product_attention
2. When comparing outputs, the Benchmark_Script shall verify numerical accuracy within acceptable tolerance
3. The Benchmark_Script shall measure memory usage reduction compared to naive attention
4. When sequence length increases, FlashAttention shall show better memory scaling than naive implementation
5. The Benchmark_Script shall report speedup ratio for various sequence lengths

### Requirement 6: Numerical Correctness Validation

**User Story:** As a developer, I want to ensure numerical correctness of all kernel implementations to trust computation results.

#### Acceptance Criteria

1. When comparing Triton_Kernel output with reference implementation, the difference shall be within 1e-3 relative tolerance
2. The Benchmark_Script shall include correctness checks before performance measurements
3. If numerical mismatch is detected, then the Benchmark_Script shall report detailed error information
4. The Triton_Kernel shall handle edge cases like zero matrices and identity operations correctly

### Requirement 7: Git Version Control

**User Story:** As a developer, I want to use Git to manage project code to track changes and collaborate on development.

#### Acceptance Criteria

1. The Project shall be initialized as a Git repository
2. The Project shall include a .gitignore file for Python and CUDA artifacts
3. The Project shall include a README.md with setup instructions and usage examples
4. When significant features are completed, the Project shall have meaningful commit messages

### Requirement 8: Modern CUDA Features Support

**User Story:** As a developer, I want to leverage the latest features from CUDA 13.x and Hopper/Blackwell architectures for optimal performance.

#### Acceptance Criteria

1. Where Hopper GPU is available, the Triton_Kernel shall utilize TMA for async data loading
2. Where FP8 is supported, the Triton_Kernel shall provide FP8 computation option (E4M3 and E5M2)
3. The Triton_Kernel shall auto-detect GPU architecture and select optimal code path via AdaptiveKernelSelector
4. If modern features are unavailable, then the Triton_Kernel shall fallback to compatible implementation
5. The Project shall support GPU architectures from Volta (SM70) to Blackwell (SM100)

### Requirement 9: Project Packaging and Automation

**User Story:** As a developer, I want the project to have standard Python packaging configuration and automation tools for easy installation, testing, and publishing.

#### Acceptance Criteria

1. The Project shall include pyproject.toml with project metadata, dependencies, and tool configurations
2. The Project shall include a Makefile with common development tasks (test, lint, benchmark, etc.)
3. The Project shall configure pytest, ruff, and mypy via pyproject.toml
4. The Project shall define project entry points for benchmark scripts

### Requirement 10: Documentation and Examples

**User Story:** As a developer, I want complete documentation and example code for quick onboarding and understanding implementation details.

#### Acceptance Criteria

1. The Project shall include API documentation covering all public interfaces
2. The Project shall include a tutorial for step-by-step learning
3. The Project shall include a cheatsheet for quick reference of Triton concepts
4. The Project shall include performance documentation with benchmarking guidelines
5. The Project shall include example scripts demonstrating key features (quick start, advanced usage, block size experiments, tiling visualization)
6. The Project shall include FAQ for common questions

### Requirement 11: Open Source Collaboration Standards

**User Story:** As an open source project maintainer, I want standardized collaboration processes to guide contributors in participating.

#### Acceptance Criteria

1. The Project shall include a CONTRIBUTING.md with contribution guidelines
2. The Project shall include a LICENSE file (MIT)
3. The Project shall include a CHANGELOG.md tracking project changes
4. The Project shall include GitHub CI configuration for automated testing

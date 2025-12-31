# Requirements Document

## Introduction

本项目旨在使用 Python 和 OpenAI Triton 复现 FlashAttention 算法，这是 LLM 中最核心的注意力机制加速算法。项目将从基础的矩阵乘法 Kernel 开始，逐步构建完整的 FlashAttention 实现，并通过 benchmark 验证性能。

项目将使用 Git 进行版本控制，并尽可能利用 CUDA 13.x 和现代 GPU 架构（Hopper/Blackwell）的最新特性，包括：
- TMA (Tensor Memory Accelerator) 异步数据传输
- Warpgroup MMA 指令
- FP8 数据类型支持
- 异步屏障和流水线优化

## Glossary

- **Triton_Kernel**: 使用 OpenAI Triton 编写的 GPU 计算核函数
- **Block_Size**: GPU 计算中的分块大小参数，影响内存访问效率和计算性能
- **Tiling**: GPU 内存分块技术，将大矩阵分割成小块进行计算
- **Coalescing**: GPU 内存合并访问技术，优化内存带宽利用率
- **TFLOPS**: 每秒万亿次浮点运算，衡量计算性能的指标
- **FlashAttention**: 一种内存高效的注意力机制实现，通过分块计算减少 HBM 访问
- **Benchmark_Script**: 性能测试脚本，用于对比不同实现的性能
- **TMA**: Tensor Memory Accelerator，Hopper 架构引入的硬件加速异步数据传输单元
- **Warpgroup_MMA**: Hopper 架构的 warpgroup 级矩阵乘累加指令
- **FP8**: 8位浮点数据类型，用于加速推理

## Requirements

### Requirement 1: 基础矩阵乘法 Kernel

**User Story:** 作为开发者，我想要实现一个基础的 Triton 矩阵乘法 Kernel，以便理解 Triton 的编程模型和 Block 指针运算。

#### Acceptance Criteria

1. THE Triton_Kernel SHALL compute matrix multiplication C = A @ B for two input matrices
2. WHEN the Triton_Kernel receives matrices A (M×K) and B (K×N), THE Triton_Kernel SHALL produce output matrix C (M×N)
3. THE Triton_Kernel SHALL use configurable Block_Size parameters for M, N, and K dimensions
4. WHEN Block_Size parameters are modified, THE Triton_Kernel SHALL still produce correct results
5. THE Triton_Kernel SHALL handle matrices where dimensions are not multiples of Block_Size

### Requirement 2: 性能 Benchmark 脚本

**User Story:** 作为开发者，我想要一个 benchmark 脚本来对比 Triton Kernel 和 PyTorch 原生实现的性能，以便量化优化效果。

#### Acceptance Criteria

1. THE Benchmark_Script SHALL compare Triton_Kernel performance against torch.matmul
2. WHEN running benchmarks, THE Benchmark_Script SHALL report TFLOPS for both implementations
3. THE Benchmark_Script SHALL test multiple matrix sizes to show scaling behavior
4. THE Benchmark_Script SHALL report execution time in milliseconds for each test case
5. WHEN benchmark completes, THE Benchmark_Script SHALL display a formatted comparison table

### Requirement 3: Block Size 参数调优

**User Story:** 作为开发者，我想要能够调整 Block Size 参数并观察性能变化，以便理解 GPU 优化的关键因素。

#### Acceptance Criteria

1. THE Triton_Kernel SHALL expose Block_Size as tunable parameters
2. WHEN different Block_Size values are used, THE Benchmark_Script SHALL report corresponding TFLOPS changes
3. THE Benchmark_Script SHALL support testing multiple Block_Size configurations in a single run
4. IF an invalid Block_Size is provided, THEN THE Triton_Kernel SHALL raise a descriptive error

### Requirement 4: FlashAttention 核心实现

**User Story:** 作为开发者，我想要实现 FlashAttention 的核心算法，以便理解其内存高效的注意力计算机制。

#### Acceptance Criteria

1. THE Triton_Kernel SHALL implement the forward pass of FlashAttention algorithm
2. WHEN computing attention, THE Triton_Kernel SHALL use online softmax to avoid materializing full attention matrix
3. THE Triton_Kernel SHALL support causal masking for autoregressive models
4. WHEN processing Q, K, V tensors, THE Triton_Kernel SHALL produce numerically correct attention output
5. THE Triton_Kernel SHALL handle variable sequence lengths within a batch

### Requirement 5: FlashAttention 性能验证

**User Story:** 作为开发者，我想要验证 FlashAttention 实现的正确性和性能，以便确保其可用于实际 LLM 推理。

#### Acceptance Criteria

1. THE Benchmark_Script SHALL compare FlashAttention against PyTorch's scaled_dot_product_attention
2. WHEN comparing outputs, THE Benchmark_Script SHALL verify numerical accuracy within acceptable tolerance
3. THE Benchmark_Script SHALL measure memory usage reduction compared to naive attention
4. WHEN sequence length increases, THE FlashAttention SHALL show better memory scaling than naive implementation
5. THE Benchmark_Script SHALL report speedup ratio for various sequence lengths

### Requirement 6: 数值正确性验证

**User Story:** 作为开发者，我想要确保所有 Kernel 实现的数值正确性，以便信任计算结果。

#### Acceptance Criteria

1. WHEN comparing Triton_Kernel output with reference implementation, THE difference SHALL be within 1e-3 relative tolerance
2. THE Benchmark_Script SHALL include correctness checks before performance measurements
3. IF numerical mismatch is detected, THEN THE Benchmark_Script SHALL report detailed error information
4. THE Triton_Kernel SHALL handle edge cases like zero matrices and identity operations correctly

### Requirement 7: Git 版本控制

**User Story:** 作为开发者，我想要使用 Git 管理项目代码，以便追踪变更和协作开发。

#### Acceptance Criteria

1. THE Project SHALL be initialized as a Git repository
2. THE Project SHALL include a .gitignore file for Python and CUDA artifacts
3. THE Project SHALL include a README.md with setup instructions and usage examples
4. WHEN significant features are completed, THE Project SHALL have meaningful commit messages

### Requirement 8: 现代 CUDA 特性支持（可选）

**User Story:** 作为开发者，我想要利用 CUDA 13.x 和 Hopper/Blackwell 架构的最新特性，以便获得最佳性能。

#### Acceptance Criteria

1. WHERE Hopper GPU is available, THE Triton_Kernel SHALL utilize TMA for async data loading
2. WHERE FP8 is supported, THE Triton_Kernel SHALL provide FP8 computation option
3. THE Triton_Kernel SHALL auto-detect GPU architecture and select optimal code path
4. IF modern features are unavailable, THEN THE Triton_Kernel SHALL fallback to compatible implementation

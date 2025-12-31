# Implementation Plan: DIY FlashAttention

## Overview

本实现计划从基础的 Triton 矩阵乘法开始，逐步构建到完整的 FlashAttention 实现。每个阶段都包含验证步骤，确保正确性。

## Tasks

- [x] 1. 项目初始化和环境配置
  - [x] 1.1 初始化 Git 仓库
    - 执行 git init
    - 创建 .gitignore 文件（Python, CUDA, IDE artifacts）
    - 创建初始 commit
    - _Requirements: 7.1, 7.2_

  - [x] 1.2 创建项目结构
    - 创建目录结构 (kernels/, benchmarks/, tests/, utils/)
    - 创建 requirements.txt 包含 torch, triton, hypothesis, pytest
    - 创建基础的 __init__.py 文件
    - _Requirements: 1.1, 2.1_

  - [x] 1.3 创建 README.md
    - 项目介绍和目标
    - 环境要求（Python, CUDA, GPU）
    - 安装和使用说明
    - _Requirements: 7.3_

  - [x] 1.4 实现 GPU 检测工具
    - 在 utils/gpu_detect.py 中实现 GPUCapabilities 和 detect_gpu
    - 检测 GPU 架构（Ampere/Hopper/Blackwell）
    - 检测 TMA, FP8, Warpgroup MMA 支持
    - _Requirements: 8.3_

- [x] 2. 实现基础矩阵乘法 Kernel
  - [x] 2.1 实现 Triton matmul kernel
    - 在 kernels/matmul.py 中实现 matmul_kernel
    - 实现 Block 指针运算和 tiling 逻辑
    - 支持可配置的 BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    - 实现 triton_matmul wrapper 函数
    - _Requirements: 1.1, 1.2, 1.3_

  - [x]* 2.2 编写矩阵乘法正确性属性测试
    - **Property 1: Matrix Multiplication Correctness**
    - **Validates: Requirements 1.1, 1.2, 6.1**

  - [x] 2.3 处理非对齐维度
    - 添加 masking 逻辑处理维度不是 block size 倍数的情况
    - _Requirements: 1.5_

  - [x]* 2.4 编写 Block Size 不变性属性测试
    - **Property 2: Block Size Invariance**
    - **Validates: Requirements 1.4**

- [x] 3. Checkpoint - 矩阵乘法验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. 实现 Benchmark 工具
  - [x] 4.1 实现 benchmark 工具类
    - 在 utils/benchmark.py 中实现 BenchmarkResult 和 BenchmarkRunner
    - 实现 TFLOPS 计算逻辑
    - 实现格式化输出表格
    - _Requirements: 2.2, 2.4, 2.5_

  - [x] 4.2 实现矩阵乘法 benchmark 脚本
    - 在 benchmarks/bench_matmul.py 中实现
    - 对比 Triton kernel 和 torch.matmul
    - 支持多种矩阵大小和 block size 配置
    - _Requirements: 2.1, 2.3, 3.2, 3.3_

  - [x]* 4.3 编写 benchmark 单元测试
    - 测试 TFLOPS 计算正确性
    - 测试输出格式
    - _Requirements: 2.2_

- [x] 5. 实现验证工具
  - [x] 5.1 实现 validation 工具
    - 在 utils/validation.py 中实现 validate_matmul 和 validate_attention
    - 实现数值比较逻辑（rtol=1e-3, atol=1e-3）
    - 实现详细错误报告
    - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 5.2 编写验证工具单元测试
    - 测试边界情况（零矩阵、单位矩阵）
    - _Requirements: 6.4_

- [x] 6. Checkpoint - Benchmark 和验证工具
  - 确保所有测试通过，如有问题请询问用户

- [x] 7. 实现 FlashAttention 核心算法
  - [x] 7.1 实现 online softmax 辅助函数
    - 实现 running max 和 running sum 更新逻辑
    - _Requirements: 4.2_

  - [x] 7.2 实现 FlashAttention forward kernel
    - 在 kernels/flash_attn.py 中实现 flash_attention_forward_kernel
    - 实现 Q, K, V 的分块加载
    - 实现 online softmax 计算
    - 实现增量输出累积
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 7.3 实现 causal masking
    - 在 kernel 中添加 IS_CAUSAL 参数
    - 实现上三角 mask 逻辑
    - _Requirements: 4.3_

  - [x] 7.4 实现 flash_attention wrapper 函数
    - 支持 batch 和 multi-head
    - 处理不同序列长度
    - _Requirements: 4.5_

  - [x]* 7.5 编写 FlashAttention 正确性属性测试
    - **Property 3: FlashAttention Correctness**
    - **Validates: Requirements 4.1, 4.4, 6.1**

  - [x]* 7.6 编写 Causal Masking 属性测试
    - **Property 4: Causal Masking Correctness**
    - **Validates: Requirements 4.3**

- [x] 8. Checkpoint - FlashAttention 核心功能
  - 确保所有测试通过，如有问题请询问用户

- [x] 9. 实现 FlashAttention Benchmark
  - [x] 9.1 实现 FlashAttention benchmark 脚本
    - 在 benchmarks/bench_flash.py 中实现
    - 对比 FlashAttention 和 PyTorch scaled_dot_product_attention
    - 测试不同序列长度的性能
    - _Requirements: 5.1, 5.5_

  - [x] 9.2 实现内存使用测量
    - 测量不同序列长度的内存使用
    - 验证 O(N) vs O(N²) 缩放
    - _Requirements: 5.3, 5.4_

  - [x]* 9.3 编写内存缩放属性测试
    - **Property 5: Memory Scaling**
    - **Validates: Requirements 5.4**

- [x] 10. 错误处理和边界情况
  - [x] 10.1 添加输入验证
    - 验证矩阵维度兼容性
    - 验证 block size 有效性
    - 验证 dtype 支持
    - _Requirements: 3.4_

  - [x]* 10.2 编写错误处理单元测试
    - 测试无效输入的错误消息
    - _Requirements: 3.4_

- [x] 11. Final Checkpoint - 完整功能验证
  - 运行所有测试确保通过
  - 运行 benchmark 脚本验证性能
  - 如有问题请询问用户

- [x] 12. 现代 CUDA 特性优化（可选）
  - [x]* 12.1 实现 TMA 数据加载（Hopper+）
    - 使用 tl.make_tensor_descriptor 创建 TMA 描述符
    - 实现异步数据预取
    - _Requirements: 8.1_

  - [x]* 12.2 添加 FP8 支持
    - 实现 FP8 matmul kernel 变体
    - 添加 FP8 FlashAttention 选项
    - _Requirements: 8.2_

  - [x]* 12.3 实现架构自适应
    - 根据 GPU 检测结果选择最优 kernel
    - 实现 fallback 逻辑
    - _Requirements: 8.3, 8.4_

## Notes

- 标记 `*` 的任务为可选任务，可跳过以加快 MVP 开发
- 每个任务都引用了具体的需求以便追溯
- Checkpoint 任务用于增量验证
- 属性测试验证普遍正确性属性
- 单元测试验证特定示例和边界情况
- 项目使用 Git 进行版本控制，建议在每个 Checkpoint 后提交
- 现代 CUDA 特性（Task 12）需要 Hopper 或更新的 GPU，在旧 GPU 上会自动 fallback

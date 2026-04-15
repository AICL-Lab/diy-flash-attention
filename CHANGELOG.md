# Changelog

本项目的所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增
- 📚 全面重构文档系统
  - 教程: 添加学习路径、可视化图表、更多代码示例
  - API 参考: 完善参数说明、返回值、异常、使用示例
  - 性能指南: 添加 Block Size 调优、数据类型选择、性能陷阱
  - 速查表: 优化结构，添加更多实用信息

---

## [1.0.2] - 2025-02-27

### 修复
- **pyproject.toml**: `requires-python` 从 `>=3.8` 更正为 `>=3.9`（代码使用 `tuple[int, int]` 等 3.9+ 语法）；新增 Python 3.12 classifier；mypy target version 同步更新
- **CI**: `.github/workflows/ci.yml` Python 矩阵从 `3.8-3.11` 更新为 `3.9-3.12`

### 新增
- **Makefile**: 新增 `lint`（ruff check）、`format`（ruff format）、`typecheck`（mypy）目标，满足 Requirement 9.2
- **test_error_handling.py**: 新增 `test_unsupported_dtype`（int32 输入应抛 TypeError）、`test_block_size_exceeds_dimension`（block size 超过矩阵维度应抛 ValueError）、`test_matmul_bfloat16_input`（bfloat16 输入应成功，验证 Req 1.7）
- **test_flash.py**: 新增 `TestFlashAttention3DInput` 测试类，包含 `test_3d_input_correctness` 和 `test_3d_input_causal_correctness`，验证 3D 输入的数值正确性（Req 4.6）

---

## [1.0.1] - 2025-02-27

### 变更
- 优化 spec 文档，使其与实际代码实现保持一致
  - **design.md**: 更新架构图（补充 docs/、examples/、scripts/、modern_features.py 等）；更新所有组件接口签名匹配实际代码；更新 GPUArch 枚举（7 种架构）和 GPUCapabilities 数据类（新增 name、num_sms、total_memory_gb）；新增 modern_features.py 组件文档；更新数据模型、autotune 配置和 block size 选择逻辑；更新 Error Handling 表（新增 GPU Detection 和更详细的错误条件）；更新 Testing Strategy 测试覆盖矩阵
  - **requirements.md**: 新增 Requirement 9（项目打包与自动化）、10（文档与示例）、11（开源协作规范）；细化 Requirement 1（新增 autotune 和多 dtype 支持）、4（新增变长序列、多维输入、head_dim 约束、log-sum-exp 存储）、8（去除"可选"标记，新增架构范围）；补充术语表（SDPA、Autotune）
  - **tasks.md**: 细化 Task 2.1（autotune、双 kernel、多 dtype）和 7.4（4D/3D 输入、变长序列、reference_attention）描述；新增 Task 13（项目打包与自动化）、14（文档与示例）、15（开源协作规范），均标记为已完成；更新 GPU 检测任务描述

---

## [1.0.0] - 2024-12-31

### 新增

#### 🎯 核心 Kernels

- **矩阵乘法 Kernel** (`kernels/matmul.py`)
  - 支持 autotune 自动选择最优配置
  - 支持手动指定 Block Size
  - 处理非对齐维度
  - L2 Cache 优化 (super-grouping)
  - 支持 float16, float32, bfloat16

- **FlashAttention Kernel** (`kernels/flash_attn.py`)
  - Online Softmax 算法实现
  - Causal Masking 支持
  - 多头注意力 (Multi-Head Attention)
  - Batch 处理
  - O(N) 内存复杂度
  - 支持 3D 和 4D 输入
  - 变长序列支持 (`seq_lens` 参数)

- **现代 GPU 特性** (`kernels/modern_features.py`)
  - TMA (Tensor Memory Accelerator) 检测
  - FP8 格式支持检测
  - 架构自适应 Kernel 选择
  - 自动 fallback 机制

#### 🔧 工具函数

- **GPU 检测** (`utils/gpu_detect.py`)
  - 支持 Volta/Turing/Ampere/Ada/Hopper/Blackwell 架构
  - 获取 GPU 能力信息
  - 架构特定最优配置

- **Benchmark 工具** (`utils/benchmark.py`)
  - TFLOPS 计算
  - 内存使用测量
  - 格式化输出表格
  - 一键生成报告

- **验证工具** (`utils/validation.py`)
  - 数值验证工具
  - 边界情况验证
  - 详细输出模式

#### 🧪 测试套件

- **单元测试** (`tests/`)
  - `test_matmul.py`: 矩阵乘法测试
  - `test_flash.py`: FlashAttention 测试
  - `test_benchmark.py`: Benchmark 工具测试
  - `test_validation.py`: 验证工具测试
  - `test_gpu_detect.py`: GPU 检测测试
  - `test_modern_features.py`: 现代特性测试
  - `test_error_handling.py`: 错误处理测试

- **属性测试** (`tests/test_properties.py`)
  - 使用 Hypothesis 进行属性测试
  - 矩阵乘法正确性属性
  - Block Size 不变性属性
  - FlashAttention 正确性属性
  - Causal Masking 属性
  - 内存缩放属性

#### 📊 Benchmark 脚本

- `benchmarks/bench_matmul.py`: 矩阵乘法性能测试
- `benchmarks/bench_flash.py`: FlashAttention 性能测试
- `scripts/run_all_benchmarks.py`: 一键运行所有 benchmark

#### 📚 示例代码

- `examples/quick_start.py`: 快速入门示例
- `examples/advanced_usage.py`: 高级用法示例
- `examples/block_size_experiment.py`: Block Size 实验
- `examples/visualize_tiling.py`: Tiling 可视化

#### 📖 文档

- `README.md` / `README.zh-CN.md`: 项目概述
- `docs/tutorial.md`: 教程
- `docs/api.md`: API 参考
- `docs/cheatsheet.md`: 速查表
- `docs/performance.md`: 性能指南
- `docs/faq.md`: 常见问题
- `CONTRIBUTING.md`: 贡献指南

### 技术规格

| 组件 | 版本要求 |
|------|---------|
| Python | >= 3.9 |
| CUDA | >= 11.0 |
| PyTorch | >= 2.0.0 |
| Triton | >= 2.1.0 |
| GPU 架构 | SM70+ (Volta 及更新) |

### 项目结构

```
diy-flash-attention/
├── kernels/               # Triton GPU Kernels
│   ├── matmul.py          # 矩阵乘法
│   ├── flash_attn.py      # FlashAttention
│   └── modern_features.py # 现代特性
├── utils/                 # 工具函数
│   ├── benchmark.py       # Benchmark 工具
│   ├── validation.py      # 验证工具
│   └── gpu_detect.py      # GPU 检测
├── tests/                 # 测试套件
├── benchmarks/            # 性能测试
├── examples/              # 示例代码
├── docs/                  # 文档
└── scripts/               # 脚本
```

---

## 未来计划

### [1.1.0] - 计划中

- [ ] FlashAttention Backward Pass
- [ ] TMA 数据加载完整实现 (Hopper+)
- [ ] FP8 矩阵乘法 Kernel
- [ ] 更多 Benchmark 场景
- [ ] 性能分析工具集成

### [1.2.0] - 计划中

- [ ] Multi-Query Attention (MQA) 支持
- [ ] Grouped-Query Attention (GQA) 支持
- [ ] 可变序列长度优化
- [ ] Flash-Decoding 实现

---

## 版本说明

- **主版本号 (Major)**: 不兼容的 API 变更
- **次版本号 (Minor)**: 向后兼容的功能新增
- **修订号 (Patch)**: 向后兼容的问题修复

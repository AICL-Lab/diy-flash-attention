# Changelog

本项目的所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [1.0.2] - 2025-02-27

### 修复
- **pyproject.toml**: `requires-python` 从 `>=3.8` 更正为 `>=3.9`（代码使用 `tuple[int, int]` 等 3.9+ 语法）；新增 Python 3.12 classifier；mypy target version 同步更新
- **CI**: `.github/workflows/ci.yml` Python 矩阵从 `3.8-3.11` 更新为 `3.9-3.12`

### 新增
- **Makefile**: 新增 `lint`（ruff check）、`format`（ruff format）、`typecheck`（mypy）目标，满足 Requirement 9.2
- **test_error_handling.py**: 新增 `test_unsupported_dtype`（int32 输入应抛 TypeError）、`test_block_size_exceeds_dimension`（block size 超过矩阵维度应抛 ValueError）、`test_matmul_bfloat16_input`（bfloat16 输入应成功，验证 Req 1.7）
- **test_flash.py**: 新增 `TestFlashAttention3DInput` 测试类，包含 `test_3d_input_correctness` 和 `test_3d_input_causal_correctness`，验证 3D 输入的数值正确性（Req 4.6）

## [1.0.1] - 2025-02-27

### 变更
- 优化 spec 文档，使其与实际代码实现保持一致
  - **design.md**: 更新架构图（补充 docs/、examples/、scripts/、modern_features.py 等）；更新所有组件接口签名匹配实际代码；更新 GPUArch 枚举（7 种架构）和 GPUCapabilities 数据类（新增 name、num_sms、total_memory_gb）；新增 modern_features.py 组件文档；更新数据模型、autotune 配置和 block size 选择逻辑；更新 Error Handling 表（新增 GPU Detection 和更详细的错误条件）；更新 Testing Strategy 测试覆盖矩阵
  - **requirements.md**: 新增 Requirement 9（项目打包与自动化）、10（文档与示例）、11（开源协作规范）；细化 Requirement 1（新增 autotune 和多 dtype 支持）、4（新增变长序列、多维输入、head_dim 约束、log-sum-exp 存储）、8（去除"可选"标记，新增架构范围）；补充术语表（SDPA、Autotune）
  - **tasks.md**: 细化 Task 2.1（autotune、双 kernel、多 dtype）和 7.4（4D/3D 输入、变长序列、reference_attention）描述；新增 Task 13（项目打包与自动化）、14（文档与示例）、15（开源协作规范），均标记为已完成；更新 GPU 检测任务描述

## [1.0.0] - 2024-12-31

### 新增
- 🎉 初始版本发布
- ✨ Triton 矩阵乘法 Kernel
  - 支持 autotune 自动选择最优配置
  - 支持手动指定 Block Size
  - 处理非对齐维度
  - L2 Cache 优化 (super-grouping)
- ✨ FlashAttention 实现
  - Online Softmax 算法
  - Causal Masking 支持
  - 多头注意力 (Multi-Head Attention)
  - Batch 处理
  - O(N) 内存复杂度
- 📊 Benchmark 工具
  - TFLOPS 计算
  - 内存使用测量
  - 格式化输出表格
  - 一键生成报告
- 🔧 工具函数
  - GPU 架构检测 (Volta/Turing/Ampere/Ada/Hopper/Blackwell)
  - 数值验证工具
  - 边界情况验证
- 🧪 完整测试套件
  - 单元测试 (pytest)
  - 属性测试 (Hypothesis)
  - 错误处理测试
  - Benchmark 工具测试
- 🚀 现代 CUDA 特性支持
  - TMA 数据加载框架 (Hopper+)
  - FP8 格式支持检测
  - 架构自适应 Kernel 选择
  - 自动 fallback 机制
- 📚 文档
  - 中文教程
  - API 参考文档
  - 贡献指南
  - 示例代码

### 技术细节
- 支持 CUDA 11.0+
- 支持 PyTorch 2.0+
- 支持 Triton 2.1+
- GPU 架构: SM70+ (Volta 及更新)
- 测试框架: pytest + hypothesis

### 项目结构
```
kernels/           - Triton GPU Kernels
├── matmul.py      - 矩阵乘法
├── flash_attn.py  - FlashAttention
└── modern_features.py - 现代特性

utils/             - 工具函数
├── benchmark.py   - Benchmark 工具
├── validation.py  - 验证工具
└── gpu_detect.py  - GPU 检测

tests/             - 测试套件
├── test_matmul.py
├── test_flash.py
├── test_properties.py
├── test_benchmark.py
├── test_validation.py
└── test_error_handling.py

examples/          - 示例代码
├── quick_start.py
├── advanced_usage.py
├── block_size_experiment.py
└── visualize_tiling.py

docs/              - 文档
├── tutorial.md
└── api.md
```

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
- [ ] 可变序列长度支持
- [ ] Flash-Decoding 优化

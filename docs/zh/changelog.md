# 变更日志

项目各版本的重要变更记录。完整历史见 [`CHANGELOG.md`](https://github.com/LessUp/diy-flash-attention/blob/master/CHANGELOG.md)。

## [1.0.3] - 2026-04-16

### 修复
- **关键修复**: GitHub Actions CI workflow YAML 语法错误
  - 将 `cache` 参数移入 `with:` 块内
  - CI 因 YAML 结构无效导致 0 个 job 执行

### 变更
- 文档全面审查，确保 changelog 和规格文档一致性

## [1.0.2] - 2025-02-27

### 修复
- `pyproject.toml` — `requires-python` 从 `>=3.8` 更正为 `>=3.9`（代码使用 3.9+ 语法）
- CI Python 矩阵从 `3.8-3.11` 更新为 `3.9-3.12`

### 新增
- Makefile 新增 `lint`（ruff check）、`format`（ruff format）、`typecheck`（mypy）目标
- 错误处理测试：`test_unsupported_dtype`、`test_block_size_exceeds_dimension`、`test_matmul_bfloat16_input`
- FlashAttention 3D 输入测试：`test_3d_input_correctness`、`test_3d_input_causal_correctness`

## [1.0.1] - 2025-02-27

### 变更
- 优化 spec 文档，使其与实际代码实现保持一致
  - `design.md` — 更新架构图、组件接口签名、GPUArch 枚举、modern_features.py 文档
  - `requirements.md` — 新增 Requirement 9-11（打包/文档/协作），细化已有需求
  - `tasks.md` — 新增 Task 13-15，细化 autotune、变长序列等描述

## [1.0.0] - 2024-12-31

### 新增
- Triton 矩阵乘法 Kernel — autotune、手动 Block Size、非对齐维度、L2 Cache 优化
- FlashAttention 实现 — Online Softmax、Causal Masking、Multi-Head Attention、O(N) 内存
- Benchmark 工具 — TFLOPS 计算、内存测量、格式化输出、一键报告
- GPU 架构检测 — Volta / Turing / Ampere / Ada / Hopper / Blackwell
- 完整测试套件 — pytest 单元测试 + Hypothesis 属性测试
- 现代 CUDA 特性 — TMA 框架、FP8 检测、架构自适应 Kernel 选择
- 中文文档 — 教程、API 参考、贡献指南、示例代码

## 未来计划

### [1.1.0] - 计划中
- FlashAttention Backward Pass
- TMA 数据加载完整实现 (Hopper+)
- FP8 矩阵乘法 Kernel

### [1.2.0] - 计划中
- Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)
- Flash-Decoding 优化

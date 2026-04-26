# 更新日志

本项目所有重大变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更
- OpenSpec 治理、文档瘦身与 archive-ready 收尾工作仍在进行中。

## [1.0.3] - 2026-04-16

### 修复
- 🐛 **严重**: GitHub Actions CI 工作流 YAML 语法错误
  - 将 `cache` 参数移至 `with:` 块内
  - CI 因无效 YAML 结构导致 0 个 job 执行
  - 影响: lint, test-cpu, docs 任务

### 变更
- 📝 **文档**: 全面的 changelog 和规格文档审查
  - 验证所有 6 个 changelog 文件的一致性
  - 审查 OpenSpec / 文档与实现的同步
  - 验证 VitePress 配置的完整性

### 基础设施
- ✅ CI/CD 管道现在正常运行
- ✅ YAML 语法已验证

---

## [1.0.2] - 2025-02-27

### 修复
- 🔧 **pyproject.toml**: 将 `requires-python` 从 `>=3.8` 更正为 `>=3.9`
  - 代码使用 `tuple[int, int]` 语法，需要 Python 3.9+
  - 防止在不兼容的 Python 版本上安装
- 🔧 **pyproject.toml**: 添加 Python 3.12 分类器
- 🔧 **pyproject.toml**: 更新 mypy 目标版本至 3.9
- 🔧 **CI**: 更新 Python 矩阵从 `3.8-3.11` 至 `3.9-3.12`

### 新增
- ✨ **Makefile**: 新增代码质量目标
  - `lint`: 运行 ruff 检查
  - `format`: 运行 ruff 格式化
  - `typecheck`: 运行 mypy 类型检查
- ✨ **tests/test_error_handling.py**: 新增测试用例
  - `test_unsupported_dtype`: 验证 int32 输入引发 TypeError
  - `test_block_size_exceeds_dimension`: 验证块大小超限引发 ValueError
  - `test_matmul_bfloat16_input`: 验证 bfloat16 数据类型支持
- ✨ **tests/test_flash.py**: 3D 输入验证
  - `test_3d_input_correctness`: 3D 张量数值精度
  - `test_3d_input_causal_correctness`: 3D 因果掩码精度

### 测试
- 覆盖率: kernels/ 85%+, utils/ 90%+
- 所有错误处理路径已验证

---

## [1.0.1] - 2025-02-27

### 变更
- 📝 **规格文档**: 更新 design.md、requirements.md、tasks.md
  - 架构图: 添加 docs/、examples/、scripts/、modern_features.py
  - GPUArch 枚举: 扩展至 7 种架构（Volta → Blackwell）
  - GPUCapabilities: 添加 name、num_sms、total_memory_gb 字段
  - 需求: 添加 Req 9-11 用于打包、文档、协作
  - 任务: 添加 Task 13-15 用于项目自动化

### 文档
- 规格文档现与代码库完全同步
- 从需求到实现的完整可追溯性

---

## [1.0.0] - 2024-12-31

### 新增

#### 核心 Kernel

- 🚀 **矩阵乘法 Kernel** (`kernels/matmul.py`)
  - Autotune 自动最优块大小选择
  - 手动块大小指定支持
  - 非对齐维度处理
  - L2 Cache 优化（super-grouping）
  - 多数据类型支持: float16、float32、bfloat16

- 🚀 **FlashAttention Kernel** (`kernels/flash_attn.py`)
  - Online Softmax 算法，O(N) 内存复杂度
  - 因果掩码用于自回归模型
  - 多头注意力支持
  - 批处理能力
  - 3D 和 4D 输入张量支持
  - 变长序列通过 `seq_lens` 参数

- 🔮 **现代 GPU 特性** (`kernels/modern_features.py`)
  - TMA (Tensor Memory Accelerator) 检测
  - FP8 格式支持检测
  - 架构自适应 kernel 选择
  - 自动回退机制

#### 工具函数

- 🔍 **GPU 检测** (`utils/gpu_detect.py`)
  - 架构检测: Volta、Turing、Ampere、Ada、Hopper、Blackwell
  - 能力信息获取
  - 架构特定最优配置

- 📊 **Benchmark 工具** (`utils/benchmark.py`)
  - TFLOPS 计算与格式化输出
  - 内存使用测量
  - 一键报告生成

- ✅ **验证工具** (`utils/validation.py`)
  - 数值正确性验证
  - 边界情况验证
  - 详细输出模式

#### 测试

- 🧪 **完整测试套件** (`tests/`)
  - 所有模块的单元测试
  - Hypothesis 属性测试
  - 错误处理测试
  - 50+ 单元测试，覆盖率 85%+

#### 文档

- 📚 **双语文档**
  - README.md / README.zh-CN.md
  - VitePress 文档站点
  - 教程、API 参考、性能指南、FAQ、速查表

#### CI/CD

- 🔄 GitHub Actions 工作流
  - CI: 代码检查、测试、类型检查
  - Pages: 文档部署

### 技术规格

| 组件 | 要求 |
|------|------|
| Python | >= 3.9 |
| CUDA | >= 11.0 |
| PyTorch | >= 2.0.0 |
| Triton | >= 2.1.0 |
| GPU | SM70+ (Volta 或更新) |

### 性能亮点

| 指标 | 数值 |
|------|------|
| 内存节省 | 最高 99% |
| 相比 PyTorch 加速 | 1.1-1.6x |
| 最大序列长度 | 8192+ |

---

## 版本历史摘要

| 版本 | 日期 | 类型 | 亮点 |
|------|------|------|------|
| 1.0.3 | 2026-04-16 | 补丁 | CI 修复、文档审查 |
| 1.0.2 | 2025-02-27 | 补丁 | Python 3.9+ 修复、新测试 |
| 1.0.1 | 2025-02-27 | 补丁 | 规格同步 |
| 1.0.0 | 2024-12-31 | 次要 | 初始发布 |

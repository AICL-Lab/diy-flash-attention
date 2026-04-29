# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-04-29

### Added
- 🚀 **FlashAttention V2**: 行级并行（条纹并行）实现，Ampere+ 上性能提升 5-15%
  - `flash_attention_v2()` 函数，支持与 V1 相同的参数
  - 更优的内存访问模式，适合大序列长度
- 🚀 **Persistent Kernels**: 持久化线程块内核实现
  - `persistent_matmul()` 函数，grid-stride loop 模式
  - 减少小矩阵的内核启动开销
- 🚀 **Mask DSL**: 块级注意力掩码抽象
  - `BlockMask` 类，支持 causal/full/sliding_window/prefix_lm 模式
  - 掩码组合（intersect/union）操作
- 🚀 **Backend Selector**: 统一内核调度注册表
  - `BackendSelector` 类，自动选择 V1/V2
  - 基于问题规模和 GPU 能力的启发式选择
- 🚀 **Profiling Toolkit**: GPU 内存分析工具
  - `GPUMemoryProfile` 类，occupancy 估算
  - Ampere/Ada/Hopper 内存层级规格

### Changed
- 📝 **CLAUDE.md**: 更新模块参考，添加 V2/persistent/mask DSL 文档

### Removed
- 🗑️ 删除冗余的 `changelog/` 目录（与 CHANGELOG.md 重复）
- 🗑️ 删除子目录中冗余的 `CLAUDE.md` 文件

---

## [1.0.4] - 2026-04-27

### Changed
- 🔧 **Python 版本**: 升级最低支持版本至 Python 3.10（3.9 已 EOL）
- 🔧 **pre-commit**: 更新 ruff 至 v0.11.7，pre-commit-hooks 至 v5.0.0

### Removed
- 🗑️ 删除冗余的 `dev/` 目录（5 个低价值临时文档）
- 🗑️ 删除冗余的 `changelog/archive/` 目录（与 CHANGELOG.md 重复）
- 🗑️ 删除 `requirements.txt`（统一使用 pyproject.toml）
- 🗑️ 删除 `.githooks/` 目录（统一使用 pre-commit 框架）

### Infrastructure
- ✅ OpenSpec change 状态清理（归档已完成的 change）
- ✅ Git Hook 配置统一为 pre-commit 框架
- ✅ 项目达到 archive-ready 状态

---

## [1.0.3] - 2026-04-16

### Fixed
- 🐛 **Critical**: GitHub Actions CI workflow YAML syntax error
  - Moved `cache` parameter inside `with:` block for `setup-python` and `setup-node` actions
  - CI was failing with 0 jobs due to invalid YAML structure
  - Affected: lint, test-cpu, and docs jobs

### Changed
- 📝 **Documentation**: Comprehensive changelog and spec documentation review
  - Validated all 6 changelog files for consistency
  - Reviewed OpenSpec/docs alignment with implementation
  - Verified VitePress configuration completeness

### Infrastructure
- ✅ CI/CD pipeline now runs successfully
- ✅ YAML syntax validated

---

## [1.0.2] - 2025-02-27

### Fixed
- 🔧 **pyproject.toml**: Corrected `requires-python` from `>=3.8` to `>=3.9`
  - Code uses `tuple[int, int]` syntax requiring Python 3.9+
  - Prevents installation on incompatible Python versions
- 🔧 **pyproject.toml**: Added Python 3.12 classifier
- 🔧 **pyproject.toml**: Updated mypy target version to 3.9
- 🔧 **CI**: Updated Python matrix from `3.8-3.11` to `3.9-3.12`

### Added
- ✨ **Makefile**: New targets for code quality
  - `lint`: Run ruff check
  - `format`: Run ruff format
  - `typecheck`: Run mypy
- ✨ **tests/test_error_handling.py**: New test cases
  - `test_unsupported_dtype`: Validates int32 input raises TypeError
  - `test_block_size_exceeds_dimension`: Validates oversized blocks raise ValueError
  - `test_matmul_bfloat16_input`: Validates bfloat16 dtype support
- ✨ **tests/test_flash.py**: 3D input validation
  - `test_3d_input_correctness`: Numerical accuracy for 3D tensors
  - `test_3d_input_causal_correctness`: 3D causal masking accuracy

### Testing
- Coverage: kernels/ 85%+, utils/ 90%+
- All error handling paths validated

---

## [1.0.1] - 2025-02-27

### Changed
- 📝 **Specifications**: Updated design.md, requirements.md, tasks.md
  - Architecture diagram: Added docs/, examples/, scripts/, modern_features.py
  - GPUArch enum: Expanded to 7 architectures (Volta → Blackwell)
  - GPUCapabilities: Added name, num_sms, total_memory_gb fields
  - Requirements: Added Req 9-11 for packaging, docs, collaboration
  - Tasks: Added Task 13-15 for project automation

### Documentation
- Spec documentation now fully synchronized with codebase
- Complete traceability from requirements to implementation

---

## [1.0.0] - 2024-12-31

### Added

#### Core Kernels

- 🚀 **Matrix Multiplication Kernel** (`kernels/matmul.py`)
  - Autotune for automatic optimal block size selection
  - Manual block size specification support
  - Non-aligned dimension handling
  - L2 Cache optimization via super-grouping
  - Multi-dtype support: float16, float32, bfloat16

- 🚀 **FlashAttention Kernel** (`kernels/flash_attn.py`)
  - Online Softmax algorithm for O(N) memory complexity
  - Causal masking for autoregressive models
  - Multi-head attention support
  - Batch processing capabilities
  - 3D and 4D input tensor support
  - Variable sequence length via `seq_lens` parameter

- 🔮 **Modern GPU Features** (`kernels/modern_features.py`)
  - TMA (Tensor Memory Accelerator) detection
  - FP8 format support detection
  - Architecture-adaptive kernel selection
  - Automatic fallback mechanism

#### Utilities

- 🔍 **GPU Detection** (`utils/gpu_detect.py`)
  - Architecture detection: Volta, Turing, Ampere, Ada, Hopper, Blackwell
  - Capability information retrieval
  - Architecture-specific optimal configurations

- 📊 **Benchmark Tools** (`utils/benchmark.py`)
  - TFLOPS calculation with formatted output
  - Memory usage measurement
  - One-click report generation

- ✅ **Validation Tools** (`utils/validation.py`)
  - Numerical correctness validation
  - Edge case validation
  - Verbose output mode

#### Testing

- 🧪 **Complete Test Suite** (`tests/`)
  - Unit tests for all modules
  - Property-based tests with Hypothesis
  - Error handling tests
  - 50+ unit tests with 85%+ coverage

#### Documentation

- 📚 **Bilingual Documentation**
  - README.md / README.zh-CN.md
  - VitePress documentation site
  - Tutorial, API Reference, Performance Guide, FAQ, Cheatsheet

#### CI/CD

- 🔄 GitHub Actions workflows
  - CI: Lint, test, typecheck
  - Pages: Documentation deployment

### Technical Specifications

| Component | Requirement |
|-----------|-------------|
| Python | >= 3.9 |
| CUDA | >= 11.0 |
| PyTorch | >= 2.0.0 |
| Triton | >= 2.1.0 |
| GPU | SM70+ (Volta or newer) |

### Performance Highlights

| Metric | Value |
|--------|-------|
| Memory Reduction | Up to 99% |
| Speedup vs PyTorch | 1.1-1.6x |
| Max Sequence Length | 8192+ |

---

## Version History Summary

| Version | Date | Type | Highlights |
|---------|------|------|------------|
| 1.0.3 | 2026-04-16 | Patch | CI fix, docs review |
| 1.0.2 | 2025-02-27 | Patch | Python 3.9+ fix, new tests |
| 1.0.1 | 2025-02-27 | Patch | Spec sync |
| 1.0.0 | 2024-12-31 | Minor | Initial release |

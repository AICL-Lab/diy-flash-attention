# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 🌍 **Internationalization**: Complete bilingual documentation (English/Chinese)
  - English documentation in `docs/en/`
  - Chinese documentation in `docs/zh/`
  - Language switcher in VitePress config
- 📚 **Enhanced Documentation**: Comprehensive professional documentation overhaul

## [1.0.3] - 2026-04-16

### Fixed
- 🐛 **Critical**: GitHub Actions CI workflow YAML syntax error
  - Moved `cache` parameter inside `with:` block for `setup-python` and `setup-node` actions
  - CI was failing with 0 jobs due to invalid YAML structure
  - Affected: lint, test-cpu, and docs jobs

### Changed
- 📝 **Documentation**: Comprehensive changelog and spec documentation review
  - Validated all 6 changelog files for consistency
  - Reviewed `.kiro/specs/` alignment with implementation
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

---

## Roadmap

### [1.1.0] - Planned

- [ ] FlashAttention Backward Pass
- [ ] TMA (Tensor Memory Accelerator) full implementation for Hopper+
- [ ] FP8 matrix multiplication kernel
- [ ] Extended benchmark scenarios with profiling integration

### [1.2.0] - Planned

- [ ] Multi-Query Attention (MQA) support
- [ ] Grouped-Query Attention (GQA) support
- [ ] Variable sequence length optimizations
- [ ] Flash-Decoding implementation

---

## Versioning Policy

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible new features
- **PATCH**: Backwards-compatible bug fixes

---

## Archives

Detailed changelogs for each release are archived in [`archive/`](./archive/).

# Changelog

Complete project changelog. For latest updates, see [CHANGELOG.md](https://github.com/LessUp/diy-flash-attention/blob/master/CHANGELOG.md).

---

## [1.0.3] - 2026-04-16

### Fixed
- **Critical**: GitHub Actions CI workflow YAML syntax error
  - Moved `cache` parameter inside `with:` block for `setup-python` and `setup-node` actions
  - CI was failing with 0 jobs due to invalid YAML structure
  - Affected: lint, test-cpu, and docs jobs

### Changed
- **Documentation**: Comprehensive changelog and spec documentation review
  - Validated all 6 changelog files for consistency
  - Reviewed .kiro specifications alignment with implementation
  - Verified VitePress configuration completeness

---

## [1.0.2] - 2025-02-27

### Fixed
- **pyproject.toml**: Corrected `requires-python` from `>=3.8` to `>=3.9` (code uses `tuple[int, int]` etc. requiring 3.9+)
- **pyproject.toml**: Added Python 3.12 classifier
- **pyproject.toml**: Updated mypy target version to 3.9
- **CI**: Updated Python matrix from `3.8-3.11` to `3.9-3.12` in `.github/workflows/ci.yml`

### Added
- **Makefile**: Added `lint` (ruff check), `format` (ruff format), `typecheck` (mypy) targets
- **tests/test_error_handling.py**: `test_unsupported_dtype` for int32 input validation
- **tests/test_error_handling.py**: `test_block_size_exceeds_dimension` for block size validation
- **tests/test_error_handling.py**: `test_matmul_bfloat16_input` for bfloat16 support
- **tests/test_flash.py**: `TestFlashAttention3DInput` class with correctness tests

---

## [1.0.1] - 2025-02-27

### Changed
- **specs**: Updated design.md, requirements.md, tasks.md to match implementation
  - Architecture diagram: Added docs/, examples/, scripts/, modern_features.py
  - GPUArch enum: 7 architectures (Volta through Blackwell)
  - GPUCapabilities: Added name, num_sms, total_memory_gb fields
  - Requirements: Added Req 9-11 for packaging, docs, collaboration
  - Tasks: Added Task 13-15 for project automation

---

## [1.0.0] - 2024-12-31

### Added

#### Core Kernels

- **Matrix Multiplication Kernel** (`kernels/matmul.py`)
  - Autotune for automatic optimal configuration
  - Manual block size specification
  - Non-aligned dimension handling
  - L2 Cache optimization (super-grouping)
  - Multi-dtype: float16, float32, bfloat16

- **FlashAttention Kernel** (`kernels/flash_attn.py`)
  - Online Softmax algorithm for O(N) memory
  - Causal masking for autoregressive models
  - Multi-head attention support
  - Batch processing
  - 3D and 4D input support
  - Variable sequence length via `seq_lens`

- **Modern GPU Features** (`kernels/modern_features.py`)
  - TMA (Tensor Memory Accelerator) detection
  - FP8 format support detection
  - Architecture-adaptive kernel selection

#### Utilities

- **GPU Detection** (`utils/gpu_detect.py`)
  - Architecture: Volta → Blackwell
  - Capability information
  - Optimal configurations per architecture

- **Benchmark Tools** (`utils/benchmark.py`)
  - TFLOPS calculation
  - Memory measurement
  - Formatted output tables
  - One-click reports

- **Validation Tools** (`utils/validation.py`)
  - Numerical correctness
  - Edge case validation

#### Testing

- **Unit Tests** (`tests/`)
  - 50+ unit tests across all modules
  - Property-based tests with Hypothesis
  - Edge case coverage

#### Documentation

- **VitePress Docs** (`docs/`)
  - Tutorial: Step-by-step learning
  - API Reference: Complete documentation
  - Performance Guide: Optimization tips
  - FAQ: Common questions
  - Cheatsheet: Quick reference

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.3 | 2026-04-16 | CI fix, docs review |
| 1.0.2 | 2025-02-27 | Python 3.9+ fix, new tests |
| 1.0.1 | 2025-02-27 | Spec sync |
| 1.0.0 | 2024-12-31 | Initial release |

---

## Roadmap

### [1.1.0] - Planned

- [ ] FlashAttention Backward Pass
- [ ] TMA implementation for Hopper+
- [ ] FP8 matrix multiplication kernel
- [ ] Extended benchmark scenarios

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

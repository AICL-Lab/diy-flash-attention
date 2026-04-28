# Tasks: Kernel Modernization Implementation (Phase A)

## Overview

This document defines all tasks for Phase A implementation of kernel modernization, structured for **test-driven development (TDD)**. Each task follows the pattern:
1. **RED**: Write failing test first
2. **VERIFY RED**: Watch test fail correctly
3. **GREEN**: Implement minimal code to pass
4. **VERIFY GREEN**: All tests pass
5. **REFACTOR**: Clean up (optional, but stay green)

**Total estimated effort**: ~30-40 implementation tasks across 5 capabilities.

---

## A.1: FlashAttention V2 Kernel Implementation

### A.1.1: Core Flash Attention V2 Kernel

**Objective**: Implement row-wise (striped) attention kernel with host descriptors.

**Prerequisites**: Triton 2.1+, CUDA toolchain, existing flash_attn v1 for reference.

**Tasks**:

#### RED: test_flash_attention_v2_basic
- Write test in `tests/test_flash_v2.py`
- Fixture: Standard input (batch=1, seq_len=128, heads=8, head_dim=64, dtype=float16)
- Assertion: `flash_attention_v2(q, k, v) == sdpa_baseline` (within rtol=1e-5)
- Expected result: **FAIL** (kernel not implemented)

#### GREEN: Implement flash_attention_v2 kernel
- Create `kernels/flash_attn_v2.py` (new file)
- Implement row-wise parallelism Triton kernel
- Register with `kernels/__init__.py`
- Minimal implementation: direct row-wise softmax, no warp specialization yet
- Expected result: **PASS**

#### RED: test_flash_attention_v2_causal
- Test: `flash_attention_v2(q, k, v, causal=True)` matches causal mask behavior
- Baseline: Compare with v1 causal output
- Expected result: **FAIL**

#### GREEN: Add causal masking to v2
- Implement causal mask in v2 kernel (set future tokens to -inf)
- Expected result: **PASS**

#### RED: test_flash_attention_v2_dtypes
- Parametrized test: float16, bfloat16, float32 inputs
- Check output dtype matches input dtype
- Expected result: **FAIL** (dtype handling incomplete)

#### GREEN: Add dtype handling
- Ensure v2 converts non-float16 → float16 for kernel, output restores original dtype
- Expected result: **PASS**

#### RED: test_flash_attention_v2_seq_lens
- Test variable sequence lengths with `seq_lens` mask
- Baseline: Compare with v1 seq_lens output
- Expected result: **FAIL**

#### GREEN: Implement seq_lens handling
- Apply per-sequence length masking in v2
- Expected result: **PASS**

### A.1.2: V2 Warp Specialization (Teaching Feature)

#### RED: test_flash_attention_v2_warp_specialize
- Test: `flash_attention_v2(..., warp_specialize=True)` produces correct output
- Expected result: **FAIL** (parameter not implemented)

#### GREEN: Add warp_specialize flag
- Minimal implementation: flag accepted but optional in Phase A (can be no-op)
- Expected result: **PASS**

### A.1.3: V2 Performance Baseline

#### RED: test_flash_attention_v2_benchmark
- Create benchmark script: `benchmarks/bench_flash_v2.py`
- Measure v1 vs v2 latency on standard sizes (seq_len=1024, batch=4)
- Expected result: **FAIL** (no baseline data)

#### GREEN: Run and record benchmark
- Execute benchmark; document v2 performance relative to v1
- Expected result: **PASS** (with performance data)

---

## A.2: Persistent Kernel Implementation

### A.2.1: Persistent Matmul Kernel

#### RED: test_persistent_matmul_basic
- Write test: `persistent_matmul(a, b)` matches `torch.mm()` output
- Shapes: (512, 512) @ (512, 512)
- Expected result: **FAIL**

#### GREEN: Implement persistent matmul
- Create `kernels/persistent_kernels.py`
- Implement persistent thread-block matmul (Triton)
- Expected result: **PASS**

#### RED: test_persistent_matmul_large_shapes
- Test larger shapes: (2048, 2048) @ (2048, 2048)
- Verify correctness and no OOM
- Expected result: **FAIL**

#### GREEN: Ensure scalability
- Tune block sizes for large shapes
- Expected result: **PASS**

#### RED: test_persistent_matmul_dtypes
- Parametrized: float16, float32, bfloat16
- Expected result: **FAIL**

#### GREEN: Add dtype support
- Expected result: **PASS**

### A.2.2: Persistent Attention Kernel

#### RED: test_persistent_attention_basic
- Test: `persistent_attention(q, k, v)` matches attention baseline
- Expected result: **FAIL**

#### GREEN: Implement persistent attention
- Create persistent attention kernel (row-wise, all heads in one pass)
- Expected result: **PASS**

#### RED: test_persistent_attention_causal
- Test causal masking
- Expected result: **FAIL**

#### GREEN: Add causal mask
- Expected result: **PASS**

### A.2.3: Profiling Integration (Deferred to A.4)

---

## A.3: Backend Selector Implementation

### A.3.1: Selector Core

#### RED: test_backend_selector_select_attention_v1
- Test: `select_attention_kernel(variant="v1")` returns v1 kernel
- Expected result: **FAIL**

#### GREEN: Implement selector
- Create `kernels/backend_selector.py`
- Implement `BackendSelector.select_attention()`
- Route to correct kernel based on variant
- Expected result: **PASS**

#### RED: test_backend_selector_select_attention_v2
- Test: `select_attention_kernel(variant="v2")` returns v2 kernel
- Expected result: **FAIL**

#### GREEN: Add v2 routing
- Expected result: **PASS**

#### RED: test_backend_selector_select_attention_persistent
- Test: persistent variant
- Expected result: **FAIL**

#### GREEN: Add persistent routing
- Expected result: **PASS**

### A.3.2: Auto Heuristics

#### RED: test_backend_selector_auto_small_problem
- Test: `select_attention_kernel(variant="auto", q_shape=(1, 256, 8, 64))` selects V1
- Expected result: **FAIL** (heuristics not implemented)

#### GREEN: Implement heuristics
- Small batch/seq → V1
- Large batch + long seq → V2 (on Ampere+)
- Expected result: **PASS**

#### RED: test_backend_selector_gpu_capability
- Test: Selector respects GPU capability (e.g., no V2 on Volta)
- Expected result: **FAIL** (capability check incomplete)

#### GREEN: Add GPU capability check
- Query `detect_gpu()`, error if variant unavailable
- Expected result: **PASS**

### A.3.3: Matmul Selector

#### RED: test_backend_selector_select_matmul_v1
- Test: matmul selector returns v1 by default
- Expected result: **FAIL**

#### GREEN: Implement matmul selector
- Expected result: **PASS**

#### RED: test_backend_selector_select_matmul_persistent
- Test: persistent matmul option available
- Expected result: **FAIL**

#### GREEN: Add persistent matmul routing
- Expected result: **PASS**

### A.3.4: Public API

#### RED: test_backend_selector_public_api
- Test: `kernels.flash_attention(..., variant="v2")` uses v2 kernel
- Expected result: **FAIL** (parameter not plumbed through)

#### GREEN: Update kernels/__init__.py
- Add `variant=` parameter to public API
- Use selector internally
- Expected result: **PASS**

---

## A.4: Profiling Toolkit Implementation

### A.4.1: GPU Memory Profile Data Structure

#### RED: test_profiling_gpu_memory_profile
- Test: `GPUMemoryProfile` dataclass can be instantiated with typical metrics
- Expected result: **FAIL** (dataclass not defined)

#### GREEN: Define GPUMemoryProfile
- Create `utils/profiling.py`
- Add `@dataclass GPUMemoryProfile` with occupancy, SMEM, registers, etc.
- Expected result: **PASS**

### A.4.2: Occupancy Estimation

#### RED: test_profiling_occupancy_estimation
- Test: `estimate_occupancy(block_size=256, registers=128, smem=32768, capability=80)` returns 0.0-100.0
- Verify result makes sense (e.g., small block + high register pressure → low occupancy)
- Expected result: **FAIL**

#### GREEN: Implement occupancy calculation
- Hardcode occupancy formulas for Ampere (and other archs)
- Expected result: **PASS**

#### RED: test_profiling_occupancy_scaling
- Test: Increasing BLOCK_M decreases occupancy (register scaling effect)
- Expected result: **FAIL** (occupancy formula may be too simple)

#### GREEN: Refine occupancy model
- Adjust formula to show proper scaling
- Expected result: **PASS**

### A.4.3: GPU Memory Hierarchy Specs

#### RED: test_profiling_gpu_hierarchy_ampere
- Test: `get_gpu_memory_hierarchy(80)` returns correct SMEM, registers, L2, etc. for Ampere
- Expected result: **FAIL**

#### GREEN: Add GPU specs database
- Implement `get_gpu_memory_hierarchy()` with hardcoded values per GPU arch
- Expected result: **PASS**

#### RED: test_profiling_gpu_hierarchy_hopper
- Test: Hopper specs available
- Expected result: **FAIL**

#### GREEN: Add Hopper specs
- Expected result: **PASS**

### A.4.4: Profiling Context Manager

#### RED: test_profiling_context_manager
- Test: `with GPUProfiler() as prof: run_kernel(); metrics = prof.get_profile()`
- Expected result: **FAIL** (context manager not implemented)

#### GREEN: Implement GPUProfiler context manager
- Use `torch.profiler.profile()` internally
- Extract GPU metrics from torch profiler output
- Expected result: **PASS**

### A.4.5: Roofline Model

#### RED: test_profiling_roofline_model
- Test: `get_roofline_model(80, float32)` returns compute + memory rooflines
- Expected result: **FAIL**

#### GREEN: Implement roofline
- Return compute peak TFLOPS and memory peak GB/s
- Expected result: **PASS**

### A.4.6: Kernel Profiling Function

#### RED: test_profiling_profile_kernel
- Test: `profile_kernel(flash_attention_v2, q, k, v, iterations=5)` returns `KernelBenchmark`
- Expected result: **FAIL**

#### GREEN: Implement profile_kernel()
- Wrapper around `torch.profiler` + occupancy estimation
- Return timing + profiling metrics
- Expected result: **PASS**

---

## A.5: BlockMask and Attention Mask DSL

### A.5.1: BlockMask Data Structure

#### RED: test_mask_dsl_block_mask_creation
- Test: `BlockMask(query_blocks=8, key_blocks=8)` instantiates
- Expected result: **FAIL** (BlockMask not defined)

#### GREEN: Define BlockMask
- Create `kernels/mask_dsl.py`
- Implement `@dataclass BlockMask` with mask matrix, metadata
- Expected result: **PASS**

### A.5.2: Causal Block Mask

#### RED: test_mask_dsl_create_causal
- Test: `create_block_mask("causal", query_len=256, key_len=256, block_size=32)` produces lower-triangular mask
- Expected result: **FAIL**

#### GREEN: Implement causal block mask
- Generate mask matrix where query block i can attend to key block j iff j <= i
- Expected result: **PASS**

#### RED: test_mask_dsl_causal_vs_token_level
- Test: Block causal mask applied to attention == token-level causal masking
- Expected result: **FAIL**

#### GREEN: Ensure token-level equivalence
- Verify numerical parity
- Expected result: **PASS**

### A.5.3: Full Block Mask

#### RED: test_mask_dsl_create_full
- Test: `create_block_mask("full", ...)` produces all-True mask
- Expected result: **FAIL**

#### GREEN: Implement full block mask
- Expected result: **PASS**

### A.5.4: Sliding Window Block Mask

#### RED: test_mask_dsl_create_sliding_window
- Test: `create_block_mask("sliding_window", ..., sliding_window=64)` produces band structure
- Expected result: **FAIL**

#### GREEN: Implement sliding window
- Expected result: **PASS**

### A.5.5: Prefix LM Block Mask

#### RED: test_mask_dsl_create_prefix_lm
- Test: `create_block_mask("prefix_lm", ..., prefix_len=64)` produces prefix-dense + causal tail
- Expected result: **FAIL**

#### GREEN: Implement prefix LM
- Expected result: **PASS**

### A.5.6: Mask Composition

#### RED: test_mask_dsl_compose_intersect
- Test: `compose_block_masks(mask1, mask2, "intersect")` produces AND of masks
- Expected result: **FAIL**

#### GREEN: Implement composition
- Expected result: **PASS**

### A.5.7: Apply Block Mask to Attention

#### RED: test_mask_dsl_apply_to_attention
- Test: `apply_block_mask_to_attention(q, k, v, block_mask)` returns correct attention output
- Baseline: Compare with token-level masking
- Expected result: **FAIL** (integration incomplete)

#### GREEN: Integrate with kernels
- Update `flash_attention()` and `flash_attention_v2()` to accept block_mask parameter
- Apply mask in kernel or before/after softmax
- Expected result: **PASS**

---

## A.6: Benchmark Updates & Documentation

### A.6.1: Benchmark Script Updates

#### RED: test_benchmarks_comprehensive
- Test: `benchmarks/bench_flash.py` runs with all kernel variants (v1, v2, persistent)
- Expected result: **FAIL** (benchmarks not updated)

#### GREEN: Update benchmarks
- Add variant parameter to bench_flash.py
- Add bench_persistent.py for persistent kernel comparison
- Expected result: **PASS**

### A.6.2: Example Scripts

#### RED: test_examples_run_without_error
- Test: `examples/attention_v2.py` runs and produces correct output
- Expected result: **FAIL** (example not created)

#### GREEN: Create example scripts
- `examples/flash_attention_v2_basic.py`: Simple v2 usage
- `examples/persistent_kernel_teaching.py`: Occupancy trade-off demo
- `examples/backend_selector_comparison.py`: Compare v1/v2/persistent
- Expected result: **PASS**

### A.6.3: Documentation

#### RED: test_documentation_builds
- Test: `npm run docs:build` succeeds with new content
- Expected result: **FAIL** (docs not updated)

#### GREEN: Add documentation
- Add `docs/en/gpu-memory-model.md`: SRAM/L2/occupancy guide
- Add `docs/en/flash-attention-v2.md`: V2 kernel tutorial
- Add `docs/en/persistent-kernels.md`: Persistent kernel patterns
- Add Chinese equivalents to `docs/zh/`
- Update API reference
- Expected result: **PASS**

---

## A.7: Integration & Regression Testing

### A.7.1: All Tests Pass

#### RED: (Pre-condition) Run full test suite
```bash
pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py
```
- Expected result: **PASS** (baseline: existing tests)

### A.7.2: Add Tests to Property Suite

#### RED: test_properties_flash_v2
- Add hypothesis-driven property tests for v2 to `tests/test_properties.py`
- Expected result: **FAIL** (tests generate failures on random inputs? Unlikely if code correct)

#### GREEN: Ensure v2 passes all properties
- Property: `f_v2(x) == f_v1(x)` for all legal shapes/dtypes
- Expected result: **PASS**

### A.7.3: CI/Local Validation

#### VERIFICATION: Final Integration
```bash
# CPU tests (must pass)
make test-cpu

# Full test suite (must pass)
make test

# Linting & type checking
make lint
make typecheck

# Benchmarks (no regression)
make bench-all

# Documentation build
npm run docs:build

# OpenSpec validation
openspec validate
```

**Expected result**: All checks **PASS**

---

## Implementation Order (Recommended Sequence)

1. **A.1.1** → A.1.2 → A.1.3 (FlashAttention v2 core + basic features)
2. **A.3.1** → A.3.2 (Backend selector; enables testing across variants)
3. **A.2.1** → A.2.2 (Persistent kernels)
4. **A.5.1** → A.5.2 → A.5.7 (BlockMask; foundational for mask support)
5. **A.4.1** → A.4.6 (Profiling; integrates with existing + new kernels)
6. **A.3.3 → A.3.4** (Matmul selector; finalize selector)
7. **A.6** (Benchmarks, examples, documentation; runs last, uses all prior code)
8. **A.7** (Final integration & regression testing)

---

## Acceptance Criteria

**All tasks must satisfy TDD discipline:**
- [ ] Each task has a failing test (RED) before implementation
- [ ] Implementation is minimal (GREEN)
- [ ] All tests pass at task completion
- [ ] No pre-existing tests broken
- [ ] Code reviewed for correctness and style
- [ ] Documentation updated for new public APIs

**Phase A completion gates**:
- [ ] 50+ new tests added, all passing
- [ ] All 5 capabilities (v2, persistent, selector, profiling, mask DSL) implemented
- [ ] Benchmarks show expected performance (v2: 5-15% faster on Ampere+)
- [ ] Documentation updated: GPU memory model, tutorials, API reference
- [ ] Zero regressions: existing tests still pass
- [ ] CI validated: CPU tests pass, GPU tests pass (if CUDA available)

---

## Notes for Implementation

### Code Organization

```
kernels/
  __init__.py                  (updated: exports new APIs)
  flash_attn.py               (existing; no changes)
  flash_attn_v2.py            (new: v2 kernel)
  persistent_kernels.py       (new: persistent matmul + attention)
  backend_selector.py         (new: kernel dispatch)
  mask_dsl.py                 (new: BlockMask + composition)

utils/
  profiling.py                (new: GPU metrics, occupancy, roofline)

tests/
  test_flash_v2.py            (new: v2 kernel tests)
  test_persistent.py          (new: persistent kernel tests)
  test_backend_selector.py    (new: selector tests)
  test_profiling.py           (new: profiling tests)
  test_mask_dsl.py            (new: mask DSL tests)
  test_properties.py          (updated: add v2 + persistent properties)

benchmarks/
  bench_flash.py              (updated: add v2, persistent variants)
  bench_persistent.py         (new: persistent kernel focus)

examples/
  attention_v2_basic.py       (new)
  persistent_kernel_teaching.py (new)
  backend_selector_comparison.py (new)

docs/
  en/gpu-memory-model.md      (new)
  en/flash-attention-v2.md    (new)
  en/persistent-kernels.md    (new)
  zh/gpu-memory-model.md      (new)
  zh/flash-attention-v2.md    (new)
  zh/persistent-kernels.md    (new)
```

### Testing Framework

- **Pytest**: Main test runner
- **Hypothesis**: Property-based testing
- **torch.profiler**: GPU metrics extraction
- **GPU markers**: `@pytest.mark.cuda` for GPU-only tests

### Documentation

- **VitePress**: Already configured; add new `.md` files to `docs/`
- **API documentation**: Update `docs/en/api.md` with new public functions
- **Examples**: Include in `docs/en/tutorial.md` or link to `examples/`

---

## Success Metrics

**By end of Phase A:**
- ✅ FlashAttention v2 implemented and benchmarked
- ✅ Persistent kernels reference implementations available
- ✅ Backend selector unifies kernel dispatch
- ✅ Profiling toolkit enables GPU memory analysis
- ✅ BlockMask foundation supports multiple mask patterns
- ✅ All code covered by TDD tests (red → green)
- ✅ Zero regressions: existing tests pass
- ✅ Documentation updated: GPU memory model, v2/persistent tutorials
- ✅ Benchmarks show expected performance improvements
- ✅ Educational value increased: learners can understand modern kernel patterns

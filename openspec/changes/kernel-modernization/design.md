# Design: Kernel Modernization for DIY-FlashAttention

## Architecture Overview

This change introduces a **layered modernization** of the FlashAttention and matmul kernels, adding contemporary GPU patterns (v2 striped parallelism, persistent kernels, TMA references, profiling) while maintaining the educational forward-only scope and archive-ready maintenance posture.

```
┌──────────────────────────────────────────────────────────┐
│          User-Facing APIs (Backward Compatible)           │
│  flash_attention(variant="v1|v2", ...) / triton_matmul()  │
│           backend_selector.select_kernel()                │
└───────────┬─────────────────────────┬────────────────────┘
            │                         │
    ┌───────▼─────┐        ┌──────────▼─────────┐
    │  v1 Kernels │        │  v2 Kernels       │
    │  (baseline) │        │  (striped par)    │
    └─────────────┘        └──────────────────┘
            │                         │
    ┌───────▼─────────────────────────▼─────────┐
    │     Persistent Kernel Variants            │
    │  (persistent matmul, persistent attentio) │
    └─────────────┬─────────────────────────────┘
                  │
    ┌─────────────▼──────────────────┐
    │   Backend Selector Logic       │
    │  (GPU capability detection)    │
    │  (Kernel dispatch routing)     │
    └─────────────┬──────────────────┘
                  │
    ┌─────────────▼──────────────────────┐
    │   Profiling & Instrumentation      │
    │  (SRAM/L2 analysis, GPU metrics)   │
    └────────────────────────────────────┘
```

## Design Decisions

### 1. FlashAttention v2 Implementation Strategy

**Decision:** Implement v2 as a **striped (row-wise) parallelism variant** alongside the existing block-column v1.

**Rationale:**
- Triton reference (`06-fused-attention.py`) demonstrates row-wise parallelism with host descriptors
- Allows side-by-side teaching of two distinct parallelization strategies
- Matches modern hardware (Hopper+) affinity for thread blocks mapped to attention heads

**Scope:**
- Core v2 kernel: `flash_attn_v2()` in `kernels/flash_attn.py`
- Support: float16, bfloat16, float32 (converted internally)
- Mask support: causal, full (sliding_window/prefix_lm deferred to Phase B)
- Optional: `warp_specialize` flag for teaching scheduling differences
- Tests: v1 vs v2 correctness on identical inputs, performance comparison

**Not in scope (Phase B+):**
- Warp group MMA / WGMMA specialization (Hopper-only advanced feature)
- FP8 output optimization (advanced teaching, optional in later phase)
- Multi-query/grouped attention (defer to inference lab)

### 2. Persistent Kernel Strategy

**Decision:** Implement persistent matmul and a minimal persistent attention reference, both with built-in profiling hooks.

**Rationale:**
- Triton's `09-persistent-matmul.py` emphasizes profiling-first design
- Teaches memory hierarchy and occupancy trade-offs explicitly
- Does **not** replace existing kernels; purely educational reference

**Scope:**
- `persistent_matmul()`: Persistent thread-block matmul with configurable occupancy
- `persistent_attention()`: Row-wise attention variant (complements v2)
- Profiling hooks: GPU memory occupancy, shared memory pressure, register count
- GPU selection: Adaptive kernel choice based on `detect_gpu()` (Ampere+ only)

**Not in scope:**
- Replacing the default matmul path (keep v1 matmul as baseline)
- TMA/async-copy specialization (reference only; optional docs)
- Multi-level GPU memory hierarchy (L2, L1 caching analysis deferred)

### 3. Backend Selector Design

**Decision:** Create a minimal **selector registry** that routes kernel calls based on:
1. GPU capability (Volta/Turing/Ampere/Ada/Hopper/Blackwell)
2. User preference (variant parameter: `"v1"`, `"v2"`, `"persistent"`)
3. Problem size heuristics (batch, seq_len, head_dim)

**Rationale:**
- Unifies kernel dispatch without adding complexity
- Aligns with `modern_features.py`'s existing detection logic
- Enables benchmarks/examples to work across all kernel variants
- Extensible for Phase B/C additions (inference lab kernels)

**Scope:**
- `backend_selector.select_kernel()`: Route to v1/v2/persistent matmul
- `backend_selector.select_attention()`: Route to v1/v2 attention
- Heuristics: Simple rules (e.g., persistent if batch > 4, seq_len > 8192)
- Configuration: Expose as module-level defaults + parameter overrides

**Not in scope:**
- JIT compilation or module caching (FlashInfer-level engineering deferred)
- Dynamic cost models (fixed heuristics only)

### 4. Profiling Toolkit Design

**Decision:** Create `utils/profiling.py` with GPU memory analysis and optional Nsight Compute integration.

**Rationale:**
- Triton's persistent matmul tutorial uses `proton` profiler; we extend this
- SRAM/register pressure is a key teaching axis (why some kernels hit hardware limits)
- Lightweight metrics extraction enables benchmark interpretation

**Scope:**
- `GPUMemoryProfile`: SRAM occupancy, register pressure, L2 efficiency
- `profile_kernel()`: Context manager for torch.profiler integration
- Nsight Compute parsing helpers (optional, read-only)
- Educational output: TFLOPS, GB/s, occupancy %, roofline plots (simple)

**Not in scope:**
- Full Nsight Compute CLI automation (reference-only examples)
- Register pressure prediction/optimization (deferred to advanced lab)
- Per-warp scheduling/divergence analysis

### 5. BlockMask / Attention Mask DSL Foundation

**Decision:** Implement `BlockMask` and mask composition helpers supporting causal, full, sliding_window, and prefix_lm.

**Rationale:**
- PyTorch FlexAttention demonstrates that mask abstraction is central to future expansion
- Phase A only implements causal; v2/persistent with full support
- Prepares for Phase B (attention abstraction capability)

**Scope (Phase A - Baseline):**
- `BlockMask` data structure (block IDs, mask operations)
- `create_block_mask()` for causal, full patterns
- Integration with existing `flash_attn()` and v2 kernels
- Tests: Correctness on masked vs full attention

**Scope (Phase B - Deferred):**
- Sliding window, prefix_lm, custom_mod patterns
- Mask fusion/optimization (combining multiple mask_mod functions)

**Not in scope:**
- Custom score_mod functions (FlexAttention feature; deferred to inference lab)
- Block-sparsity optimization hints (automatic; deferred to advanced phase)

### 6. Testing & Validation Strategy

**Test structure (TDD):**
1. **Correctness tests** (test_flash.py, test_matmul.py)
   - Each new kernel variant: `test_flash_attention_v2()`, `test_persistent_matmul()`, etc.
   - Correctness against PyTorch baseline (SDPA, F.linear for matmul)
   - Mask correctness: causal vs full vs sliding_window
   - Dtype and shape coverage: float16/32/bf16, various seq_lens

2. **Property-based tests** (test_properties.py)
   - Hypothesis-driven random shapes, dtypes, masks
   - Invariants: `f(x) == torch_baseline(x)` for all legal inputs

3. **Performance tests** (benchmarks/)
   - v1 vs v2 vs persistent matmul/attention
   - TFLOPS, GB/s, occupancy metrics
   - Scaling: seq_len, batch_size, head_dim

4. **Integration tests**
   - Selector logic: ensure correct kernel path taken
   - Profiling integration: metrics extracted without errors
   - All existing tests continue to pass

**CI strategy:**
- CPU tests (no CUDA): Core correctness and logic
- GPU tests (optional): Benchmark and profiling validation
- Marking: `@pytest.mark.cuda` for GPU-only tests

## Implementation Order

1. **Phase A.1**: Implement `flash_attention_v2()` kernel + tests
2. **Phase A.2**: Implement `persistent_matmul()` + `persistent_attention()` + tests
3. **Phase A.3**: Create `backend_selector` module + tests
4. **Phase A.4**: Add profiling utilities + integration with benchmarks
5. **Phase A.5**: Implement `BlockMask` foundation + mask composition helpers
6. **Phase A.6**: Update benchmarks and documentation

## Backward Compatibility

- Existing `flash_attention()` and `triton_matmul()` remain default and unchanged
- New variants opt-in via `variant=` parameter
- Selector logic transparent to existing code paths
- All existing tests pass without modification

## Open Questions / Deferred Decisions

1. **Warp specialization scope**: Teaching tool or optional optimization?
   - Decision: Teaching tool with on/off flag
2. **FP8 output path**: Include in v2 or advanced lab?
   - Decision: Deferred to Phase B (infrastructure ready, implementation optional)
3. **Persistent kernel occupancy tuning**: Static heuristics or dynamic?
   - Decision: Static heuristics per GPU arch; dynamic deferred to advanced lab
4. **Profiling tool dependencies**: Nsight Compute CLI or torch.profiler only?
   - Decision: torch.profiler primary; Nsight integration optional/example-only

## Success Criteria

- [ ] FlashAttention v2 kernel passes correctness tests (vs PyTorch SDPA)
- [ ] Persistent matmul/attention variants pass correctness tests
- [ ] Backend selector correctly routes based on GPU capability + variant
- [ ] Profiling utilities extract and display GPU metrics
- [ ] BlockMask abstraction supports causal, full, sliding_window, prefix_lm patterns
- [ ] All new code covered by TDD tests (red → green → refactor)
- [ ] Benchmarks updated to compare v1/v2/persistent variants
- [ ] Documentation updated: GPU memory model guide, v2/persistent tutorials
- [ ] All existing tests continue to pass
- [ ] CI validates correctness on CPU; benchmarks run (GPU optional)

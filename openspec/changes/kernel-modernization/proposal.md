## Why

The repository currently implements FlashAttention v1-style forward pass with basic autotune, but lacks modern GPU kernel patterns (FlashAttention v2 striped parallelism, persistent kernels, TMA/FP8 support) and has no profiling/performance analysis teaching layer. This limits educational value and leaves Hopper/Blackwell hardware capabilities unexploited. The modernization introduces contemporary kernel design patterns while preserving the educational focus and archive-ready maintenance constraints.

## What Changes

- Add `flash_attention_v2()` kernel implementing striped attention (row-wise parallelism) alongside v1 block-column approach
- Introduce `persistent_matmul()` kernel and TMA matmul reference implementations with profiling integration
- Add `utils/profiling.py` module with GPU memory hierarchy analysis and Nsight Compute parsing helpers
- Expand GPU memory model documentation with SRAM/L2 budgeting and register pressure calculations
- Create `backend_selector` module for unified kernel dispatch across baseline/v2/persistent variants
- Implement `BlockMask` / mask DSL foundation supporting causal, full, sliding_window, and prefix_lm modes
- Update benchmarks to compare v1 vs v2, persistent variants, and profiling output
- Ensure all new kernels have comprehensive tests (TDD) and pass existing validation suite

## Capabilities

### New Capabilities

- `flash-attention-v2`: Striped attention variant with row-wise parallelism, host descriptors, and optional warp specialization
- `persistent-kernels`: Persistent thread-block matmul and attention reference implementations with profiling
- `profiling-toolkit`: GPU memory profiling, performance counter analysis, and educational metrics extraction
- `attention-mask-dsl`: Composable mask abstraction (BlockMask-style) supporting multiple attention patterns
- `kernel-selector`: Unified backend dispatch for choosing between baseline/v2/persistent/lab kernels

### Modified Capabilities

- `flashattention-kernels`: Adds v2 variant and modern feature path; maintains v1 for backward compatibility

## Impact

- **Code**: New modules in `kernels/`, `utils/`, `benchmarks/`; updates to existing kernel APIs
- **APIs**: Backward compatible; adds `flash_attention_v2()`, `variant=` parameter, backend selection
- **Dependencies**: None new (profiling uses existing torch.profiler; Triton 2.1+ already required)
- **Documentation**: GPU memory model guide, FA2/persistent kernel tutorials, profiling examples
- **Tests**: TDD coverage for all new kernels and utilities; all existing tests continue to pass

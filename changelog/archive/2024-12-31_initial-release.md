# Changelog: 2024-12-31 Initial Release

**Date**: 2024-12-31  
**Type**: Release  
**Version**: 1.0.0

## Overview

Initial public release of DIY FlashAttention - a from-scratch implementation of FlashAttention using Python and OpenAI Triton.

## Features

### Core Kernels

#### Matrix Multiplication (`kernels/matmul.py`)

| Feature | Description |
|---------|-------------|
| Autotune | Automatic optimal block size selection |
| Manual Tuning | Custom block size specification |
| L2 Optimization | Super-grouping for cache efficiency |
| Multi-dtype | float16, float32, bfloat16 support |
| Edge Handling | Non-aligned dimension support |

#### FlashAttention (`kernels/flash_attn.py`)

| Feature | Description |
|---------|-------------|
| Online Softmax | O(N) memory complexity |
| Causal Masking | Autoregressive model support |
| Multi-Head | Parallel head computation |
| Variable Length | Per-sample sequence lengths |
| 3D/4D Input | Flexible tensor shapes |

#### Modern GPU Features (`kernels/modern_features.py`)

| Feature | Description |
|---------|-------------|
| Architecture Detection | Volta → Blackwell |
| TMA Detection | Hopper+ async loading |
| FP8 Detection | Hopper+ low precision |
| Adaptive Selection | Best kernel per GPU |

### Utilities

| Module | Purpose |
|--------|---------|
| `gpu_detect.py` | GPU capability detection |
| `benchmark.py` | Performance measurement |
| `validation.py` | Numerical correctness |

### Testing

| Category | Count |
|----------|-------|
| Unit Tests | 50+ |
| Property Tests | 5 |
| Edge Case Tests | 20+ |

### Documentation

| Doc | Content |
|-----|---------|
| Tutorial | Step-by-step learning |
| API Reference | Complete API docs |
| Performance Guide | Optimization tips |
| FAQ | Common questions |
| Cheatsheet | Quick reference |

## Project Structure

```
diy-flash-attention/
├── kernels/         # GPU kernels
├── utils/           # Utilities
├── tests/           # Test suite
├── benchmarks/      # Performance tests
├── examples/        # Usage examples
├── docs/            # VitePress docs
├── changelog/       # Detailed changes
└── scripts/         # Automation
```

## Technical Requirements

| Component | Version |
|-----------|---------|
| Python | >= 3.9 |
| CUDA | >= 11.0 |
| PyTorch | >= 2.0.0 |
| Triton | >= 2.1.0 |

## Supported GPUs

| Architecture | Compute Capability | Status |
|--------------|-------------------|--------|
| Volta | SM70 | ✅ Basic |
| Turing | SM75 | ✅ Basic |
| Ampere | SM80+ | ✅ Full |
| Ada | SM89 | ✅ Full |
| Hopper | SM90 | ✅ Advanced |
| Blackwell | SM100 | ✅ Advanced |

## Performance Highlights

| Metric | Value |
|--------|-------|
| Memory Reduction | Up to 99% |
| Speedup vs PyTorch | 1.1-1.6x |
| Max Sequence Length | 8192+ |

## License

MIT License - Open source, free to use and modify.

# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.1+-orange.svg)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/docs.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/docs.yml)

English | [简体中文](README.zh-CN.md)

> 📖 **Docs**: [https://lessup.github.io/diy-flash-attention/](https://lessup.github.io/diy-flash-attention/)

Implement FlashAttention from scratch using Python + OpenAI Triton.

## Goals

1. **Learn Triton programming** — Block pointer arithmetic and tiling via matrix multiplication kernels
2. **Reproduce FlashAttention** — The core attention acceleration algorithm in LLMs
3. **Performance benchmarking** — Quantify optimization impact, explore block size effects
4. **Modern GPU support** — Auto-detect GPU architecture, support Hopper/Blackwell features
5. **Full test coverage** — Unit tests and property-based testing (Hypothesis)

## Requirements

- Python >= 3.9, CUDA >= 11.0, PyTorch >= 2.0.0, Triton >= 2.1.0
- NVIDIA GPU (Ampere or newer recommended)

## Installation

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from kernels import triton_matmul, flash_attention

# Matrix multiplication (float16/bfloat16)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## Project Structure

```
diy-flash-attention/
├── kernels/               # Triton GPU Kernels
│   ├── matmul.py          # Matrix multiplication (autotune, float16/bfloat16)
│   ├── flash_attn.py      # FlashAttention (online softmax, causal)
│   └── modern_features.py # GPU feature detection (FP8, TMA)
├── benchmarks/            # Performance benchmarks
├── tests/                 # Test suite (unit + property-based)
├── utils/                 # Benchmark, validation, GPU detection
├── examples/              # Usage examples
├── docs/                  # Documentation
└── Makefile               # Convenience commands
```

## Commands

```bash
make demo          # Quick demo
make bench-all     # Run all benchmarks
make test          # Run all tests
make lint          # Code check (ruff)
make format        # Code format (ruff)
```

## Core Concepts

- **Tiling**: Split large matrices into blocks that fit in SRAM
- **Online Softmax**: Compute softmax incrementally — O(N) instead of O(N²) memory
- **Block Size Tuning**: Key parameter affecting performance

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)

## License

MIT

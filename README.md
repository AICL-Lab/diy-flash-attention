# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.1+-orange.svg)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml)

📖 **Documentation**: [https://lessup.github.io/diy-flash-attention/](https://lessup.github.io/diy-flash-attention/)

Implement FlashAttention from scratch using Python + OpenAI Triton.

## 🎯 Goals

1. **Learn Triton Programming** — Block pointer arithmetic, tiling, and autotuning
2. **Reproduce FlashAttention** — The core attention acceleration algorithm in LLMs
3. **Performance Benchmarking** — Quantify optimization impact, explore block size effects
4. **Modern GPU Support** — Auto-detect GPU architecture, support Hopper/Blackwell features
5. **Full Test Coverage** — Unit tests and property-based testing (Hypothesis)

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### Usage

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

## 📁 Project Structure

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

## 🚀 Commands

```bash
make demo          # Quick demo
make bench-all     # Run all benchmarks
make test          # Run all tests
make lint          # Code check (ruff)
make format        # Code format (ruff)
```

## 📊 Performance

| Sequence Length | PyTorch SDPA | FlashAttention | Memory Saved |
|----------------|--------------|----------------|--------------|
| 1024 | 8 MB | 0.5 MB | 94% |
| 4096 | 128 MB | 2 MB | 98% |
| 8192 | 512 MB | 4 MB | 99% |

## 🔑 Key Concepts

- **Tiling**: Split large matrices into blocks that fit in SRAM
- **Online Softmax**: Compute softmax incrementally — O(N) memory instead of O(N²)
- **Block Size Tuning**: Key parameter affecting performance

## 📚 Documentation

- [Tutorial](https://lessup.github.io/diy-flash-attention/tutorial) — Learn FlashAttention from scratch
- [API Reference](https://lessup.github.io/diy-flash-attention/api) — Complete API documentation
- [Performance Guide](https://lessup.github.io/diy-flash-attention/performance) — Optimization tips
- [FAQ](https://lessup.github.io/diy-flash-attention/faq) — Common questions

## 📋 Requirements

- Python >= 3.9
- CUDA >= 11.0
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- NVIDIA GPU (Ampere or newer recommended)

## 📖 References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)

## 📄 License

MIT

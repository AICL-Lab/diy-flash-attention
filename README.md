# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![CUDA 11+](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Triton 2.1+](https://img.shields.io/badge/Triton-2.1%2B-orange)](https://triton-lang.org/)
[![PyPI](https://img.shields.io/badge/install-pip%20install%20diy--flash--attention-3776ab)](https://pypi.org/project/diy-flash-attention/)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/diy-flash-attention/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

<div align="center">
  <h3>Learn Triton by Building FlashAttention from Scratch</h3>
  <p>
    <a href="https://lessup.github.io/diy-flash-attention/">📖 Docs</a> •
    <a href="https://lessup.github.io/diy-flash-attention/en/tutorial">📚 Tutorial</a> •
    <a href="https://lessup.github.io/diy-flash-attention/en/api">🔧 API</a> •
    <a href="./README.zh-CN.md">🇨🇳 中文</a>
  </p>
</div>

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.10+ |
| CUDA | 11.0+ | 12.0+ |
| GPU | NVIDIA Volta (SM70)+ | RTX 40xx / A100 / H100 |
| VRAM | 4 GB | 16 GB+ |
| OS | Linux | Ubuntu 22.04 |

> ⚠️ **NVIDIA GPU Required**: Triton only supports CUDA devices. AMD/Intel GPUs are not supported.

---

## Why FlashAttention?

Standard attention computes **O(N²)** memory — a hard limit on sequence length. FlashAttention changes the game:

| | Standard | FlashAttention |
|---|---|---|
| Memory | O(N²) | **O(N)** |
| Max seq len | ~2K tokens | **64K+ tokens** |
| HBM accesses | Writes full attention matrix | **Never materializes it** |
| Precision | Full | Exact, no approximation |

> At seq_len=8192, FlashAttention uses **99% less memory** (512 MB → 4 MB).

---

## What's Inside

- **Triton kernels** — [`matmul`](./kernels/matmul.py) with autotune, [`flash_attention`](./kernels/flash_attn.py) with online softmax & causal masking
- **GPU auto-detection** — Volta → Blackwell, picks optimal config per architecture
- **Property-based tests** — Hypothesis validates correctness across infinite input spaces
- **Benchmark suite** — Compare against PyTorch SDPA with TFLOPS & memory metrics ([source](./benchmarks/))
- **Production-grade** — Type hints, ruff lint, mypy checks, CI on every push

> 🔬 **Forward-only**: This is an educational implementation focusing on the forward pass. For training, you'd need to implement the backward kernel.

---

## Project Focus

This repository is designed as a **hands-on learning artifact**, not a generic CUDA portal:

- learn Triton through a real matrix multiplication kernel
- understand FlashAttention by reading a compact forward implementation
- compare against PyTorch SDPA with reproducible benchmark scripts
- inspect architecture-aware helpers for Volta → Blackwell GPUs

If you want a compact reference for studying attention-kernel design, this repository is the entrypoint. If you need a full production training stack, treat this project as the teaching scaffold rather than the final system.

---

## Quick Start

### 1. Setup Environment

```bash
# Using conda (recommended)
conda create -n flash python=3.10
conda activate flash

# Or using venv
python -m venv flash_env
source flash_env/bin/activate  # Windows: flash_env\Scripts\activate
```

### 2. Install

```bash
pip install diy-flash-attention

# Or install from source for development
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
python -c "from kernels import flash_attention; print('✓ Installation successful')"
```

### 4. Run Example

```python
import torch
from kernels import triton_matmul, flash_attention

# Matrix multiplication (autotune finds optimal block sizes)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention — 99% less memory for long sequences
# Shape: (batch, heads, seq_len, head_dim)
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)  # GPT-style causal mask

print(f"Output shape: {out.shape}")  # [2, 8, 4096, 64]
```

### 5. Run Benchmarks

```bash
make demo        # matmul + flash_attention smoke test
make bench-all   # full benchmark suite against PyTorch SDPA
make report      # generate markdown report
```

---

## Benchmarks (RTX 4090)

### Memory Usage

| Seq Len | Standard | FlashAttention | Saved |
|---------|---------|----------------|-------|
| 1,024   | 8 MB    | 0.5 MB         | 94% ↓ |
| 4,096   | 128 MB  | 2 MB           | 98% ↓ |
| 8,192   | 512 MB  | 4 MB           | 99% ↓ |

### Throughput

| Kernel | PyTorch SDPA | Ours (Triton) | Speedup |
|--------|-------------|---------------|---------|
| MatMul 4096² | 120 TFLOPS | 140 TFLOPS | **1.17x** |
| Attention 4096 | 35.0 ms | 22.0 ms | **1.59x** |

---

## Who Is This For?

| You are... | You'll get... |
|---|---|
| ML engineer | Understand the kernel behind your transformer framework |
| CUDA beginner | Learn Triton without reading 500 pages of docs |
| Researcher | Reproduce and modify FlashAttention for your experiments |
| Performance engineer | Study autotune configs and block-size trade-offs |

---

## GPU Support

| Architecture | GPUs | Support Level |
|---|---|---|
| Volta (SM70) | V100 | ✅ Basic |
| Turing (SM75) | RTX 20xx | ✅ Basic |
| Ampere (SM80) | A100, RTX 30xx | ✅ Full |
| Ada (SM89) | RTX 40xx | ✅ Full |
| Hopper (SM90) | H100 | ✅ TMA, FP8 |
| Blackwell (SM100) | B100/B200 | ✅ Latest |

Auto-detection runs at import time — no config needed.

---

## Project Structure

| Directory | Purpose |
|-----------|---------|
| [`kernels/`](./kernels/) | Triton GPU kernels (matmul, flash_attn, feature detection) |
| [`utils/`](./utils/) | Benchmark, validation, GPU detection utilities |
| [`tests/`](./tests/) | Unit tests & property-based tests with Hypothesis |
| [`benchmarks/`](./benchmarks/) | CLI benchmark tools vs PyTorch SDPA |
| [`examples/`](./examples/) | Usage demos and tutorials |
| [`openspec/`](./openspec/) | OpenSpec change management, capability specs, and active change tasks |
| [`docs/`](./docs/) | Bilingual documentation (EN / ZH) |

---

## Development

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
pip install -e ".[dev]"

make test-cpu    # CPU-safe validation path
make test-gpu    # full GPU test suite
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
make docs        # build GitHub Pages site
make hooks-install
make clean       # remove caches
```

This project follows **OpenSpec-driven development**. For non-trivial work:

```bash
openspec list --json
# choose the active change from the list above
openspec status --change <change-name> --json
openspec instructions apply --change <change-name> --json
```

See [AGENTS.md](./AGENTS.md), [CLAUDE.md](./CLAUDE.md), and [openspec/specs/README.md](./openspec/specs/README.md) for the workflow contract.

---

## Learn More

**Original papers**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

**Tutorials & guides**
- [Step-by-step tutorial](https://lessup.github.io/diy-flash-attention/en/tutorial) — from GPU basics to FlashAttention
- [API reference](https://lessup.github.io/diy-flash-attention/en/api) — kernel signatures, autotune configs
- [Performance guide](https://lessup.github.io/diy-flash-attention/en/performance) — block size tuning, memory profiling
- [Cheatsheet](https://lessup.github.io/diy-flash-attention/en/cheatsheet) — quick reference for common patterns

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvcc not found` | Install CUDA toolkit: `conda install -c nvidia cuda-toolkit` |
| `triton` install fails | Ensure Python ≥3.9 and pip ≥21.0: `pip install --upgrade pip` |
| `CUDA out of memory` | Reduce batch size or sequence length; check GPU memory with `nvidia-smi` |
| `No module named 'kernels'` | Make sure you're in the project root and installed with `pip install -e ".[dev]"` |
| Import error on AMD/Intel GPU | Triton requires NVIDIA GPU. This project does not support other vendors. |

See [GitHub Issues](https://github.com/LessUp/diy-flash-attention/issues) for more help.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. The most useful contributions usually keep the repository clearer and more trustworthy:

- 📝 Improve docs, examples, or cross-links between README and Pages
- 🐛 Fix edge cases or tighten error handling
- ✅ Add or refine tests around current forward-only behavior
- 🧹 Remove stale process/docs/config clutter that no longer helps readers

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute.

<div align="center">
  <p>
    ⭐ Star this repo if you find it helpful!<br>
    <a href="https://github.com/LessUp/diy-flash-attention/issues">🐛 Report a bug</a> •
    <a href="https://github.com/LessUp/diy-flash-attention/discussions">💡 Request a feature</a>
  </p>
</div>

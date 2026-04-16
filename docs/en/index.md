---
layout: home

hero:
  name: DIY FlashAttention
  text: FlashAttention from Scratch
  tagline: Hands-on Implementation Using Python + OpenAI Triton
  actions:
    - theme: brand
      text: Tutorial
      link: /en/tutorial
    - theme: alt
      text: API Reference
      link: /en/api
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - title: Triton Programming Model
    details: Learn block pointer arithmetic, tiling, and autotune automatic optimization through implementing matrix multiplication kernels
    icon: ⚡
  - title: FlashAttention Re-implementation
    details: Implement the core attention acceleration algorithm in LLMs with O(N) memory complexity, supporting Causal Masking and variable-length sequences
    icon: 🧠
  - title: Performance Benchmarking
    details: Quantify optimization effects through benchmarking, compare with PyTorch SDPA, and explore Block Size impact on performance
    icon: 📊
  - title: Modern GPU Support
    details: Auto-detect GPU architectures (Volta → Blackwell), supports TMA, FP8, and Warpgroup MMA feature detection
    icon: 🖥️
  - title: Complete Test Coverage
    details: Unit tests + Property-based testing (Hypothesis), covering correctness, edge cases, and memory scaling
    icon: ✅
  - title: Bilingual Documentation
    details: Complete documentation available in both English and Chinese - Tutorial, API Reference, Performance Guide, Cheatsheet, and FAQ
    icon: 📖
---

## Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Language | Python 3.9+ | Primary language |
| GPU Programming | OpenAI Triton 2.1+ | GPU Kernel development framework |
| Deep Learning | PyTorch 2.0+ | Tensor operations and reference implementation |
| GPU Runtime | CUDA 11.0+ | GPU compute driver |
| Testing | pytest + Hypothesis | Unit tests + Property-based tests |
| Code Quality | Ruff + mypy | Lint + Type checking |

## Quick Start

```python
import torch
from kernels import triton_matmul, flash_attention

# Matrix Multiplication (float16/bfloat16 supported, autotune for optimal config)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention (batch=2, heads=8, seq_len=512, head_dim=64)
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## Key Concepts

- **Tiling** — Split large matrices into tiles for SRAM computation, reducing HBM access
- **Online Softmax** — Compute softmax incrementally by tiles, reducing memory from O(N²) to O(N)
- **Autotune** — Automatically search for optimal Block Size configurations, adapting to different matrix sizes
- **Architecture Adaptation** — Automatically detect GPU architecture and select optimal kernel implementations

# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.1+-orange.svg)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml)

📖 **文档站**：[https://lessup.github.io/diy-flash-attention/](https://lessup.github.io/diy-flash-attention/)

使用 Python + OpenAI Triton 从零实现 FlashAttention 算法。

## 🎯 项目目标

1. **理解 Triton 编程模型** — Block 指针运算、Tiling、Autotune
2. **复现 FlashAttention** — LLM 核心注意力加速算法
3. **性能对比** — 量化优化效果，探索 Block Size 影响
4. **现代 GPU 支持** — 自动检测架构，支持 Hopper/Blackwell 特性
5. **完整测试覆盖** — 单元测试 + 属性测试 (Hypothesis)

## ⚡ 快速开始

### 安装

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### 使用

```python
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法 (支持 float16/bfloat16)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## 📁 项目结构

```
diy-flash-attention/
├── kernels/               # Triton GPU Kernels
│   ├── matmul.py          # 矩阵乘法 (autotune, float16/bfloat16)
│   ├── flash_attn.py      # FlashAttention (online softmax, causal)
│   └── modern_features.py # GPU 特性检测 (FP8, TMA)
├── benchmarks/            # 性能测试
├── tests/                 # 测试套件 (单元 + 属性测试)
├── utils/                 # Benchmark、验证、GPU 检测
├── examples/              # 示例代码
├── docs/                  # 文档
└── Makefile               # 便捷命令
```

## 🚀 常用命令

```bash
make demo          # 快速演示
make bench-all     # 运行所有 benchmark
make test          # 运行所有测试
make lint          # 代码检查 (ruff)
make format        # 代码格式化 (ruff)
```

## 📊 性能

| 序列长度 | PyTorch SDPA | FlashAttention | 内存节省 |
|---------|--------------|----------------|---------|
| 1024 | 8 MB | 0.5 MB | 94% |
| 4096 | 128 MB | 2 MB | 98% |
| 8192 | 512 MB | 4 MB | 99% |

## 🔑 核心概念

- **Tiling（分块）**：将大矩阵分割成小块在 SRAM 中计算
- **Online Softmax**：分块增量计算 softmax，O(N) 内存
- **Block Size 调优**：影响性能的关键参数

## 📚 文档

- [教程](https://lessup.github.io/diy-flash-attention/tutorial) — 从零学习 FlashAttention
- [API 参考](https://lessup.github.io/diy-flash-attention/api) — 完整 API 文档
- [性能指南](https://lessup.github.io/diy-flash-attention/performance) — 优化技巧
- [FAQ](https://lessup.github.io/diy-flash-attention/faq) — 常见问题

## 📋 环境要求

- Python >= 3.9
- CUDA >= 11.0
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- NVIDIA GPU（推荐 Ampere 或更新架构）

## 📖 参考资料

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)

## 📄 许可证

MIT

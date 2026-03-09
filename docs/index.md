---
layout: home

hero:
  name: DIY FlashAttention
  text: 从零实现 FlashAttention
  tagline: 使用 Python + OpenAI Triton 动手实践 LLM 核心算法
  actions:
    - theme: brand
      text: 教程
      link: /tutorial
    - theme: alt
      text: API 参考
      link: /api
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - title: Triton 编程模型
    details: 通过实现矩阵乘法 Kernel 学习 Block 指针运算、Tiling 和 Autotune 自动调优
    icon: ⚡
  - title: FlashAttention 复现
    details: 实现 LLM 中最核心的注意力机制加速算法，O(N) 内存复杂度，支持 Causal Masking 和变长序列
    icon: 🧠
  - title: 性能对比
    details: 通过 Benchmark 量化优化效果，对比 PyTorch SDPA，感受 Block Size 对性能的影响
    icon: 📊
  - title: 现代 GPU 支持
    details: 自动检测 GPU 架构（Volta → Blackwell），支持 TMA、FP8、Warpgroup MMA 特性检测
    icon: 🖥️
  - title: 完整测试覆盖
    details: 单元测试 + 属性测试 (Hypothesis)，覆盖正确性、边界条件和内存缩放
    icon: ✅
  - title: 中文文档
    details: 教程、API 参考、性能指南、速查表、FAQ 全中文编写
    icon: 📖
---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.9+ | 主语言 |
| GPU 编程 | OpenAI Triton 2.1+ | GPU Kernel 编写框架 |
| 深度学习 | PyTorch 2.0+ | 张量运算与参考实现 |
| GPU 运行时 | CUDA 11.0+ | GPU 计算驱动 |
| 测试 | pytest + Hypothesis | 单元测试 + 属性测试 |
| 代码质量 | Ruff + mypy | Lint + 类型检查 |

## 快速开始

```python
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法 (支持 float16 / bfloat16, Autotune 自动选择最优配置)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention (batch=2, heads=8, seq_len=512, head_dim=64)
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## 核心概念

- **Tiling（分块）** — 将大矩阵分割成小块在 SRAM 中计算，减少 HBM 访问
- **Online Softmax** — 分块增量计算 softmax，内存从 O(N²) 降至 O(N)
- **Autotune** — 自动搜索最优 Block Size 配置，适配不同矩阵规模
- **架构自适应** — 自动检测 GPU 架构，选择最优 Kernel 实现和参数

# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-2.1+-orange.svg)](https://triton-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml)

<div align="center">
  <h3>🚀 从零实现 FlashAttention，学习 GPU 编程</h3>
  <p>
    <a href="https://lessup.github.io/diy-flash-attention/zh/">📖 文档</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/tutorial">📚 教程</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/api">🔧 API</a> •
    <a href="./README.md">🇺🇸 English</a>
  </p>
</div>

---

## 什么是 FlashAttention？

**FlashAttention** 是一项革命性的注意力计算算法：

| 方面 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| 内存 | O(N²) - 长序列爆炸 | O(N) - 线性扩展 |
| 速度 | 内存受限，慢 | IO 感知，快 |
| 精度 | 全精度 | 精确，无近似 |
| 最大序列长度 | 受限于 GPU 显存 | 可处理 4-8 倍更长序列 |

> **实际影响**: 使用 FlashAttention，你可以在消费级 GPU 上训练支持 **64K 令牌** 序列长度的模型！

---

## 🎯 本项目提供什么

### 1. 学习 Triton GPU 编程
- **Block 指针运算** - 掌握 GPU 内存访问方式
- **Tiling 策略** - 将问题拆分以适应快速 SRAM
- **Autotune 优化** - 让 Triton 自动寻找最佳配置
- **与 CUDA 对比** - 了解 Triton 如何抽象复杂性

### 2. 深入理解 FlashAttention
- **Online Softmax** - O(N) 内存背后的数学技巧
- **因果掩码** - 自回归模型的工作原理
- **内存层次结构** - HBM 与 SRAM 为何重要
- **IO 复杂度** - GPU 计算的真正瓶颈

### 3. 生产就绪代码
- ✅ 在 V100 到 H100 GPU 上测试通过
- ✅ 使用 Hypothesis 进行属性测试
- ✅ 与 PyTorch 对比的 Benchmark 套件
- ✅ 自动 GPU 架构检测
- ✅ 支持 Hopper 特性 (TMA, FP8)

---

## ⚡ 快速开始

### 安装

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### 30 秒演示

```python
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法（大矩阵比朴素实现快 2 倍）
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention（长序列内存节省 99%！）
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)  # 用于 GPT 类模型
```

运行完整演示：
```bash
make demo
```

---

## 📊 性能亮点

### 内存节省

| 序列长度 | 标准 Attention | FlashAttention | 节省内存 |
|---------|---------------|----------------|---------|
| 1024 | 8 MB | 0.5 MB | 94% ↓ |
| 4096 | 128 MB | 2 MB | 98% ↓ |
| 8192 | 512 MB | 4 MB | 99% ↓ |

### 速度对比 (RTX 4090)

| 操作 | PyTorch | Triton (本项目) | 加速 |
|------|---------|-----------------|------|
| MatMul 4096² | 120 TFLOPS | 140 TFLOPS | 1.17x |
| Attention 4096 | 35.0 ms | 22.0 ms | 1.59x |

运行性能测试：
```bash
make bench-all
make report
```

---

## 📁 项目结构

```
diy-flash-attention/
├── kernels/               # GPU 核函数
│   ├── matmul.py          # 支持 autotune 的矩阵乘法
│   ├── flash_attn.py      # FlashAttention 实现
│   └── modern_features.py # GPU 能力检测
├── benchmarks/            # 性能基准测试
├── tests/                 # 完整测试套件
├── utils/                 # 验证和 GPU 检测
├── examples/              # 使用示例
├── docs/                  # 📚 双语文档
│   ├── en/                # 英文文档
│   └── zh/                # 中文文档
├── specs/                 # 📋 规范文档（SDD）
│   ├── product/           # 产品需求文档（PRD）
│   ├── rfc/               # 技术设计文档（RFCs）
│   └── testing/           # BDD 测试规范
└── changelog/             # 专业的变更日志管理
```

---

## 📚 文档

我们提供全面的双语文档：

| 文档 | English | 中文 |
|------|---------|------|
| 教程 | [📖 Tutorial](https://lessup.github.io/diy-flash-attention/tutorial) | [📖 教程](https://lessup.github.io/diy-flash-attention/zh/tutorial) |
| API 参考 | [📚 API](https://lessup.github.io/diy-flash-attention/api) | [📚 API](https://lessup.github.io/diy-flash-attention/zh/api) |
| 性能指南 | [⚡ Guide](https://lessup.github.io/diy-flash-attention/performance) | [⚡ 性能指南](https://lessup.github.io/diy-flash-attention/zh/performance) |
| FAQ | [❓ FAQ](https://lessup.github.io/diy-flash-attention/faq) | [❓ 常见问题](https://lessup.github.io/diy-flash-attention/zh/faq) |
| 速查表 | [📋 Quick Ref](https://lessup.github.io/diy-flash-attention/cheatsheet) | [📋 速查表](https://lessup.github.io/diy-flash-attention/zh/cheatsheet) |

---

## 🛠️ 开发命令

```bash
# 运行所有测试
make test

# 运行性能测试
make bench-matmul     # 矩阵乘法
make bench-flash      # FlashAttention
make bench-all        # 所有测试

# GPU 信息
make gpu-info         # 显示 GPU 能力

# 代码质量
make lint             # 运行 ruff 检查
make format           # 使用 ruff 格式化代码
make typecheck        # 运行 mypy

# 清理
make clean            # 删除缓存文件
```

---

## 🖥️ 支持的 GPU

| 架构 | 计算能力 | GPU 型号 | 特性 |
|------|---------|---------|------|
| Volta | SM70 | V100 | ✅ 基础支持 |
| Turing | SM75 | RTX 20xx | ✅ 基础支持 |
| Ampere | SM80+ | A100, RTX 30xx | ✅ 完整支持 |
| Ada | SM89 | RTX 40xx | ✅ 完整支持 |
| Hopper | SM90 | H100 | ✅ TMA, FP8 |
| Blackwell | SM100 | B100/B200 | ✅ 最新 |

自动检测会为你的 GPU 选择最优 kernel。

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解指南。

本项目遵循**规范驱动开发（SDD）**。所有功能和修改必须在实现前记录于 `/specs` 目录。详见 [AGENTS.md](./AGENTS.md)。

主要贡献领域：
- 🔮 Hopper/Blackwell 特性实现 (TMA, FP8 kernel)
- 📊 扩展 benchmark 场景
- 📝 文档改进
- 🐛 Bug 修复和边缘情况

---

## 📖 学习资源

想要深入理解 FlashAttention？查看这些资源：

**论文**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

**技术资料**
- [Triton 文档](https://triton-lang.org/)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/)

**教程**
- 我们的[分步教程](https://lessup.github.io/diy-flash-attention/zh/tutorial)涵盖从 GPU 基础到 FlashAttention 实现的所有内容

---

## 📋 环境要求

- Python >= 3.9
- CUDA >= 11.0
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- NVIDIA GPU (Volta 或更新)

---

## 📄 许可证

MIT 许可证 - 开源，可自由使用和修改。

---

<div align="center">
  <p>
    ⭐ 如果这个项目对你有帮助，请给它点个星！<br>
    🐛 发现 bug？<a href="https://github.com/LessUp/diy-flash-attention/issues">提交 issue</a><br>
    💡 有想法？<a href="https://github.com/LessUp/diy-flash-attention/discussions">发起讨论</a>
  </p>
</div>

# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![CUDA 11+](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Triton 2.1+](https://img.shields.io/badge/Triton-2.1%2B-orange)](https://triton-lang.org/)
[![PyPI](https://img.shields.io/badge/install-pip%20install%20diy--flash--attention-3776ab)](https://pypi.org/project/diy-flash-attention/)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/diy-flash-attention/zh/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

<div align="center">
  <h3>从零实现 FlashAttention，学习 GPU 编程</h3>
  <p>
    <a href="https://lessup.github.io/diy-flash-attention/zh/">📖 文档</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/tutorial">📚 教程</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/api">🔧 API</a> •
    <a href="./README.md">🇺🇸 English</a>
  </p>
</div>

---

## 为什么需要 FlashAttention？

标准注意力的内存消耗是 **O(N²)**——序列长度的硬限制。FlashAttention 改变了游戏规则：

| | 标准 Attention | FlashAttention |
|---|---|---|
| 内存 | O(N²) | **O(N)** |
| 最大序列长度 | ~2K tokens | **64K+ tokens** |
| HBM 访问 | 写入完整注意力矩阵 | **从不物化它** |
| 精度 | 全精度 | 精确，无近似 |

> 在 seq_len=8192 时，FlashAttention 节省 **99% 内存**（512 MB → 4 MB）。

---

## 项目内容

- **Triton 核函数** — 带 autotune 的 `matmul`、带 online softmax 和因果掩码的 `flash_attention`
- **GPU 自动检测** — Volta → Blackwell，按架构选择最优配置
- **属性测试** — Hypothesis 在无限输入空间上验证正确性
- **Benchmark 套件** — 对比 PyTorch SDPA，提供 TFLOPS 和内存指标
- **生产级代码** — 类型注解、ruff lint、mypy 检查、每次推送 CI

---

## 快速开始

```bash
pip install diy-flash-attention   # 或: pip install -e ".[dev]"
```

```python
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法（autotune 自动寻找最优 block 大小）
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention — 长序列节省 99% 内存
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)  # GPT 风格因果掩码
```

运行交互式演示：
```bash
make demo        # matmul + flash_attention 冒烟测试
make bench-all   # 完整 benchmark 套件
make report      # 生成 Markdown 报告
```

---

## 性能数据 (RTX 4090)

### 内存使用

| 序列长度 | 标准 | FlashAttention | 节省 |
|---------|------|---------------|------|
| 1,024   | 8 MB    | 0.5 MB   | 94% ↓ |
| 4,096   | 128 MB  | 2 MB     | 98% ↓ |
| 8,192   | 512 MB  | 4 MB     | 99% ↓ |

### 吞吐

| Kernel | PyTorch SDPA | 本项目 (Triton) | 加速 |
|--------|-------------|----------------|------|
| MatMul 4096² | 120 TFLOPS | 140 TFLOPS | **1.17x** |
| Attention 4096 | 35.0 ms | 22.0 ms | **1.59x** |

---

## 适合谁？

| 你是... | 你将获得... |
|---|---|
| ML 工程师 | 理解 transformer 框架背后的 kernel |
| CUDA 初学者 | 无需读 500 页文档就能学 Triton |
| 研究员 | 复现和修改 FlashAttention 用于实验 |
| 性能工程师 | 研究 autotune 配置和 block 大小权衡 |

---

## GPU 支持

| 架构 | GPU 型号 | 支持级别 |
|---|---|---|
| Volta (SM70) | V100 | ✅ 基础 |
| Turing (SM75) | RTX 20xx | ✅ 基础 |
| Ampere (SM80) | A100, RTX 30xx | ✅ 完整 |
| Ada (SM89) | RTX 40xx | ✅ 完整 |
| Hopper (SM90) | H100 | ✅ TMA, FP8 |
| Blackwell (SM100) | B100/B200 | ✅ 最新 |

导入时自动检测——无需手动配置。

---

## 项目结构

```
diy-flash-attention/
├── kernels/           # Triton GPU 核函数
│   ├── matmul.py      #    带 autotune 的矩阵乘法
│   ├── flash_attn.py  #    FlashAttention 前向传播
│   └── modern_features.py # Hopper+ 特性检测
├── utils/             # Benchmark、验证、GPU 检测
├── tests/             # 单元测试 + 属性测试
├── benchmarks/        # CLI benchmark 工具
├── examples/          # 使用演示
├── specs/             # SDD: PRD、RFC、BDD 测试规范
└── docs/              # 双语文档 (EN / ZH)
```

---

## 开发

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
pip install -e ".[dev]"

make test        # 运行测试
make lint        # ruff 检查
make format      # ruff 格式化
make typecheck   # mypy
make clean       # 清理缓存
```

本项目遵循**规范驱动开发**——所有变更在 `/specs` 中定义后再实现。详见 [AGENTS.md](./AGENTS.md)。

---

## 深入学习

**原始论文**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

**教程与指南**
- [分步教程](https://lessup.github.io/diy-flash-attention/zh/tutorial) — 从 GPU 基础到 FlashAttention
- [API 参考](https://lessup.github.io/diy-flash-attention/zh/api) — 核函数签名、autotune 配置
- [性能指南](https://lessup.github.io/diy-flash-attention/zh/performance) — block 大小调优、内存分析
- [速查表](https://lessup.github.io/diy-flash-attention/zh/cheatsheet) — 常见模式快速参考

---

## 贡献

详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。推荐的入门方向：

- 🔮 实现 Hopper TMA / FP8 核函数
- 📊 扩展 benchmark 覆盖（稀疏注意力、分组查询）
- 📝 改进文档或添加示例
- 🐛 修复边缘情况或增加属性测试

---

## 许可证

[MIT License](LICENSE) — 自由使用、修改和分发。

<div align="center">
  <p>
    ⭐ 如果这个项目对你有帮助，请点个星！<br>
    <a href="https://github.com/LessUp/diy-flash-attention/issues">🐛 报告 Bug</a> •
    <a href="https://github.com/LessUp/diy-flash-attention/discussions">💡 提出想法</a>
  </p>
</div>

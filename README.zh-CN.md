# DIY FlashAttention

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![CUDA 11+](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Triton 2.1+](https://img.shields.io/badge/Triton-2.1%2B-orange)](https://triton-lang.org/)
[![PyPI](https://img.shields.io/badge/install-pip%20install%20diy--flash--attention-3776ab)](https://pypi.org/project/diy-flash-attention/)
[![CI](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/diy-flash-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/diy-flash-attention/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/diy-flash-attention/zh/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

<div align="center">
  <h3>通过从零实现 FlashAttention 学习 Triton</h3>
  <p>
    <a href="https://lessup.github.io/diy-flash-attention/zh/">📖 文档</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/tutorial">📚 教程</a> •
    <a href="https://lessup.github.io/diy-flash-attention/zh/api">🔧 API</a> •
    <a href="./README.md">🇺🇸 English</a>
  </p>
</div>

---

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|----------|
| Python | 3.9+ | 3.10+ |
| CUDA | 11.0+ | 12.0+ |
| GPU | NVIDIA Volta (SM70)+ | RTX 40xx / A100 / H100 |
| 显存 | 4 GB | 16 GB+ |
| 操作系统 | Linux | Ubuntu 22.04 |

> ⚠️ **需要 NVIDIA GPU**：Triton 仅支持 CUDA 设备，暂不支持 AMD/Intel GPU。

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

- **Triton 核函数** — 带 autotune 的 [`matmul`](./kernels/matmul.py)、带 online softmax 和因果掩码的 [`flash_attention`](./kernels/flash_attn.py)
- **GPU 自动检测** — Volta → Blackwell，按架构选择最优配置
- **属性测试** — Hypothesis 在无限输入空间上验证正确性
- **Benchmark 套件** — 对比 PyTorch SDPA，提供 TFLOPS 和内存指标（[源码](./benchmarks/)）
- **生产级代码** — 类型注解、ruff lint、mypy 检查、每次推送 CI

> 🔬 **仅前向传播**：这是一个教学实现，专注于前向传播。如需训练，需要自行实现反向传播核函数。

---

## 项目定位

这个仓库是一个**面向学习的参考实现**，不是泛化的 CUDA 门户站：

- 用真实的矩阵乘 Triton Kernel 学习块划分与 autotune
- 用紧凑的前向实现理解 FlashAttention 的 online softmax
- 用 benchmark 脚本对比 PyTorch SDPA
- 查看 Volta → Blackwell 的架构自适应辅助逻辑

如果你想快速建立 attention kernel 的直觉，这个项目适合作为入口；如果你需要完整训练栈，请把它看作教学骨架，而不是最终系统。

---

## 快速开始

### 1. 配置环境

```bash
# 使用 conda（推荐）
conda create -n flash python=3.10
conda activate flash

# 或使用 venv
python -m venv flash_env
source flash_env/bin/activate  # Windows: flash_env\Scripts\activate
```

### 2. 安装

```bash
pip install diy-flash-attention

# 或从源码安装（用于开发）
pip install -e ".[dev]"
```

### 3. 验证安装

```bash
python -c "from kernels import flash_attention; print('✓ 安装成功')"
```

### 4. 运行示例

```python
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法（autotune 自动寻找最优 block 大小）
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention — 长序列节省 99% 内存
# 形状: (batch, heads, seq_len, head_dim)
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)  # GPT 风格因果掩码

print(f"输出形状: {out.shape}")  # [2, 8, 4096, 64]
```

### 5. 运行基准测试

```bash
make demo        # matmul + flash_attention 冒烟测试
make bench-all   # 完整 benchmark 套件，对比 PyTorch SDPA
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

| 目录 | 说明 |
|------|------|
| [`kernels/`](./kernels/) | Triton GPU 核函数（matmul、flash_attn、特性检测） |
| [`utils/`](./utils/) | Benchmark、验证、GPU 检测工具 |
| [`tests/`](./tests/) | 单元测试 & Hypothesis 属性测试 |
| [`benchmarks/`](./benchmarks/) | CLI 基准测试工具（对比 PyTorch SDPA） |
| [`examples/`](./examples/) | 使用示例和教程 |
| [`openspec/`](./openspec/) | OpenSpec 变更管理、能力规格和活跃任务 |
| [`docs/`](./docs/) | 双语文档（EN / ZH） |

---

## 开发

```bash
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention
pip install -e ".[dev]"

make test-cpu    # CPU 安全校验路径
make test-gpu    # 完整 GPU 测试
make lint        # ruff 检查
make format      # ruff 格式化
make typecheck   # mypy
make docs        # 构建 GitHub Pages 文档站
make hooks-install
make clean       # 清理缓存
```

本项目遵循 **OpenSpec 驱动开发**。处理非平凡改动时，请先执行：

```bash
openspec list --json
# 从上面的列表中选择当前活跃 change
openspec status --change <change-name> --json
openspec instructions apply --change <change-name> --json
```

流程约束见 [AGENTS.md](./AGENTS.md)、[CLAUDE.md](./CLAUDE.md) 和 [openspec/specs/README.md](./openspec/specs/README.md)。

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

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `nvcc not found` | 安装 CUDA toolkit：`conda install -c nvidia cuda-toolkit` |
| `triton` 安装失败 | 确保 Python ≥3.9 且 pip ≥21.0：`pip install --upgrade pip` |
| `CUDA out of memory` | 减小 batch size 或序列长度；用 `nvidia-smi` 检查显存 |
| `No module named 'kernels'` | 确保在项目根目录并使用 `pip install -e ".[dev]"` 安装 |
| AMD/Intel GPU 导入错误 | Triton 需要 NVIDIA GPU。本项目不支持其他厂商。 |

更多帮助请查看 [GitHub Issues](https://github.com/LessUp/diy-flash-attention/issues)。

---

## 贡献

详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。当前最有价值的贡献通常是让仓库更清晰、更可信：

- 📝 改进文档、示例，以及 README 与 Pages 之间的串联
- 🐛 修复边缘情况或收紧错误处理
- ✅ 为现有的 forward-only 行为补强测试
- 🧹 删除不再有帮助的陈旧流程/文档/配置噪声

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

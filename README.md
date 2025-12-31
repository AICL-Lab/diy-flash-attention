# DIY FlashAttention 🚀

使用 Python + OpenAI Triton 从零实现 FlashAttention 算法。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目目标

1. **理解 Triton 编程模型** - 通过实现矩阵乘法 Kernel 学习 Block 指针运算和 Tiling
2. **复现 FlashAttention** - 实现 LLM 中最核心的注意力机制加速算法
3. **性能对比** - 通过 Benchmark 量化优化效果，感受 Block Size 对性能的影响
4. **支持现代 GPU** - 自动检测 GPU 架构，支持 Hopper/Blackwell 特性

## 环境要求

- Python >= 3.8
- CUDA >= 11.0
- PyTorch >= 2.0.0
- OpenAI Triton >= 2.1.0
- NVIDIA GPU (推荐 Ampere 或更新架构)

### 推荐 GPU

| 架构 | GPU 型号 | 特性支持 |
|------|---------|---------|
| Ampere (SM80) | A100, RTX 30xx | 基础 Triton 支持 |
| Hopper (SM90) | H100 | TMA, Warpgroup MMA, FP8 |
| Blackwell (SM100) | B100/B200 | 最新优化 |

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd diy-flash-attention

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或者使用 pip 安装（开发模式）
pip install -e ".[dev]"
```

## 快速开始

```bash
# 运行快速演示
python examples/quick_start.py
```

```python
# 或者在 Python 中使用
import torch
from kernels import triton_matmul, flash_attention

# 矩阵乘法
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## 项目结构

```
diy-flash-attention/
├── kernels/           # Triton GPU Kernels
│   ├── matmul.py      # 矩阵乘法 Kernel (含 autotune)
│   └── flash_attn.py  # FlashAttention Kernel (含 online softmax)
├── benchmarks/        # 性能测试脚本
│   ├── bench_matmul.py
│   └── bench_flash.py
├── tests/             # 单元测试
│   ├── test_matmul.py
│   └── test_flash.py
├── utils/             # 工具函数
│   ├── benchmark.py   # Benchmark 工具类
│   ├── validation.py  # 数值验证工具
│   └── gpu_detect.py  # GPU 检测 (支持 Hopper/Blackwell)
├── examples/          # 示例代码
│   └── quick_start.py
├── pyproject.toml     # 项目配置
└── requirements.txt
```

## 快速开始

### 1. 检测 GPU 能力

```python
from utils import detect_gpu, print_gpu_info

caps = detect_gpu()
print_gpu_info(caps)
# 输出:
# GPU Information
# Name:                NVIDIA GeForce RTX 4090
# Architecture:        sm_89
# TMA 支持:            ✗
# FP8 支持:            ✗
```

### 2. 运行矩阵乘法 Benchmark

```bash
python benchmarks/bench_matmul.py
python benchmarks/bench_matmul.py --test-block-sizes  # 测试不同 block size
```

输出示例：
```
Matrix Multiplication Benchmark
================================
Size (M×K×N)    | Triton (TFLOPS) | PyTorch (TFLOPS) | Speedup
----------------|-----------------|------------------|--------
1024×1024×1024  | 45.2            | 42.1             | 1.07x
2048×2048×2048  | 78.5            | 71.3             | 1.10x
4096×4096×4096  | 112.3           | 98.7             | 1.14x
```

### 3. 运行 FlashAttention Benchmark

```bash
python benchmarks/bench_flash.py
```

### 4. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_matmul.py -v
```

## 核心概念

### Tiling (分块)

GPU 计算的核心优化策略，将大矩阵分割成小块在 SRAM 中计算：

```
┌─────────────────────────────────────────┐
│  HBM (慢, 大)  ──→  SRAM (快, 小)  ──→  计算  │
└─────────────────────────────────────────┘
```

### Online Softmax

FlashAttention 的核心算法，允许分块计算 softmax 而无需存储完整的 N×N attention matrix：

```
标准 Attention: O(N²) 内存
FlashAttention: O(N) 内存
```

### Block Size 调优

Block Size 是影响性能的关键参数：

```python
# 尝试不同的 Block Size
from kernels import triton_matmul

# 小 Block Size - 更多并行度，但更多内存访问
c1 = triton_matmul(a, b, block_m=64, block_n=64, block_k=32)

# 大 Block Size - 更好的数据复用，但可能受限于 SRAM
c2 = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)
```

## 参考资料

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Triton Documentation](https://triton-lang.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## License

MIT

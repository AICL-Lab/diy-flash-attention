# DIY FlashAttention 速查表

快速查找常用 API、命令和配置。

## 🚀 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行演示
make demo

# 运行测试
make test
```

## 📦 核心 API

### 矩阵乘法

```python
from kernels import triton_matmul

# 基本用法 (自动选择最优配置)
c = triton_matmul(a, b)

# 指定 block size
c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)

# 支持的数据类型
a = torch.randn(..., dtype=torch.float16)   # ✅ 推荐
a = torch.randn(..., dtype=torch.bfloat16)  # ✅ 支持
a = torch.randn(..., dtype=torch.float32)   # ⚠️ 内部转 float16
```

### FlashAttention

```python
from kernels import flash_attention

# 基本用法
out = flash_attention(q, k, v)

# 因果注意力 (用于自回归模型)
out = flash_attention(q, k, v, causal=True)

# 变长序列
seq_lens = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
out = flash_attention(q, k, v, seq_lens=seq_lens)

# 3D 输入: (batch*heads, seq_len, head_dim)
q_3d = torch.randn(16, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q_3d, k_3d, v_3d)
```

### GPU 检测

```python
from utils import detect_gpu, print_gpu_info

caps = detect_gpu()
print_gpu_info(caps)

# 检查特性
print(f"TMA: {caps.has_tma}")
print(f"FP8: {caps.has_fp8}")
```

### Benchmark

```python
from utils import BenchmarkRunner

runner = BenchmarkRunner(warmup=10, rep=50)

# 矩阵乘法
results = runner.benchmark_matmul(
    triton_matmul,
    sizes=[(1024, 1024, 1024), (2048, 2048, 2048)],
)
runner.print_comparison_table(results)

# FlashAttention
results = runner.benchmark_attention(
    flash_attention,
    seq_lengths=[512, 1024, 2048],
)
```

### 验证

```python
from utils import validate_matmul, validate_attention

# 验证矩阵乘法
is_valid, max_diff = validate_matmul(
    triton_matmul, m=1024, n=1024, k=1024
)

# 验证 FlashAttention
is_valid, max_diff = validate_attention(
    flash_attention, batch=2, heads=8, seq_len=512, head_dim=64
)
```

## 🔧 常用命令

| 命令 | 说明 |
|------|------|
| `make demo` | 快速演示 |
| `make test` | 运行所有测试 |
| `make bench-all` | 运行所有 benchmark |
| `make gpu-info` | 显示 GPU 信息 |
| `make experiment` | Block Size 实验 |
| `make lint` | 代码检查 |
| `make format` | 代码格式化 |
| `make clean` | 清理缓存 |

## 📐 输入形状

### 矩阵乘法

```
A: (M, K)  ×  B: (K, N)  →  C: (M, N)
```

### FlashAttention

```
4D 输入: (batch, heads, seq_len, head_dim)
3D 输入: (batch*heads, seq_len, head_dim)

seq_lens: (batch,) 指定每个样本的有效长度
head_dim: 支持 32 或 64
```

## 📊 Block Size 推荐

| 矩阵大小 | BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-1024 | 64 | 64 | 32 |
| 1024-2048 | 64 | 128 | 32 |
| 2048-4096 | 128 | 128 | 32 |
| > 4096 | 128 | 256 | 64 |

**提示**: 使用 autotune (不指定 block size) 可自动选择最优配置。

## 🎨 数据类型

| 类型 | 精度 | 性能 | 推荐场景 |
|------|------|------|---------|
| `float16` | 中 | 2x | 训练/推理 (推荐) |
| `bfloat16` | 中 | 2x | 训练 (更稳定) |
| `float32` | 高 | 1x | 调试/高精度 |
| `float8` | 低 | 4x | 推理 (Hopper+) |

## 🖥️ GPU 架构

| 架构 | SM | GPU | 特性 |
|------|-----|-----|------|
| Ampere | 80 | A100, RTX 30xx | 完整支持 |
| Ada | 89 | RTX 40xx | 完整支持 |
| Hopper | 90 | H100 | TMA, FP8 |
| Blackwell | 100 | B100 | 最新 |

## 💾 内存复杂度

| 方法 | 内存 | 说明 |
|------|------|------|
| 标准 Attention | O(N²) | 存储完整 attention matrix |
| FlashAttention | O(N) | 分块计算，不存储完整矩阵 |

**内存节省**: 长序列可节省 **99%** 内存！

## ⚠️ 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `Expected 2D tensors` | matmul 输入不是 2D | 使用 `.view()` 或 `.reshape()` |
| `Incompatible dimensions` | A.shape[1] != B.shape[0] | 检查矩阵维度 |
| `CUDA tensors required` | 输入在 CPU 上 | 使用 `.cuda()` 或 `.to("cuda")` |
| `Q, K, V shapes must match` | 形状不一致 | 确保 Q, K, V 形状相同 |
| `Expected 3D or 4D tensors` | attention 输入维度错误 | 检查输入形状 |
| `Unsupported head_dim` | head_dim 不是 32/64 | 使用 32 或 64 |
| `Unsupported dtype` | dtype 不支持 | 使用 float16/bfloat16/float32 |
| `dtypes must match` | dtype 不一致 | 统一 dtype |

## ✅ 性能检查清单

```
□ 使用 FP16 或 BF16
□ 使用 autotune (不指定 block size)
□ 预热 kernel (运行几次后计时)
□ 确保输入是连续的 (.is_contiguous())
□ 数据保持在 GPU 上
□ 矩阵足够大 (> 512)
□ 避免循环内同步
□ 避免循环内 CPU-GPU 数据移动
```

## 📁 文件结构

```
kernels/
├── matmul.py          # 矩阵乘法
├── flash_attn.py      # FlashAttention
└── modern_features.py # 现代特性

utils/
├── benchmark.py       # Benchmark 工具
├── validation.py      # 验证工具
└── gpu_detect.py      # GPU 检测

tests/
├── test_matmul.py     # 矩阵乘法测试
├── test_flash.py      # FlashAttention 测试
├── test_properties.py # 属性测试
└── test_error_handling.py # 错误处理测试
```

## 🔗 链接

- 📖 [教程](tutorial.md)
- 📚 [API 参考](api.md)
- 📊 [性能指南](performance.md)
- ❓ [FAQ](faq.md)
- 💻 [GitHub](https://github.com/LessUp/diy-flash-attention)

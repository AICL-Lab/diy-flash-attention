# DIY FlashAttention 速查表

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 运行演示
make demo

# 运行测试
make test
```

## 核心 API

### 矩阵乘法

```python
from kernels import triton_matmul

# 基本用法
c = triton_matmul(a, b)

# 指定 block size
c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)
```

### FlashAttention

```python
from kernels import flash_attention

# 基本用法
out = flash_attention(q, k, v)

# Causal attention
out = flash_attention(q, k, v, causal=True)
```

### GPU 检测

```python
from utils import detect_gpu, print_gpu_info

caps = detect_gpu()
print_gpu_info(caps)
```

### Benchmark

```python
from utils import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.benchmark_matmul(triton_matmul, sizes=[(1024, 1024, 1024)])
runner.print_comparison_table(results)
```

### 验证

```python
from utils import validate_matmul, validate_attention

is_valid, max_diff = validate_matmul(triton_matmul, m=512, n=512, k=512)
is_valid, max_diff = validate_attention(flash_attention, batch=2, heads=4, seq_len=128, head_dim=64)
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `make demo` | 快速演示 |
| `make test` | 运行测试 |
| `make bench-all` | 运行所有 benchmark |
| `make gpu-info` | 显示 GPU 信息 |
| `make experiment` | Block Size 实验 |
| `make advanced` | 高级用法示例 |
| `make report` | 生成 benchmark 报告 |
| `make clean` | 清理缓存 |

## 输入形状

### 矩阵乘法

```
A: (M, K)  ×  B: (K, N)  →  C: (M, N)
```

### FlashAttention

```
Q, K, V: (batch, heads, seq_len, head_dim)
Output:  (batch, heads, seq_len, head_dim)
```

## Block Size 推荐

| 矩阵大小 | BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-2048 | 64 | 128 | 32 |
| > 2048 | 128 | 256 | 64 |

## 数据类型

| 类型 | 精度 | 性能 | 用途 |
|------|------|------|------|
| float32 | 高 | 1x | 高精度计算 |
| float16 | 中 | 2x | 训练/推理 |
| bfloat16 | 中 | 2x | 训练 |
| float8 | 低 | 4x | 推理 (Hopper+) |

## GPU 架构

| 架构 | SM | GPU | 特性 |
|------|-----|-----|------|
| Volta | 70 | V100 | 基础 |
| Turing | 75 | RTX 20xx | 基础 |
| Ampere | 80+ | A100, RTX 30xx | 完整 |
| Ada | 89 | RTX 40xx | 完整 |
| Hopper | 90 | H100 | TMA, FP8 |
| Blackwell | 100 | B100 | 最新 |

## 内存复杂度

| 方法 | 内存 | 说明 |
|------|------|------|
| 标准 Attention | O(N²) | 存储完整 attention matrix |
| FlashAttention | O(N) | 分块计算，不存储完整矩阵 |

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Incompatible dimensions` | A.shape[1] != B.shape[0] | 检查矩阵维度 |
| `Expected 2D tensors` | 输入不是 2D | 使用 2D 张量 |
| `Block sizes must be positive` | block size <= 0 | 使用正数 |
| `Q, K, V shapes must match` | 形状不一致 | 确保 Q, K, V 形状相同 |
| `Expected 3D or 4D tensors` | attention 输入维度错误 | 使用 3D 或 4D 张量 |

## 性能提示

1. ✅ 使用 FP16
2. ✅ 使用 autotune
3. ✅ 预热 kernel
4. ✅ 批量处理
5. ❌ 避免小矩阵
6. ❌ 避免频繁同步
7. ❌ 避免 CPU-GPU 数据移动

## 文件结构

```
kernels/
├── matmul.py          # 矩阵乘法
├── flash_attn.py      # FlashAttention
└── modern_features.py # 现代特性

utils/
├── benchmark.py       # Benchmark
├── validation.py      # 验证
└── gpu_detect.py      # GPU 检测

tests/
├── test_matmul.py     # 矩阵乘法测试
├── test_flash.py      # FlashAttention 测试
└── test_properties.py # 属性测试
```

## 链接

- [教程](tutorial.md)
- [API 参考](api.md)
- [性能指南](performance.md)
- [FAQ](faq.md)

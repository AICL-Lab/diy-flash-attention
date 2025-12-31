# 性能指南

本文档介绍如何优化 DIY FlashAttention 的性能。

## 性能基准

### 矩阵乘法 (MatMul)

典型性能数据 (RTX 4090, FP16):

| 矩阵大小 | PyTorch (TFLOPS) | Triton (TFLOPS) | 加速比 |
|---------|-----------------|-----------------|--------|
| 1024×1024 | 45 | 48 | 1.07x |
| 2048×2048 | 85 | 95 | 1.12x |
| 4096×4096 | 120 | 140 | 1.17x |
| 8192×8192 | 150 | 175 | 1.17x |

### FlashAttention

典型性能数据 (RTX 4090, FP16, batch=4, heads=8, head_dim=64):

| 序列长度 | PyTorch SDPA (ms) | FlashAttention (ms) | 加速比 |
|---------|------------------|---------------------|--------|
| 512 | 0.8 | 0.7 | 1.14x |
| 1024 | 2.5 | 2.0 | 1.25x |
| 2048 | 9.0 | 6.5 | 1.38x |
| 4096 | 35.0 | 22.0 | 1.59x |

### 内存使用

| 序列长度 | 标准 Attention | FlashAttention | 节省 |
|---------|---------------|----------------|------|
| 1024 | 8 MB | 0.5 MB | 94% |
| 2048 | 32 MB | 1 MB | 97% |
| 4096 | 128 MB | 2 MB | 98% |
| 8192 | 512 MB | 4 MB | 99% |

## 优化策略

### 1. Block Size 调优

Block Size 是影响性能的最关键参数。

#### 原则

- **小 Block Size**: 更多并行度，但更多内存访问
- **大 Block Size**: 更好数据复用，但可能超出 SRAM

#### 推荐配置

```python
# 小矩阵 (< 1024)
config_small = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}

# 中等矩阵 (1024-4096)
config_medium = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}

# 大矩阵 (> 4096)
config_large = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}
```

#### 自动调优

```python
# 使用 autotune (推荐)
c = triton_matmul(a, b)  # 自动选择最优配置

# 或者运行实验找到最优配置
python examples/block_size_experiment.py
```

### 2. 数据类型选择

| 数据类型 | 精度 | 性能 | 推荐场景 |
|---------|------|------|---------|
| FP32 | 高 | 基准 | 需要高精度 |
| FP16 | 中 | 2x | 训练/推理 |
| BF16 | 中 | 2x | 训练 |
| FP8 | 低 | 4x | 推理 (Hopper+) |

```python
# 使用 FP16 (推荐)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# 使用 BF16 (训练更稳定)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
```

### 3. 内存访问优化

#### 确保连续内存

```python
# 不好: 非连续张量
a = some_tensor.transpose(0, 1)  # 可能非连续

# 好: 确保连续
a = some_tensor.transpose(0, 1).contiguous()
```

#### 避免不必要的拷贝

```python
# 不好: 创建新张量
b = a.to(torch.float16)  # 如果 a 已经是 float16，这会创建拷贝

# 好: 检查后转换
if a.dtype != torch.float16:
    a = a.to(torch.float16)
```

### 4. Batch 处理

#### 合并小 batch

```python
# 不好: 多次小 batch 调用
for i in range(10):
    out = flash_attention(q[i:i+1], k[i:i+1], v[i:i+1])

# 好: 一次大 batch 调用
out = flash_attention(q[:10], k[:10], v[:10])
```

#### 选择合适的 batch size

```python
# 根据 GPU 内存选择 batch size
# RTX 3090 (24GB): batch=8-16
# A100 (40GB): batch=16-32
# H100 (80GB): batch=32-64
```

### 5. 预热和缓存

#### Kernel 预热

```python
# Triton 首次运行需要编译
# 预热以获得稳定性能
for _ in range(10):
    _ = triton_matmul(a, b)
torch.cuda.synchronize()

# 然后进行实际计算
result = triton_matmul(a, b)
```

#### 清理缓存

```python
# 在 benchmark 前清理缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

## 性能分析

### 使用 Triton Profiler

```python
import triton

# 启用性能分析
with triton.profiler.profile() as prof:
    result = triton_matmul(a, b)

# 打印结果
prof.print()
```

### 使用 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    result = triton_matmul(a, b)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 使用 NVIDIA Nsight

```bash
# 使用 Nsight Systems
nsys profile python benchmarks/bench_matmul.py

# 使用 Nsight Compute
ncu --set full python benchmarks/bench_matmul.py
```

## GPU 架构优化

### Ampere (SM80)

```python
# 推荐配置
config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 3,
    "num_warps": 8,
}
```

### Hopper (SM90)

```python
# 利用更大的 SRAM
config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 4,  # 更多 stages
    "num_warps": 8,
}

# 启用 TMA (如果可用)
from kernels import check_hopper_features
if check_hopper_features()["tma_available"]:
    # 使用 TMA 优化版本
    pass
```

## 常见性能陷阱

### 1. 过小的矩阵

```python
# 不好: 矩阵太小，kernel launch 开销占主导
a = torch.randn(32, 32, device="cuda")

# 好: 使用足够大的矩阵
a = torch.randn(1024, 1024, device="cuda")
```

### 2. 频繁的 CPU-GPU 同步

```python
# 不好: 每次操作后同步
for i in range(100):
    result = triton_matmul(a, b)
    torch.cuda.synchronize()  # 阻塞!

# 好: 批量操作后同步
for i in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()  # 只同步一次
```

### 3. 不必要的数据移动

```python
# 不好: 频繁在 CPU 和 GPU 之间移动
for i in range(100):
    a_gpu = a_cpu.cuda()
    result = triton_matmul(a_gpu, b_gpu)
    result_cpu = result.cpu()

# 好: 保持数据在 GPU 上
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()
for i in range(100):
    result = triton_matmul(a_gpu, b_gpu)
result_cpu = result.cpu()  # 最后才移动
```

## 运行 Benchmark

```bash
# 矩阵乘法 benchmark
make bench-matmul

# FlashAttention benchmark
make bench-flash

# 所有 benchmark
make bench-all

# 生成报告
make report
```

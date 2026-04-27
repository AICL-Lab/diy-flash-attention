# 性能指南

本指南详细介绍如何优化 DIY FlashAttention 的性能，涵盖配置调优、最佳实践和常见陷阱。

## 目录

- [性能基准](#性能基准)
- [Block Size 调优](#block-size-调优)
- [数据类型选择](#数据类型选择)
- [内存优化](#内存优化)
- [GPU 架构优化](#gpu-架构优化)
- [性能分析工具](#性能分析工具)
- [常见性能陷阱](#常见性能陷阱)
- [性能检查清单](#性能检查清单)

---

## 性能基准

### 矩阵乘法 (MatMul)

典型性能数据 (RTX 4090, FP16):

| 矩阵大小 | PyTorch (TFLOPS) | Triton (TFLOPS) | 加速比 | 说明 |
|---------|-----------------|-----------------|--------|------|
| 512×512 | 25 | 28 | 1.12x | 小矩阵，kernel 开销占比大 |
| 1024×1024 | 45 | 48 | 1.07x | |
| 2048×2048 | 85 | 95 | 1.12x | |
| 4096×4096 | 120 | 140 | 1.17x | 大矩阵，优势明显 |
| 8192×8192 | 150 | 175 | 1.17x | |

### FlashAttention

典型性能数据 (RTX 4090, FP16, batch=4, heads=8, head_dim=64):

| 序列长度 | PyTorch SDPA (ms) | FlashAttention (ms) | 加速比 | 内存节省 |
|---------|------------------|---------------------|--------|---------|
| 512 | 0.8 | 0.7 | 1.14x | 94% |
| 1024 | 2.5 | 2.0 | 1.25x | 97% |
| 2048 | 9.0 | 6.5 | 1.38x | 98% |
| 4096 | 35.0 | 22.0 | 1.59x | 99% |

### 内存使用对比

| 序列长度 | 标准 Attention | FlashAttention | 节省 |
|---------|---------------|----------------|------|
| 512 | 2 MB | 0.25 MB | 88% |
| 1024 | 8 MB | 0.5 MB | 94% |
| 2048 | 32 MB | 1 MB | 97% |
| 4096 | 128 MB | 2 MB | 98% |
| 8192 | 512 MB | 4 MB | 99% |

---

## Block Size 调优

Block Size 是影响 Triton kernel 性能的最关键参数。

### 核心原理

```
┌─────────────────────────────────────────────────────────────┐
│                    Block Size 权衡                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  小 Block Size:              大 Block Size:                 │
│  ┌───┬───┬───┐              ┌───────────┐                  │
│  │   │   │   │              │           │                  │
│  ├───┼───┼───┤              │           │                  │
│  │   │   │   │              │    单块   │                  │
│  ├───┼───┼───┤              │           │                  │
│  │   │   │   │              │           │                  │
│  └───┴───┴───┘              └───────────┘                  │
│                                                             │
│  ✅ 更多并行 block           ✅ 更好的数据复用               │
│  ✅ 适合小矩阵               ✅ 更少的 HBM 访问               │
│  ❌ 更多 HBM 访问            ❌ 可能超出 SRAM                │
│  ❌ 更低的数据复用           ❌ 更少的并行度                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 推荐配置

| 矩阵大小范围 | BLOCK_M | BLOCK_N | BLOCK_K | num_stages | num_warps |
|-------------|---------|---------|---------|------------|-----------|
| < 512 | 32 | 32 | 32 | 4 | 4 |
| 512 - 1024 | 64 | 64 | 32 | 4 | 4 |
| 1024 - 2048 | 64 | 128 | 32 | 4 | 4 |
| 2048 - 4096 | 128 | 128 | 32 | 4 | 4 |
| > 4096 | 128 | 256 | 64 | 3 | 8 |

### Autotune 使用

**推荐方式**：使用内置 autotune

```python
from kernels import triton_matmul

# 不指定 block size，自动选择最优配置
c = triton_matmul(a, b)
```

**实验方式**：手动测试不同配置

```bash
# 运行 Block Size 实验
python examples/block_size_experiment.py
```

```python
# 手动测试不同配置
configs = [
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
]

for config in configs:
    c = triton_matmul(a, b,
        block_m=config["BLOCK_M"],
        block_n=config["BLOCK_N"],
        block_k=config["BLOCK_K"]
    )
```

### SRAM 容量限制

检查 Block Size 是否超出 SRAM：

```python
def check_sram_usage(block_m, block_n, block_k, dtype_bytes=2):
    """
    估算 SRAM 使用量。

    SRAM 限制:
    - Ampere (SM80): ~164 KB per SM
    - Ada (SM89): ~192 KB per SM
    - Hopper (SM90): ~228 KB per SM
    """
    # 两个输入块 + 一个累加器
    a_sram = block_m * block_k * dtype_bytes
    b_sram = block_k * block_n * dtype_bytes
    acc_sram = block_m * block_n * 4  # float32 累加器

    total = a_sram + b_sram + acc_sram
    print(f"SRAM 使用量: {total / 1024:.1f} KB")

    if total > 164 * 1024:
        print("⚠️ 可能超出 Ampere SRAM 限制!")

    return total

check_sram_usage(128, 256, 64)
```

---

## 数据类型选择

### 类型对比

| 数据类型 | 范围 | 精度位 | 相对性能 | 推荐场景 |
|---------|------|-------|---------|---------|
| FP32 | ±3.4e38 | 23 位 mantissa | 1x (基准) | 高精度需求、调试 |
| FP16 | ±65504 | 10 位 mantissa | ~2x | 训练、推理 |
| BF16 | ±3.4e38 | 7 位 mantissa | ~2x | 训练 (更稳定) |
| FP8 E4M3 | ±448 | 3 位 mantissa | ~4x | 推理 (Hopper+) |
| FP8 E5M2 | ±57344 | 2 位 mantissa | ~4x | 梯度存储 (Hopper+) |

### 选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                    数据类型选择流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────┐                          │
│                    │ 需要高精度？ │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
│              ┌────────────┴────────────┐                    │
│              ▼                         ▼                    │
│         ┌────────┐               ┌────────────┐             │
│         │  FP32  │               │ Hopper GPU?│             │
│         └────────┘               └──────┬─────┘             │
│                                         │                   │
│                            ┌────────────┴────────────┐      │
│                            ▼                         ▼      │
│                       ┌────────┐               ┌────────┐   │
│                       │  FP8   │               │ 训练中? │   │
│                       └────────┘               └────┬───┘   │
│                                                     │       │
│                                        ┌────────────┴───┐   │
│                                        ▼                ▼   │
│                                   ┌────────┐       ┌──────┐ │
│                                   │  BF16  │       │ FP16 │ │
│                                   └────────┘       └──────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
# FP16 - 最常用
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# BF16 - 训练更稳定 (避免梯度溢出)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

# FP8 - Hopper+ GPU 推理
from kernels import to_fp8_e4m3
a_fp8 = to_fp8_e4m3(a)
```

---

## 内存优化

### 确保连续内存

```python
# ❌ 不好: 非连续张量会触发额外的复制
a = some_tensor.transpose(0, 1)
c = triton_matmul(a, b)  # 内部会调用 .contiguous()

# ✅ 好: 显式确保连续
a = some_tensor.transpose(0, 1).contiguous()
c = triton_matmul(a, b)
```

### 避免不必要的类型转换

```python
# ❌ 不好: 即使 a 已经是 float16，也会创建新张量
a = a.to(torch.float16)

# ✅ 好: 检查后转换
if a.dtype != torch.float16:
    a = a.to(torch.float16)
```

### 清理缓存

```python
# 在 benchmark 或大计算前清理
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

### 监控内存使用

```python
def memory_report():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"已分配: {allocated:.2f} GB")
    print(f"已预留: {reserved:.2f} GB")
    print(f"峰值:   {peak:.2f} GB")

memory_report()
```

---

## GPU 架构优化

### Ampere (SM80) - A100, RTX 30xx

```python
# 推荐配置
ampere_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 3,
    "num_warps": 8,
}

# SRAM 限制: ~164 KB per SM
```

### Ada (SM89) - RTX 40xx

```python
ada_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 4,  # 更大的 SRAM
    "num_warps": 8,
}

# SRAM: ~192 KB per SM
```

### Hopper (SM90) - H100

```python
from kernels import check_hopper_features

features = check_hopper_features()

if features["tma_available"]:
    print("TMA 可用，可使用异步数据加载")

if features["fp8_available"]:
    print("FP8 可用，可使用低精度计算")

hopper_config = {
    "BLOCK_M": 128,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "num_stages": 4,
    "num_warps": 8,
}

# SRAM: ~228 KB per SM
```

### 自动架构适配

```python
from kernels import get_matmul_config, get_attention_config

# 自动获取当前 GPU 的最优配置
matmul_config = get_matmul_config()
attention_config = get_attention_config()

print(f"MatMul 配置: {matmul_config}")
print(f"Attention 配置: {attention_config}")
```

---

## 性能分析工具

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("triton_matmul"):
        result = triton_matmul(a, b)

# 打印结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Chrome trace
prof.export_chrome_trace("trace.json")
```

### Triton 内置分析

```python
# 运行 benchmark
from utils.benchmark import benchmark_fn

median_ms, p20_ms, p80_ms = benchmark_fn(
    triton_matmul, a, b,
    warmup=25,
    rep=100,
)
print(f"Median: {median_ms:.3f} ms, P20: {p20_ms:.3f} ms, P80: {p80_ms:.3f} ms")
```

### NVIDIA Nsight

```bash
# Nsight Systems - 整体时间线分析
nsys profile -o report python benchmarks/bench_matmul.py

# 查看报告
nsys-ui report.nsys-rep

# Nsight Compute - 详细 kernel 分析
ncu --set full -o report python benchmarks/bench_matmul.py

# 查看报告
ncu-ui report.ncu-rep
```

---

## 常见性能陷阱

### 1. 过小的矩阵

```python
# ❌ 不好: 小矩阵 kernel launch 开销占主导
a = torch.randn(32, 32, device="cuda")
for _ in range(1000):
    c = triton_matmul(a, a)

# ✅ 好: 使用足够大的矩阵
a = torch.randn(1024, 1024, device="cuda")
c = triton_matmul(a, a)
```

### 2. 频繁的 CPU-GPU 同步

```python
# ❌ 不好: 每次操作后同步
for _ in range(100):
    result = triton_matmul(a, b)
    torch.cuda.synchronize()  # 阻塞等待!

# ✅ 好: 批量操作后同步
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()  # 只同步一次
```

### 3. 不必要的数据移动

```python
# ❌ 不好: 频繁 CPU-GPU 数据移动
for _ in range(100):
    a_gpu = a_cpu.cuda()       # CPU → GPU
    result = triton_matmul(a_gpu, b)
    result_cpu = result.cpu()  # GPU → CPU

# ✅ 好: 数据保持在 GPU
a_gpu = a_cpu.cuda()  # 只移动一次
for _ in range(100):
    result = triton_matmul(a_gpu, b)
result_cpu = result.cpu()  # 最后才移动
```

### 4. 首次运行冷启动

```python
# ❌ 不好: 首次运行包含编译时间
import time
start = time.time()
result = triton_matmul(a, b)  # 包含 JIT 编译!
print(f"Time: {time.time() - start:.3f}s")

# ✅ 好: 预热后计时
# 预热
for _ in range(10):
    _ = triton_matmul(a, b)
torch.cuda.synchronize()

# 计时
start = time.time()
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) / 100 * 1000:.3f} ms")
```

### 5. 非连续内存

```python
# ❌ 不好: 转置后非连续
a_t = a.t()  # 非连续!
# 内部会触发 .contiguous()，增加额外开销

# ✅ 好: 显式处理
a_t = a.t().contiguous()
```

---

## 性能检查清单

运行 Benchmark 前，确保：

```
□ 数据类型
  ├─ ☑ 使用 FP16 或 BF16
  ├─ ☑ 避免 FP32 (除非需要高精度)
  └─ ☑ 输入输出 dtype 一致

□ 内存
  ├─ ☑ 输入张量是连续的
  ├─ ☑ 数据已在 GPU 上
  └─ ☑ 调用前清理缓存 (benchmark 时)

□ 配置
  ├─ ☑ 使用 autotune (不指定 block size)
  ├─ ☑ 或根据矩阵大小选择合适的 block size
  └─ ☑ 检查 SRAM 容量限制

□ 测量
  ├─ ☑ 进行预热 (10+ 次)
  ├─ ☑ 测量多次取平均
  ├─ ☑ 使用 torch.cuda.synchronize() 确保完成
  └─ ☑ 使用 GPU 时间而非 CPU 时间

□ 代码
  ├─ ☑ 避免循环内同步
  ├─ ☑ 避免循环内 CPU-GPU 数据移动
  └─ ☑ 矩阵足够大 (> 512)
```

---

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

# Block Size 实验
make experiment
```

---

## 参考资料

- [Triton Performance Guide](https://triton-lang.org/main/programming-guide/chapter-3/performance.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [GPU Performance Background](https://developer.nvidia.com/blog/cuda-pro-tip-write-efficient-kernels-cuda-compiler/)

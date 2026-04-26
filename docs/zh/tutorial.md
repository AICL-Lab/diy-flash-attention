# DIY FlashAttention 教程

本教程将带你从零开始理解和实现 FlashAttention 算法。无论你是 GPU 编程新手还是经验丰富的开发者，都能从中获得价值。

## 📚 学习路径

```
入门 ──→ 进阶 ──→ 实战
 │         │        │
 ▼         ▼        ▼
GPU基础   FlashAttention  性能优化
Triton    原理实现       Benchmark
```

---

## 第一部分：GPU 编程基础

### 1.1 为什么需要 GPU 加速？

在大语言模型（LLM）时代，Attention 机制是最核心的计算之一：

```python
# 标准 Attention 计算
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

对于一个 **序列长度 N=8192** 的输入：
- Attention Matrix 大小：8192 × 8192 × 2 bytes = **128 MB**
- 训练时需要存储用于反向传播 → **内存爆炸**！

GPU 的并行计算能力正是解决这个问题的关键。

### 1.2 GPU 内存层次结构

理解 GPU 内存层次是优化性能的基础：

```
┌─────────────────────────────────────────────────────────────┐
│                      HBM (High Bandwidth Memory)            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  容量: 40-80 GB (A100/H100)                         │   │
│  │  带宽: 1.5-3.35 TB/s                                │   │
│  │  延迟: ~500 cycles (慢!)                            │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      L2 Cache (共享)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  容量: 40-60 MB                                     │   │
│  │  带宽: ~4 TB/s                                      │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│           SRAM (Shared Memory, 每个 SM 独立)                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  容量: 164-228 KB per SM                            │   │
│  │  带宽: ~19 TB/s (最快!)                             │   │
│  │  ⚡ FlashAttention 的关键优化目标                   │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Registers                              │
│  │  容量: ~256 KB per SM                               │   │
│  │  延迟: 1 cycle                                      │   │
└─────────────────────────────────────────────────────────────┘
```

**关键洞察**: 
- HBM 容量大但慢，SRAM 容量小但快
- FlashAttention 的核心思想：**让数据尽可能留在 SRAM 中**

### 1.3 GPU 执行模型

```
┌─────────────────────────────────────────────────────────────┐
│                         Grid                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Block 0   │  │   Block 1   │  │   Block 2   │  ...    │
│  │ ┌───┬───┐   │  │ ┌───┬───┐   │  │ ┌───┬───┐   │         │
│  │ │Warp│Warp│  │  │ │Warp│Warp│  │  │ │Warp│Warp│  │         │
│  │ │ 0 │ 1 │   │  │ │ 0 │ 1 │   │  │ │ 0 │ 1 │   │         │
│  │ └───┴───┘   │  │ └───┴───┘   │  │ └───┴───┘   │         │
│  │  Shared     │  │  Shared     │  │  Shared     │         │
│  │  Memory     │  │  Memory     │  │  Memory     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘

Thread: 最小执行单元
Warp:  32 个 thread，同时执行相同指令 (SIMT)
Block: 多个 warp，共享 SRAM
Grid:  多个 block，覆盖整个计算
```

**Triton 中的对应关系**：

```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)  # 当前 block 的 ID
    # 每个 "program" 对应一个 block
    # block 内部的并行由 Triton 自动处理
```

---

## 第二部分：Triton 入门

### 2.1 为什么选择 Triton？

| 特性 | CUDA C++ | Triton |
|------|----------|--------|
| 内存分块 | 手动管理 | 自动处理 |
| 合并访问 | 需要精心设计 | 自动优化 |
| 共享内存 | 手动分配 | 自动管理 |
| 同步 | 手动 `__syncthreads()` | 自动处理 |
| 学习曲线 | 陡峭 | 平缓 |

**结论**: Triton 让你专注于算法，而非底层优化。

### 2.2 第一个 Triton Kernel

让我们从一个简单的向量加法开始：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,        # 输入向量 X 的指针
    y_ptr,        # 输入向量 Y 的指针
    output_ptr,   # 输出向量的指针
    n_elements,   # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    """
    向量加法: output = x + y
    
    关键概念:
    1. tl.program_id(0): 获取当前 block 的 ID
    2. tl.arange(): 创建索引序列
    3. mask: 处理边界条件
    4. tl.load/tl.store: 内存读写
    """
    # 1. 获取当前 block 的 ID
    pid = tl.program_id(0)
    
    # 2. 计算这个 block 负责的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. 创建 mask 处理边界 (当 n_elements 不是 BLOCK_SIZE 的倍数)
    mask = offsets < n_elements
    
    # 4. 加载数据 (只加载有效的元素)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # 5. 计算
    output = x + y
    
    # 6. 存储结果 (只存储有效的元素)
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法的 wrapper 函数"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 计算 grid 大小
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动 kernel
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=1024,  # 每个 block 处理 1024 个元素
    )
    
    return output


# 使用示例
if __name__ == "__main__":
    x = torch.randn(10000, device="cuda", dtype=torch.float16)
    y = torch.randn(10000, device="cuda", dtype=torch.float16)
    
    result = add(x, y)
    expected = x + y
    
    print(f"Max diff: {(result - expected).abs().max().item():.2e}")
```

### 2.3 Triton 核心概念详解

#### `tl.constexpr` - 编译时常量

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE 在编译时确定，可以用于:
    # 1. 数组大小
    offsets = tl.arange(0, BLOCK_SIZE)  # ✅ 正确
    
    # 2. 控制流
    if BLOCK_SIZE > 64:  # ✅ 编译时分支
        ...
```

#### `tl.program_id()` - Block 标识

```python
# 1D Grid
pid = tl.program_id(0)  # 第 0 维的 block ID

# 2D Grid
pid_m = tl.program_id(0)  # 第 0 维
pid_n = tl.program_id(1)  # 第 1 维
```

#### `tl.load()` / `tl.store()` - 内存访问

```python
# 基本用法
data = tl.load(ptr + offsets)

# 带边界检查
data = tl.load(ptr + offsets, mask=offsets < n)

# 带默认值 (mask 为 False 时使用)
data = tl.load(ptr + offsets, mask=offsets < n, other=0.0)

# 存储数据
tl.store(ptr + offsets, data, mask=offsets < n)
```

---

## 第三部分：矩阵乘法 Kernel

### 3.1 问题分析

矩阵乘法 `C = A @ B`，其中 `A` 是 `(M, K)`，`B` 是 `(K, N)`：

```
朴素算法:
for i in range(M):
    for j in range(N):
        C[i,j] = sum(A[i,k] * B[k,j] for k in range(K))
```

**问题**：每个输出元素需要读取 `A` 的一整行和 `B` 的一整列，内存访问效率极低。

### 3.2 分块算法 (Tiling)

核心思想：将矩阵分成小块，每个 block 计算一个输出块。

```
┌─────────────────────────────────────────────────────────────┐
│                     矩阵乘法分块示意                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    A (M×K)              B (K×N)            C (M×N)          │
│   ┌─────┐              ┌─────┐           ┌─────┐           │
│   │ A₀₀ │              │ B₀₀ │ B₀₁       │ C₀₀ │ C₀₁       │
│   │─────│ A₀₁          │─────│─────      │─────│─────      │
│   │ A₁₀ │ A₁₁          │ B₁₀ │ B₁₁       │ C₁₀ │ C₁₁       │
│   └─────┘              └─────┘           └─────┘           │
│                                                             │
│   C₀₀ = A₀₀ @ B₀₀ + A₀₁ @ B₁₀                              │
│                                                             │
│   每个 block 在 SRAM 中计算，减少 HBM 访问                   │
└─────────────────────────────────────────────────────────────┘
```

**伪代码**：

```python
for m in range(0, M, BLOCK_M):
    for n in range(0, N, BLOCK_N):
        acc = zeros(BLOCK_M, BLOCK_N)  # 在 SRAM 中
        for k in range(0, K, BLOCK_K):
            # 加载小块到 SRAM
            a_tile = A[m:m+BLOCK_M, k:k+BLOCK_K]  # HBM → SRAM
            b_tile = B[k:k+BLOCK_K, n:n+BLOCK_N]  # HBM → SRAM
            # 在 SRAM 中计算
            acc += a_tile @ b_tile                  # SRAM 计算
        C[m:m+BLOCK_M, n:n+BLOCK_N] = acc          # SRAM → HBM
```

### 3.3 Block Size 的影响

| Block Size | 优点 | 缺点 | 适用场景 |
|------------|------|------|---------|
| 小 (32×32) | 更多并行 block | 更多 HBM 访问 | 小矩阵 |
| 中 (128×128) | 平衡 | 平衡 | 通用 |
| 大 (256×256) | 更好数据复用 | 可能超出 SRAM | 大矩阵 |

**自动调优**：使用 Triton 的 `autotune` 自动选择最优配置：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],  # 根据矩阵大小选择配置
)
@triton.jit
def matmul_kernel(...):
    ...
```

### 3.4 L2 Cache 优化 (Super-grouping)

简单遍历顺序会导致 L2 cache 命中率低：

```
问题: 行优先遍历
Block 0 → Block 1 → Block 2 → Block 3 → ...
每次切换都丢失 A 的缓存数据
```

**解决方案**：Super-grouping

```
优化: 分组遍历 (GROUP_SIZE_M = 4)
Group 0: Block 0, 4, 8, 12  (共享 A 的行)
Group 1: Block 1, 5, 9, 13
Group 2: Block 2, 6, 10, 14
Group 3: Block 3, 7, 11, 15

相邻 block 共享 A 的数据 → L2 cache 命中率提升
```

---

## 第四部分：FlashAttention 原理

### 4.1 标准 Attention 的问题

```python
def standard_attention(Q, K, V):
    # Q: (batch, heads, seq_len, head_dim)
    
    # 步骤 1: 计算 attention scores
    S = Q @ K.transpose(-2, -1) / sqrt(d)  # (batch, heads, seq_len, seq_len)
    # 内存: O(N²) - 对于 N=8192, 需要 128 MB per head!
    
    # 步骤 2: Softmax
    P = softmax(S, dim=-1)  # O(N²)
    
    # 步骤 3: 加权求和
    O = P @ V  # O(N²)
    
    return O
```

**内存复杂度**: O(N² × batch × heads × head_dim)

对于 LLM 训练，这是不可接受的！

### 4.2 FlashAttention 的核心创新

**核心思想**: 不存储完整的 attention matrix，而是分块计算并使用 online softmax。

```
┌─────────────────────────────────────────────────────────────┐
│                  FlashAttention vs Standard                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Standard Attention:          FlashAttention:              │
│  ┌─────────────┐              ┌───┬───┬───┬───┐            │
│  │             │              │ Q₁│ Q₂│ Q₃│ Q₄│            │
│  │   N × N     │              ├───┼───┼───┼───┤            │
│  │  Attention  │    ──→       │   │   │   │   │  分块      │
│  │   Matrix    │              │ K │ V │块 │对 │  计算      │
│  │   存储在    │              │   │   │   │   │            │
│  │    HBM      │              └───┴───┴───┴───┘            │
│  └─────────────┘                   ↓                       │
│   O(N²) 内存                  O(N) 内存                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Online Softmax 算法

标准 softmax 需要两遍扫描：
1. 第一遍：找最大值 (数值稳定性)
2. 第二遍：计算 exp 和归一化

**Online Softmax** 可以一遍完成：

```python
def online_softmax(Q, K, V):
    """
    Online Softmax 算法
    
    关键洞察:
    - 维护 running max 和 running sum
    - 可以增量更新，无需存储完整矩阵
    """
    # 初始化 running 状态
    m = -inf      # running max (每个 query 位置)
    l = 0         # running sum of exp
    O = 0         # running output
    
    # 对每个 K, V 块
    for K_j, V_j in blocks(K, V):
        # 1. 计算当前块的 attention scores
        S_j = Q @ K_j.T / sqrt(d)  # (BLOCK_M, BLOCK_N)
        
        # 2. 更新 running max
        m_new = maximum(m, max(S_j, axis=1))
        
        # 3. 更新 running sum (需要修正之前的值)
        # exp(m - m_new) 修正因子，因为 max 变了
        l_new = exp(m - m_new) * l + sum(exp(S_j - m_new[:, None]), axis=1)
        
        # 4. 更新 output (同样需要修正)
        O_new = (exp(m - m_new)[:, None] * O * l[:, None] + 
                 exp(S_j - m_new[:, None]) @ V_j) / l_new[:, None]
        
        # 5. 更新状态
        m, l, O = m_new, l_new, O_new
    
    return O
```

### 4.4 内存复杂度对比

| 方法 | 内存复杂度 | N=1024 | N=4096 | N=8192 |
|------|-----------|--------|--------|--------|
| Standard | O(N²) | 8 MB | 128 MB | 512 MB |
| FlashAttention | O(N) | 0.5 MB | 2 MB | 4 MB |

**节省比例高达 99%！**

### 4.5 Causal Masking

对于自回归模型（如 GPT），位置 `i` 只能看到位置 `≤ i` 的 token：

```
Causal Mask 示例 (seq_len = 4):

     j=0  j=1  j=2  j=3
i=0 [ ✓   ✗    ✗    ✗  ]
i=1 [ ✓   ✓    ✗    ✗  ]
i=2 [ ✓   ✓    ✓    ✗  ]
i=3 [ ✓   ✓    ✓    ✓  ]

✓ = 可见 (attention score 保留)
✗ = 不可见 (attention score = -inf)
```

```python
# Triton 实现
if IS_CAUSAL:
    # 创建 causal mask
    causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    # 应用 mask
    qk = tl.where(causal_mask, qk, float("-inf"))
```

---

## 第五部分：性能优化技巧

### 5.1 Block Size 调优指南

```bash
# 运行 Block Size 实验
python examples/block_size_experiment.py
```

**推荐配置**：

| 矩阵大小 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|---------|---------|---------|---------|------|
| < 512 | 32 | 32 | 32 | 小矩阵，更多并行度 |
| 512-2048 | 64 | 128 | 32 | 中等矩阵 |
| 2048-4096 | 128 | 128 | 64 | 大矩阵 |
| > 4096 | 128 | 256 | 64 | 超大矩阵 |

### 5.2 数据类型选择

| 类型 | 范围 | 精度 | 性能 | 推荐场景 |
|------|------|------|------|---------|
| FP32 | ±3.4e38 | 高 | 1x | 高精度需求 |
| FP16 | ±65504 | 中 | 2x | 训练/推理 |
| BF16 | ±3.4e38 | 中 | 2x | 训练 (更稳定) |
| FP8 | ±448 | 低 | 4x | 推理 (Hopper+) |

```python
# 推荐使用 FP16
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# 训练场景推荐 BF16 (避免溢出)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
```

### 5.3 性能检查清单

| 优化项 | 检查方法 | 影响 |
|--------|---------|------|
| ✅ 使用 FP16/BF16 | `tensor.dtype` | 2x 加速 |
| ✅ 启用 autotune | 不指定 block size | 自动最优 |
| ✅ 预热 kernel | 运行几次后计时 | 避免编译开销 |
| ✅ 批量处理 | 合并小 batch | 减少启动开销 |
| ✅ 连续内存 | `tensor.is_contiguous()` | 提升带宽 |
| ❌ 避免小矩阵 | 矩阵 > 512 | kernel 开销 |
| ❌ 避免频繁同步 | 减少 `.cpu()` 调用 | 避免 CPU-GPU 等待 |

### 5.4 性能分析工具

```python
# 使用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    result = triton_matmul(a, b)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

```bash
# 使用 NVIDIA Nsight Systems
nsys profile python benchmarks/bench_matmul.py

# 使用 NVIDIA Nsight Compute
ncu --set full python benchmarks/bench_matmul.py
```

---

## 实战练习

### 练习 1：运行快速演示

```bash
make demo
```

### 练习 2：Block Size 实验

```bash
python examples/block_size_experiment.py
```

观察不同 Block Size 对性能的影响。

### 练习 3：内存对比

```bash
python benchmarks/bench_flash.py --memory-test
```

验证 FlashAttention 的 O(N) 内存复杂度。

### 练习 4：运行 Benchmark

```bash
make bench-all
make report
```

---

## 下一步学习

1. **深入阅读论文**
   - [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
   - [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)

2. **修改源码实验**
   - 尝试不同的 Block Size
   - 添加新的 autotune 配置
   - 在你自己的 GPU 上对比 benchmark 结果

3. **探索超出当前仓库范围的高级主题**
   - FlashAttention Backward Pass
   - TMA (Tensor Memory Accelerator)
   - FP8 计算

---

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Online Softmax Paper](https://arxiv.org/abs/1805.02867)

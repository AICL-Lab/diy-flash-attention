# DIY FlashAttention 教程

本教程将带你从零开始理解和实现 FlashAttention 算法。

## 目录

1. [GPU 编程基础](#1-gpu-编程基础)
2. [Triton 入门](#2-triton-入门)
3. [矩阵乘法 Kernel](#3-矩阵乘法-kernel)
4. [FlashAttention 原理](#4-flashattention-原理)
5. [性能优化技巧](#5-性能优化技巧)

---

## 1. GPU 编程基础

### 1.1 GPU 内存层次结构

GPU 有多层内存，速度和容量各不相同：

```
┌─────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)            │
│  容量: ~40-80 GB                        │
│  带宽: ~1.5-3.35 TB/s                   │
│  延迟: 高                               │
├─────────────────────────────────────────┤
│  L2 Cache                               │
│  容量: ~40 MB                           │
│  带宽: ~4 TB/s                          │
├─────────────────────────────────────────┤
│  SRAM (Shared Memory)                   │
│  容量: ~192-228 KB per SM               │
│  带宽: ~19 TB/s                         │
├─────────────────────────────────────────┤
│  Registers                              │
│  容量: ~256 KB per SM                   │
│  带宽: 最快                             │
└─────────────────────────────────────────┘
```

**关键洞察**: FlashAttention 的核心优化就是尽量让数据留在快速的 SRAM 中，减少对慢速 HBM 的访问。

### 1.2 GPU 执行模型

- **Thread**: 最小执行单元
- **Warp**: 32 个 thread 一起执行（SIMT）
- **Block**: 多个 warp 组成，共享 SRAM
- **Grid**: 多个 block 组成，覆盖整个计算

```python
# Triton 中的对应关系
@triton.jit
def kernel(...):
    pid = tl.program_id(0)  # 当前 block 的 ID
    # 每个 program 对应一个 block
```

---

## 2. Triton 入门

### 2.1 为什么用 Triton？

传统 CUDA 编程需要手动处理：
- 内存分块 (Tiling)
- 合并访问 (Coalescing)
- 共享内存管理
- 同步

Triton 自动处理这些，让你专注于算法逻辑。

### 2.2 第一个 Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 获取当前 program 的 ID
    pid = tl.program_id(0)
    
    # 计算这个 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建 mask 处理边界
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2.3 关键概念

1. **`tl.constexpr`**: 编译时常量，用于 block size
2. **`tl.program_id()`**: 获取当前 block ID
3. **`tl.arange()`**: 创建索引数组
4. **`tl.load()/tl.store()`**: 内存读写
5. **`mask`**: 处理边界条件

---

## 3. 矩阵乘法 Kernel

### 3.1 朴素算法

```
C[i,j] = sum(A[i,k] * B[k,j] for k in range(K))
```

问题：每个输出元素需要读取整行 A 和整列 B，内存访问效率低。

### 3.2 分块算法 (Tiling)

核心思想：将矩阵分成小块，每个 block 计算一个输出块。

```
for m in range(0, M, BLOCK_M):
    for n in range(0, N, BLOCK_N):
        acc = zeros(BLOCK_M, BLOCK_N)
        for k in range(0, K, BLOCK_K):
            # 加载 A 和 B 的小块到 SRAM
            a_block = A[m:m+BLOCK_M, k:k+BLOCK_K]
            b_block = B[k:k+BLOCK_K, n:n+BLOCK_N]
            # 在 SRAM 中计算
            acc += a_block @ b_block
        C[m:m+BLOCK_M, n:n+BLOCK_N] = acc
```

### 3.3 Block Size 的影响

| Block Size | 优点 | 缺点 |
|------------|------|------|
| 小 (32×32) | 更多并行度 | 更多内存访问 |
| 大 (256×256) | 更好数据复用 | 可能超出 SRAM |

**最佳实践**: 通过 autotune 自动选择最优配置。

### 3.4 L2 Cache 优化

问题：简单的行优先遍历会导致 L2 cache 命中率低。

解决：使用 "super-grouping" 策略，让相邻的 block 共享更多数据。

```python
# 不好的遍历顺序
# Block 0, 1, 2, 3, 4, 5, 6, 7, 8 ...

# 好的遍历顺序 (GROUP_SIZE_M = 4)
# Block 0, 4, 8, 12 | 1, 5, 9, 13 | 2, 6, 10, 14 | 3, 7, 11, 15
```

---

## 4. FlashAttention 原理

### 4.1 标准 Attention 的问题

```python
# 标准实现
S = Q @ K.T / sqrt(d)  # O(N²) 内存
P = softmax(S)          # O(N²) 内存
O = P @ V               # O(N²) 内存
```

对于长序列 (N=8192)，attention matrix 需要 8192² × 2 bytes = 128 MB！

### 4.2 FlashAttention 的解决方案

核心思想：**不存储完整的 attention matrix**，而是分块计算并使用 online softmax。

### 4.3 Online Softmax

标准 softmax 需要两遍扫描：
1. 找最大值 (数值稳定性)
2. 计算 exp 和归一化

Online softmax 可以一遍完成：

```python
# 初始化
m = -inf  # running max
l = 0     # running sum
o = 0     # running output

# 对每个 K, V 块
for j in range(num_blocks):
    # 计算当前块的 attention scores
    s_j = Q @ K_j.T / sqrt(d)
    
    # 更新 running max
    m_new = max(m, max(s_j))
    
    # 更新 running sum (需要修正之前的值)
    l_new = exp(m - m_new) * l + sum(exp(s_j - m_new))
    
    # 更新 output (需要修正之前的值)
    o_new = (l * exp(m - m_new) * o + exp(s_j - m_new) @ V_j) / l_new
    
    m, l, o = m_new, l_new, o_new
```

### 4.4 内存复杂度

| 方法 | 内存复杂度 |
|------|-----------|
| 标准 Attention | O(N²) |
| FlashAttention | O(N) |

这就是为什么 FlashAttention 可以处理更长的序列！

### 4.5 Causal Masking

对于自回归模型，位置 i 只能看到位置 ≤ i 的 token：

```python
if IS_CAUSAL:
    # 只处理 j ≤ i 的位置
    mask = i >= j
    s_j = where(mask, s_j, -inf)
```

---

## 5. 性能优化技巧

### 5.1 选择合适的 Block Size

```python
# 运行实验
python examples/block_size_experiment.py
```

一般规律：
- 小矩阵：小 block size (32-64)
- 大矩阵：大 block size (128-256)
- 受限于 SRAM：减小 block size

### 5.2 使用 Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],  # 根据这些参数选择最优配置
)
@triton.jit
def kernel(...):
    ...
```

### 5.3 数值稳定性

- 使用 float32 累加器
- Softmax 前减去最大值
- 避免 exp 溢出

```python
# 不好
p = exp(s)

# 好
m = max(s)
p = exp(s - m)
```

### 5.4 内存访问模式

- 确保合并访问 (coalesced access)
- 避免 bank conflict
- 预取数据 (software pipelining)

---

## 下一步

1. 运行示例代码，感受性能差异
2. 修改 block size，观察 TFLOPS 变化
3. 阅读 FlashAttention 论文
4. 尝试实现 backward pass

```bash
# 开始实验
make demo
make experiment
make bench-all
```

## 参考资料

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)
- [Online Softmax](https://arxiv.org/abs/1805.02867)

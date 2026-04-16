# DIY FlashAttention Tutorial

A comprehensive guide to understanding and implementing FlashAttention from scratch. Whether you're new to GPU programming or an experienced developer, you'll find value here.

## Learning Path

```
Basics ──→ Advanced ──→ Hands-on
    │          │           │
    ▼          ▼           ▼
GPU Basics   FlashAttention    Performance
Triton       Implementation    Benchmarking
```

---

## Part 1: GPU Programming Fundamentals

### 1.1 Why GPU Acceleration?

In the era of Large Language Models (LLMs), attention is one of the core computations:

```python
# Standard Attention computation
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

For **sequence length N=8192**:
- Attention matrix size: 8192 × 8192 × 2 bytes = **128 MB**
- Required for backpropagation → **Memory explosion!**

GPU's parallel computing power is key to solving this problem.

### 1.2 GPU Memory Hierarchy

Understanding GPU memory hierarchy is fundamental for optimization:

<GpuArchitectureVisualizer />

**Interactive**: Click on different GPU architectures above to see their memory hierarchy and feature support.

```
┌─────────────────────────────────────────────────────────────┐
│                   HBM (High Bandwidth Memory)               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Capacity: 40-80 GB (A100/H100)                     │   │
│  │  Bandwidth: 1.5-3.35 TB/s                          │   │
│  │  Latency: ~500 cycles (slow!)                      │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      L2 Cache (Shared)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Capacity: 40-60 MB                                 │   │
│  │  Bandwidth: ~4 TB/s                                │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│        SRAM (Shared Memory, per SM)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Capacity: 164-228 KB per SM                        │   │
│  │  Bandwidth: ~19 TB/s (fastest!)                    │   │
│  │  ⚡ Key optimization target for FlashAttention     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Registers                              │
│  │  Capacity: ~256 KB per SM                         │   │
│  │  Latency: 1 cycle                                 │   │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: HBM has large capacity but is slow; SRAM is small but fast. FlashAttention's core idea: **keep data in SRAM as much as possible**.

---

## Part 2: Getting Started with Triton

### 2.1 Why Triton?

| Feature | CUDA C++ | Triton |
|---------|----------|--------|
| Memory tiling | Manual | Automatic |
| Coalesced access | Requires careful design | Auto-optimized |
| Shared memory | Manual allocation | Auto-managed |
| Synchronization | Manual `__syncthreads()` | Auto-handled |
| Learning curve | Steep | Gentle |

**Conclusion**: Triton lets you focus on the algorithm, not low-level optimizations.

### 2.2 Your First Triton Kernel

A simple vector addition example:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition: output = x + y
    
    Key concepts:
    1. tl.program_id(0): Get current block ID
    2. tl.arange(): Create index sequence
    3. mask: Handle boundary conditions
    4. tl.load/tl.store: Memory read/write
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vector addition wrapper"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=1024,
    )
    return output
```

---

## Part 3: FlashAttention Principles

### 3.1 The Problem with Standard Attention

```python
def standard_attention(Q, K, V):
    # Q: (batch, heads, seq_len, head_dim)
    
    # Step 1: Compute attention scores
    S = Q @ K.transpose(-2, -1) / sqrt(d)  # O(N²) memory
    
    # Step 2: Softmax
    P = softmax(S, dim=-1)  # O(N²)
    
    # Step 3: Weighted sum
    O = P @ V  # O(N²)
    
    return O
```

**Memory complexity**: O(N² × batch × heads × head_dim)

For LLM training, this is unacceptable!

### 3.2 FlashAttention's Core Innovation

**Core idea**: Don't store the full attention matrix; compute tiles with online softmax.

```
┌─────────────────────────────────────────────────────────────┐
│              FlashAttention vs Standard                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Standard Attention:          FlashAttention:              │
│  ┌─────────────┐              ┌───┬───┬───┬───┐            │
│  │             │              │ Q₁│ Q₂│ Q₃│ Q₄│            │
│  │   N × N     │              ├───┼───┼───┼───┤            │
│  │  Attention  │    ──→       │   │   │   │   │ Tiled      │
│  │   Matrix    │              │ K │ V │   │   │ Compute    │
│  │   Stored    │              │   │   │   │   │            │
│  │    in HBM   │              └───┴───┴───┴───┘            │
│  └─────────────┘                    ↓                       │
│   O(N²) memory                O(N) memory                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Online Softmax Algorithm

Standard softmax requires two passes:
1. Find max value (numerical stability)
2. Compute exp and normalize

**Online Softmax** does it in one pass:

```python
def online_softmax(Q, K, V):
    """
    Online Softmax Algorithm
    
    Key insight: Maintain running max and running sum
    Can update incrementally without storing full matrix
    """
    m = -inf      # running max
    l = 0         # running sum of exp
    O = 0         # running output
    
    for K_j, V_j in blocks(K, V):
        # 1. Compute current block's attention scores
        S_j = Q @ K_j.T / sqrt(d)
        
        # 2. Update running max
        m_new = max(m, max(S_j, axis=1))
        
        # 3. Update running sum (correct previous values)
        l_new = exp(m - m_new) * l + sum(exp(S_j - m_new[:, None]), axis=1)
        
        # 4. Update output (also requires correction)
        O_new = (exp(m - m_new)[:, None] * O * l[:, None] + 
                 exp(S_j - m_new[:, None]) @ V_j) / l_new[:, None]
        
        m, l, O = m_new, l_new, O_new
    
    return O
```

### 3.4 Memory Complexity Comparison

| Method | Memory | N=1024 | N=4096 | N=8192 |
|--------|--------|--------|--------|--------|
| Standard | O(N²) | 8 MB | 128 MB | 512 MB |
| FlashAttention | O(N) | 0.5 MB | 2 MB | 4 MB |

**Up to 99% memory savings!**

### 3.5 Causal Masking

For autoregressive models (like GPT), position `i` can only see positions `≤ i`:

```
Causal Mask Example (seq_len = 4):

     j=0  j=1  j=2  j=3
i=0 [ ✓   ✗    ✗    ✗  ]
i=1 [ ✓   ✓    ✗    ✗  ]
i=2 [ ✓   ✓    ✓    ✗  ]
i=3 [ ✓   ✓    ✓    ✓  ]

✓ = Visible (attention score kept)
✗ = Invisible (attention score = -inf)
```

```python
# Triton implementation
if IS_CAUSAL:
    causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(causal_mask, qk, float("-inf"))
```

---

## Part 4: Performance Optimization

### 4.1 Block Size Tuning Guide

Block Size is the most critical parameter for Triton kernel performance.

**Trade-offs**:

| Block Size | Pros | Cons | Use Case |
|------------|------|------|----------|
| Small (32×32) | More parallel blocks | More HBM access | Small matrices |
| Medium (128×128) | Balanced | Balanced | General purpose |
| Large (256×256) | Better data reuse | May exceed SRAM | Large matrices |

**Recommended configuration**:

| Matrix Size | BLOCK_M | BLOCK_N | BLOCK_K |
|-------------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-2048 | 64 | 128 | 32 |
| 2048-4096 | 128 | 128 | 64 |
| > 4096 | 128 | 256 | 64 |

### 4.2 Data Type Selection

| Type | Range | Precision | Performance | Recommended |
|------|-------|-----------|-------------|-------------|
| FP32 | ±3.4e38 | High | 1x | High precision needs |
| FP16 | ±65504 | Medium | 2x | Training/Inference |
| BF16 | ±3.4e38 | Medium | 2x | Training (more stable) |
| FP8 | ±448 | Low | 4x | Inference (Hopper+) |

```python
# Recommended: FP16
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Training: BF16 (avoids overflow)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
```

---

## Practice Exercises

### Exercise 1: Run Quick Demo

```bash
make demo
```

### Exercise 2: Block Size Experiment

```bash
python examples/block_size_experiment.py
```

Observe how different Block Sizes affect performance.

### Exercise 3: Memory Comparison

```bash
python benchmarks/bench_flash.py --memory-test
```

Verify FlashAttention's O(N) memory complexity.

### Exercise 4: Run Benchmarks

```bash
make bench-all
make report
```

---

## Next Steps

1. **Read the Papers**
   - [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
   - [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)

2. **Experiment with Source Code**
   - Try different Block Sizes
   - Add new autotune configs
   - Implement other GPU kernels

3. **Explore Advanced Topics**
   - FlashAttention Backward Pass
   - TMA (Tensor Memory Accelerator)
   - FP8 computation

---

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Triton Documentation](https://triton-lang.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

# System Architecture Design

This document provides a comprehensive overview of the DIY FlashAttention system architecture, including GPU memory hierarchy, kernel design, and performance optimization strategies.

## Overview

DIY FlashAttention is an educational implementation of the FlashAttention algorithm using OpenAI Triton. The system is designed to:

1. **Teach GPU programming concepts** through real, production-quality code
2. **Demonstrate FlashAttention's memory efficiency** with actual benchmarks
3. **Provide architecture-aware optimization** across Volta → Blackwell GPUs

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Q["Q Tensor [B, H, N, D]"]
        K["K Tensor [B, H, N, D]"]
        V["V Tensor [B, H, N, D]"]
    end

    subgraph Core["Core Kernels"]
        FA["flash_attention() - V1 Column Parallel"]
        FA2["flash_attention_v2() - V2 Row Parallel"]
        MM["triton_matmul() - Optimized GEMM"]
        BE["Backend Selector - Auto Dispatch"]
    end

    subgraph Utils["Utility Layer"]
        GD["GPU Detect - Architecture Identification"]
        CF["Config - Block Size Tuning"]
        BM["Benchmark - Performance Measurement"]
        VL["Validation - Correctness Checking"]
    end

    subgraph Output["Output Layer"]
        O["Output Tensor [B, H, N, D]"]
        L["Logsumexp [B, H, Nc]"]
    end

    Q --> BE
    K --> BE
    V --> BE
    
    BE --> FA
    BE --> FA2
    
    GD --> BE
    CF --> FA
    CF --> FA2
    
    FA --> O
    FA2 --> O
    FA --> L
    FA2 --> L
    
    MM -.-> FA
    MM -.-> FA2
    BM --> CF
    VL --> FA
    VL --> FA2

    style Core fill:#0e7490,stroke:#06b6d4,color:#fff
    style Utils fill:#1e293b,stroke:#475569,color:#fff
    style Input fill:#065f46,stroke:#10b981,color:#fff
    style Output fill:#7c3aed,stroke:#a78bfa,color:#fff
```

---

## GPU Memory Hierarchy

Understanding GPU memory hierarchy is essential for optimizing FlashAttention. The key insight is that **memory bandwidth, not compute, is the bottleneck** for attention computation.

### Memory Levels

```mermaid
flowchart LR
    subgraph Registers["⚡ Registers (Fastest)"]
        R1["32K 32-bit registers per thread"]
        R2["~1 cycle latency"]
        R3["~20+ TB/s effective bandwidth"]
    end

    subgraph SRAM["💾 Shared Memory / SRAM"]
        S1["228 KB per SM (A100)"]
        S2["~20-30 cycles latency"]
        S3["~19 TB/s bandwidth"]
    end

    subgraph L2["📦 L2 Cache"]
        L1["40 MB (A100)"]
        L2["~200-300 cycles latency"]
        L3["~4 TB/s bandwidth"]
    end

    subgraph HBM["📀 HBM2e (Slowest, Largest)"]
        H1["80 GB (A100 80GB)"]
        H2["~400-600 cycles latency"]
        H3["~3.35 TB/s bandwidth"]
    end

    Registers <--> SRAM <--> L2 <--> HBM

    style Registers fill:#dc2626,stroke:#ef4444,color:#fff
    style SRAM fill:#ea580c,stroke:#f97316,color:#fff
    style L2 fill:#ca8a04,stroke:#eab308,color:#fff
    style HBM fill:#16a34a,stroke:#22c55e,color:#fff
```

### Memory Hierarchy Table

| Level | Capacity | Latency | Bandwidth | Purpose |
|-------|----------|---------|-----------|---------|
| **Registers** | 256 KB/SM | 1 cycle | 20+ TB/s | Thread-local computation |
| **Shared Memory (SRAM)** | 228 KB/SM | 20-30 cycles | 19 TB/s | Block-level data sharing |
| **L2 Cache** | 40 MB | 200-300 cycles | 4 TB/s | Global data caching |
| **HBM** | 80 GB | 400-600 cycles | 3.35 TB/s | Main GPU memory |

### FlashAttention's Memory Strategy

FlashAttention achieves O(N) memory complexity by:

1. **Never materializing the full N×N attention matrix** in HBM
2. **Computing attention in blocks** that fit in SRAM
3. **Using online softmax** to accumulate results incrementally

```mermaid
flowchart TB
    subgraph Traditional["Traditional Attention"]
        T1["Load Q, K from HBM"]
        T2["Compute S = QK^T (N×N matrix)"]
        T3["Store S to HBM ❌"]
        T4["Load S from HBM"]
        T5["Compute P = softmax(S)"]
        T6["Store P to HBM ❌"]
        T7["Load P, V from HBM"]
        T8["Compute O = PV"]
        T9["Store O to HBM"]
    end

    subgraph FlashAttn["FlashAttention"]
        F1["Load Q_i, K_j, V_j blocks to SRAM"]
        F2["Compute S_ij = Q_i K_j^T in SRAM ✅"]
        F3["Online softmax in SRAM ✅"]
        F4["Accumulate O_i in SRAM ✅"]
        F5["Store O_i to HBM (once)"]
    end

    T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8 --> T9
    F1 --> F2 --> F3 --> F4 --> F5

    style Traditional fill:#991b1b,stroke:#dc2626,color:#fff
    style FlashAttn fill:#065f46,stroke:#10b981,color:#fff
```

---

## Kernel Design

### FlashAttention V1: Column-Parallel

```mermaid
flowchart LR
    subgraph Input["Input Tensors"]
        Q["Q: [B, H, N, D]"]
        K["K: [B, H, N, D]"]
        V["V: [B, H, N, D]"]
    end

    subgraph Blocks["Block Processing"]
        direction TB
        B1["Block 1: Q[0:Br, :]"]
        B2["Block 2: Q[Br:2Br, :]"]
        B3["Block 3: Q[2Br:3Br, :]"]
        BN["Block Nc: Q[(Nc-1)Br:N, :]"]
    end

    subgraph Compute["Per-Block Computation"]
        C1["Load K, V fully"]
        C2["For each Q block:"]
        C3["Compute attention"]
        C4["Accumulate output"]
    end

    Q --> Blocks
    Blocks --> Compute

    style Blocks fill:#1e293b,stroke:#475569,color:#fff
    style Compute fill:#0e7490,stroke:#06b6d4,color:#fff
```

**Characteristics:**
- **Parallelization**: Over query blocks (Br rows each)
- **Memory Access**: K, V loaded once per block; Q streamed
- **Best for**: Shorter sequences, older architectures

### FlashAttention V2: Row-Parallel (Stripe-Parallel)

```mermaid
flowchart TB
    subgraph Input["Input Tensors"]
        Q["Q: [B, H, N, D]"]
        K["K: [B, H, N, D]"]
        V["V: [B, H, N, D]"]
    end

    subgraph V2["V2 Stripe Processing"]
        direction LR
        S1["Stripe 1: Q[0:Br, :] × K, V"]
        S2["Stripe 2: Q[Br:2Br, :] × K, V"]
        S3["Stripe 3: Q[2Br:3Br, :] × K, V"]
        SN["Stripe Nc: Q[(Nc-1)Br:N, :] × K, V"]
    end

    subgraph Optimizations["V2 Optimizations"]
        O1["Reduced non-matmul FLOPs"]
        O2["Better work partitioning"]
        O3["Sequential attention loops"]
    end

    Q --> V2
    K --> V2
    V --> V2
    V2 --> O1
    V2 --> O2
    V2 --> O3

    style V2 fill:#0e7490,stroke:#06b6d4,color:#fff
    style Optimizations fill:#065f46,stroke:#10b981,color:#fff
```

**Characteristics:**
- **Parallelization**: Better work distribution across thread blocks
- **Memory Access**: Optimized HBM access patterns
- **Performance**: 5-15% faster than V1 on Ampere+ GPUs
- **Best for**: Longer sequences, modern architectures (Ampere, Ada, Hopper)

### Block Size Selection

Block sizes (Br, Bc) are critical for performance:

```mermaid
flowchart TB
    subgraph Decision["Block Size Decision"]
        D1["Determine SRAM capacity"]
        D2["Calculate available space per block"]
        D3["Select Br, Bc to maximize occupancy"]
        D4["Apply architecture-specific tuning"]
    end

    subgraph Constraints["Constraints"]
        C1["Br × Bc × dtype_size ≤ SRAM_limit"]
        C2["Br ≤ N (sequence length)"]
        C3["Bc ≤ N (sequence length)"]
    end

    subgraph Defaults["Default Values"]
        A1["FP16: Br=128, Bc=64"]
        A2["BF16: Br=128, Bc=64"]
        A3["FP32: Br=64, Bc=32"]
    end

    D1 --> D2 --> D3 --> D4
    D3 --> C1
    D3 --> C2
    D3 --> C3
    D4 --> A1
    D4 --> A2
    D4 --> A3

    style Decision fill:#0e7490,stroke:#06b6d4,color:#fff
    style Constraints fill:#991b1b,stroke:#dc2626,color:#fff
    style Defaults fill:#065f46,stroke:#10b981,color:#fff
```

---

## Architecture Adaptation

The system automatically detects and adapts to different GPU architectures:

### Supported Architectures

| Architecture | GPUs | Compute Capability | Features |
|--------------|------|-------------------|----------|
| **Volta** | V100 | SM70 | Tensor Cores, FP16 |
| **Turing** | RTX 20xx | SM75 | Tensor Cores, FP16 |
| **Ampere** | A100, RTX 30xx | SM80 | Tensor Cores, BF16, FP16 |
| **Ada** | RTX 40xx | SM89 | Tensor Cores, BF16, FP16 |
| **Hopper** | H100 | SM90 | TMA, FP8, Tensor Memory |
| **Blackwell** | B100/B200 | SM100 | Latest features |

### Feature Detection Flow

```mermaid
flowchart TB
    Start["Detect GPU via CUDA"]
    
    CheckSM["Get Compute Capability"]
    
    Volta["SM70: Volta - Basic Support"]
    Turing["SM75: Turing - Basic Support"]
    Ampere["SM80: Ampere - Full Support + BF16"]
    Ada["SM89: Ada - Full Support + BF16"]
    Hopper["SM90: Hopper - TMA, FP8"]
    Blackwell["SM100: Blackwell - Latest"]
    
    Config["Apply Architecture-Specific Config"]
    
    Start --> CheckSM
    
    CheckSM -->|"SM70"| Volta --> Config
    CheckSM -->|"SM75"| Turing --> Config
    CheckSM -->|"SM80"| Ampere --> Config
    CheckSM -->|"SM89"| Ada --> Config
    CheckSM -->|"SM90"| Hopper --> Config
    CheckSM -->|"SM100"| Blackwell --> Config

    style Hopper fill:#7c3aed,stroke:#a78bfa,color:#fff
    style Blackwell fill:#7c3aed,stroke:#a78bfa,color:#fff
```

---

## Design Decisions

### Why Triton Instead of CUDA C++?

| Aspect | Triton | CUDA C++ |
|--------|--------|----------|
| **Learning Curve** | Gentle (Python-like) | Steep (low-level) |
| **Memory Management** | Automatic tiling | Manual shared memory |
| **Portability** | Architecture-agnostic | Architecture-specific |
| **Debugging** | Python tooling | Limited tooling |
| **Performance** | ~90-95% of hand-tuned CUDA | Maximum potential |

**Decision**: Triton was chosen for its **educational value** while maintaining production-quality performance.

### Why Support Both V1 and V2?

1. **Educational Value**: V1 is simpler to understand; V2 shows optimization techniques
2. **Compatibility**: V1 works better on older architectures
3. **Performance Comparison**: Users can benchmark both approaches

### Why Forward-Only?

1. **Educational Focus**: Forward pass contains the core algorithmic innovations
2. **Simplified Codebase**: Easier to understand without backward pass complexity
3. **Reference Value**: Most users want to understand the algorithm, not train models

---

## Performance Characteristics

### Memory Complexity

| Method | Memory Complexity | HBM Accesses |
|--------|------------------|--------------|
| Standard Attention | O(N²) | N² reads/writes |
| FlashAttention | O(N) | ~N reads/writes |

### Bandwidth Utilization

On A100 (3.35 TB/s HBM bandwidth):

| Operation | Theoretical Peak | FlashAttention Achieves |
|-----------|-----------------|------------------------|
| Memory Reads | 3.35 TB/s | ~2.8 TB/s (84%) |
| Attention Compute | 312 TFLOPS | ~280 TFLOPS (90%) |

---

## See Also

- [Algorithm Deep Dive](/en/algorithm) - Mathematical foundations and algorithm details
- [Performance Guide](/en/performance) - Tuning and optimization strategies
- [API Reference](/en/api) - Complete function signatures and examples

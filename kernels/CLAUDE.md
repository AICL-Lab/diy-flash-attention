# kernels/ - Triton GPU 内核

本目录包含 DIY FlashAttention 项目的核心 Triton GPU 内核实现。

## 文件结构

```
kernels/
├── __init__.py           # 包入口，统一导出所有公开 API
├── flash_attn.py         # FlashAttention V1（列并行）
├── flash_attn_v2.py      # FlashAttention V2（行并行，Ampere+ 优化）
├── matmul.py             # 高性能矩阵乘法，autotune
├── persistent_kernels.py # 持久化线程块内核
├── backend_selector.py   # 统一内核调度注册表
├── mask_dsl.py           # BlockMask 掩码抽象
├── modern_features.py    # Hopper+ 特性检测
└── py.typed              # PEP 561 类型标记
```

## 公开 API

### 注意力内核

| 函数 | 描述 |
|------|------|
| `flash_attention(q, k, v, causal=False, ...)` | V1 列并行实现，通用支持 |
| `flash_attention_v2(q, k, v, causal=False, ...)` | V2 行并行，Ampere+ 快 5-15% |

### 矩阵乘法

| 函数 | 描述 |
|------|------|
| `triton_matmul(a, b, ...)` | Autotune 矩阵乘法 |
| `persistent_matmul(a, b, ...)` | 持久化线程块实现 |

### 内核选择

| 函数/类 | 描述 |
|---------|------|
| `BackendSelector` | 基于 GPU 能力和问题规模自动选择内核 |
| `select_attention_kernel(variant, ...)` | 选择注意力内核 |

### Mask DSL

| 函数/类 | 描述 |
|---------|------|
| `BlockMask` | 块级注意力掩码抽象 |
| `create_block_mask(pattern, ...)` | 创建 causal/full/sliding_window/prefix_lm 掩码 |
| `compose_block_masks(m1, m2, op)` | 组合掩码（intersect/union） |

### 架构检测

| 函数 | 描述 |
|------|------|
| `check_hopper_features()` | 检测 Hopper+ GPU 特性 |
| `supports_fp8()` | 检查 FP8 支持 |
| `get_matmul_config()` | 获取最优 matmul 配置 |
| `get_attention_config()` | 获取最优 attention 配置 |

## V1 vs V2 对比

| 特性 | V1 (flash_attn.py) | V2 (flash_attn_v2.py) |
|------|-------------------|----------------------|
| 并行模式 | 列并行（每个 block 处理一个 query block） | 行并行（每个 block 处理一个 query row） |
| 最优架构 | 所有架构 | Ampere+ (SM80+) |
| 性能 | 基线 | 快 5-15% |
| 推荐 | 通用场景 | 大序列、Ampere+ GPU |

## 设计原则

1. **在线 Softmax**：避免实例化完整注意力矩阵，O(N) 内存复杂度
2. **分块计算**：利用 SRAM 高带宽，减少 HBM 访问
3. **架构自适应**：自动检测 GPU 能力，选择最优配置
4. **教育性代码**：紧凑可读，注释详尽

## 数据类型支持

- float16 ✅
- bfloat16 ✅
- float32 ✅（内部转换为 float16 计算）

## head_dim 支持

- 32 ✅
- 64 ✅

---

**导航**: [← 项目根目录](../CLAUDE.md)

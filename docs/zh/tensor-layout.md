# 张量布局指南

理解 FlashAttention V1 和 V2 之间的张量布局差异。

## 关键差异

⚠️ **V1 和 V2 使用不同的张量布局** - `heads` 和 `seq_len` 维度互换了位置！

| 版本 | 布局 |
|------|------|
| V1 (`flash_attention`) | `(batch, heads, seq_len, head_dim)` |
| V2 (`flash_attention_v2`) | `(batch, seq_len, heads, head_dim)` |

## 为什么存在差异？

FlashAttention V2 使用行级（条纹）并行，这要求 `seq_len` 维度在内存中连续，以在 Ampere+ GPU 上获得最优的内存访问模式。

V1 使用列并行处理，`heads` 连续存储更自然。

## 示例：正确用法

### V1 (flash_attention)

```python
import torch
from kernels import flash_attention

batch, heads, seq_len, head_dim = 2, 8, 512, 64

# V1 期望: (batch, heads, seq_len, head_dim)
q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)
print(out.shape)  # (2, 8, 512, 64)
```

### V2 (flash_attention_v2)

```python
import torch
from kernels import flash_attention_v2

batch, heads, seq_len, head_dim = 2, 8, 512, 64

# V2 期望: (batch, seq_len, heads, head_dim)
q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)

out = flash_attention_v2(q, k, v, causal=True)
print(out.shape)  # (2, 512, 8, 64)
```

## 版本间转换

在 V1 和 V2 之间切换时，需要转置第 1 和第 2 维度：

```python
# 从 V1 格式转换为 V2 格式
q_v2 = q_v1.transpose(1, 2)  # (b, h, s, d) -> (b, s, h, d)
k_v2 = k_v1.transpose(1, 2)
v_v2 = v_v1.transpose(1, 2)

# 运行 V2
out_v2 = flash_attention_v2(q_v2, k_v2, v_v2, causal=True)

# 转换回 V1 格式
out_v1 = out_v2.transpose(1, 2)  # (b, s, h, d) -> (b, h, s, d)
```

## 应该使用哪个版本？

| 场景 | 推荐 |
|------|------|
| Ampere+ GPU (A100, RTX 30xx, RTX 40xx) | V2，性能提升 5-15% |
| Volta/Turing GPU (V100, RTX 20xx) | V1（V2 未针对旧架构优化）|
| 大批次 + 长序列 | V2 |
| 代码兼容性优先 | V1（标准 PyTorch 注意力布局）|

## BackendSelector

如果需要自动选择，可以使用 `BackendSelector`：

```python
from kernels import BackendSelector, flash_attention, flash_attention_v2

# 选择器内部处理布局差异
selector = BackendSelector()
kernel = selector.select_attention(batch=2, heads=8, seq_len=1024, head_dim=64)

# 或使用带 variant 参数的 flash_attention
from kernels import flash_attention
out = flash_attention(q, k, v, causal=True, variant="auto")
```

## 总结

- **V1**: `(batch, heads, seq_len, head_dim)` - 标准布局，通用支持
- **V2**: `(batch, seq_len, heads, head_dim)` - 针对 Ampere+ 优化，快 5-15%
- **切换版本时务必检查张量形状！**

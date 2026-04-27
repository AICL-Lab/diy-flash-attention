# 常见问题 (FAQ)

本文档汇总了 DIY FlashAttention 使用过程中的常见问题和解决方案。

## 目录

- [安装问题](#安装问题)
- [性能问题](#性能问题)
- [正确性问题](#正确性问题)
- [GPU 兼容性](#gpu-兼容性)
- [开发问题](#开发问题)
- [错误排查](#错误排查)

---

## 安装问题

### Q: 安装 Triton 时报错怎么办？

**A:** Triton 需要特定版本的 CUDA 和 PyTorch，推荐以下安装方式：

```bash
# 方式 1: 通过 PyTorch 安装 (推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton

# 方式 2: 使用 conda
conda install -c conda-forge triton

# 方式 3: 从源码安装 (最新版)
pip install triton-nightly
```

**环境要求**：
| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.9 | 3.10-3.11 |
| CUDA | 11.0 | 12.1 |
| PyTorch | 2.0.0 | 2.2+ |
| Triton | 2.1.0 | 最新版 |

---

### Q: 提示 "CUDA not available" 怎么办？

**A:** 按以下步骤排查：

```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi
# 应显示 GPU 信息和驱动版本

# 2. 检查 CUDA 版本
nvcc --version
# 或
nvidia-smi | grep "CUDA Version"

# 3. 检查 PyTorch CUDA 支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

**常见原因**：
1. NVIDIA 驱动未安装或版本过低
2. PyTorch 安装的是 CPU 版本
3. CUDA toolkit 版本与 PyTorch 不匹配

**解决方案**：
```bash
# 重新安装 PyTorch CUDA 版本
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Q: 内存不足 (OOM) 怎么办？

**A:** 尝试以下方法：

```python
# 1. 减小 batch size
batch = 1  # 而不是 4 或 8

# 2. 减小序列长度
seq_len = 512  # 而不是 2048 或 4096

# 3. 使用 float16
dtype = torch.float16

# 4. 清理缓存
torch.cuda.empty_cache()

# 5. 使用梯度检查点 (训练时)
# torch.utils.checkpoint.checkpoint()

# 6. 监控内存使用
def print_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## 性能问题

### Q: Triton kernel 比 PyTorch 慢？

**A:** 可能的原因和解决方案：

| 原因 | 症状 | 解决方案 |
|------|------|---------|
| 矩阵太小 | < 512 维度 | 使用 PyTorch 或增大矩阵 |
| 首次运行 | 第一次慢，之后快 | 预热 kernel |
| Block size 不合适 | 性能波动 | 使用 autotune |
| 非连续内存 | 意外的慢 | 确保 `.is_contiguous()` |

```python
# 预热示例
for _ in range(10):
    _ = triton_matmul(a, b)
torch.cuda.synchronize()

# 然后计时
import time
start = time.time()
for _ in range(100):
    result = triton_matmul(a, b)
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) / 100 * 1000:.3f} ms")
```

---

### Q: 如何选择最优 Block Size？

**A:**

**推荐方式**：使用 autotune（默认）

```python
from kernels import triton_matmul

# 不指定 block size，自动选择最优配置
c = triton_matmul(a, b)
```

**手动选择指南**：

| 矩阵大小 | BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|---------|
| < 512 | 32 | 32 | 32 |
| 512-1024 | 64 | 64 | 32 |
| 1024-2048 | 64 | 128 | 32 |
| > 2048 | 128 | 256 | 64 |

```bash
# 运行实验找到最优配置
python examples/block_size_experiment.py
```

---

### Q: FlashAttention 内存使用没有减少？

**A:** 检查以下几点：

1. **确认使用了正确的函数**

```python
# ✅ 正确 - 使用 FlashAttention
from kernels import flash_attention
out = flash_attention(q, k, v)

# ❌ 错误 - 使用标准 attention
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

2. **序列长度足够长**

```python
# 内存节省在长序列 (> 256) 时才明显
# N=256: 节省 ~75%
# N=1024: 节省 ~94%
# N=4096: 节省 ~98%
```

3. **正确测量峰值内存**

```python
torch.cuda.reset_peak_memory_stats()
out = flash_attention(q, k, v)
peak_memory = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak_memory / 1e6:.1f} MB")
```

---

## 正确性问题

### Q: 结果与 PyTorch 不完全一致？

**A:** 这是正常现象，原因如下：

1. **浮点精度差异**
   - FP16 有精度损失
   - 不同计算顺序导致舍入误差累积

2. **可接受的容差**

```python
# 验证正确性
import torch

# 推荐容差
rtol = 1e-2  # 相对容差
atol = 1e-2  # 绝对容差

assert torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol)

# 或查看最大差异
max_diff = (triton_out - torch_out).abs().max().item()
print(f"Max diff: {max_diff:.2e}")
# 通常 < 1e-2 是可接受的
```

---

### Q: Causal attention 结果不对？

**A:** 排查步骤：

```python
import torch
from kernels import flash_attention

# 1. 确认设置了 causal=True
out = flash_attention(q, k, v, causal=True)

# 2. 检查输入形状
assert q.dim() == 4, f"Expected 4D, got {q.dim()}D"
assert q.shape == k.shape == v.shape, "Q, K, V shapes must match"

# 3. 与参考实现对比
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
max_diff = (out - ref).abs().max().item()
print(f"Max diff from reference: {max_diff:.2e}")

# 4. 验证因果性: 修改未来不应影响过去
q_test = q.clone()
k_modified = k.clone()
k_modified[:, :, seq_len//2:, :] = torch.randn_like(k_modified[:, :, seq_len//2:, :])

out_orig = flash_attention(q_test, k, v, causal=True)
out_mod = flash_attention(q_test, k_modified, v, causal=True)

# 前半部分应该相同
first_half_diff = (out_orig[:, :, :seq_len//2, :] - out_mod[:, :, :seq_len//2, :]).abs().max()
print(f"Causality check: {first_half_diff.item():.2e} (should be ~0)")
```

---

## GPU 兼容性

### Q: 支持哪些 GPU？

**A:**

| GPU 架构 | 计算能力 | 代表型号 | 支持状态 |
|---------|---------|---------|---------|
| Volta | SM70 | V100 | ✅ 基础支持 |
| Turing | SM75 | RTX 2080 | ✅ 基础支持 |
| Ampere | SM80 | A100, RTX 3090 | ✅ 完整支持 |
| Ada Lovelace | SM89 | RTX 4090 | ✅ 完整支持 |
| Hopper | SM90 | H100 | ✅ 高级特性 (FP8, TMA) |
| Blackwell | SM100 | B100/B200 | ✅ 高级特性 |

---

### Q: 如何检测 GPU 特性？

**A:**

```python
from utils import detect_gpu, print_gpu_info
from kernels import check_hopper_features

# 方式 1: 使用 detect_gpu
caps = detect_gpu()
print(f"GPU: {caps.name}")
print(f"架构: {caps.arch.value}")
print(f"计算能力: {caps.compute_capability}")
print(f"TMA: {caps.has_tma}")
print(f"FP8: {caps.has_fp8}")

# 方式 2: 打印完整信息
print_gpu_info(caps)

# 方式 3: 检查 Hopper+ 特性
features = check_hopper_features()
print(f"TMA: {features['tma_available']}")
print(f"FP8: {features['fp8_available']}")
print(f"Warpgroup MMA: {features['wgmma_available']}")
```

---

### Q: 在 AMD GPU 上能运行吗？

**A:** 目前不支持。Triton 主要面向 NVIDIA GPU (CUDA)。AMD GPU 需要 ROCm 版本的 Triton，本项目暂未适配。

---

## 开发问题

### Q: 如何调试 Triton kernel？

**A:**

**方法 1: 使用小输入测试**

```python
# 使用小矩阵便于手动验证
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float16)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=torch.float16)

result = triton_matmul(a, b)
expected = torch.matmul(a.float(), b.float()).half()

print(f"Result:\n{result}")
print(f"Expected:\n{expected}")
print(f"Diff:\n{(result - expected).abs()}")
```

**方法 2: 检查边界条件**

```python
# 测试非对齐维度
a = torch.randn(33, 47, device="cuda", dtype=torch.float16)
b = torch.randn(47, 61, device="cuda", dtype=torch.float16)
result = triton_matmul(a, b)
assert result.shape == (33, 61)
```

**方法 3: 使用验证工具**

```python
from utils import validate_matmul, validate_matmul_edge_cases

# 验证单个尺寸
is_valid, max_diff = validate_matmul(triton_matmul, m=128, n=128, k=128, verbose=True)

# 验证边界情况
all_passed, results = validate_matmul_edge_cases(triton_matmul, verbose=True)
```

---

### Q: 如何添加新的 kernel？

**A:** 参考以下步骤：

```
1. 在 kernels/ 目录创建新文件
   └── kernels/new_kernel.py

2. 实现 Triton kernel
   @triton.jit
   def _new_kernel(...):
       ...

3. 添加 wrapper 函数处理输入验证
   def new_function(...):
       # 输入验证
       # 准备输出
       # 启动 kernel
       return output

4. 在 kernels/__init__.py 中导出
   from .new_kernel import new_function
   __all__.append("new_function")

5. 添加测试
   └── tests/test_new_kernel.py

6. 添加文档
   └── docs/api.md (更新 API 文档)
```

---

## 错误排查

### 常见错误速查表

| 错误类型 | 错误信息 | 原因 | 解决方案 |
|---------|---------|------|---------|
| `ValueError` | `Expected 2D tensors` | matmul 输入不是 2D | `a.view(M, K)` |
| `ValueError` | `Incompatible dimensions` | A.shape[1] != B.shape[0] | 检查矩阵维度 |
| `ValueError` | `CUDA tensors required` | 输入在 CPU 上 | `a.cuda()` |
| `ValueError` | `same device` | 输入在不同 GPU | 统一到同一设备 |
| `ValueError` | `Expected 3D or 4D` | attention 输入维度错误 | 检查形状 |
| `ValueError` | `Unsupported head_dim` | head_dim 不是 32/64 | 使用 32 或 64 |
| `ValueError` | `seq_lens values must be positive` | seq_lens 含非正值 | 检查 seq_lens |
| `TypeError` | `Unsupported dtype` | dtype 不支持 | 使用 fp16/bf16/fp32 |
| `TypeError` | `dtypes must match` | dtype 不一致 | 统一 dtype |
| `ModuleNotFoundError` | `No module named 'triton'` | Triton 未安装 | `pip install triton` |
| `RuntimeError` | `CUDA out of memory` | 显存不足 | 减小 batch/seq_len |

### 获取帮助

1. **查看文档**: [API 参考](api.md) | [教程](tutorial.md) | [性能指南](performance.md)

2. **搜索 Issues**: [GitHub Issues](https://github.com/LessUp/diy-flash-attention/issues)

3. **提交 Bug 报告**: 请包含以下信息
   ```markdown
   ## 环境
   - Python 版本:
   - PyTorch 版本:
   - Triton 版本:
   - CUDA 版本:
   - GPU 型号:

   ## 复现步骤
   1. ...
   2. ...

   ## 错误信息
   ```
   ```
   错误堆栈...
   ```

   ## 期望行为
   ...
   ```

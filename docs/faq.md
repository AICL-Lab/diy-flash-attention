# 常见问题 (FAQ)

## 安装问题

### Q: 安装 Triton 时报错

**A:** Triton 需要特定版本的 CUDA 和 PyTorch。推荐安装方式：

```bash
# 方式 1: 通过 PyTorch 安装 (推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton

# 方式 2: 从源码安装
pip install triton-nightly
```

### Q: 提示 "CUDA not available"

**A:** 检查以下几点：

1. 确认 NVIDIA 驱动已安装：`nvidia-smi`
2. 确认 PyTorch CUDA 版本：`python -c "import torch; print(torch.cuda.is_available())"`
3. 确认 CUDA 版本兼容：`nvcc --version`

### Q: 内存不足 (OOM)

**A:** 尝试以下方法：

```python
# 1. 减小 batch size
batch = 1  # 而不是 4

# 2. 减小序列长度
seq_len = 512  # 而不是 2048

# 3. 使用 float16
dtype = torch.float16

# 4. 清理缓存
torch.cuda.empty_cache()
```

## 性能问题

### Q: Triton kernel 比 PyTorch 慢？

**A:** 可能的原因：

1. **矩阵太小**: Triton 在大矩阵上优势明显，小矩阵可能有 kernel launch 开销
2. **Block size 不合适**: 尝试使用 autotune 或调整 block size
3. **首次运行**: Triton 需要编译 kernel，首次运行会慢

```python
# 使用 autotune
c = triton_matmul(a, b)  # 不指定 block size

# 或者预热
for _ in range(10):
    _ = triton_matmul(a, b)
```

### Q: 如何选择最优 Block Size？

**A:** 一般规律：

| 矩阵大小 | 推荐 Block Size |
|---------|----------------|
| < 512 | 32×32×32 |
| 512-2048 | 64×64×32 或 128×128×32 |
| > 2048 | 128×256×64 |

最佳方式是使用 autotune 或运行实验：

```bash
python examples/block_size_experiment.py
```

### Q: FlashAttention 内存使用没有减少？

**A:** 确认以下几点：

1. 使用的是 `flash_attention` 而不是标准 attention
2. 序列长度足够长（>256）才能看到明显差异
3. 使用 `torch.cuda.max_memory_allocated()` 测量峰值内存

## 正确性问题

### Q: 结果与 PyTorch 不完全一致

**A:** 这是正常的，原因：

1. **浮点精度**: FP16 计算有精度损失
2. **计算顺序**: 不同的计算顺序会导致舍入误差累积
3. **容差范围**: 通常 rtol=1e-2, atol=1e-2 是可接受的

```python
# 验证正确性
assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
```

### Q: Causal attention 结果不对

**A:** 检查以下几点：

1. 确认 `causal=True` 参数已设置
2. 检查输入形状是否正确：`(batch, heads, seq_len, head_dim)`
3. 验证参考实现：

```python
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## GPU 兼容性

### Q: 支持哪些 GPU？

**A:** 

| GPU 架构 | 计算能力 | 支持状态 |
|---------|---------|---------|
| Volta (V100) | SM70 | ✓ 基础支持 |
| Turing (RTX 20xx) | SM75 | ✓ 基础支持 |
| Ampere (A100, RTX 30xx) | SM80+ | ✓ 完整支持 |
| Ada (RTX 40xx) | SM89 | ✓ 完整支持 |
| Hopper (H100) | SM90 | ✓ 高级特性 |
| Blackwell (B100) | SM100 | ✓ 高级特性 |

### Q: 如何检测 GPU 特性？

**A:**

```python
from utils import detect_gpu, print_gpu_info

caps = detect_gpu()
print_gpu_info(caps)

# 检查特定特性
print(f"TMA 支持: {caps.has_tma}")
print(f"FP8 支持: {caps.has_fp8}")
```

## 开发问题

### Q: 如何调试 Triton kernel？

**A:**

1. **打印中间值** (仅用于调试):
```python
@triton.jit
def kernel(...):
    # 使用 tl.device_print (仅调试)
    tl.device_print("value:", some_value)
```

2. **使用小输入测试**:
```python
# 使用小矩阵便于手动验证
a = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.float16)
```

3. **检查边界条件**:
```python
# 测试非对齐维度
a = torch.randn(33, 47, device="cuda", dtype=torch.float16)
```

### Q: 如何添加新的 kernel？

**A:** 参考现有实现：

1. 在 `kernels/` 目录创建新文件
2. 使用 `@triton.jit` 装饰器
3. 添加 wrapper 函数处理输入验证
4. 在 `kernels/__init__.py` 中导出
5. 添加测试到 `tests/`

### Q: 测试失败怎么办？

**A:**

1. 检查 GPU 是否可用
2. 检查依赖版本
3. 运行单个测试定位问题：

```bash
pytest tests/test_matmul.py::TestMatmulBasic::test_square_matrix -v
```

## 其他问题

### Q: 如何贡献代码？

**A:** 参考 [CONTRIBUTING.md](../CONTRIBUTING.md)

### Q: 在哪里报告 Bug？

**A:** 在 GitHub Issues 中报告，请包含：
- 环境信息 (Python, PyTorch, Triton, CUDA 版本)
- 复现步骤
- 错误信息

### Q: 有中文社区吗？

**A:** 欢迎在 GitHub Discussions 中用中文讨论！

# API 参考文档

本文档提供 DIY FlashAttention 的完整 API 参考。

## 目录

- [Kernels](#kernels)
  - [triton_matmul](#triton_matmul)
  - [flash_attention](#flash_attention)
  - [reference_attention](#reference_attention)
- [GPU 检测](#gpu-检测)
  - [detect_gpu](#detect_gpu)
  - [GPUCapabilities](#gpucapabilities)
  - [GPUArch](#gpuarch)
- [Benchmark 工具](#benchmark-工具)
  - [BenchmarkRunner](#benchmarkrunner)
  - [BenchmarkResult](#benchmarkresult)
  - [benchmark_fn](#benchmark_fn)
- [验证工具](#验证工具)
  - [validate_matmul](#validate_matmul)
  - [validate_attention](#validate_attention)
- [现代特性检测](#现代特性检测)
  - [check_hopper_features](#check_hopper_features)
  - [AdaptiveKernelSelector](#adaptivekernelselector)

---

## Kernels

### `triton_matmul`

高性能 Triton 矩阵乘法，支持 autotune 和多种数据类型。

```python
from kernels import triton_matmul

def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    使用 Triton 执行矩阵乘法 C = A @ B。

    参数:
        a (torch.Tensor): 输入矩阵 A，形状 (M, K)。
            支持数据类型: float16, float32, bfloat16。
            必须是 2D CUDA tensor。
        
        b (torch.Tensor): 输入矩阵 B，形状 (K, N)。
            支持数据类型: float16, float32, bfloat16。
            必须是 2D CUDA tensor，且与 a 同 dtype、同 device。
        
        block_m (int, optional): M 维度的 block size。
            如果指定，必须同时指定 block_n 和 block_k。
            默认使用 autotune 自动选择。
        
        block_n (int, optional): N 维度的 block size。
        
        block_k (int, optional): K 维度的 block size。
        
        use_autotune (bool): 是否使用 autotune。
            默认 True。仅在未指定 block size 时生效。

    返回:
        torch.Tensor: 输出矩阵 C，形状 (M, N)。
            - float16/bfloat16 输入 → 保持原 dtype
            - float32 输入 → 输出 float16（内部转换为 float16 计算）

    抛出:
        ValueError: 
            - 输入不是 2D tensor
            - 输入不是 CUDA tensor
            - 输入不在同一 device
            - 维度不兼容 (A.shape[1] != B.shape[0])
            - block size 非法 (非正数或超过矩阵维度)
        
        TypeError:
            - 输入 dtype 不受支持
            - 输入 dtype 不一致

    示例:
        基本用法::

            import torch
            from kernels import triton_matmul

            # 创建输入矩阵
            a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
            b = torch.randn(512, 2048, device="cuda", dtype=torch.float16)

            # 使用 autotune (推荐)
            c = triton_matmul(a, b)

        指定 block size::

            # 手动指定 block size
            c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)

        不同数据类型::

            # BF16 支持
            a_bf16 = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
            b_bf16 = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
            c_bf16 = triton_matmul(a_bf16, b_bf16)  # 输出也是 bfloat16

    注意:
        - 输入会被自动转换为连续内存布局
        - float32 输入会转换为 float16 计算，输出为 float16
        - 对于小矩阵 (< 64)，使用 PyTorch 可能更快

    性能提示:
        - 使用 autotune 可获得最佳性能
        - 大矩阵 (> 2048) 性能优势更明显
        - 确保输入已在 GPU 上，避免 CPU-GPU 数据传输
    """
```

---

### `flash_attention`

FlashAttention 前向实现，O(N) 内存复杂度。

```python
from kernels import flash_attention

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention forward pass，实现高效的注意力计算。

    计算: softmax(Q @ K^T / sqrt(d)) @ V

    参数:
        q (torch.Tensor): Query tensor。
            形状: (batch, heads, seq_len, head_dim) 或 (batch*heads, seq_len, head_dim)。
            支持数据类型: float16, float32, bfloat16。
            必须是 CUDA tensor。
        
        k (torch.Tensor): Key tensor。形状和 dtype 必须与 q 相同。
        
        v (torch.Tensor): Value tensor。形状和 dtype 必须与 q 相同。
        
        causal (bool): 是否使用因果 masking（用于自回归模型）。
            默认 False。
            当 True 时，位置 i 只能关注位置 <= i 的 token。
        
        sm_scale (float, optional): Softmax 缩放因子。
            默认 1 / sqrt(head_dim)。
            对于标准 attention，建议使用默认值。
        
        seq_lens (torch.Tensor, optional): 每个样本的有效序列长度。
            形状: (batch,)。
            数据类型: int32。
            用于处理变长序列，超过有效长度的位置输出为 0。
            必须满足: 0 < seq_lens[i] <= seq_len。

    返回:
        torch.Tensor: Attention 输出，形状与输入 q 相同。
            内部使用 float16 计算。

    抛出:
        ValueError:
            - 输入不是 3D 或 4D tensor
            - Q, K, V 形状不一致
            - 输入不是 CUDA tensor
            - 输入不在同一 device
            - head_dim 不是 32 或 64
            - seq_lens 形状或值非法
        
        TypeError:
            - 输入 dtype 不受支持
            - 输入 dtype 不一致

    示例:
        基本用法::

            import torch
            from kernels import flash_attention

            # 创建输入 (batch=2, heads=8, seq_len=512, head_dim=64)
            q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)

            # 非因果 attention
            out = flash_attention(q, k, v)

        因果 attention::

            # 用于自回归模型 (如 GPT)
            out = flash_attention(q, k, v, causal=True)

        变长序列::

            # 处理不同长度的序列
            seq_lens = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
            out = flash_attention(q, k, v, seq_lens=seq_lens)

        3D 输入::

            # 3D 输入: (batch*heads, seq_len, head_dim)
            q_3d = torch.randn(16, 512, 64, device="cuda", dtype=torch.float16)
            k_3d = torch.randn(16, 512, 64, device="cuda", dtype=torch.float16)
            v_3d = torch.randn(16, 512, 64, device="cuda", dtype=torch.float16)
            out_3d = flash_attention(q_3d, k_3d, v_3d)

    注意:
        - 当前仅支持 head_dim = 32 或 64
        - 非 float16 输入会转换为 float16 计算
        - 内存复杂度 O(N)，相比标准 attention 的 O(N²)

    性能提示:
        - 长序列 (> 512) 内存节省更明显
        - 启用 causal 可减少约一半计算量
        - 建议使用 batch > 1 以充分利用 GPU
    """
```

---

### `reference_attention`

PyTorch 参考实现，用于验证正确性。

```python
from kernels.flash_attn import reference_attention

def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    标准 attention 实现，用于验证 FlashAttention 的正确性。

    参数:
        q, k, v: 输入 tensor，任意形状兼容 torch.matmul
        causal: 是否使用因果 masking

    返回:
        torch.Tensor: Attention 输出

    示例:
        验证 FlashAttention 正确性::

            import torch
            from kernels import flash_attention
            from kernels.flash_attn import reference_attention

            q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)

            flash_out = flash_attention(q, k, v, causal=True)
            ref_out = reference_attention(q, k, v, causal=True)

            assert torch.allclose(flash_out, ref_out, rtol=1e-2, atol=1e-2)
    """
```

---

## GPU 检测

### `detect_gpu`

检测 GPU 能力和特性。

```python
from utils import detect_gpu

def detect_gpu(device_id: int = 0) -> GPUCapabilities:
    """
    检测指定 GPU 的能力。

    参数:
        device_id (int): CUDA 设备 ID。默认 0。

    返回:
        GPUCapabilities: GPU 能力信息对象。

    抛出:
        RuntimeError: CUDA 不可用。

    示例:
        ::

            from utils import detect_gpu, print_gpu_info

            caps = detect_gpu()
            print(f"GPU: {caps.name}")
            print(f"架构: {caps.arch.value}")
            print(f"计算能力: {caps.compute_capability}")
            print(f"显存: {caps.total_memory_gb:.1f} GB")
            print(f"TMA 支持: {caps.has_tma}")
            print(f"FP8 支持: {caps.has_fp8}")
    """
```

---

### `GPUCapabilities`

GPU 能力信息数据类。

```python
from utils import GPUCapabilities
from dataclasses import dataclass

@dataclass
class GPUCapabilities:
    """GPU 能力信息。"""

    name: str
        # GPU 名称，如 "NVIDIA GeForce RTX 4090"

    arch: GPUArch
        # GPU 架构枚举值

    compute_capability: tuple[int, int]
        # 计算能力，如 (8, 9) 表示 SM 89

    has_tma: bool
        # 是否支持 Tensor Memory Accelerator (Hopper+)

    has_fp8: bool
        # 是否支持 FP8 数据类型 (Hopper+)

    has_warpgroup_mma: bool
        # 是否支持 Warpgroup MMA (Hopper+)

    sram_per_sm: int
        # 每个 SM 的共享内存大小 (bytes)

    num_sms: int
        # SM 数量

    total_memory_gb: float
        # 总显存大小 (GB)
```

---

### `GPUArch`

GPU 架构枚举。

```python
from utils import GPUArch

class GPUArch(Enum):
    """GPU 架构枚举。"""
    
    VOLTA = "sm_70"      # V100
    TURING = "sm_75"     # RTX 20xx
    AMPERE = "sm_80"     # A100, RTX 30xx
    ADA = "sm_89"        # RTX 40xx
    HOPPER = "sm_90"     # H100
    BLACKWELL = "sm_100" # B100/B200
    UNKNOWN = "unknown"
```

---

## Benchmark 工具

### `BenchmarkRunner`

Benchmark 运行器。

```python
from utils import BenchmarkRunner

class BenchmarkRunner:
    """
    运行和管理 benchmark。
    
    参数:
        device (str): 运行设备。默认 "cuda"。
        warmup (int): 预热迭代次数。默认 25。
        rep (int): 重复测量次数。默认 100。
    
    示例:
        矩阵乘法 benchmark::
        
            from utils import BenchmarkRunner
            from kernels import triton_matmul

            runner = BenchmarkRunner(warmup=10, rep=50)
            results = runner.benchmark_matmul(
                triton_matmul,
                sizes=[(1024, 1024, 1024), (2048, 2048, 2048)],
            )
            runner.print_comparison_table(results)

        FlashAttention benchmark::

            results = runner.benchmark_attention(
                flash_attention,
                seq_lengths=[512, 1024, 2048],
                batch_size=4,
                num_heads=8,
                head_dim=64,
            )
    """

    def benchmark_matmul(
        self,
        triton_fn: Callable,
        sizes: list[tuple[int, int, int]],
        block_configs: Optional[list[dict]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """
        运行矩阵乘法 benchmark。
        
        参数:
            triton_fn: Triton matmul 函数
            sizes: 矩阵大小列表 [(M, N, K), ...]
            block_configs: 可选的 block size 配置列表
            dtype: 数据类型
        
        返回:
            BenchmarkResult 列表
        """

    def benchmark_attention(
        self,
        flash_fn: Callable,
        seq_lengths: list[int],
        batch_size: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        causal: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        """
        运行 FlashAttention benchmark。
        """

    def print_comparison_table(
        self,
        results: Optional[list[BenchmarkResult]] = None,
        title: str = "Benchmark Results",
    ) -> None:
        """打印格式化的比较表格。"""

    def clear_results(self) -> None:
        """清除已存储的结果。"""
```

---

### `BenchmarkResult`

Benchmark 结果数据类。

```python
from utils import BenchmarkResult
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """单个 benchmark 结果。"""

    name: str
        # 实现名称，如 "PyTorch", "Triton", "FlashAttention"

    size: tuple
        # 问题大小

    time_ms: float
        # 执行时间 (毫秒)

    tflops: float
        # 计算吞吐量 (TFLOPS)

    memory_mb: float = 0.0
        # 峰值内存使用 (MB)

    block_config: Optional[dict] = None
        # Block size 配置 (如果手动指定)
```

---

### `benchmark_fn`

单函数 benchmark 工具。

```python
from utils.benchmark import benchmark_fn

def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 25,
    rep: int = 100,
    quantiles: Optional[list[float]] = None,
    **kwargs,
) -> tuple[float, float, float]:
    """
    对单个函数进行 benchmark。
    
    参数:
        fn: 要测试的函数
        *args: 传递给 fn 的位置参数
        warmup: 预热次数
        rep: 测量次数
        quantiles: 分位数列表
        **kwargs: 传递给 fn 的关键字参数
    
    返回:
        tuple[float, float, float]: (median_ms, p20_ms, p80_ms)
    
    示例:
        ::

            from utils.benchmark import benchmark_fn
            from kernels import triton_matmul

            a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

            median_ms, p20_ms, p80_ms = benchmark_fn(triton_matmul, a, b)
            print(f"Median: {median_ms:.3f} ms")
    """
```

---

## 验证工具

### `validate_matmul`

验证矩阵乘法正确性。

```python
from utils import validate_matmul

def validate_matmul(
    triton_fn: Callable,
    m: int,
    n: int,
    k: int,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[bool, float]:
    """
    验证 Triton matmul 与 PyTorch 参考实现的一致性。
    
    参数:
        triton_fn: Triton matmul 函数
        m, n, k: 矩阵维度
        rtol: 相对容差
        atol: 绝对容差
        dtype: 数据类型
        device: 设备
        verbose: 是否打印详细信息
    
    返回:
        tuple[bool, float]: (是否通过验证, 最大差异)
    
    示例:
        ::

            from utils import validate_matmul
            from kernels import triton_matmul

            is_valid, max_diff = validate_matmul(
                triton_matmul, m=1024, n=1024, k=1024, verbose=True
            )
            print(f"验证{'通过' if is_valid else '失败'}, 最大差异: {max_diff:.2e}")
    """
```

---

### `validate_attention`

验证 FlashAttention 正确性。

```python
from utils import validate_attention

def validate_attention(
    flash_fn: Callable,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[bool, float]:
    """
    验证 FlashAttention 与 PyTorch SDPA 的一致性。
    
    参数:
        flash_fn: FlashAttention 函数
        batch: 批次大小
        heads: 注意力头数
        seq_len: 序列长度
        head_dim: 每个头的维度
        causal: 是否使用因果 masking
        rtol: 相对容差
        atol: 绝对容差
        dtype: 数据类型
        device: 设备
        verbose: 是否打印详细信息
    
    返回:
        tuple[bool, float]: (是否通过验证, 最大差异)
    
    示例:
        ::

            from utils import validate_attention
            from kernels import flash_attention

            # 验证非因果 attention
            is_valid, max_diff = validate_attention(
                flash_attention, batch=2, heads=8, seq_len=512, head_dim=64
            )

            # 验证因果 attention
            is_valid, max_diff = validate_attention(
                flash_attention, batch=2, heads=8, seq_len=512, head_dim=64, causal=True
            )
    """
```

---

## 现代特性检测

### `check_hopper_features`

检测 Hopper+ GPU 特性。

```python
from kernels import check_hopper_features

def check_hopper_features() -> dict[str, Any]:
    """
    检测当前 GPU 的 Hopper+ 特性支持情况。
    
    返回:
        dict: 包含以下键:
            - tma_available (bool): TMA 支持
            - fp8_available (bool): FP8 支持
            - wgmma_available (bool): Warpgroup MMA 支持
            - arch (str): 架构名称
            - compute_capability (tuple): 计算能力
    
    示例:
        ::

            from kernels import check_hopper_features

            features = check_hopper_features()
            print(f"TMA: {features['tma_available']}")
            print(f"FP8: {features['fp8_available']}")
            print(f"架构: {features['arch']}")
    """
```

---

### `AdaptiveKernelSelector`

自适应 kernel 选择器。

```python
from kernels import AdaptiveKernelSelector

class AdaptiveKernelSelector:
    """
    根据 GPU 架构选择最优 kernel 实现。
    
    示例:
        ::

            from kernels import AdaptiveKernelSelector

            selector = AdaptiveKernelSelector()

            # 获取最优配置
            matmul_config = selector.get_matmul_config()
            attention_config = selector.get_attention_config()

            # 获取最优 kernel
            matmul_fn = selector.select_matmul_kernel()
            attention_fn = selector.select_attention_kernel()
    """

    def get_matmul_config(self) -> dict[str, Any]:
        """获取最优 matmul 配置。"""

    def get_attention_config(self) -> dict[str, Any]:
        """获取最优 attention 配置。"""

    def select_matmul_kernel(self) -> Callable:
        """选择最优 matmul kernel。"""

    def select_attention_kernel(self) -> Callable:
        """选择最优 attention kernel。"""
```

---

## 快捷函数

```python
from kernels import (
    get_matmul_config,      # 获取最优 matmul 配置
    get_attention_config,   # 获取最优 attention 配置
    get_optimal_matmul,     # 获取最优 matmul 函数
    get_optimal_attention,  # 获取最优 attention 函数
    supports_fp8,           # 检测 FP8 支持
)
```

---

## 错误处理

### 常见错误及解决方案

| 错误类型 | 错误信息 | 原因 | 解决方案 |
|---------|---------|------|---------|
| `ValueError` | `Expected 2D tensors` | matmul 输入不是 2D | 使用 `.view()` 或 `.reshape()` |
| `ValueError` | `Incompatible dimensions` | 矩阵维度不匹配 | 检查 `A.shape[1] == B.shape[0]` |
| `ValueError` | `CUDA tensors required` | 输入在 CPU 上 | 使用 `.to("cuda")` 或 `.cuda()` |
| `ValueError` | `Expected 3D or 4D tensors` | attention 输入维度错误 | 检查输入形状 |
| `ValueError` | `Unsupported head_dim` | head_dim 不是 32/64 | 使用 head_dim=32 或 64 |
| `TypeError` | `Unsupported dtype` | 使用了不支持的 dtype | 使用 float16/bfloat16/float32 |
| `TypeError` | `dtypes must match` | Q, K, V dtype 不一致 | 确保相同 dtype |
| `ModuleNotFoundError` | `triton is required` | Triton 未安装 | `pip install triton` |

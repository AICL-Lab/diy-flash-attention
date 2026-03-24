# API 参考文档

## Kernels

### `triton_matmul`

高性能 Triton 矩阵乘法。

```python
from kernels import triton_matmul

def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    使用 Triton 执行矩阵乘法 C = A @ B。

    参数:
        a: 输入矩阵 A，形状 (M, K)，支持 float16 / float32 / bfloat16
        b: 输入矩阵 B，形状 (K, N)，支持 float16 / float32 / bfloat16
        block_m: M 维度的 block size（可选）
        block_n: N 维度的 block size（可选）
        block_k: K 维度的 block size（可选）
        use_autotune: 是否使用 autotune；若未完整指定 block size，则默认启用

    返回:
        输出矩阵 C，形状 (M, N)。
        - float16 / bfloat16 输入保留对应计算 dtype
        - float32 输入会下转为 float16 计算，因此输出为 float16

    说明:
        - 输入必须是 2D CUDA tensor
        - 两个输入必须位于同一 device，且 dtype 必须一致
    """
```

### `triton_matmul_fp32`

```python
from kernels.matmul import triton_matmul_fp32

def triton_matmul_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    以 float16 运行 Triton kernel，再将结果转换为 float32 输出。

    注意：这不是“真 FP32 kernel 计算”，仅用于更方便地与 PyTorch reference 做比较。
    """
```

### `flash_attention`

FlashAttention 前向实现，支持 causal masking 和可变有效长度。

```python
from kernels import flash_attention

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    seq_lens: torch.Tensor = None,
) -> torch.Tensor:
    """
    FlashAttention forward pass。

    参数:
        q: Query 张量，形状 (batch, heads, seq_len, head_dim) 或 (batch*heads, seq_len, head_dim)
        k: Key 张量，形状与 q 相同
        v: Value 张量，形状与 q 相同
        causal: 是否使用因果 masking
        sm_scale: Softmax 缩放因子，默认 1/sqrt(head_dim)
        seq_lens: 可选，形状 (batch,) 的 int32 张量，表示每个样本的有效序列长度

    返回:
        输出张量，形状与输入 q 保持一致，kernel 使用 float16 计算/输出

    说明:
        - 当前仅支持 head_dim 为 32 或 64
        - q / k / v 必须是同 dtype、同 device 的 CUDA tensor
        - 支持输入 dtype：float16 / float32 / bfloat16
        - 非 float16 输入会统一转换为 float16 执行
        - 当 seq_lens 小于 seq_len 时，超过有效长度的位置输出为 0
    """
```

## Utils

### GPU 检测

```python
from utils import detect_gpu, GPUCapabilities, GPUArch, print_gpu_info

def detect_gpu(device_id: int = 0) -> GPUCapabilities:
    """
    检测当前 GPU 的能力。
    """

@dataclass
class GPUCapabilities:
    name: str
    arch: GPUArch
    compute_capability: tuple[int, int]
    has_tma: bool
    has_fp8: bool
    has_warpgroup_mma: bool
    sram_per_sm: int
    num_sms: int
    total_memory_gb: float

class GPUArch(Enum):
    VOLTA = "sm_70"
    TURING = "sm_75"
    AMPERE = "sm_80"
    ADA = "sm_89"
    HOPPER = "sm_90"
    BLACKWELL = "sm_100"
    UNKNOWN = "unknown"

def print_gpu_info(caps: GPUCapabilities | None = None) -> None:
    """打印 GPU 信息。"""
```

### Benchmark 工具

```python
from utils.benchmark import BenchmarkResult, BenchmarkRunner, benchmark_fn

@dataclass
class BenchmarkResult:
    name: str
    size: tuple
    time_ms: float
    tflops: float
    memory_mb: float = 0.0
    block_config: dict | None = None


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 25,
    rep: int = 100,
    quantiles: list[float] | None = None,
    **kwargs,
) -> tuple[float, float, float]:
    """
    返回 `(median_ms, p20_ms, p80_ms)`。
    默认 quantiles 为 [0.5, 0.2, 0.8]。
    """


class BenchmarkRunner:
    def __init__(self, device: str = "cuda", warmup: int = 25, rep: int = 100):
        ...

    def benchmark_matmul(
        self,
        triton_fn: Callable,
        sizes: list[tuple[int, int, int]],
        block_configs: list[dict] | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> list[BenchmarkResult]:
        ...

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
        ...

    def print_comparison_table(
        self,
        results: list[BenchmarkResult] | None = None,
        title: str = "Benchmark Results",
    ) -> None:
        ...
```

### 验证工具

```python
from utils import validate_matmul, validate_attention

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
    返回 `(is_valid, max_diff)`。
    """


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
    返回 `(is_valid, max_diff)`。
    """
```

## 常见用法

### 基础矩阵乘法

```python
import torch
from kernels import triton_matmul
from utils import validate_matmul

ok, max_diff = validate_matmul(triton_matmul, m=1024, n=1024, k=1024)
print(ok, max_diff)
```

### FlashAttention

```python
import torch
from kernels import flash_attention
from utils import validate_attention

q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)

ok, max_diff = validate_attention(flash_attention, batch=2, heads=8, seq_len=512, head_dim=64)
print(ok, max_diff)
```

### 架构自适应与现代 CUDA 特性

```python
from kernels import (
    AdaptiveKernelSelector,
    check_hopper_features,
    get_attention_config,
    get_matmul_config,
    get_optimal_attention,
    get_optimal_matmul,
    supports_fp8,
)

features = check_hopper_features()
print(features)

matmul_config = get_matmul_config()
attention_config = get_attention_config()
print(matmul_config)
print(attention_config)

selector = AdaptiveKernelSelector()
matmul_fn = selector.select_matmul_kernel()
attention_fn = selector.select_attention_kernel()

print(callable(matmul_fn), callable(attention_fn))
print(supports_fp8())
```

## 错误处理

- `triton_matmul` 会在以下情况下抛出异常：
  - 输入不是 2D tensor
  - 输入不是 CUDA tensor
  - 输入不在同一 device
  - 输入 dtype 不受支持或不一致
  - 维度不兼容
  - block size 非法

- `flash_attention` 会在以下情况下抛出异常：
  - 输入不是 3D/4D tensor
  - q/k/v 形状不一致
  - q/k/v 不是 CUDA tensor
  - q/k/v 不在同一 device
  - q/k/v dtype 不受支持或不一致
  - head_dim 不在支持范围内
  - `seq_lens` 长度或取值非法

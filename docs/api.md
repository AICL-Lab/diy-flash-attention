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
    block_k: int = None
) -> torch.Tensor:
    """
    使用 Triton 执行矩阵乘法 C = A @ B
    
    参数:
        a: 输入矩阵 A，形状 (M, K)
        b: 输入矩阵 B，形状 (K, N)
        block_m: M 维度的 block size (可选，默认使用 autotune)
        block_n: N 维度的 block size (可选)
        block_k: K 维度的 block size (可选)
    
    返回:
        输出矩阵 C，形状 (M, N)
    
    示例:
        >>> a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
        >>> b = torch.randn(512, 768, device="cuda", dtype=torch.float16)
        >>> c = triton_matmul(a, b)  # 使用 autotune
        >>> c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)  # 手动指定
    """
```

### `flash_attention`

FlashAttention 实现，支持 causal masking。

```python
from kernels import flash_attention

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None
) -> torch.Tensor:
    """
    FlashAttention forward pass
    
    参数:
        q: Query 张量，形状 (batch, heads, seq_len, head_dim)
        k: Key 张量，形状 (batch, heads, seq_len, head_dim)
        v: Value 张量，形状 (batch, heads, seq_len, head_dim)
        causal: 是否使用因果 masking (用于自回归模型)
        sm_scale: Softmax 缩放因子 (默认 1/sqrt(head_dim))
    
    返回:
        输出张量，形状 (batch, heads, seq_len, head_dim)
    
    示例:
        >>> q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> out = flash_attention(q, k, v, causal=True)
    """
```

## Utils

### GPU 检测

```python
from utils import detect_gpu, GPUCapabilities, GPUArch, print_gpu_info

def detect_gpu() -> GPUCapabilities:
    """
    检测当前 GPU 的能力
    
    返回:
        GPUCapabilities 对象，包含:
        - name: GPU 名称
        - arch: GPU 架构 (GPUArch 枚举)
        - compute_capability: 计算能力 (如 (8, 0))
        - supports_tma: 是否支持 TMA
        - supports_fp8: 是否支持 FP8
        - supports_wgmma: 是否支持 Warpgroup MMA
    """

class GPUArch(Enum):
    """GPU 架构枚举"""
    UNKNOWN = "unknown"
    AMPERE = "ampere"      # SM80, SM86, SM87
    HOPPER = "hopper"      # SM90
    BLACKWELL = "blackwell"  # SM100+

def print_gpu_info(caps: GPUCapabilities) -> None:
    """打印 GPU 信息"""
```

### Benchmark 工具

```python
from utils import BenchmarkResult, BenchmarkRunner

@dataclass
class BenchmarkResult:
    """Benchmark 结果"""
    name: str
    time_ms: float
    tflops: float = None
    memory_mb: float = None

class BenchmarkRunner:
    """Benchmark 运行器"""
    
    def __init__(self, warmup: int = 10, repeat: int = 100):
        """
        参数:
            warmup: 预热次数
            repeat: 重复次数
        """
    
    def benchmark(
        self,
        fn: Callable,
        name: str,
        flops: int = None
    ) -> BenchmarkResult:
        """
        运行 benchmark
        
        参数:
            fn: 要测试的函数
            name: 测试名称
            flops: 浮点运算次数 (用于计算 TFLOPS)
        
        返回:
            BenchmarkResult 对象
        """
    
    def print_results(self, results: List[BenchmarkResult]) -> None:
        """打印格式化的结果表格"""
```

### 验证工具

```python
from utils import validate_matmul, validate_attention

def validate_matmul(
    triton_result: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Tuple[bool, str]:
    """
    验证矩阵乘法结果
    
    参数:
        triton_result: Triton kernel 的输出
        reference: 参考实现的输出 (如 torch.matmul)
        rtol: 相对容差
        atol: 绝对容差
    
    返回:
        (is_valid, message) 元组
    """

def validate_attention(
    flash_result: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Tuple[bool, str]:
    """
    验证 attention 结果
    
    参数:
        flash_result: FlashAttention 的输出
        reference: 参考实现的输出
        rtol: 相对容差
        atol: 绝对容差
    
    返回:
        (is_valid, message) 元组
    """
```

## 常见用法

### 基础矩阵乘法

```python
import torch
from kernels import triton_matmul

# 创建输入
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# 使用 autotune (推荐)
c = triton_matmul(a, b)

# 手动指定 block size
c = triton_matmul(a, b, block_m=128, block_n=256, block_k=64)

# 验证正确性
from utils import validate_matmul
ref = torch.matmul(a, b)
is_valid, msg = validate_matmul(c, ref)
print(f"验证: {msg}")
```

### FlashAttention

```python
import torch
from kernels import flash_attention

# 创建输入 (batch=2, heads=8, seq_len=512, head_dim=64)
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)

# 标准 attention
out = flash_attention(q, k, v)

# Causal attention (用于 GPT 等自回归模型)
out_causal = flash_attention(q, k, v, causal=True)

# 验证正确性
from utils import validate_attention
import torch.nn.functional as F
ref = F.scaled_dot_product_attention(q, k, v)
is_valid, msg = validate_attention(out, ref)
print(f"验证: {msg}")
```

### Benchmark

```python
from utils import BenchmarkRunner
import torch
from kernels import triton_matmul

runner = BenchmarkRunner(warmup=10, repeat=100)

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

# 计算 FLOPS: 2 * M * N * K
flops = 2 * 4096 * 4096 * 4096

results = [
    runner.benchmark(lambda: triton_matmul(a, b), "Triton", flops),
    runner.benchmark(lambda: torch.matmul(a, b), "PyTorch", flops),
]

runner.print_results(results)
```

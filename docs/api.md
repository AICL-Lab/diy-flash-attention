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
    sm_scale: float = None,
    seq_lens: torch.Tensor = None
) -> torch.Tensor:
    """
    FlashAttention forward pass
    
    参数:
        q: Query 张量，形状 (batch, heads, seq_len, head_dim)
        k: Key 张量，形状 (batch, heads, seq_len, head_dim)
        v: Value 张量，形状 (batch, heads, seq_len, head_dim)
        causal: 是否使用因果 masking (用于自回归模型)
        sm_scale: Softmax 缩放因子 (默认 1/sqrt(head_dim))
        seq_lens: 可选，形状 (batch,) 的 int32 张量，表示每个样本的有效序列长度
    
    返回:
        输出张量，形状 (batch, heads, seq_len, head_dim)

    说明:
        - 当前仅支持 head_dim 为 32 或 64
        - 当 seq_lens 小于 seq_len 时，超过有效长度的位置输出为 0
    
    示例:
        >>> q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
        >>> out = flash_attention(q, k, v, causal=True)
        >>> seq_lens = torch.tensor([256, 512], device="cuda", dtype=torch.int32)
        >>> out = flash_attention(q, k, v, seq_lens=seq_lens)
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


## 现代 CUDA 特性

### 架构自适应

```python
from kernels import (
    get_optimal_matmul,
    get_optimal_attention,
    get_matmul_config,
    get_attention_config,
    check_hopper_features,
    AdaptiveKernelSelector,
)

# 检查 Hopper+ 特性支持
features = check_hopper_features()
print(f"TMA 支持: {features['tma_available']}")
print(f"FP8 支持: {features['fp8_available']}")
print(f"Warpgroup MMA: {features['wgmma_available']}")

# 获取当前 GPU 的最优配置
matmul_config = get_matmul_config()
print(f"最优 MatMul 配置: {matmul_config}")

attention_config = get_attention_config()
print(f"最优 Attention 配置: {attention_config}")

# 获取最优实现
optimal_matmul = get_optimal_matmul()
optimal_attention = get_optimal_attention()

# 使用 AdaptiveKernelSelector
selector = AdaptiveKernelSelector()
matmul_fn = selector.select_matmul_kernel()
attention_fn = selector.select_attention_kernel()
```

### FP8 支持

```python
from kernels.modern_features import supports_fp8, to_fp8_e4m3, to_fp8_e5m2

# 检查 FP8 支持
if supports_fp8():
    print("当前 GPU 支持 FP8!")
    
    # 转换为 FP8 E4M3 (适合权重和激活)
    tensor_fp8 = to_fp8_e4m3(tensor)
    
    # 转换为 FP8 E5M2 (适合梯度)
    grad_fp8 = to_fp8_e5m2(gradient)
else:
    print("FP8 不可用，将使用 FP16 fallback")
```

### 打印特性状态

```python
from kernels.modern_features import print_feature_status, benchmark_feature_impact

# 打印当前 GPU 的特性支持状态
print_feature_status()

# 运行特性影响 benchmark
benchmark_feature_impact()
```

## 测试工具

### 属性测试 (Property-Based Testing)

项目使用 Hypothesis 进行属性测试，验证以下正确性属性：

1. **Property 1: 矩阵乘法正确性** - Triton matmul 结果应与 torch.matmul 一致
2. **Property 2: Block Size 不变性** - 不同 block size 应产生相同结果
3. **Property 3: FlashAttention 正确性** - FlashAttention 应与参考实现一致
4. **Property 4: Causal Masking 正确性** - 因果 masking 应正确阻止未来信息
5. **Property 5: 内存缩放** - 内存使用应为 O(N) 而非 O(N²)

```python
# 运行属性测试
pytest tests/test_properties.py -v

# 运行特定属性测试
pytest tests/test_properties.py::TestMatmulCorrectnessProperty -v
```

### 验证工具扩展

```python
from utils.validation import (
    validate_matmul_edge_cases,
    validate_attention_edge_cases,
)

# 验证矩阵乘法边界情况
all_passed, results = validate_matmul_edge_cases(triton_matmul, verbose=True)

# 验证 attention 边界情况
all_passed, results = validate_attention_edge_cases(flash_attention, verbose=True)
```

## 错误处理

### 矩阵乘法错误

```python
from kernels import triton_matmul

# 维度不兼容
try:
    a = torch.randn(64, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)  # K 不匹配
    triton_matmul(a, b)
except ValueError as e:
    print(f"错误: {e}")  # "Incompatible matrix dimensions..."

# 无效的 block size
try:
    triton_matmul(a, b, block_m=0, block_n=64, block_k=32)
except ValueError as e:
    print(f"错误: {e}")  # "Block sizes must be positive..."
```

### FlashAttention 错误

```python
from kernels import flash_attention

# 形状不匹配
try:
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16)  # seq_len 不匹配
    v = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
    flash_attention(q, k, v)
except ValueError as e:
    print(f"错误: {e}")  # "Q, K, V shapes must match..."

# 无效维度
try:
    q = torch.randn(128, 64, device="cuda", dtype=torch.float16)  # 2D 而非 3D/4D
    flash_attention(q, q, q)
except ValueError as e:
    print(f"错误: {e}")  # "Expected 3D or 4D tensors..."
```

## 性能调优指南

### 选择最优 Block Size

```python
from kernels import triton_matmul
from utils import BenchmarkRunner

# 测试不同 block size 配置
configs = [
    (32, 32, 32),
    (64, 64, 32),
    (128, 128, 32),
    (128, 256, 64),
]

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

for bm, bn, bk in configs:
    result = triton_matmul(a, b, block_m=bm, block_n=bn, block_k=bk)
    # 测量性能...
```

### 内存优化

```python
# FlashAttention 的内存优势
# 标准 Attention: O(N²) 内存
# FlashAttention: O(N) 内存

# 对于 seq_len=8192:
# 标准: 8192² × 2 bytes = 128 MB (仅 attention matrix)
# Flash: 8192 × 64 × 2 bytes ≈ 1 MB (中间状态)
```

### GPU 架构优化

```python
from utils import detect_gpu, get_optimal_config

caps = detect_gpu()

# 根据 GPU 架构获取最优配置
if caps.arch.value == "sm_90":  # Hopper
    print("使用 Hopper 优化配置")
    config = get_optimal_config(caps, "flash_attention")
elif caps.arch.value >= "sm_80":  # Ampere+
    print("使用 Ampere 优化配置")
    config = get_optimal_config(caps, "flash_attention")
```

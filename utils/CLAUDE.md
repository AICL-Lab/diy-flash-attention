# utils/ - 工具函数

本目录包含 GPU 检测、基准测试、验证、配置管理和性能分析等工具函数。

## 文件结构

```
utils/
├── __init__.py       # 包入口
├── config.py         # 配置常量
├── gpu_detect.py     # GPU 架构检测
├── benchmark.py      # 基准测试工具
├── validation.py     # 正确性验证
├── profiling.py      # GPU 内存分析
└── py.typed          # PEP 561 类型标记
```

## 公开 API

### gpu_detect.py - GPU 架构检测

| 导出 | 类型 | 描述 |
|------|------|------|
| `GPUArch` | Enum | GPU 架构枚举 (Volta, Turing, Ampere, Ada, Hopper, Blackwell) |
| `GPUCapabilities` | Dataclass | GPU 能力信息（架构、SM 数量、内存等） |
| `detect_gpu()` | Function | 检测当前 GPU |
| `get_optimal_config()` | Function | 获取最优内核配置 |
| `print_gpu_info()` | Function | 打印 GPU 信息 |

### benchmark.py - 基准测试

| 导出 | 类型 | 描述 |
|------|------|------|
| `BenchmarkResult` | Dataclass | 基准测试结果（时间、TFLOPS、内存） |
| `BenchmarkRunner` | Class | 基准测试运行器 |
| `benchmark_fn()` | Function | 单函数计时 |
| `calculate_matmul_flops()` | Function | MatMul FLOPs 计算 |
| `calculate_attention_flops()` | Function | Attention FLOPs 计算 |

### validation.py - 正确性验证

| 导出 | 类型 | 描述 |
|------|------|------|
| `validate_matmul()` | Function | MatMul 正确性验证 |
| `validate_attention()` | Function | Attention 正确性验证 |
| `validate_matmul_edge_cases()` | Function | MatMul 边界测试 |
| `validate_attention_edge_cases()` | Function | Attention 边界测试 |

### profiling.py - GPU 内存分析

| 导出 | 类型 | 描述 |
|------|------|------|
| `GPUMemoryProfile` | Dataclass | GPU 内存层级指标 |
| `KernelBenchmark` | Dataclass | 内核性能指标 |
| `get_gpu_memory_hierarchy()` | Function | 获取 GPU 内存层级规格 |
| `estimate_occupancy()` | Function | 估算 GPU occupancy |

### config.py - 配置常量

| 常量 | 描述 |
|------|------|
| `MATMUL_BLOCK_M/N/K` | MatMul 默认块大小 |
| `FLASH_ATTN_BLOCK_M/N` | FlashAttention 默认块大小 |
| `BENCHMARK_WARMUP/REP` | 基准测试默认参数 |

## 支持的 GPU 架构

| 架构 | GPU | 支持级别 |
|------|-----|----------|
| Volta (SM70) | V100 | Basic |
| Turing (SM75) | RTX 20xx | Basic |
| Ampere (SM80) | A100, RTX 30xx | Full |
| Ada (SM89) | RTX 40xx | Full |
| Hopper (SM90) | H100 | TMA, FP8 |
| Blackwell (SM100) | B100/B200 | Latest |

---

**导航**: [← 项目根目录](../CLAUDE.md)

# Changelog

本项目的所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [1.0.0] - 2024-12-31

### 新增
- 🎉 初始版本发布
- ✨ Triton 矩阵乘法 Kernel
  - 支持 autotune 自动选择最优配置
  - 支持手动指定 Block Size
  - 处理非对齐维度
  - L2 Cache 优化 (super-grouping)
- ✨ FlashAttention 实现
  - Online Softmax 算法
  - Causal Masking 支持
  - 多头注意力 (Multi-Head Attention)
  - Batch 处理
  - O(N) 内存复杂度
- 📊 Benchmark 工具
  - TFLOPS 计算
  - 内存使用测量
  - 格式化输出表格
  - 一键生成报告
- 🔧 工具函数
  - GPU 架构检测 (Volta/Turing/Ampere/Ada/Hopper/Blackwell)
  - 数值验证工具
  - 边界情况验证
- 🧪 完整测试套件
  - 单元测试 (pytest)
  - 属性测试 (Hypothesis)
  - 错误处理测试
  - Benchmark 工具测试
- 🚀 现代 CUDA 特性支持
  - TMA 数据加载框架 (Hopper+)
  - FP8 格式支持检测
  - 架构自适应 Kernel 选择
  - 自动 fallback 机制
- 📚 文档
  - 中文教程
  - API 参考文档
  - 贡献指南
  - 示例代码

### 技术细节
- 支持 CUDA 11.0+
- 支持 PyTorch 2.0+
- 支持 Triton 2.1+
- GPU 架构: SM70+ (Volta 及更新)
- 测试框架: pytest + hypothesis

### 项目结构
```
kernels/           - Triton GPU Kernels
├── matmul.py      - 矩阵乘法
├── flash_attn.py  - FlashAttention
└── modern_features.py - 现代特性

utils/             - 工具函数
├── benchmark.py   - Benchmark 工具
├── validation.py  - 验证工具
└── gpu_detect.py  - GPU 检测

tests/             - 测试套件
├── test_matmul.py
├── test_flash.py
├── test_properties.py
├── test_benchmark.py
├── test_validation.py
└── test_error_handling.py

examples/          - 示例代码
├── quick_start.py
├── advanced_usage.py
├── block_size_experiment.py
└── visualize_tiling.py

docs/              - 文档
├── tutorial.md
└── api.md
```

## 未来计划

### [1.1.0] - 计划中
- [ ] FlashAttention Backward Pass
- [ ] TMA 数据加载完整实现 (Hopper+)
- [ ] FP8 矩阵乘法 Kernel
- [ ] 更多 Benchmark 场景
- [ ] 性能分析工具集成

### [1.2.0] - 计划中
- [ ] Multi-Query Attention (MQA) 支持
- [ ] Grouped-Query Attention (GQA) 支持
- [ ] 可变序列长度支持
- [ ] Flash-Decoding 优化

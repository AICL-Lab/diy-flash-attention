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
- ✨ FlashAttention 实现
  - Online Softmax 算法
  - Causal Masking 支持
  - 多头注意力 (Multi-Head Attention)
  - Batch 处理
- 📊 Benchmark 工具
  - TFLOPS 计算
  - 内存使用测量
  - 格式化输出表格
- 🔧 工具函数
  - GPU 架构检测 (Ampere/Hopper/Blackwell)
  - 数值验证工具
- 📚 文档
  - 中文教程
  - 示例代码
  - API 文档

### 技术细节
- 支持 CUDA 11.0+
- 支持 PyTorch 2.0+
- 支持 Triton 2.1+
- GPU 架构: SM80+ (Ampere 及更新)

## 未来计划

### [1.1.0] - 计划中
- [ ] TMA 数据加载 (Hopper+)
- [ ] FP8 支持
- [ ] 架构自适应优化
- [ ] Backward Pass 实现
- [ ] 更多 Benchmark 场景

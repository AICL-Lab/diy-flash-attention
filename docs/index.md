---
layout: home

hero:
  name: DIY FlashAttention
  text: 从零实现 FlashAttention
  tagline: 使用 Python + OpenAI Triton 动手实践 LLM 核心算法
  actions:
    - theme: brand
      text: 教程
      link: /tutorial
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - title: Triton 编程模型
    details: 通过实现矩阵乘法 Kernel 学习 Block 指针运算和 Tiling
    icon: ⚡
  - title: FlashAttention 复现
    details: 实现 LLM 中最核心的注意力机制加速算法，O(N) 内存复杂度
    icon: 🧠
  - title: 性能对比
    details: 通过 Benchmark 量化优化效果，感受 Block Size 对性能的影响
    icon: 📊
  - title: 现代 GPU 支持
    details: 自动检测 GPU 架构，支持 Ampere / Hopper / Blackwell 特性
    icon: 🖥️
  - title: 完整测试覆盖
    details: 单元测试 + 属性测试 (Hypothesis)，确保正确性
    icon: ✅
  - title: 中文文档
    details: 教程、API 参考、速查表、FAQ 全中文编写
    icon: 📖
---
